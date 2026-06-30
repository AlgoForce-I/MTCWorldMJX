"""Sawyer nut disassembly environment (disassemble-v3)."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from mujoco import mjx

from MTCWorldMJX import mjx_env
from MTCWorldMJX.asset_paths import sawyer_xml_path
from MTCWorldMJX.envs._helpers import as_f32, sample_hstack_rand_vec, set_free_joint_xyz
from MTCWorldMJX.envs.sawyer_assembly_peg_v3 import SawyerNutAssemblyEnvV3
from MTCWorldMJX.envs.sawyer_xyz import SawyerXYZConfig
from MTCWorldMJX.utils import reward as reward_utils


class SawyerNutDisassembleEnvV3(SawyerNutAssemblyEnvV3):
    """Lift a nut off a peg."""

    def hand_init_pos(self) -> jax.Array:
        return as_f32([0.0, 0.4, 0.2])

    def random_reset_bounds(self) -> tuple[jax.Array, jax.Array]:
        obj_low = as_f32([0.0, 0.6, 0.025])
        obj_high = as_f32([0.1, 0.75, 0.02501])
        goal_low = as_f32([-0.1, 0.6, 0.1699])
        goal_high = as_f32([0.1, 0.75, 0.1701])
        return jnp.concatenate([obj_low, goal_low]), jnp.concatenate([obj_high, goal_high])

    def reset_state_from_rand_vec(self, rand_vec: jax.Array) -> dict[str, jax.Array]:
        obj_init_pos = rand_vec[:3]
        goal_pos = obj_init_pos + jnp.array([0.0, 0.0, 0.15])
        peg_pos = obj_init_pos + jnp.array([0.0, 0.0, 0.03])
        peg_top = obj_init_pos + jnp.array([0.0, 0.0, 0.08])
        return {
            "obj_init_pos": obj_init_pos,
            "goal_pos": goal_pos,
            "peg_pos": peg_pos,
            "peg_top": peg_top,
            "rand_vec": rand_vec,
        }

    def sample_reset_state(self, rng: jax.Array) -> dict[str, jax.Array]:
        low, high = self.random_reset_bounds()
        vec, _ = sample_hstack_rand_vec(rng, low, high, min_xy_dist=0.1)
        return self.reset_state_from_rand_vec(vec)

    def apply_reset_state(self, model, data, reset_state):
        peg_pos = reset_state["peg_pos"]
        peg_top = reset_state["peg_top"]
        body_pos = model.body_pos.at[self._peg_body_id].set(peg_pos)
        site_pos = model.site_pos.at[self._peg_top_site_id].set(peg_top)
        model = model.replace(body_pos=body_pos, site_pos=site_pos)
        data = set_free_joint_xyz(
            data, self._obj_qposadr, self._obj_qveladr, reset_state["obj_init_pos"], ndim=3
        )
        return model, data

    @staticmethod
    def _reward_pos(wrench_center: jax.Array, target_pos: jax.Array) -> jax.Array:
        pos_error = target_pos + jnp.array([0.0, 0.0, 0.1]) - wrench_center
        lifted = wrench_center[2] > 0.02
        return 0.1 * lifted.astype(jnp.float32) + 0.9 * reward_utils.tolerance(
            jnp.linalg.norm(pos_error), bounds=(0.0, 0.02), margin=0.2, sigmoid="long_tail"
        )

    def __init__(self, config: SawyerXYZConfig | None = None):
        super().__init__(config)
        self._wrench_center_site_id = mjx_env.site_id(self._mj_model, "RoundNut")

    def compute_reward(self, data, action, info) -> tuple[jax.Array, dict[str, jax.Array]]:
        hand = self._hand_pos(data)
        wrench = self.get_obj_pos(data)
        wrench_center = mjx_env.site_xpos(data, jnp.array(self._wrench_center_site_id))
        wrench_threshed = wrench.at[0].set(
            jnp.where(
                jnp.abs(wrench[0] - hand[0]) < self.WRENCH_HANDLE_LENGTH / 2.0,
                hand[0],
                wrench[0],
            )
        )
        reward_quat = self._reward_quat(
            mjx_env.body_xquat(data, jnp.array(self._roundnut_body_id))
        )
        reward_grab = self._gripper_caging_reward(
            data, action, wrench_threshed, info["obj_init_pos"], info["init_tcp"],
            obj_radius=0.015, pad_success_thresh=0.02, object_reach_radius=0.01,
            xz_thresh=0.01, high_density=True,
        )
        reward_in_place = self._reward_pos(wrench_center, info["goal_pos"])
        reward = (2.0 * reward_grab + 6.0 * reward_in_place) * reward_quat
        success = wrench[2] > info["goal_pos"][2]
        reward = jnp.where(success, 10.0, reward)
        metrics = {
            "success": success.astype(jnp.float32),
            "near_object": reward_quat,
            "grasp_success": (reward_grab >= 0.5).astype(jnp.float32),
            "grasp_reward": reward_grab,
            "in_place_reward": reward_in_place,
            "obj_to_target": jnp.array(0.0),
        }
        return reward, metrics
