"""Sawyer hammer environment (hammer-v3)."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from mujoco import mjx

from MTCWorldMJX import mjx_env
from MTCWorldMJX.asset_paths import sawyer_xml_path
from MTCWorldMJX.envs._helpers import as_f32, pack_interleaved_obj_obs, set_free_joint_xyz
from MTCWorldMJX.envs.sawyer_xyz import (
    CURR_OBS_DIM,
    OBS_OBJ_MAX_LEN,
    SawyerXYZConfig,
    SawyerXYZEnv,
)
from MTCWorldMJX.utils import reward as reward_utils


class SawyerHammerEnvV3(SawyerXYZEnv):
    HAMMER_HANDLE_LENGTH = 0.14

    def __init__(self, config: SawyerXYZConfig | None = None):
        super().__init__(config)
        self._hammer_body_id = mjx_env.body_id(self._mj_model, "hammer")
        self._nail_body_id = mjx_env.body_id(self._mj_model, "nail_link")
        self._box_body_id = mjx_env.body_id(self._mj_model, "box")
        self._goal_site_id = mjx_env.site_id(self._mj_model, "goal")
        self._nail_joint_adr = int(self._mj_model.jnt_qposadr[10])
        self._hammer_qposadr = int(self._mj_model.jnt_qposadr[9])
        self._hammer_qveladr = int(self._mj_model.jnt_dofadr[9])

    @property
    def xml_path(self):
        return sawyer_xml_path("sawyer_hammer.xml")

    def hand_init_pos(self) -> jax.Array:
        return as_f32([0.0, 0.4, 0.2])

    def mocap_low(self) -> jax.Array:
        return as_f32([-0.5, 0.40, 0.05])

    def mocap_high(self) -> jax.Array:
        return as_f32([0.5, 1.0, 0.5])

    def random_reset_bounds(self) -> tuple[jax.Array, jax.Array]:
        return as_f32([-0.1, 0.4, 0.0]), as_f32([0.1, 0.5, 0.0])

    def get_obj_pos(self, data: mjx.Data) -> jax.Array:
        hammer = mjx_env.body_xpos(data, jnp.array(self._hammer_body_id))
        nail = mjx_env.body_xpos(data, jnp.array(self._nail_body_id))
        return jnp.concatenate([hammer, nail])

    def get_obj_quat(self, data: mjx.Data) -> jax.Array:
        hammer = mjx_env.body_xquat(data, jnp.array(self._hammer_body_id))
        nail = mjx_env.body_xquat(data, jnp.array(self._nail_body_id))
        return jnp.concatenate([hammer, nail])

    def _get_curr_obs_no_goal(self, data: mjx.Data) -> jax.Array:
        pos_hand = self._hand_pos(data)
        gripper = self._gripper_opening(data)[None]
        obj_pos = self.get_obj_pos(data)
        obj_quat = self.get_obj_quat(data)
        obs_obj = pack_interleaved_obj_obs(obj_pos, obj_quat, max_len=OBS_OBJ_MAX_LEN)
        return jnp.concatenate([pos_hand, gripper, obs_obj])

    def reset_state_from_rand_vec(self, rand_vec: jax.Array) -> dict[str, jax.Array]:
        return {
            "obj_init_pos": rand_vec[:3],
            "goal_pos": as_f32([0.24, 0.74, 0.11]),
            "hammer_init_pos": rand_vec[:3],
            "rand_vec": rand_vec,
        }

    def sample_reset_state(self, rng: jax.Array) -> dict[str, jax.Array]:
        low, high = self.random_reset_bounds()
        vec = jax.random.uniform(rng, shape=low.shape, minval=low, maxval=high)
        return self.reset_state_from_rand_vec(vec)

    def apply_reset_state(self, model, data, reset_state):
        hammer_pos = reset_state["hammer_init_pos"]
        body_pos = model.body_pos.at[self._box_body_id].set(as_f32([0.24, 0.85, 0.0]))
        model = model.replace(body_pos=body_pos)
        data = set_free_joint_xyz(
            data, self._hammer_qposadr, self._hammer_qveladr, hammer_pos, ndim=3
        )
        return model, data

    def _resolve_goal_pos(self, model, data, reset_state):
        del model, reset_state
        return mjx_env.site_xpos(data, jnp.array(self._goal_site_id))

    @staticmethod
    def _reward_quat(hammer_quat: jax.Array) -> jax.Array:
        ideal = jnp.array([1.0, 0.0, 0.0, 0.0])
        error = jnp.linalg.norm(hammer_quat - ideal)
        return jnp.maximum(1.0 - error / 0.4, 0.0)

    def compute_reward(self, data, action, info):
        hand = self._hand_pos(data)
        hammer = mjx_env.body_xpos(data, jnp.array(self._hammer_body_id))
        hammer_head = hammer + jnp.array([0.16, 0.06, 0.0])
        hammer_threshed = hammer.at[0].set(
            jnp.where(
                jnp.abs(hammer[0] - hand[0]) < self.HAMMER_HANDLE_LENGTH / 2.0,
                hand[0],
                hammer[0],
            )
        )
        reward_quat = self._reward_quat(
            mjx_env.body_xquat(data, jnp.array(self._hammer_body_id))
        )
        reward_grab = self._gripper_caging_reward(
            data, action, hammer_threshed, info["obj_init_pos"], info["init_tcp"],
            obj_radius=0.015, pad_success_thresh=0.02, object_reach_radius=0.01,
            xz_thresh=0.01, high_density=True,
        )
        lifted = hammer_head[2] > 0.02
        reward_in_place = 0.1 * lifted.astype(jnp.float32) + 0.9 * reward_utils.tolerance(
            jnp.linalg.norm(info["goal_pos"] - hammer_head),
            bounds=(0.0, 0.02), margin=0.2, sigmoid="long_tail",
        )
        reward = (2.0 * reward_grab + 6.0 * reward_in_place) * reward_quat
        nail_qpos = data.qpos[self._nail_joint_adr]
        success = (nail_qpos > 0.09) & (reward > 5.0)
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
