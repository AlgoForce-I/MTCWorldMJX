"""Sawyer nut assembly environment (assembly-v3)."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from mujoco import mjx

from MTCWorldMJX import mjx_env
from MTCWorldMJX.asset_paths import sawyer_xml_path
from MTCWorldMJX.envs.sawyer_xyz import SawyerXYZConfig, SawyerXYZEnv
from MTCWorldMJX.utils import reward as reward_utils


class SawyerNutAssemblyEnvV3(SawyerXYZEnv):
    """Place a round nut/wrench onto a peg."""

    WRENCH_HANDLE_LENGTH = 0.02

    def __init__(self, config: SawyerXYZConfig | None = None):
        super().__init__(config)
        self._roundnut_site_id = mjx_env.site_id(self._mj_model, "RoundNut-8")
        self._roundnut_body_id = mjx_env.body_id(self._mj_model, "RoundNut")
        self._peg_body_id = mjx_env.body_id(self._mj_model, "peg")
        self._peg_top_site_id = mjx_env.site_id(self._mj_model, "pegTop")
        self._default_goal = jnp.array([0.1, 0.8, 0.1], dtype=jnp.float32)

    @property
    def xml_path(self):
        return sawyer_xml_path("sawyer_assembly_peg.xml")

    def hand_init_pos(self) -> jax.Array:
        return jnp.array([0.0, 0.6, 0.2], dtype=jnp.float32)

    def mocap_low(self) -> jax.Array:
        return jnp.array([-0.5, 0.40, 0.05], dtype=jnp.float32)

    def mocap_high(self) -> jax.Array:
        return jnp.array([0.5, 1.0, 0.5], dtype=jnp.float32)

    def random_reset_bounds(self) -> tuple[jax.Array, jax.Array]:
        obj_low = jnp.array([0.0, 0.6, 0.02], dtype=jnp.float32)
        obj_high = jnp.array([0.0, 0.6, 0.02], dtype=jnp.float32)
        goal_low = jnp.array([-0.1, 0.75, 0.1], dtype=jnp.float32)
        goal_high = jnp.array([0.1, 0.85, 0.1], dtype=jnp.float32)
        return jnp.concatenate([obj_low, goal_low]), jnp.concatenate([obj_high, goal_high])

    def get_obj_pos(self, data: mjx.Data) -> jax.Array:
        return mjx_env.site_xpos(data, jnp.array(self._roundnut_site_id))

    def get_obj_quat(self, data: mjx.Data) -> jax.Array:
        return mjx_env.body_xquat(data, jnp.array(self._roundnut_body_id))

    def reset_state_from_rand_vec(self, rand_vec: jax.Array) -> dict[str, jax.Array]:
        obj_init_pos = rand_vec[:3]
        goal_pos = rand_vec[-3:]
        peg_pos = goal_pos - jnp.array([0.0, 0.0, 0.05])
        return {
            "obj_init_pos": obj_init_pos,
            "goal_pos": goal_pos,
            "peg_pos": peg_pos,
            "rand_vec": rand_vec,
        }

    def sample_reset_state(self, rng: jax.Array) -> dict[str, jax.Array]:
        low, high = self.random_reset_bounds()
        key, subkey = jax.random.split(rng)
        goal_vec = jax.random.uniform(subkey, shape=low.shape, minval=low, maxval=high)

        def cond_fn(carry):
            vec, key = carry
            too_close = jnp.linalg.norm(vec[:2] - vec[-3:-1]) < 0.1
            return too_close

        def body_fn(carry):
            vec, key = carry
            key, subkey = jax.random.split(key)
            vec = jax.random.uniform(subkey, shape=low.shape, minval=low, maxval=high)
            return vec, key

        goal_vec, _ = jax.lax.while_loop(cond_fn, body_fn, (goal_vec, key))

        obj_init_pos = goal_vec[:3]
        goal_pos = goal_vec[-3:]
        peg_pos = goal_pos - jnp.array([0.0, 0.0, 0.05])
        return {
            "obj_init_pos": obj_init_pos,
            "goal_pos": goal_pos,
            "peg_pos": peg_pos,
            "rand_vec": goal_vec,
        }

    def apply_reset_state(
        self,
        model: mjx.Model,
        data: mjx.Data,
        reset_state: dict[str, jax.Array],
    ) -> tuple[mjx.Model, mjx.Data]:
        obj_pos = reset_state["obj_init_pos"]
        peg_pos = reset_state["peg_pos"]
        goal_pos = reset_state["goal_pos"]

        qpos = data.qpos.at[self._obj_qposadr : self._obj_qposadr + 3].set(obj_pos)
        qvel = data.qvel.at[self._obj_qveladr : self._obj_qveladr + 6].set(0.0)
        data = data.replace(qpos=qpos, qvel=qvel)

        body_pos = model.body_pos.at[self._peg_body_id].set(peg_pos)
        site_pos = model.site_pos.at[self._peg_top_site_id].set(goal_pos)
        model = model.replace(body_pos=body_pos, site_pos=site_pos)
        return model, data

    @staticmethod
    def _reward_quat(obj_quat: jax.Array) -> jax.Array:
        ideal = jnp.array([0.707, 0.0, 0.0, 0.707])
        error = jnp.linalg.norm(obj_quat - ideal)
        return jnp.maximum(1.0 - error / 0.4, 0.0)

    @staticmethod
    def _reward_pos(
        wrench_center: jax.Array, target_pos: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        pos_error = target_pos - wrench_center
        radius = jnp.linalg.norm(pos_error[:2])
        aligned = radius < 0.02
        hooked = pos_error[2] > 0.0
        success = aligned & hooked

        threshold = jnp.where(success, 0.02, 0.01)
        target_height = jnp.where(
            radius > threshold,
            0.02 * jnp.log(jnp.maximum(radius - threshold, 1e-8)) + 0.2,
            0.0,
        )
        pos_error = pos_error.at[2].set(target_height - wrench_center[2])
        scale = jnp.array([1.0, 1.0, 3.0])
        lifted = jnp.logical_or(wrench_center[2] > 0.02, radius < threshold)
        in_place = 0.1 * lifted.astype(jnp.float32) + 0.9 * reward_utils.tolerance(
            jnp.linalg.norm(pos_error * scale),
            bounds=(0.0, 0.02),
            margin=0.4,
            sigmoid="long_tail",
        )
        return in_place, success.astype(jnp.float32)

    def compute_reward(
        self,
        data: mjx.Data,
        action: jax.Array,
        info: dict[str, Any],
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        hand = self._hand_pos(data)
        wrench = self.get_obj_pos(data)
        wrench_center = mjx_env.site_xpos(data, jnp.array(self._roundnut_site_id))
        target_pos = info["goal_pos"]

        threshold = self.WRENCH_HANDLE_LENGTH / 2.0
        close_x = jnp.abs(wrench[0] - hand[0]) < threshold
        wrench_threshed = wrench.at[0].set(jnp.where(close_x, hand[0], wrench[0]))

        obj_quat = self.get_obj_quat(data)
        reward_quat = self._reward_quat(obj_quat)
        reward_grab = self._gripper_caging_reward(
            data,
            action,
            wrench_threshed,
            info["obj_init_pos"],
            info["init_tcp"],
            obj_radius=0.015,
            pad_success_thresh=0.02,
            object_reach_radius=0.01,
            xz_thresh=0.01,
            medium_density=True,
        )
        reward_in_place, success = self._reward_pos(wrench_center, target_pos)
        reward = (2.0 * reward_grab + 6.0 * reward_in_place) * reward_quat
        reward = jnp.where(success > 0.0, 10.0, reward)

        metrics = {
            "success": success,
            "near_object": reward_quat,
            "grasp_success": (reward_grab >= 0.5).astype(jnp.float32),
            "grasp_reward": reward_grab,
            "in_place_reward": reward_in_place,
            "obj_to_target": jnp.array(0.0),
        }
        return reward, metrics
