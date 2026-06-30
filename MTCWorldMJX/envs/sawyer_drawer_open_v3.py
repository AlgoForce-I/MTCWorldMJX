"""Sawyer drawer open environment (drawer-open-v3)."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from mujoco import mjx

from MTCWorldMJX import mjx_env
from MTCWorldMJX.asset_paths import sawyer_xml_path
from MTCWorldMJX.envs._helpers import as_f32, sample_obj_only_rand_vec
from MTCWorldMJX.envs.sawyer_xyz import SawyerXYZConfig, SawyerXYZEnv
from MTCWorldMJX.utils import reward as reward_utils


class SawyerDrawerOpenEnvV3(SawyerXYZEnv):
    """Pull a drawer open (drawer-open-v3)."""

    _HANDLE_OFFSET = as_f32([0.0, -0.16, 0.0])
    _GOAL_OFFSET = as_f32([0.0, -0.16, 0.09])
    _MAX_DIST = 0.2

    def __init__(self, config: SawyerXYZConfig | None = None):
        super().__init__(config)
        self._drawer_body_id = mjx_env.body_id(self._mj_model, "drawer")
        self._drawer_link_body_id = mjx_env.body_id(self._mj_model, "drawer_link")
        self._goal_site_id = mjx_env.site_id(self._mj_model, "goal")

    @property
    def xml_path(self):
        return sawyer_xml_path("drawer")

    def hand_init_pos(self) -> jax.Array:
        return as_f32([0.0, 0.6, 0.2])

    def mocap_low(self) -> jax.Array:
        return as_f32([-0.5, 0.40, 0.05])

    def mocap_high(self) -> jax.Array:
        return as_f32([0.5, 1.0, 0.5])

    def random_reset_bounds(self) -> tuple[jax.Array, jax.Array]:
        low = as_f32([-0.1, 0.9, 0.0])
        high = as_f32([0.1, 0.9, 0.0])
        return low, high

    def get_obj_pos(self, data: mjx.Data) -> jax.Array:
        return data.xpos[self._drawer_link_body_id] + self._HANDLE_OFFSET

    def get_obj_quat(self, data: mjx.Data) -> jax.Array:
        return data.xquat[self._drawer_link_body_id]

    def reset_state_from_rand_vec(self, rand_vec: jax.Array) -> dict[str, jax.Array]:
        obj_init_pos = rand_vec[:3]
        goal_pos = obj_init_pos + self._GOAL_OFFSET + as_f32([0.0, -self._MAX_DIST, 0.0])
        return {
            "obj_init_pos": obj_init_pos,
            "goal_pos": goal_pos,
            "rand_vec": rand_vec,
        }

    def sample_reset_state(self, rng: jax.Array) -> dict[str, jax.Array]:
        low, high = self.random_reset_bounds()
        vec = sample_obj_only_rand_vec(rng, low, high)
        return self.reset_state_from_rand_vec(vec)

    def apply_reset_state(
        self,
        model: mjx.Model,
        data: mjx.Data,
        reset_state: dict[str, jax.Array],
    ) -> tuple[mjx.Model, mjx.Data]:
        obj_pos = reset_state["obj_init_pos"]
        goal_pos = reset_state["goal_pos"]
        body_pos = model.body_pos.at[self._drawer_body_id].set(obj_pos)
        site_pos = model.site_pos.at[self._goal_site_id].set(goal_pos)
        return model.replace(body_pos=body_pos, site_pos=site_pos), data

    def compute_reward(
        self,
        data: mjx.Data,
        action: jax.Array,
        info: dict[str, Any],
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        del action
        gripper = self._hand_pos(data)
        handle = self.get_obj_pos(data)
        handle_error = jnp.linalg.norm(handle - info["goal_pos"])
        opening_reward = reward_utils.tolerance(
            handle_error,
            bounds=(0.0, 0.02),
            margin=self._MAX_DIST,
            sigmoid="long_tail",
        )
        handle_pos_init = info["goal_pos"] + as_f32([0.0, self._MAX_DIST, 0.0])
        scale = as_f32([3.0, 3.0, 1.0])
        gripper_error = (handle - gripper) * scale
        gripper_error_init = (handle_pos_init - info["init_tcp"]) * scale
        caging_reward = reward_utils.tolerance(
            jnp.linalg.norm(gripper_error),
            bounds=(0.0, 0.01),
            margin=jnp.linalg.norm(gripper_error_init),
            sigmoid="long_tail",
        )
        reward = (caging_reward + opening_reward) * 5.0
        metrics = {
            "success": (handle_error <= 0.03).astype(jnp.float32),
            "near_object": (jnp.linalg.norm(handle - gripper) <= 0.03).astype(jnp.float32),
            "grasp_success": (self._gripper_opening(data) > 0.0).astype(jnp.float32),
            "grasp_reward": caging_reward,
            "in_place_reward": opening_reward,
            "obj_to_target": handle_error,
        }
        return reward, metrics
