"""Sawyer push-wall environment (push-wall-v3)."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from mujoco import mjx

from MTCWorldMJX import mjx_env
from MTCWorldMJX.asset_paths import sawyer_xml_path
from MTCWorldMJX.envs._helpers import (
    as_f32,
    calibrate_hand_settle_geom_z_offset,
    geom_quat_xyzw,
    set_free_joint_xyz,
)
from MTCWorldMJX.envs.sawyer_push_v3 import _PushXYResetMixin
from MTCWorldMJX.envs.sawyer_xyz import SawyerXYZConfig, SawyerXYZEnv
from MTCWorldMJX.utils import reward as reward_utils


class SawyerPushWallEnvV3(_PushXYResetMixin, SawyerXYZEnv):
    """Push a puck around a wall to a goal (push-wall-v3)."""

    TARGET_RADIUS = 0.05

    def __init__(self, config: SawyerXYZConfig | None = None):
        super().__init__(config)
        self._obj_geom_id = mjx_env.geom_id(self._mj_model, "objGeom")
        self._goal_site_id = mjx_env.site_id(self._mj_model, "goal")
        self._geom_z_offset = calibrate_hand_settle_geom_z_offset(
            self._mj_model,
            self._mjx_model,
            self._init_qpos,
            self._init_qvel,
            self.hand_init_pos(),
            self._obj_geom_id,
            hand_reset_steps=self.config.hand_reset_steps,
            frame_skip=self.config.frame_skip,
            settle_hand=self._settle_hand,
        )

    @property
    def xml_path(self):
        return sawyer_xml_path("push_wall_v3")

    def hand_init_pos(self) -> jax.Array:
        return as_f32([0.0, 0.6, 0.2])

    def mocap_low(self) -> jax.Array:
        return as_f32([-0.5, 0.40, 0.05])

    def mocap_high(self) -> jax.Array:
        return as_f32([0.5, 1.0, 0.5])

    def random_reset_bounds(self) -> tuple[jax.Array, jax.Array]:
        obj_low = as_f32([-0.05, 0.6, 0.015])
        obj_high = as_f32([0.05, 0.65, 0.015])
        goal_low = as_f32([-0.05, 0.85, 0.01])
        goal_high = as_f32([0.05, 0.9, 0.02])
        return jnp.concatenate([obj_low, goal_low]), jnp.concatenate([obj_high, goal_high])

    def _obj_reset_z(self, data: mjx.Data) -> jax.Array:
        return data.geom_xpos[self._obj_geom_id, 2]

    def get_obj_pos(self, data: mjx.Data) -> jax.Array:
        return data.geom_xpos[self._obj_geom_id]

    def get_obj_quat(self, data: mjx.Data) -> jax.Array:
        return geom_quat_xyzw(data, self._obj_geom_id)

    def apply_reset_state(
        self,
        model: mjx.Model,
        data: mjx.Data,
        reset_state: dict[str, jax.Array],
    ) -> tuple[mjx.Model, mjx.Data]:
        obj_pos = reset_state["obj_init_pos"]
        goal_pos = reset_state["goal_pos"]
        data = set_free_joint_xyz(
            data, self._obj_qposadr, self._obj_qveladr, obj_pos, ndim=3
        )
        site_pos = model.site_pos.at[self._goal_site_id].set(goal_pos)
        return model.replace(site_pos=site_pos), data

    def compute_reward(
        self,
        data: mjx.Data,
        action: jax.Array,
        info: dict[str, Any],
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        tcp = self._tcp_center(data)
        obj = self.get_obj_pos(data)
        tcp_opened = self._gripper_opening(data)
        target = info["goal_pos"]
        midpoint = jnp.array([-0.05, 0.77, obj[2]], dtype=jnp.float32)
        in_place_scaling = jnp.array([3.0, 1.0, 1.0], dtype=jnp.float32)

        tcp_to_obj = jnp.linalg.norm(obj - tcp)
        obj_to_midpoint = jnp.linalg.norm((obj - midpoint) * in_place_scaling)
        obj_to_midpoint_init = jnp.linalg.norm(
            (info["obj_init_pos"] - midpoint) * in_place_scaling
        )
        obj_to_target = jnp.linalg.norm(obj - target)
        obj_to_target_init = jnp.linalg.norm(info["obj_init_pos"] - target)

        in_place_part1 = reward_utils.tolerance(
            obj_to_midpoint,
            bounds=(0.0, self.TARGET_RADIUS),
            margin=obj_to_midpoint_init,
            sigmoid="long_tail",
        )
        in_place_part2 = reward_utils.tolerance(
            obj_to_target,
            bounds=(0.0, self.TARGET_RADIUS),
            margin=obj_to_target_init,
            sigmoid="long_tail",
        )
        object_grasped = self._gripper_caging_reward(
            data,
            action,
            obj,
            info["obj_init_pos"],
            info["init_tcp"],
            obj_radius=0.015,
            pad_success_thresh=0.05,
            object_reach_radius=0.01,
            xz_thresh=0.005,
            high_density=True,
        )
        reward = 2.0 * object_grasped
        near_tcp = (tcp_to_obj < 0.02) & (tcp_opened > 0)
        reward = jnp.where(
            near_tcp,
            2.0 * object_grasped + 1.0 + 4.0 * in_place_part1,
            reward,
        )
        reward = jnp.where(
            near_tcp & (obj[1] > 0.75),
            2.0 * object_grasped + 1.0 + 4.0 + 3.0 * in_place_part2,
            reward,
        )
        reward = jnp.where(obj_to_target < self.TARGET_RADIUS, 10.0, reward)

        metrics = {
            "success": (obj_to_target <= 0.07).astype(jnp.float32),
            "near_object": (tcp_to_obj <= 0.03).astype(jnp.float32),
            "grasp_success": (
                (tcp_opened > 0) & (obj[2] - 0.02 > info["obj_init_pos"][2])
            ).astype(jnp.float32),
            "grasp_reward": object_grasped,
            "in_place_reward": in_place_part2,
            "obj_to_target": obj_to_target,
        }
        return reward, metrics
