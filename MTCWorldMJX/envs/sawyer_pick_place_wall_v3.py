"""Sawyer pick-and-place wall environment (pick-place-wall-v3)."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from mujoco import mjx

from MTCWorldMJX import mjx_env
from MTCWorldMJX.asset_paths import sawyer_xml_path
from MTCWorldMJX.envs._helpers import (
    as_f32,
    geom_quat_xyzw,
    sample_hstack_rand_vec,
    set_free_joint_xyz,
)
from MTCWorldMJX.envs.sawyer_xyz import SawyerXYZConfig, SawyerXYZEnv
from MTCWorldMJX.utils import reward as reward_utils


class SawyerPickPlaceWallEnvV3(SawyerXYZEnv):
    """Pick up a puck and place it behind a wall (pick-place-wall-v3)."""

    TARGET_RADIUS = 0.05

    def __init__(self, config: SawyerXYZConfig | None = None):
        super().__init__(config)
        self._obj_geom_id = mjx_env.geom_id(self._mj_model, "objGeom")
        self._goal_site_id = mjx_env.site_id(self._mj_model, "goal")

    @property
    def xml_path(self):
        return sawyer_xml_path("pick_place_wall_v3")

    def hand_init_pos(self) -> jax.Array:
        return as_f32([0.0, 0.6, 0.2])

    def mocap_low(self) -> jax.Array:
        return as_f32([-0.5, 0.40, 0.05])

    def mocap_high(self) -> jax.Array:
        return as_f32([0.5, 1.0, 0.5])

    def random_reset_bounds(self) -> tuple[jax.Array, jax.Array]:
        obj_low = as_f32([-0.05, 0.6, 0.015])
        obj_high = as_f32([0.05, 0.65, 0.015])
        goal_low = as_f32([-0.05, 0.85, 0.05])
        goal_high = as_f32([0.05, 0.9, 0.3])
        return jnp.concatenate([obj_low, goal_low]), jnp.concatenate([obj_high, goal_high])

    def get_obj_pos(self, data: mjx.Data) -> jax.Array:
        return data.geom_xpos[self._obj_geom_id]

    def get_obj_quat(self, data: mjx.Data) -> jax.Array:
        return geom_quat_xyzw(data, self._obj_geom_id)

    def reset_state_from_rand_vec(self, rand_vec: jax.Array) -> dict[str, jax.Array]:
        return {
            "obj_init_pos": rand_vec[:3],
            "goal_pos": rand_vec[-3:],
            "rand_vec": rand_vec,
        }

    def sample_reset_state(self, rng: jax.Array) -> dict[str, jax.Array]:
        low, high = self.random_reset_bounds()
        vec, _ = sample_hstack_rand_vec(rng, low, high, min_xy_dist=0.15)
        return self.reset_state_from_rand_vec(vec)

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
        midpoint = jnp.array([target[0], 0.77, 0.25], dtype=jnp.float32)
        in_place_scaling = jnp.array([1.0, 1.0, 3.0], dtype=jnp.float32)

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
            high_density=False,
        )
        in_place_and_object_grasped = reward_utils.hamacher_product(
            object_grasped, in_place_part1
        )
        reward = in_place_and_object_grasped

        lifted = obj[2] - 0.015 > info["obj_init_pos"][2]
        near = (tcp_to_obj < 0.02) & (tcp_opened > 0) & lifted
        reward = jnp.where(
            near,
            in_place_and_object_grasped + 1.0 + 4.0 * in_place_part1,
            reward,
        )
        reward = jnp.where(
            near & (obj[1] > 0.75),
            in_place_and_object_grasped + 1.0 + 4.0 + 3.0 * in_place_part2,
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
