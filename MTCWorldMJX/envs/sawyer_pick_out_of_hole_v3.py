"""Sawyer pick-out-of-hole environment (pick-out-of-hole-v3)."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from mujoco import mjx

from MTCWorldMJX import mjx_env
from MTCWorldMJX.asset_paths import sawyer_xml_path
from MTCWorldMJX.envs._helpers import (
    as_f32,
    sample_hstack_rand_vec,
    set_free_joint_xyz,
)
from MTCWorldMJX.envs.sawyer_xyz import SawyerXYZConfig, SawyerXYZEnv
from MTCWorldMJX.utils import reward as reward_utils


class SawyerPickOutOfHoleEnvV3(SawyerXYZEnv):
    """Pick a puck out of a hole and place it at a goal (pick-out-of-hole-v3)."""

    TARGET_RADIUS = 0.02

    def __init__(self, config: SawyerXYZConfig | None = None):
        super().__init__(config)
        self._obj_body_id = mjx_env.body_id(self._mj_model, "obj")
        self._goal_site_id = mjx_env.site_id(self._mj_model, "goal")

    @property
    def xml_path(self):
        return sawyer_xml_path("pick_out_of_hole")

    def hand_init_pos(self) -> jax.Array:
        return as_f32([0.0, 0.6, 0.2])

    def mocap_low(self) -> jax.Array:
        return as_f32([-0.5, 0.40, -0.05])

    def mocap_high(self) -> jax.Array:
        return as_f32([0.5, 1.0, 0.5])

    def random_reset_bounds(self) -> tuple[jax.Array, jax.Array]:
        obj_low = as_f32([0.0, 0.75, 0.02])
        obj_high = as_f32([0.0, 0.75, 0.02])
        goal_low = as_f32([-0.1, 0.5, 0.15])
        goal_high = as_f32([0.1, 0.6, 0.3])
        return jnp.concatenate([obj_low, goal_low]), jnp.concatenate([obj_high, goal_high])

    def get_obj_pos(self, data: mjx.Data) -> jax.Array:
        return mjx_env.body_xpos(data, jnp.array(self._obj_body_id))

    def get_obj_quat(self, data: mjx.Data) -> jax.Array:
        return mjx_env.body_xquat(data, jnp.array([self._obj_body_id]))[0]

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
        obj = self.get_obj_pos(data)
        gripper = self._tcp_center(data)
        tcp_opened = self._gripper_opening(data)

        obj_to_target = jnp.linalg.norm(obj - info["goal_pos"])
        tcp_to_obj = jnp.linalg.norm(obj - gripper)
        in_place_margin = jnp.linalg.norm(info["obj_init_pos"] - info["goal_pos"])

        threshold = 0.03
        radius = jnp.linalg.norm(gripper[:2] - info["obj_init_pos"][:2])
        floor = jnp.where(
            radius <= threshold,
            0.0,
            0.015 * jnp.log(jnp.maximum(radius - threshold, 1e-8)) + 0.15,
        )
        above_floor = jnp.where(
            gripper[2] >= floor,
            1.0,
            reward_utils.tolerance(
                jnp.maximum(floor - gripper[2], 0.0),
                bounds=(0.0, 0.01),
                margin=0.02,
                sigmoid="long_tail",
            ),
        )

        object_grasped = self._gripper_caging_reward(
            data,
            action,
            obj,
            info["obj_init_pos"],
            info["init_tcp"],
            obj_radius=0.015,
            pad_success_thresh=0.02,
            object_reach_radius=0.01,
            xz_thresh=0.03,
            desired_gripper_effort=0.1,
            high_density=True,
        )
        in_place = reward_utils.tolerance(
            obj_to_target,
            bounds=(0.0, 0.02),
            margin=in_place_margin,
            sigmoid="long_tail",
        )
        reward = reward_utils.hamacher_product(object_grasped, in_place)

        near_object = tcp_to_obj < 0.04
        pinched_without_obj = tcp_opened < 0.33
        lifted = obj[2] - 0.02 > info["obj_init_pos"][2]
        grasp_success = near_object & lifted & (~pinched_without_obj)
        reward = jnp.where(
            grasp_success,
            reward + 1.0 + 5.0 * reward_utils.hamacher_product(in_place, above_floor),
            reward,
        )
        reward = jnp.where(obj_to_target < self.TARGET_RADIUS, 10.0, reward)

        metrics = {
            "success": (obj_to_target <= 0.07).astype(jnp.float32),
            "near_object": (tcp_to_obj <= 0.03).astype(jnp.float32),
            "grasp_success": grasp_success.astype(jnp.float32),
            "grasp_reward": object_grasped,
            "in_place_reward": in_place,
            "obj_to_target": obj_to_target,
        }
        return reward, metrics
