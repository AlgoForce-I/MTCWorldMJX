"""Sawyer pick-and-place environment (pick-place-v3)."""

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


class SawyerPickPlaceEnvV3(SawyerXYZEnv):
    """Pick up a puck and place it at a goal (pick-place-v3)."""

    TARGET_RADIUS = 0.05

    def __init__(self, config: SawyerXYZConfig | None = None):
        super().__init__(config)
        self._obj_body_id = mjx_env.body_id(self._mj_model, "obj")
        self._obj_geom_id = mjx_env.geom_id(self._mj_model, "objGeom")
        self._goal_site_id = mjx_env.site_id(self._mj_model, "goal")

    @property
    def xml_path(self):
        return sawyer_xml_path("pick_place_v3")

    def hand_init_pos(self) -> jax.Array:
        return as_f32([0.0, 0.6, 0.2])

    def mocap_low(self) -> jax.Array:
        return as_f32([-0.5, 0.40, 0.05])

    def mocap_high(self) -> jax.Array:
        return as_f32([0.5, 1.0, 0.5])

    def random_reset_bounds(self) -> tuple[jax.Array, jax.Array]:
        obj_low = as_f32([-0.1, 0.6, 0.02])
        obj_high = as_f32([0.1, 0.7, 0.02])
        goal_low = as_f32([-0.1, 0.8, 0.05])
        goal_high = as_f32([0.1, 0.9, 0.3])
        return jnp.concatenate([obj_low, goal_low]), jnp.concatenate([obj_high, goal_high])

    def get_obj_pos(self, data: mjx.Data) -> jax.Array:
        return mjx_env.body_xpos(data, jnp.array(self._obj_body_id))

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

    def _pick_place_caging_reward(
        self,
        data: mjx.Data,
        action: jax.Array,
        obj_pos: jax.Array,
        obj_init_pos: jax.Array,
        init_tcp: jax.Array,
        init_left_pad: jax.Array,
        init_right_pad: jax.Array,
    ) -> jax.Array:
        pad_success_margin = 0.05
        x_z_success_margin = 0.005
        obj_radius = 0.015

        tcp = self._tcp_center(data)
        left_pad = mjx_env.body_xpos(data, jnp.array(self._leftpad_body_id))
        right_pad = mjx_env.body_xpos(data, jnp.array(self._rightpad_body_id))

        delta_object_y_left_pad = left_pad[1] - obj_pos[1]
        delta_object_y_right_pad = obj_pos[1] - right_pad[1]
        right_caging_margin = jnp.abs(
            jnp.abs(obj_pos[1] - init_right_pad[1]) - pad_success_margin
        )
        left_caging_margin = jnp.abs(
            jnp.abs(obj_pos[1] - init_left_pad[1]) - pad_success_margin
        )

        right_caging = reward_utils.tolerance(
            delta_object_y_right_pad,
            bounds=(obj_radius, pad_success_margin),
            margin=right_caging_margin,
            sigmoid="long_tail",
        )
        left_caging = reward_utils.tolerance(
            delta_object_y_left_pad,
            bounds=(obj_radius, pad_success_margin),
            margin=left_caging_margin,
            sigmoid="long_tail",
        )
        y_caging = reward_utils.hamacher_product(left_caging, right_caging)

        y_offset = jnp.array([0.0, -1.0, 0.0], dtype=jnp.float32)
        tcp_xz = tcp + y_offset * tcp[1]
        obj_position_x_z = obj_pos + y_offset * obj_pos[1]
        init_obj_x_z = obj_init_pos + y_offset * obj_init_pos[1]
        init_tcp_x_z = init_tcp + y_offset * init_tcp[1]
        tcp_obj_norm_x_z = jnp.linalg.norm(tcp_xz - obj_position_x_z)
        tcp_obj_x_z_margin = (
            jnp.linalg.norm(init_obj_x_z - init_tcp_x_z) - x_z_success_margin
        )
        x_z_caging = reward_utils.tolerance(
            tcp_obj_norm_x_z,
            bounds=(0.0, x_z_success_margin),
            margin=tcp_obj_x_z_margin,
            sigmoid="long_tail",
        )

        gripper_closed = jnp.clip(action[-1], 0.0, 1.0)
        caging = reward_utils.hamacher_product(y_caging, x_z_caging)
        gripping = jnp.where(caging > 0.97, gripper_closed, 0.0)
        caging_and_gripping = reward_utils.hamacher_product(caging, gripping)
        return (caging_and_gripping + caging) / 2.0

    def compute_reward(
        self,
        data: mjx.Data,
        action: jax.Array,
        info: dict[str, Any],
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        obj = self.get_obj_pos(data)
        tcp = self._tcp_center(data)
        tcp_opened = self._gripper_opening(data)
        target = info["goal_pos"]

        obj_to_target = jnp.linalg.norm(obj - target)
        tcp_to_obj = jnp.linalg.norm(obj - tcp)
        in_place_margin = jnp.linalg.norm(info["obj_init_pos"] - target)
        in_place = reward_utils.tolerance(
            obj_to_target,
            bounds=(0.0, self.TARGET_RADIUS),
            margin=in_place_margin,
            sigmoid="long_tail",
        )
        object_grasped = self._pick_place_caging_reward(
            data,
            action,
            obj,
            info["obj_init_pos"],
            info["init_tcp"],
            info["init_left_pad"],
            info["init_right_pad"],
        )
        reward = reward_utils.hamacher_product(object_grasped, in_place)

        lifted = obj[2] - 0.01 > info["obj_init_pos"][2]
        near = (tcp_to_obj < 0.02) & (tcp_opened > 0) & lifted
        reward = jnp.where(near, reward + 1.0 + 5.0 * in_place, reward)
        reward = jnp.where(obj_to_target < self.TARGET_RADIUS, 10.0, reward)

        metrics = {
            "success": (obj_to_target <= 0.07).astype(jnp.float32),
            "near_object": (tcp_to_obj <= 0.03).astype(jnp.float32),
            "grasp_success": lifted.astype(jnp.float32),
            "grasp_reward": object_grasped,
            "in_place_reward": in_place,
            "obj_to_target": obj_to_target,
        }
        return reward, metrics
