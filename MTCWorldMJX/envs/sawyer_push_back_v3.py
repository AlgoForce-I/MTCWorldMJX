"""Sawyer push-back environment (push-back-v3)."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from mujoco import mjx

from MTCWorldMJX import mjx_env
from MTCWorldMJX.asset_paths import sawyer_xml_path
from MTCWorldMJX.envs._helpers import as_f32, geom_quat_xyzw, set_free_joint_xyz
from MTCWorldMJX.envs.sawyer_push_v3 import _PushXYResetMixin
from MTCWorldMJX.envs.sawyer_xyz import SawyerXYZConfig, SawyerXYZEnv
from MTCWorldMJX.utils import reward as reward_utils


class SawyerPushBackEnvV3(_PushXYResetMixin, SawyerXYZEnv):
    """Push a puck from the back of the table to a nearer goal (push-back-v3)."""

    OBJ_RADIUS = 0.007
    TARGET_RADIUS = 0.05

    def __init__(self, config: SawyerXYZConfig | None = None):
        super().__init__(config)
        self._obj_geom_id = mjx_env.geom_id(self._mj_model, "objGeom")
        self._goal_site_id = mjx_env.site_id(self._mj_model, "goal")

    @property
    def xml_path(self):
        return sawyer_xml_path("push_back_v3")

    def hand_init_pos(self) -> jax.Array:
        return as_f32([0.0, 0.6, 0.2])

    def mocap_low(self) -> jax.Array:
        return as_f32([-0.5, 0.40, 0.05])

    def mocap_high(self) -> jax.Array:
        return as_f32([0.5, 1.0, 0.5])

    def random_reset_bounds(self) -> tuple[jax.Array, jax.Array]:
        obj_low = as_f32([-0.1, 0.8, 0.02])
        obj_high = as_f32([0.1, 0.85, 0.02])
        goal_low = as_f32([-0.1, 0.6, 0.0199])
        goal_high = as_f32([0.1, 0.7, 0.0201])
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

    def _push_back_caging_reward(
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
        grip_success_margin = self.OBJ_RADIUS + 0.003
        x_z_success_margin = 0.01

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
            bounds=(self.OBJ_RADIUS, pad_success_margin),
            margin=right_caging_margin,
            sigmoid="long_tail",
        )
        left_caging = reward_utils.tolerance(
            delta_object_y_left_pad,
            bounds=(self.OBJ_RADIUS, pad_success_margin),
            margin=left_caging_margin,
            sigmoid="long_tail",
        )
        right_gripping = reward_utils.tolerance(
            delta_object_y_right_pad,
            bounds=(self.OBJ_RADIUS, grip_success_margin),
            margin=right_caging_margin,
            sigmoid="long_tail",
        )
        left_gripping = reward_utils.tolerance(
            delta_object_y_left_pad,
            bounds=(self.OBJ_RADIUS, grip_success_margin),
            margin=left_caging_margin,
            sigmoid="long_tail",
        )

        y_caging = reward_utils.hamacher_product(right_caging, left_caging)
        y_gripping = reward_utils.hamacher_product(right_gripping, left_gripping)

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
        gripping = jnp.where(caging > 0.95, y_gripping, 0.0)
        return (caging + gripping) / 2.0

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

        tcp_to_obj = jnp.linalg.norm(obj - tcp)
        target_to_obj = jnp.linalg.norm(obj - target)
        target_to_obj_init = jnp.linalg.norm(info["obj_init_pos"] - target)

        in_place = reward_utils.tolerance(
            target_to_obj,
            bounds=(0.0, self.TARGET_RADIUS),
            margin=target_to_obj_init,
            sigmoid="long_tail",
        )
        object_grasped = self._push_back_caging_reward(
            data,
            action,
            obj,
            info["obj_init_pos"],
            info["init_tcp"],
            info["init_left_pad"],
            info["init_right_pad"],
        )
        reward = reward_utils.hamacher_product(object_grasped, in_place)

        bonus = (
            (tcp_to_obj < 0.01)
            & (tcp_opened > 0)
            & (tcp_opened < 0.55)
            & (target_to_obj_init - target_to_obj > 0.01)
        )
        reward = jnp.where(bonus, reward + 1.0 + 5.0 * in_place, reward)
        reward = jnp.where(target_to_obj < self.TARGET_RADIUS, 10.0, reward)

        metrics = {
            "success": (target_to_obj <= 0.07).astype(jnp.float32),
            "near_object": (tcp_to_obj <= 0.03).astype(jnp.float32),
            "grasp_success": (
                (tcp_opened > 0) & (obj[2] - 0.02 > info["obj_init_pos"][2])
            ).astype(jnp.float32),
            "grasp_reward": object_grasped,
            "in_place_reward": in_place,
            "obj_to_target": target_to_obj,
        }
        return reward, metrics
