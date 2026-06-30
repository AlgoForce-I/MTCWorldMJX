"""Sawyer push environment (push-v3)."""

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
    push_xy_positions_from_rand_vec,
    sample_hstack_rand_vec,
    set_free_joint_xyz,
)
from MTCWorldMJX.envs.sawyer_xyz import SawyerXYZConfig, SawyerXYZEnv
from MTCWorldMJX.utils import reward as reward_utils


class _PushXYResetMixin:
    """Shared reset logic for push tasks that preserve object Z after hand settle."""

    def reset_state_from_rand_vec(self, rand_vec: jax.Array) -> dict[str, jax.Array]:
        return {"rand_vec": rand_vec}

    def sample_reset_state(self, rng: jax.Array) -> dict[str, jax.Array]:
        low, high = self.random_reset_bounds()
        vec, _ = sample_hstack_rand_vec(rng, low, high, min_xy_dist=0.15)
        return {"rand_vec": vec}

    def _obj_reset_z(self, data: mjx.Data) -> jax.Array:
        raise NotImplementedError

    def _hand_settle_geom_z_offset(self) -> jax.Array:
        return getattr(self, "_geom_z_offset", jnp.float32(0.0))

    def prepare_reset_state(
        self,
        model: mjx.Model,
        data: mjx.Data,
        reset_state: dict[str, jax.Array],
    ) -> tuple[mjx.Model, mjx.Data, dict[str, jax.Array]]:
        obj_z = self._obj_reset_z(data) + self._hand_settle_geom_z_offset()
        obj_init_pos, goal_pos = push_xy_positions_from_rand_vec(
            reset_state["rand_vec"], obj_z=obj_z
        )
        return model, data, {
            **reset_state,
            "obj_init_pos": obj_init_pos,
            "goal_pos": goal_pos,
        }

    def finalize_reset(
        self,
        model: mjx.Model,
        data: mjx.Data,
        reset_state: dict[str, jax.Array],
    ) -> tuple[mjx.Model, mjx.Data, dict[str, jax.Array]]:
        return model, data, reset_state

    def _resolve_goal_pos(self, model, data, reset_state):
        del model, reset_state
        return mjx_env.site_xpos(data, jnp.array(self._goal_site_id))


class SawyerPushEnvV3(_PushXYResetMixin, SawyerXYZEnv):
    """Push a puck to a goal position (push-v3)."""

    TARGET_RADIUS = 0.05

    def __init__(self, config: SawyerXYZConfig | None = None):
        super().__init__(config)
        self._obj_body_id = mjx_env.body_id(self._mj_model, "obj")
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
        return sawyer_xml_path("push_v3")

    def hand_init_pos(self) -> jax.Array:
        return as_f32([0.0, 0.6, 0.2])

    def mocap_low(self) -> jax.Array:
        return as_f32([-0.5, 0.40, 0.05])

    def mocap_high(self) -> jax.Array:
        return as_f32([0.5, 1.0, 0.5])

    def random_reset_bounds(self) -> tuple[jax.Array, jax.Array]:
        obj_low = as_f32([-0.1, 0.6, 0.02])
        obj_high = as_f32([0.1, 0.7, 0.02])
        goal_low = as_f32([-0.1, 0.8, 0.01])
        goal_high = as_f32([0.1, 0.9, 0.02])
        return jnp.concatenate([obj_low, goal_low]), jnp.concatenate([obj_high, goal_high])

    def _obj_reset_z(self, data: mjx.Data) -> jax.Array:
        return mjx_env.geom_xpos(data, jnp.array([self._obj_geom_id]))[0, 2]

    def get_obj_pos(self, data: mjx.Data) -> jax.Array:
        return mjx_env.body_xpos(data, jnp.array(self._obj_body_id))

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
        reward = jnp.where(near_tcp, reward + 1.0 + reward + 5.0 * in_place, reward)
        reward = jnp.where(target_to_obj < self.TARGET_RADIUS, 10.0, reward)

        metrics = {
            "success": (target_to_obj <= self.TARGET_RADIUS).astype(jnp.float32),
            "near_object": (tcp_to_obj <= 0.03).astype(jnp.float32),
            "grasp_success": (
                (tcp_opened > 0) & (obj[2] - 0.02 > info["obj_init_pos"][2])
            ).astype(jnp.float32),
            "grasp_reward": object_grasped,
            "in_place_reward": in_place,
            "obj_to_target": target_to_obj,
        }
        return reward, metrics
