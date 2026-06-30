"""Sawyer reach environments (reach-v3, reach-wall-v3)."""

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
    reach_reward,
    sample_hstack_rand_vec,
    set_free_joint_xyz,
)
from MTCWorldMJX.envs.sawyer_xyz import SawyerXYZConfig, SawyerXYZEnv


class SawyerReachEnvV3(SawyerXYZEnv):
    """Reach a goal position with the gripper (reach-v3)."""

    def __init__(self, config: SawyerXYZConfig | None = None):
        super().__init__(config)
        self._obj_body_id = mjx_env.body_id(self._mj_model, "obj")
        self._obj_geom_id = mjx_env.geom_id(self._mj_model, "objGeom")
        self._goal_site_id = mjx_env.site_id(self._mj_model, "goal")

    @property
    def xml_path(self):
        return sawyer_xml_path("reach_v3")

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
        obj_init_pos = rand_vec[:3]
        goal_pos = rand_vec[-3:]
        return {
            "obj_init_pos": obj_init_pos,
            "goal_pos": goal_pos,
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
        del action
        tcp = self._tcp_center(data)
        target = info["goal_pos"]
        reward, reach_dist, in_place = reach_reward(
            tcp, target, self.hand_init_pos(), target_radius=0.05
        )
        metrics = {
            "success": (reach_dist <= 0.05).astype(jnp.float32),
            "near_object": reach_dist,
            "grasp_success": jnp.array(1.0),
            "grasp_reward": reach_dist,
            "in_place_reward": in_place,
            "obj_to_target": reach_dist,
        }
        return reward, metrics


class SawyerReachWallEnvV3(SawyerReachEnvV3):
    """Reach a goal position behind a wall (reach-wall-v3)."""

    @property
    def xml_path(self):
        return sawyer_xml_path("reach_wall_v3")

    def random_reset_bounds(self) -> tuple[jax.Array, jax.Array]:
        obj_low = as_f32([-0.05, 0.6, 0.015])
        obj_high = as_f32([0.05, 0.65, 0.015])
        goal_low = as_f32([-0.05, 0.85, 0.05])
        goal_high = as_f32([0.05, 0.9, 0.3])
        return jnp.concatenate([obj_low, goal_low]), jnp.concatenate([obj_high, goal_high])

    def compute_reward(
        self,
        data: mjx.Data,
        action: jax.Array,
        info: dict[str, Any],
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        del action
        tcp = self._tcp_center(data)
        target = info["goal_pos"]
        reward, reach_dist, in_place = reach_reward(
            tcp, target, self.hand_init_pos(), target_radius=0.05
        )
        metrics = {
            "success": (reach_dist <= 0.05).astype(jnp.float32),
            "near_object": jnp.array(0.0),
            "grasp_success": jnp.array(0.0),
            "grasp_reward": jnp.array(0.0),
            "in_place_reward": in_place,
            "obj_to_target": reach_dist,
        }
        return reward, metrics
