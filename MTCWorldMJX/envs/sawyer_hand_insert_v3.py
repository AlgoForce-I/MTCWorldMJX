"""Sawyer hand-insert environment (hand-insert-v3)."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from mujoco import mjx

from MTCWorldMJX import mjx_env
from MTCWorldMJX.asset_paths import sawyer_xml_path
from MTCWorldMJX.envs._helpers import as_f32, sample_hstack_rand_vec, set_free_joint_xyz
from MTCWorldMJX.envs.sawyer_xyz import SawyerXYZConfig, SawyerXYZEnv
from MTCWorldMJX.utils import reward as reward_utils


class SawyerHandInsertEnvV3(SawyerXYZEnv):
    TARGET_RADIUS = 0.05

    def __init__(self, config: SawyerXYZConfig | None = None):
        super().__init__(config)
        self._obj_body_id = mjx_env.body_id(self._mj_model, "obj")
        self._goal_site_id = mjx_env.site_id(self._mj_model, "goal")
        self._obj_z = jnp.array(0.05, dtype=jnp.float32)

    @property
    def xml_path(self):
        return sawyer_xml_path("sawyer_table_with_hole.xml")

    def hand_init_pos(self) -> jax.Array:
        return as_f32([0.0, 0.6, 0.2])

    def mocap_low(self) -> jax.Array:
        return as_f32([-0.5, 0.40, -0.15])

    def mocap_high(self) -> jax.Array:
        return as_f32([0.5, 1.0, 0.5])

    def random_reset_bounds(self) -> tuple[jax.Array, jax.Array]:
        obj_low = as_f32([-0.1, 0.6, 0.05])
        obj_high = as_f32([0.1, 0.7, 0.05])
        goal_low = as_f32([-0.04, 0.8, -0.0201])
        goal_high = as_f32([0.04, 0.88, -0.0199])
        return jnp.concatenate([obj_low, goal_low]), jnp.concatenate([obj_high, goal_high])

    def get_obj_pos(self, data: mjx.Data) -> jax.Array:
        return mjx_env.body_xpos(data, jnp.array(self._obj_body_id))

    def get_obj_quat(self, data: mjx.Data) -> jax.Array:
        return mjx_env.body_xquat(data, jnp.array(self._obj_body_id))

    def reset_state_from_rand_vec(self, rand_vec: jax.Array) -> dict[str, jax.Array]:
        obj_init_pos = jnp.concatenate([rand_vec[:2], jnp.array([self._obj_z])])
        goal_pos = rand_vec[3:6]
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
        obj = self.get_obj_pos(data)
        target = info["goal_pos"]
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
        reward = reward_utils.hamacher_product(object_grasped, in_place)
        tcp_opened = self._gripper_opening(data)
        tcp_to_obj = jnp.linalg.norm(obj - self._tcp_center(data))
        near = (tcp_to_obj < 0.02) & (tcp_opened > 0)
        reward = jnp.where(near, reward + 1.0 + 7.0 * in_place, reward)
        reward = jnp.where(target_to_obj < self.TARGET_RADIUS, 10.0, reward)
        lifted = obj[2] - 0.02 > info["obj_init_pos"][2]
        metrics = {
            "success": (target_to_obj <= 0.05).astype(jnp.float32),
            "near_object": (tcp_to_obj <= 0.03).astype(jnp.float32),
            "grasp_success": (near & lifted).astype(jnp.float32),
            "grasp_reward": object_grasped,
            "in_place_reward": in_place,
            "obj_to_target": target_to_obj,
        }
        return reward, metrics
