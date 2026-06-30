"""Sawyer basketball environment (basketball-v3)."""

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


class SawyerBasketballEnvV3(SawyerXYZEnv):
    TARGET_RADIUS = 0.08

    def __init__(self, config: SawyerXYZConfig | None = None):
        super().__init__(config)
        self._bsktball_body_id = mjx_env.body_id(self._mj_model, "bsktball")
        self._basket_goal_body_id = mjx_env.body_id(self._mj_model, "basket_goal")
        self._goal_site_id = mjx_env.site_id(self._mj_model, "goal")
        self._obj_z = jnp.array(0.03, dtype=jnp.float32)

    @property
    def xml_path(self):
        return sawyer_xml_path("sawyer_basketball.xml")

    def hand_init_pos(self) -> jax.Array:
        return as_f32([0.0, 0.6, 0.2])

    def mocap_low(self) -> jax.Array:
        return as_f32([-0.5, 0.40, 0.05])

    def mocap_high(self) -> jax.Array:
        return as_f32([0.5, 1.0, 0.5])

    def random_reset_bounds(self) -> tuple[jax.Array, jax.Array]:
        obj_low = as_f32([-0.1, 0.6, 0.0299])
        obj_high = as_f32([0.1, 0.7, 0.0301])
        goal_low = as_f32([-0.1, 0.85, 0.0])
        goal_high = as_f32([0.1, 0.9, 0.0])
        return jnp.concatenate([obj_low, goal_low]), jnp.concatenate([obj_high, goal_high])

    def get_obj_pos(self, data: mjx.Data) -> jax.Array:
        return mjx_env.body_xpos(data, jnp.array(self._bsktball_body_id))

    def get_obj_quat(self, data: mjx.Data) -> jax.Array:
        return mjx_env.body_xquat(data, jnp.array(self._bsktball_body_id))

    def reset_state_from_rand_vec(self, rand_vec: jax.Array) -> dict[str, jax.Array]:
        obj_init_pos = jnp.concatenate([rand_vec[:2], jnp.array([self._obj_z])])
        basket_pos = rand_vec[3:6]
        return {
            "obj_init_pos": obj_init_pos,
            "goal_pos": basket_pos,
            "basket_pos": basket_pos,
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
        basket_pos = reset_state["basket_pos"]
        data = set_free_joint_xyz(
            data, self._obj_qposadr, self._obj_qveladr, obj_pos, ndim=3
        )
        body_pos = model.body_pos.at[self._basket_goal_body_id].set(basket_pos)
        return model.replace(body_pos=body_pos), data

    def finalize_reset(
        self,
        model: mjx.Model,
        data: mjx.Data,
        reset_state: dict[str, jax.Array],
    ) -> tuple[mjx.Model, mjx.Data, dict[str, jax.Array]]:
        data = mjx.forward(model, data)
        goal_pos = mjx_env.site_xpos(data, jnp.array(self._goal_site_id))
        site_pos = model.site_pos.at[self._goal_site_id].set(goal_pos)
        model = model.replace(site_pos=site_pos)
        return model, data, {**reset_state, "goal_pos": goal_pos, "skip_final_forward": True}

    def compute_reward(
        self,
        data: mjx.Data,
        action: jax.Array,
        info: dict[str, Any],
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        obj = self.get_obj_pos(data)
        target = info["goal_pos"].at[2].set(0.3)
        scale = jnp.array([1.0, 1.0, 2.0])
        target_to_obj = jnp.linalg.norm((obj - target) * scale)
        target_to_obj_init = jnp.linalg.norm((info["obj_init_pos"] - target) * scale)
        in_place = reward_utils.tolerance(
            target_to_obj,
            bounds=(0.0, self.TARGET_RADIUS),
            margin=target_to_obj_init,
            sigmoid="long_tail",
        )
        tcp = self._tcp_center(data)
        tcp_opened = self._gripper_opening(data)
        tcp_to_obj = jnp.linalg.norm(obj - tcp)
        object_grasped = self._gripper_caging_reward(
            data,
            action,
            obj,
            info["obj_init_pos"],
            info["init_tcp"],
            obj_radius=0.025,
            pad_success_thresh=0.06,
            object_reach_radius=0.01,
            xz_thresh=0.005,
            high_density=True,
        )
        lifted = obj[2] - 0.01 > info["obj_init_pos"][2]
        object_grasped = jnp.where(
            (tcp_to_obj < 0.035) & (tcp_opened > 0) & lifted,
            1.0,
            object_grasped,
        )
        reward = reward_utils.hamacher_product(object_grasped, in_place)
        near = (tcp_to_obj < 0.035) & (tcp_opened > 0) & lifted
        reward = jnp.where(near, reward + 1.0 + 5.0 * in_place, reward)
        reward = jnp.where(target_to_obj < self.TARGET_RADIUS, 10.0, reward)
        metrics = {
            "success": (target_to_obj <= self.TARGET_RADIUS).astype(jnp.float32),
            "near_object": (tcp_to_obj <= 0.05).astype(jnp.float32),
            "grasp_success": (lifted & (tcp_opened > 0)).astype(jnp.float32),
            "grasp_reward": object_grasped,
            "in_place_reward": in_place,
            "obj_to_target": target_to_obj,
        }
        return reward, metrics
