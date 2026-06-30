"""Sawyer bin-picking environment (bin-picking-v3)."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from mujoco import mjx

from MTCWorldMJX import mjx_env
from MTCWorldMJX.asset_paths import sawyer_xml_path
from MTCWorldMJX.envs._helpers import as_f32, set_free_joint_xyz
from MTCWorldMJX.envs.sawyer_xyz import SawyerXYZConfig, SawyerXYZEnv
from MTCWorldMJX.utils import reward as reward_utils


class SawyerBinPickingEnvV3(SawyerXYZEnv):
    TARGET_RADIUS = 0.05

    def __init__(self, config: SawyerXYZConfig | None = None):
        super().__init__(config)
        self._obj_body_id = mjx_env.body_id(self._mj_model, "obj")
        self._bin_goal_body_id = mjx_env.body_id(self._mj_model, "bin_goal")

    @property
    def xml_path(self):
        return sawyer_xml_path("sawyer_bin_picking.xml")

    def hand_init_pos(self) -> jax.Array:
        return as_f32([0.0, 0.6, 0.2])

    def mocap_low(self) -> jax.Array:
        return as_f32([-0.5, 0.40, 0.07])

    def mocap_high(self) -> jax.Array:
        return as_f32([0.5, 1.0, 0.5])

    def random_reset_bounds(self) -> tuple[jax.Array, jax.Array]:
        obj_low = as_f32([-0.21, 0.65, 0.02])
        obj_high = as_f32([-0.03, 0.75, 0.02])
        goal_low = as_f32([0.1199, 0.699, -0.001])
        goal_high = as_f32([0.1201, 0.701, 0.001])
        return jnp.concatenate([obj_low, goal_low]), jnp.concatenate([obj_high, goal_high])

    def get_obj_pos(self, data: mjx.Data) -> jax.Array:
        return mjx_env.body_xpos(data, jnp.array(self._obj_body_id))

    def get_obj_quat(self, data: mjx.Data) -> jax.Array:
        return mjx_env.body_xquat(data, jnp.array(self._obj_body_id))

    def reset_state_from_rand_vec(self, rand_vec: jax.Array) -> dict[str, jax.Array]:
        return {
            "obj_xy": rand_vec[:2],
            "goal_pos": as_f32([0.12, 0.7, 0.02]),
            "rand_vec": rand_vec,
            "target_to_obj_init": jnp.array(-1.0, dtype=jnp.float32),
        }

    def sample_reset_state(self, rng: jax.Array) -> dict[str, jax.Array]:
        low, high = self.random_reset_bounds()
        vec = jax.random.uniform(rng, shape=low.shape, minval=low, maxval=high)
        return self.reset_state_from_rand_vec(vec)

    def prepare_reset_state(
        self,
        model: mjx.Model,
        data: mjx.Data,
        reset_state: dict[str, jax.Array],
    ) -> tuple[mjx.Model, mjx.Data, dict[str, jax.Array]]:
        obj_z = mjx_env.body_xpos(data, jnp.array(self._obj_body_id))[2]
        obj_init_pos = jnp.concatenate([reset_state["obj_xy"], obj_z[None]])
        return model, data, {**reset_state, "obj_init_pos": obj_init_pos}

    def apply_reset_state(
        self,
        model: mjx.Model,
        data: mjx.Data,
        reset_state: dict[str, jax.Array],
    ) -> tuple[mjx.Model, mjx.Data]:
        data = set_free_joint_xyz(
            data,
            self._obj_qposadr,
            self._obj_qveladr,
            reset_state["obj_init_pos"],
            ndim=3,
        )
        return model, data

    def finalize_reset(
        self,
        model: mjx.Model,
        data: mjx.Data,
        reset_state: dict[str, jax.Array],
    ) -> tuple[mjx.Model, mjx.Data, dict[str, jax.Array]]:
        data = mjx.forward(model, data)
        goal_pos = mjx_env.body_xpos(data, jnp.array(self._bin_goal_body_id))
        return model, data, {**reset_state, "goal_pos": goal_pos, "skip_final_forward": True}

    def compute_reward(
        self,
        data: mjx.Data,
        action: jax.Array,
        info: dict[str, Any],
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        hand = self._hand_pos(data)
        obj = self.get_obj_pos(data)
        target = info["goal_pos"]
        target_to_obj = jnp.linalg.norm(obj - target)
        target_to_obj_init = jnp.where(
            info["target_to_obj_init"] < 0,
            target_to_obj,
            info["target_to_obj_init"],
        )
        in_place = reward_utils.tolerance(
            target_to_obj,
            bounds=(0.0, self.TARGET_RADIUS),
            margin=target_to_obj_init,
            sigmoid="long_tail",
        )
        threshold = 0.03
        radii = jnp.array(
            [
                jnp.linalg.norm(hand[:2] - info["obj_init_pos"][:2]),
                jnp.linalg.norm(hand[:2] - target[:2]),
            ]
        )
        floor = jnp.min(
            jnp.where(
                radii > threshold,
                0.02 * jnp.log(jnp.maximum(radii - threshold, 1e-8)) + 0.2,
                0.0,
            )
        )
        above_floor = jnp.where(
            hand[2] >= floor,
            1.0,
            reward_utils.tolerance(
                jnp.maximum(floor - hand[2], 0.0),
                bounds=(0.0, 0.01),
                margin=0.05,
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
            pad_success_thresh=0.05,
            object_reach_radius=0.01,
            xz_thresh=0.01,
            desired_gripper_effort=0.7,
            high_density=True,
        )
        reward = reward_utils.hamacher_product(object_grasped, in_place)
        near_object = jnp.linalg.norm(obj - hand) < 0.04
        pinched = self._gripper_opening(data) < 0.43
        lifted = obj[2] - 0.02 > info["obj_init_pos"][2]
        grasp_success = near_object & lifted & (~pinched)
        reward = jnp.where(
            grasp_success,
            reward + 1.0 + 5.0 * reward_utils.hamacher_product(above_floor, in_place),
            reward,
        )
        reward = jnp.where(target_to_obj < self.TARGET_RADIUS, 10.0, reward)
        metrics = {
            "success": (target_to_obj <= 0.05).astype(jnp.float32),
            "near_object": near_object.astype(jnp.float32),
            "grasp_success": grasp_success.astype(jnp.float32),
            "grasp_reward": object_grasped,
            "in_place_reward": in_place,
            "obj_to_target": target_to_obj,
        }
        return reward, metrics

    def step(self, state, action):
        next_state = super().step(state, action)
        info = dict(next_state.info)
        obj = self.get_obj_pos(next_state.data)
        new_target_to_obj_init = jnp.linalg.norm(obj - state.info["goal_pos"])
        info["target_to_obj_init"] = jnp.where(
            state.info["target_to_obj_init"] < 0,
            new_target_to_obj_init,
            state.info["target_to_obj_init"],
        )
        return next_state.replace(info=info)
