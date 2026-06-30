"""Sawyer soccer environment (soccer-v3)."""

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
    xz_plane_gripper_caging_reward,
)
from MTCWorldMJX.envs.sawyer_xyz import SawyerXYZConfig, SawyerXYZEnv
from MTCWorldMJX.utils import reward as reward_utils


class SawyerSoccerEnvV3(SawyerXYZEnv):
    OBJ_RADIUS = 0.013
    TARGET_RADIUS = 0.07

    def __init__(self, config: SawyerXYZConfig | None = None):
        super().__init__(config)
        self._ball_body_id = mjx_env.body_id(self._mj_model, "soccer_ball")
        self._goal_body_id = mjx_env.body_id(self._mj_model, "goal_whole")
        self._ball_geom_id = mjx_env.geom_id(self._mj_model, "objGeom")
        self._obj_z = jnp.array(0.03, dtype=jnp.float32)

    @property
    def xml_path(self):
        return sawyer_xml_path("sawyer_soccer.xml")

    def hand_init_pos(self) -> jax.Array:
        return as_f32([0.0, 0.6, 0.2])

    def mocap_low(self) -> jax.Array:
        return as_f32([-0.5, 0.40, 0.05])

    def mocap_high(self) -> jax.Array:
        return as_f32([0.5, 1.0, 0.5])

    def random_reset_bounds(self):
        obj_low = as_f32([-0.1, 0.6, 0.03])
        obj_high = as_f32([0.1, 0.7, 0.03])
        goal_low = as_f32([-0.1, 0.8, 0.0])
        goal_high = as_f32([0.1, 0.9, 0.0])
        return jnp.concatenate([obj_low, goal_low]), jnp.concatenate([obj_high, goal_high])

    def get_obj_pos(self, data):
        return mjx_env.body_xpos(data, jnp.array(self._ball_body_id))

    def get_obj_quat(self, data):
        return geom_quat_xyzw(data, self._ball_geom_id)

    def reset_state_from_rand_vec(self, rand_vec):
        obj_init_pos = jnp.concatenate([rand_vec[:2], jnp.array([self._obj_z])])
        return {
            "obj_init_pos": obj_init_pos,
            "goal_pos": rand_vec[3:6],
            "goal_body_pos": rand_vec[3:6],
            "rand_vec": rand_vec,
        }

    def sample_reset_state(self, rng):
        low, high = self.random_reset_bounds()
        vec, _ = sample_hstack_rand_vec(rng, low, high, min_xy_dist=0.15)
        return self.reset_state_from_rand_vec(vec)

    def apply_reset_state(self, model, data, reset_state):
        body_pos = model.body_pos.at[self._goal_body_id].set(reset_state["goal_body_pos"])
        model = model.replace(body_pos=body_pos)
        data = set_free_joint_xyz(
            data, self._obj_qposadr, self._obj_qveladr, reset_state["obj_init_pos"], ndim=3
        )
        return model, data

    def compute_reward(self, data, action, info):
        obj = self.get_obj_pos(data)
        tcp = self._tcp_center(data)
        scale = jnp.array([3.0, 1.0, 1.0])
        target_to_obj = jnp.linalg.norm((obj - info["goal_pos"]) * scale)
        target_to_obj_init = jnp.linalg.norm((obj - info["obj_init_pos"]) * scale)
        in_place = reward_utils.tolerance(
            target_to_obj, bounds=(0.0, self.TARGET_RADIUS), margin=target_to_obj_init, sigmoid="long_tail",
        )
        goal_line = info["goal_pos"][1] - 0.1
        offside = (obj[1] > goal_line) & (jnp.abs(obj[0] - info["goal_pos"][0]) > 0.10)
        in_place = jnp.where(
            offside,
            jnp.clip(in_place - 2.0 * ((obj[1] - goal_line) / (1.0 - goal_line)), 0.0, 1.0),
            in_place,
        )
        object_grasped = xz_plane_gripper_caging_reward(
            data, action, obj, info["obj_init_pos"], info["init_tcp"],
            info["init_left_pad"], info["init_right_pad"],
            self._leftpad_body_id, self._rightpad_body_id, tcp, obj_radius=self.OBJ_RADIUS,
        )
        reward = 3.0 * object_grasped + 6.5 * in_place
        reward = jnp.where(target_to_obj < self.TARGET_RADIUS, 10.0, reward)
        obj_to_target = jnp.linalg.norm(obj - info["goal_pos"])
        tcp_to_obj = jnp.linalg.norm(obj - tcp)
        metrics = {
            "success": (target_to_obj <= 0.07).astype(jnp.float32),
            "near_object": (tcp_to_obj <= 0.03).astype(jnp.float32),
            "grasp_success": jnp.array(0.0),
            "grasp_reward": object_grasped,
            "in_place_reward": in_place,
            "obj_to_target": obj_to_target,
        }
        return reward, metrics
