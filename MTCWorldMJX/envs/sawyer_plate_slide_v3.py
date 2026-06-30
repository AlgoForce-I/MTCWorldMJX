"""Shared plate-slide task logic for Sawyer plate environments."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from mujoco import mjx

from MTCWorldMJX import mjx_env
from MTCWorldMJX.envs._helpers import as_f32, geom_quat_xyzw, set_slide_joint_xy
from MTCWorldMJX.envs.sawyer_xyz import SawyerXYZConfig, SawyerXYZEnv
from MTCWorldMJX.utils import reward as reward_utils


class _SawyerPlateSlideBase(SawyerXYZEnv):
    TARGET_RADIUS = 0.05

    def __init__(self, config: SawyerXYZConfig | None = None):
        super().__init__(config)
        self._puck_geom_id = mjx_env.geom_id(self._mj_model, "puck")
        self._puck_goal_body_id = mjx_env.body_id(self._mj_model, "puck_goal")
        self._goal_site_id = mjx_env.site_id(self._mj_model, "goal")
        self._slide_qposadr = int(self._mj_model.jnt_qposadr[9])
        self._slide_qveladr = int(self._mj_model.jnt_dofadr[9])

    def hand_init_pos(self) -> jax.Array:
        return as_f32([0.0, 0.6, 0.2])

    def mocap_low(self) -> jax.Array:
        return as_f32([-0.5, 0.40, 0.05])

    def mocap_high(self) -> jax.Array:
        return as_f32([0.5, 1.0, 0.5])

    def get_obj_pos(self, data: mjx.Data) -> jax.Array:
        return mjx_env.geom_xpos(data, jnp.array(self._puck_geom_id))

    def get_obj_quat(self, data: mjx.Data) -> jax.Array:
        return geom_quat_xyzw(data, self._puck_geom_id)

    def _set_puck_slide(self, data: mjx.Data, xy: jax.Array) -> mjx.Data:
        return set_slide_joint_xy(data, self._slide_qposadr, self._slide_qveladr, xy)

    def reset_state_from_rand_vec(self, rand_vec: jax.Array) -> dict[str, jax.Array]:
        return {
            "obj_init_pos": rand_vec[:3],
            "goal_pos": rand_vec[3:6],
            "slide_xy": self._default_slide_xy(rand_vec),
            "rand_vec": rand_vec,
        }

    def sample_reset_state(self, rng: jax.Array) -> dict[str, jax.Array]:
        low, high = self.random_reset_bounds()
        vec = jax.random.uniform(rng, shape=low.shape, minval=low, maxval=high)
        return self.reset_state_from_rand_vec(vec)

    def apply_reset_state(self, model, data, reset_state):
        body_pos = model.body_pos.at[self._puck_goal_body_id].set(reset_state["goal_pos"])
        site_pos = model.site_pos.at[self._goal_site_id].set(reset_state["goal_pos"])
        model = model.replace(body_pos=body_pos, site_pos=site_pos)
        data = self._set_puck_slide(data, reset_state["slide_xy"])
        return model, data

    def _default_slide_xy(self, rand_vec: jax.Array) -> jax.Array:
        del rand_vec
        return jnp.zeros(2)

    def _plate_reward(self, data, info, tcp_opened, *, subtract_target_radius: bool = False):
        tcp = self._tcp_center(data)
        obj = self.get_obj_pos(data)
        target = info["goal_pos"]
        obj_to_target = jnp.linalg.norm(obj - target)
        in_place_margin = jnp.linalg.norm(info["obj_init_pos"] - target)
        obj_grasped_margin = jnp.linalg.norm(info["init_tcp"] - info["obj_init_pos"])
        if subtract_target_radius:
            in_place_margin = in_place_margin - self.TARGET_RADIUS
            obj_grasped_margin = obj_grasped_margin - self.TARGET_RADIUS
        in_place = reward_utils.tolerance(
            obj_to_target,
            bounds=(0.0, self.TARGET_RADIUS),
            margin=in_place_margin,
            sigmoid="long_tail",
        )
        tcp_to_obj = jnp.linalg.norm(tcp - obj)
        object_grasped = reward_utils.tolerance(
            tcp_to_obj,
            bounds=(0.0, self.TARGET_RADIUS),
            margin=obj_grasped_margin,
            sigmoid="long_tail",
        )
        return obj, tcp, tcp_opened, obj_to_target, in_place, object_grasped, tcp_to_obj


class SawyerPlateSlideEnvV3(_SawyerPlateSlideBase):
    @property
    def xml_path(self):
        from MTCWorldMJX.asset_paths import sawyer_xml_path
        return sawyer_xml_path("sawyer_plate_slide.xml")

    def random_reset_bounds(self):
        obj = as_f32([0.0, 0.6, 0.0])
        goal_low = as_f32([-0.1, 0.85, 0.0])
        goal_high = as_f32([0.1, 0.9, 0.0])
        return jnp.concatenate([obj, goal_low]), jnp.concatenate([obj, goal_high])

    def _default_slide_xy(self, rand_vec):
        del rand_vec
        return jnp.zeros(2)

    def compute_reward(self, data, action, info):
        del action
        tcp_opened = self._gripper_opening(data)
        obj, tcp, tcp_opened, obj_to_target, in_place, object_grasped, tcp_to_obj = self._plate_reward(
            data, info, tcp_opened
        )
        reward = 8.0 * reward_utils.hamacher_product(object_grasped, in_place)
        reward = jnp.where(obj_to_target < self.TARGET_RADIUS, 10.0, reward)
        metrics = {
            "success": (obj_to_target <= 0.07).astype(jnp.float32),
            "near_object": (tcp_to_obj <= 0.03).astype(jnp.float32),
            "grasp_success": jnp.array(0.0),
            "grasp_reward": object_grasped,
            "in_place_reward": in_place,
            "obj_to_target": obj_to_target,
        }
        return reward, metrics


class SawyerPlateSlideBackEnvV3(_SawyerPlateSlideBase):
    @property
    def xml_path(self):
        from MTCWorldMJX.asset_paths import sawyer_xml_path
        return sawyer_xml_path("sawyer_plate_slide.xml")

    def random_reset_bounds(self):
        obj = as_f32([0.0, 0.85, 0.0])
        goal_low = as_f32([-0.1, 0.6, 0.015])
        goal_high = as_f32([0.1, 0.6, 0.015])
        return jnp.concatenate([obj, goal_low]), jnp.concatenate([obj, goal_high])

    def _default_slide_xy(self, rand_vec):
        del rand_vec
        return jnp.array([0.0, 0.15])

    def compute_reward(self, data, action, info):
        del action
        tcp_opened = self._gripper_opening(data)
        obj, tcp, tcp_opened, obj_to_target, in_place, object_grasped, tcp_to_obj = self._plate_reward(
            data, info, tcp_opened, subtract_target_radius=True
        )
        reward = 1.5 * object_grasped
        low_push = (tcp[2] <= 0.03) & (tcp_to_obj < 0.07)
        reward = jnp.where(low_push, 2.0 + 7.0 * in_place, reward)
        reward = jnp.where(obj_to_target < self.TARGET_RADIUS, 10.0, reward)
        metrics = {
            "success": (obj_to_target <= 0.07).astype(jnp.float32),
            "near_object": (tcp_to_obj <= 0.03).astype(jnp.float32),
            "grasp_success": jnp.array(0.0),
            "grasp_reward": object_grasped,
            "in_place_reward": in_place,
            "obj_to_target": obj_to_target,
        }
        return reward, metrics


class SawyerPlateSlideSideEnvV3(_SawyerPlateSlideBase):
    @property
    def xml_path(self):
        from MTCWorldMJX.asset_paths import sawyer_xml_path
        return sawyer_xml_path("sawyer_plate_slide_sideway.xml")

    def random_reset_bounds(self):
        obj = as_f32([0.0, 0.6, 0.0])
        goal_low = as_f32([-0.3, 0.54, 0.0])
        goal_high = as_f32([-0.25, 0.66, 0.0])
        return jnp.concatenate([obj, goal_low]), jnp.concatenate([obj, goal_high])

    def compute_reward(self, data, action, info):
        del action
        tcp_opened = self._gripper_opening(data)
        obj, tcp, tcp_opened, obj_to_target, in_place, object_grasped, tcp_to_obj = self._plate_reward(
            data, info, tcp_opened, subtract_target_radius=True
        )
        reward = 1.5 * object_grasped
        low_push = (tcp[2] <= 0.03) & (tcp_to_obj < 0.07)
        reward = jnp.where(low_push, 2.0 + 7.0 * in_place, reward)
        reward = jnp.where(obj_to_target < self.TARGET_RADIUS, 10.0, reward)
        metrics = {
            "success": (obj_to_target <= 0.07).astype(jnp.float32),
            "near_object": (tcp_to_obj <= 0.03).astype(jnp.float32),
            "grasp_success": jnp.array(0.0),
            "grasp_reward": object_grasped,
            "in_place_reward": in_place,
            "obj_to_target": obj_to_target,
        }
        return reward, metrics


class SawyerPlateSlideBackSideEnvV3(_SawyerPlateSlideBase):
    @property
    def xml_path(self):
        from MTCWorldMJX.asset_paths import sawyer_xml_path
        return sawyer_xml_path("sawyer_plate_slide_sideway.xml")

    def random_reset_bounds(self):
        obj = as_f32([-0.25, 0.6, 0.0])
        goal_low = as_f32([-0.05, 0.6, 0.015])
        goal_high = as_f32([0.15, 0.6, 0.015])
        return jnp.concatenate([obj, goal_low]), jnp.concatenate([obj, goal_high])

    def apply_reset_state(self, model, data, reset_state):
        body_pos = model.body_pos.at[self._puck_goal_body_id].set(reset_state["obj_init_pos"])
        site_pos = model.site_pos.at[self._goal_site_id].set(reset_state["goal_pos"])
        model = model.replace(body_pos=body_pos, site_pos=site_pos)
        data = self._set_puck_slide(data, reset_state["slide_xy"])
        return model, data

    def _default_slide_xy(self, rand_vec):
        del rand_vec
        return jnp.array([-0.15, 0.0])

    def compute_reward(self, data, action, info):
        del action
        tcp_opened = self._gripper_opening(data)
        obj, tcp, tcp_opened, obj_to_target, in_place, object_grasped, tcp_to_obj = self._plate_reward(
            data, info, tcp_opened, subtract_target_radius=True
        )
        reward = 1.5 * object_grasped
        low_push = (tcp[2] <= 0.03) & (tcp_to_obj < 0.07)
        reward = jnp.where(low_push, 2.0 + 7.0 * in_place, reward)
        reward = jnp.where(obj_to_target < self.TARGET_RADIUS, 10.0, reward)
        metrics = {
            "success": (obj_to_target <= 0.07).astype(jnp.float32),
            "near_object": (tcp_to_obj <= 0.03).astype(jnp.float32),
            "grasp_success": jnp.array(0.0),
            "grasp_reward": object_grasped,
            "in_place_reward": in_place,
            "obj_to_target": obj_to_target,
        }
        return reward, metrics
