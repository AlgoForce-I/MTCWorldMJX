"""Sawyer stick push/pull environments (stick-push-v3, stick-pull-v3)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from mujoco import mjx

from MTCWorldMJX import mjx_env
from MTCWorldMJX.asset_paths import sawyer_xml_path
from MTCWorldMJX.envs._helpers import (
    as_f32,
    geom_quat_xyzw,
    pack_interleaved_obj_obs,
    sample_hstack_rand_vec,
    set_free_joint_xyz,
)
from MTCWorldMJX.envs.sawyer_xyz import OBS_OBJ_MAX_LEN, SawyerXYZConfig, SawyerXYZEnv
from MTCWorldMJX.utils import reward as reward_utils


class _SawyerStickBase(SawyerXYZEnv):
    TARGET_RADIUS = 0.12

    def __init__(self, config: SawyerXYZConfig | None = None):
        super().__init__(config)
        self._stick_body_id = mjx_env.body_id(self._mj_model, "stick")
        self._object_body_id = mjx_env.body_id(self._mj_model, "object")
        self._insertion_site_id = mjx_env.site_id(self._mj_model, "insertion")
        self._stick_end_site_id = mjx_env.site_id(self._mj_model, "stick_end")
        self._goal_site_id = mjx_env.site_id(self._mj_model, "goal")
        self._stick_qposadr = int(self._mj_model.jnt_qposadr[9])
        self._stick_qveladr = int(self._mj_model.jnt_dofadr[9])
        self._obj_slide_qposadr = int(self._mj_model.jnt_qposadr[10])
        self._obj_slide_qveladr = int(self._mj_model.jnt_dofadr[10])
        self._default_obj_qpos = jnp.array([0.0, 0.0])

    def hand_init_pos(self):
        return as_f32([0.0, 0.6, 0.2])

    def mocap_low(self):
        return as_f32([-0.5, 0.40, 0.05])

    def mocap_high(self):
        return as_f32([0.5, 1.0, 0.5])

    @property
    def xml_path(self):
        return sawyer_xml_path("sawyer_stick_obj.xml")

    def _get_curr_obs_no_goal(self, data: mjx.Data) -> jax.Array:
        pos_hand = self._hand_pos(data)
        gripper = self._gripper_opening(data)[None]
        stick = mjx_env.body_xpos(data, jnp.array(self._stick_body_id))
        container = self._container_pos(data)
        stick_quat = geom_quat_xyzw(data, mjx_env.geom_id(self._mj_model, "stick"))
        obs_obj = pack_interleaved_obj_obs(
            jnp.concatenate([stick, container]),
            jnp.concatenate([stick_quat, jnp.zeros(4)]),
            max_len=OBS_OBJ_MAX_LEN,
        )
        return jnp.concatenate([pos_hand, gripper, obs_obj])

    def _handle_pos(self, data: mjx.Data) -> jax.Array:
        return mjx_env.site_xpos(data, jnp.array(self._insertion_site_id))

    def _container_pos(self, data: mjx.Data) -> jax.Array:
        return self._handle_pos(data) + jnp.array([0.0, 0.09, 0.0])

    def get_obj_pos(self, data):
        return mjx_env.body_xpos(data, jnp.array(self._stick_body_id))

    def get_obj_quat(self, data):
        return geom_quat_xyzw(data, mjx_env.geom_id(self._mj_model, "stick"))

    def apply_reset_state(self, model, data, reset_state):
        site_pos = model.site_pos.at[self._goal_site_id].set(reset_state["goal_pos"])
        model = model.replace(site_pos=site_pos)
        data = set_free_joint_xyz(
            data, self._stick_qposadr, self._stick_qveladr, reset_state["stick_init_pos"], ndim=3
        )
        qpos = data.qpos.at[self._obj_slide_qposadr : self._obj_slide_qposadr + 2].set(
            reset_state["obj_slide_qpos"]
        )
        # MetaWorld only zeros qvel[16] (slide-y), not slide-x at qvel[15].
        qvel = data.qvel.at[self._obj_slide_qveladr + 1].set(0.0)
        return model, data.replace(qpos=qpos, qvel=qvel)

    def finalize_reset(self, model, data, reset_state):
        obj_init_pos = mjx_env.body_xpos(data, jnp.array(self._object_body_id))
        reset_state = dict(reset_state)
        reset_state["obj_init_pos"] = obj_init_pos
        return model, data, reset_state


class SawyerStickPushEnvV3(_SawyerStickBase):
    def random_reset_bounds(self):
        obj_low = as_f32([-0.08, 0.58, 0.0])
        obj_high = as_f32([-0.03, 0.62, 0.001])
        goal_low = as_f32([0.399, 0.55, 0.1319])
        goal_high = as_f32([0.401, 0.6, 0.1321])
        return jnp.concatenate([obj_low, goal_low]), jnp.concatenate([obj_high, goal_high])

    def reset_state_from_rand_vec(self, rand_vec):
        stick_init = jnp.concatenate([rand_vec[:2], jnp.array([0.02])])
        goal_pos = jnp.concatenate([rand_vec[3:5], jnp.array([0.131])])
        return {
            "obj_init_pos": as_f32([0.2, 0.6, 0.0]),
            "goal_pos": goal_pos,
            "stick_init_pos": stick_init,
            "obj_slide_qpos": self._default_obj_qpos,
            "rand_vec": rand_vec,
        }

    def finalize_reset(self, model, data, reset_state):
        model, data, reset_state = super().finalize_reset(model, data, reset_state)
        insertion_z = mjx_env.site_xpos(data, jnp.array(self._insertion_site_id))[2]
        reset_state = dict(reset_state)
        reset_state["goal_pos"] = jnp.concatenate([reset_state["goal_pos"][:2], jnp.array([insertion_z])])
        return model, data, reset_state

    def sample_reset_state(self, rng):
        low, high = self.random_reset_bounds()
        vec, _ = sample_hstack_rand_vec(rng, low, high, min_xy_dist=0.1)
        return self.reset_state_from_rand_vec(vec)

    def compute_reward(self, data, action, info):
        stick = self.get_obj_pos(data) + jnp.array([0.015, 0.0, 0.0])
        container = self._container_pos(data)
        tcp = self._tcp_center(data)
        tcp_opened = self._gripper_opening(data)
        target = info["goal_pos"]
        tcp_to_stick = jnp.linalg.norm(stick - tcp)
        stick_in_place = reward_utils.tolerance(
            jnp.linalg.norm(stick - target), bounds=(0.0, self.TARGET_RADIUS),
            margin=jnp.linalg.norm(info["stick_init_pos"] - target) - self.TARGET_RADIUS,
            sigmoid="long_tail",
        )
        container_in_place = reward_utils.tolerance(
            jnp.linalg.norm(container - target), bounds=(0.0, self.TARGET_RADIUS),
            margin=jnp.linalg.norm(info["obj_init_pos"] - target) - self.TARGET_RADIUS,
            sigmoid="long_tail",
        )
        object_grasped = self._gripper_caging_reward(
            data, action, stick, info["stick_init_pos"], info["init_tcp"],
            obj_radius=0.04, pad_success_thresh=0.05, object_reach_radius=0.01, xz_thresh=0.01,
            high_density=True,
        )
        reward = object_grasped
        grasped = (
            (tcp_to_stick < 0.02) & (tcp_opened > 0) & (stick[2] - 0.01 > info["stick_init_pos"][2])
        )
        reward = jnp.where(
            grasped,
            2.0 + 5.0 * stick_in_place + 3.0 * container_in_place,
            reward,
        )
        container_to_target = jnp.linalg.norm(container - target)
        reward = jnp.where(container_to_target <= self.TARGET_RADIUS, 10.0, reward)
        metrics = {
            "success": (grasped & (container_to_target <= self.TARGET_RADIUS)).astype(jnp.float32),
            "near_object": (tcp_to_stick <= 0.03).astype(jnp.float32),
            "grasp_success": grasped.astype(jnp.float32),
            "grasp_reward": jnp.where(grasped, 1.0, object_grasped),
            "in_place_reward": stick_in_place,
            "obj_to_target": container_to_target,
        }
        return reward, metrics


class SawyerStickPullEnvV3(_SawyerStickBase):
    def __init__(self, config=None):
        super().__init__(config)
        self._default_obj_qpos = jnp.array([0.0, 0.09])

    def _container_pos(self, data: mjx.Data) -> jax.Array:
        # MetaWorld stick-pull uses raw insertion site pos (no +Y offset).
        return self._handle_pos(data)

    def mocap_low(self):
        return as_f32([-0.5, 0.35, 0.05])

    def random_reset_bounds(self):
        obj_low = as_f32([-0.1, 0.55, 0.0])
        obj_high = as_f32([0.0, 0.65, 0.001])
        goal_low = as_f32([0.35, 0.45, 0.0199])
        goal_high = as_f32([0.45, 0.55, 0.0201])
        return jnp.concatenate([obj_low, goal_low]), jnp.concatenate([obj_high, goal_high])

    def reset_state_from_rand_vec(self, rand_vec):
        stick_init = jnp.concatenate([rand_vec[:2], jnp.array([0.02])])
        goal_pos = jnp.concatenate([rand_vec[3:5], jnp.array([0.02])])
        return {
            "obj_init_pos": as_f32([0.2, 0.69, 0.04]),
            "goal_pos": goal_pos,
            "stick_init_pos": stick_init,
            "obj_slide_qpos": self._default_obj_qpos,
            "rand_vec": rand_vec,
        }

    def sample_reset_state(self, rng):
        low, high = self.random_reset_bounds()
        vec, _ = sample_hstack_rand_vec(rng, low, high, min_xy_dist=0.1)
        return self.reset_state_from_rand_vec(vec)

    @staticmethod
    def _stick_inserted(handle, end_of_stick):
        return (
            (end_of_stick[0] >= handle[0])
            & (jnp.abs(end_of_stick[1] - handle[1]) <= 0.040)
            & (jnp.abs(end_of_stick[2] - handle[2]) <= 0.060)
        )

    def compute_reward(self, data, action, info):
        stick = self.get_obj_pos(data)
        handle = self._handle_pos(data)
        end_of_stick = mjx_env.site_xpos(data, jnp.array(self._stick_end_site_id))
        container = handle + jnp.array([0.05, 0.0, 0.0])
        tcp = self._tcp_center(data)
        tcp_opened = self._gripper_opening(data)
        target = info["goal_pos"]
        tcp_to_stick = jnp.linalg.norm(stick - tcp)
        yz = jnp.array([1.0, 1.0, 2.0])
        stick_in_place = reward_utils.tolerance(
            jnp.linalg.norm((stick - container) * yz), bounds=(0.0, 0.05),
            margin=jnp.linalg.norm((info["stick_init_pos"] - (info["obj_init_pos"] + jnp.array([0.05, 0.0, 0.0]))) * yz),
            sigmoid="long_tail",
        )
        stick_in_place_2 = reward_utils.tolerance(
            jnp.linalg.norm(stick - target), bounds=(0.0, 0.05),
            margin=jnp.linalg.norm(info["stick_init_pos"] - target), sigmoid="long_tail",
        )
        container_in_place = reward_utils.tolerance(
            jnp.linalg.norm(container - target), bounds=(0.0, 0.05),
            margin=jnp.linalg.norm(info["obj_init_pos"] - target), sigmoid="long_tail",
        )
        object_grasped = self._gripper_caging_reward(
            data, action, stick, info["obj_init_pos"], info["init_tcp"],
            obj_radius=0.014, pad_success_thresh=0.05, object_reach_radius=0.01, xz_thresh=0.01,
            high_density=True,
        )
        grasped = (
            (tcp_to_stick < 0.02) & (tcp_opened > 0) & (stick[2] - 0.01 > info["stick_init_pos"][2])
        )
        object_grasped = jnp.where(grasped, 1.0, object_grasped)
        reward = reward_utils.hamacher_product(object_grasped, stick_in_place)
        reward = jnp.where(grasped, 1.0 + reward + 5.0 * stick_in_place, reward)
        inserted = self._stick_inserted(handle, end_of_stick)
        reward = jnp.where(
            grasped & inserted,
            1.0 + reward + 5.0 + 2.0 * stick_in_place_2 + 1.0 * container_in_place,
            reward,
        )
        handle_to_target = jnp.linalg.norm(handle - target)
        reward = jnp.where(handle_to_target <= 0.12, 10.0, reward)
        metrics = {
            "success": (inserted & (handle_to_target <= 0.12)).astype(jnp.float32),
            "near_object": (tcp_to_stick <= 0.03).astype(jnp.float32),
            "grasp_success": grasped.astype(jnp.float32),
            "grasp_reward": object_grasped,
            "in_place_reward": stick_in_place,
            "obj_to_target": handle_to_target,
        }
        return reward, metrics
