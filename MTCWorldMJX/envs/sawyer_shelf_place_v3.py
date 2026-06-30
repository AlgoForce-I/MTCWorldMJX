"""Sawyer shelf-place environment (shelf-place-v3)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from mujoco import mjx

from MTCWorldMJX import mjx_env
from MTCWorldMJX.asset_paths import sawyer_xml_path
from MTCWorldMJX.envs._helpers import as_f32, geom_quat_xyzw, sample_hstack_rand_vec, set_free_joint_xyz
from MTCWorldMJX.envs.sawyer_xyz import SawyerXYZConfig, SawyerXYZEnv
from MTCWorldMJX.utils import reward as reward_utils


class SawyerShelfPlaceEnvV3(SawyerXYZEnv):
    TARGET_RADIUS = 0.05

    def __init__(self, config: SawyerXYZConfig | None = None):
        super().__init__(config)
        self._obj_body_id = mjx_env.body_id(self._mj_model, "obj")
        self._obj_geom_id = mjx_env.geom_id(self._mj_model, "objGeom")
        self._shelf_body_id = mjx_env.body_id(self._mj_model, "shelf")
        self._goal_site_id = mjx_env.site_id(self._mj_model, "goal")
        self._default_obj_z = jnp.array(0.02, dtype=jnp.float32)

    @property
    def xml_path(self):
        return sawyer_xml_path("sawyer_shelf_placing.xml")

    def hand_init_pos(self):
        return as_f32([0.0, 0.6, 0.2])

    def mocap_low(self):
        return as_f32([-0.5, 0.40, 0.05])

    def mocap_high(self):
        return as_f32([0.5, 1.0, 0.5])

    def random_reset_bounds(self):
        obj_low = as_f32([-0.1, 0.5, 0.019])
        obj_high = as_f32([0.1, 0.6, 0.021])
        goal_low = as_f32([-0.1, 0.8, 0.299])
        goal_high = as_f32([0.1, 0.9, 0.301])
        return jnp.concatenate([obj_low, goal_low]), jnp.concatenate([obj_high, goal_high])

    def get_obj_pos(self, data):
        return mjx_env.body_xpos(data, jnp.array(self._obj_body_id))

    def get_obj_quat(self, data):
        return geom_quat_xyzw(data, self._obj_geom_id)

    def reset_state_from_rand_vec(self, rand_vec):
        shelf_pos = rand_vec[3:6] - jnp.array([0.0, 0.0, 0.3])
        return {
            "obj_xy": rand_vec[:2],
            "goal_pos": rand_vec[3:6],
            "shelf_pos": shelf_pos,
            "rand_vec": rand_vec,
        }

    def prepare_reset_state(self, model, data, reset_state):
        obj_z = mjx_env.body_xpos(data, jnp.array(self._obj_body_id))[2]
        obj_init_pos = jnp.concatenate([reset_state["obj_xy"], jnp.array([obj_z])])
        return model, data, {**reset_state, "obj_init_pos": obj_init_pos}

    def sample_reset_state(self, rng):
        low, high = self.random_reset_bounds()
        vec, _ = sample_hstack_rand_vec(rng, low, high, min_xy_dist=0.1)
        return self.reset_state_from_rand_vec(vec)

    def apply_reset_state(self, model, data, reset_state):
        body_pos = model.body_pos.at[self._shelf_body_id].set(reset_state["shelf_pos"])
        model = model.replace(body_pos=body_pos)
        data = set_free_joint_xyz(
            data, self._obj_qposadr, self._obj_qveladr, reset_state["obj_init_pos"], ndim=3
        )
        return model, data

    def _resolve_goal_pos(self, model, data, reset_state):
        del model, reset_state
        return mjx_env.site_xpos(data, jnp.array(self._goal_site_id))

    def compute_reward(self, data, action, info):
        obj = self.get_obj_pos(data)
        tcp = self._tcp_center(data)
        tcp_opened = self._gripper_opening(data)
        target = info["goal_pos"]
        obj_to_target = jnp.linalg.norm(obj - target)
        in_place = reward_utils.tolerance(
            obj_to_target, bounds=(0.0, self.TARGET_RADIUS),
            margin=jnp.linalg.norm(info["obj_init_pos"] - target), sigmoid="long_tail",
        )
        object_grasped = self._gripper_caging_reward(
            data, action, obj, info["obj_init_pos"], info["init_tcp"],
            obj_radius=0.02, pad_success_thresh=0.05, object_reach_radius=0.01, xz_thresh=0.01,
        )
        reward = reward_utils.hamacher_product(object_grasped, in_place)
        behind = (obj[1] > target[1]) & (0.0 < obj[2]) & (obj[2] < 0.24)
        in_place = jnp.where(behind, 0.0, in_place)
        near = (jnp.linalg.norm(obj - tcp) < 0.025) & (tcp_opened > 0) & (obj[2] - 0.01 > info["obj_init_pos"][2])
        reward = jnp.where(near, reward + 1.0 + 5.0 * in_place, reward)
        reward = jnp.where(obj_to_target < self.TARGET_RADIUS, 10.0, reward)
        tcp_to_obj = jnp.linalg.norm(obj - tcp)
        metrics = {
            "success": (obj_to_target <= 0.07).astype(jnp.float32),
            "near_object": (tcp_to_obj <= 0.03).astype(jnp.float32),
            "grasp_success": near.astype(jnp.float32),
            "grasp_reward": object_grasped,
            "in_place_reward": in_place,
            "obj_to_target": obj_to_target,
        }
        return reward, metrics
