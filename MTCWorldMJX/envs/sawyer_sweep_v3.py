"""Sawyer sweep environment (sweep-v3)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from mujoco import mjx

from MTCWorldMJX import mjx_env
from MTCWorldMJX.asset_paths import sawyer_xml_path
from MTCWorldMJX.envs._helpers import as_f32, set_free_joint_xyz, xz_plane_gripper_caging_reward
from MTCWorldMJX.envs.sawyer_xyz import SawyerXYZConfig, SawyerXYZEnv
from MTCWorldMJX.utils import reward as reward_utils


class SawyerSweepEnvV3(SawyerXYZEnv):
    OBJ_RADIUS = 0.02
    TARGET_RADIUS = 0.05

    def __init__(self, config: SawyerXYZConfig | None = None):
        super().__init__(config)
        self._obj_body_id = mjx_env.body_id(self._mj_model, "obj")
        self._goal_site_id = mjx_env.site_id(self._mj_model, "goal")
        self._obj_z = jnp.array(0.02, dtype=jnp.float32)

    @property
    def xml_path(self):
        return sawyer_xml_path("sawyer_sweep_v3.xml")

    def hand_init_pos(self):
        return as_f32([0.0, 0.6, 0.2])

    def mocap_low(self):
        return as_f32([-0.5, 0.40, 0.05])

    def mocap_high(self):
        return as_f32([0.5, 1.0, 0.5])

    def random_reset_bounds(self):
        return as_f32([-0.1, 0.6, 0.02]), as_f32([0.1, 0.7, 0.02])

    def get_obj_pos(self, data):
        return mjx_env.body_xpos(data, jnp.array(self._obj_body_id))

    def get_obj_quat(self, data):
        return mjx_env.body_xquat(data, jnp.array(self._obj_body_id))

    def reset_state_from_rand_vec(self, rand_vec):
        obj_init_pos = jnp.concatenate([rand_vec[:2], jnp.array([self._obj_z])])
        goal_pos = jnp.array([0.5, rand_vec[1], 0.01])
        return {"obj_init_pos": obj_init_pos, "goal_pos": goal_pos, "rand_vec": rand_vec}

    def sample_reset_state(self, rng):
        low, high = self.random_reset_bounds()
        vec = jax.random.uniform(rng, shape=low.shape, minval=low, maxval=high)
        return self.reset_state_from_rand_vec(vec)

    def apply_reset_state(self, model, data, reset_state):
        site_pos = model.site_pos.at[self._goal_site_id].set(reset_state["goal_pos"])
        model = model.replace(site_pos=site_pos)
        return model, set_free_joint_xyz(
            data, self._obj_qposadr, self._obj_qveladr, reset_state["obj_init_pos"], ndim=3
        )

    def compute_reward(self, data, action, info):
        obj = self.get_obj_pos(data)
        tcp = self._tcp_center(data)
        target = info["goal_pos"]
        obj_to_target = jnp.linalg.norm(obj - target)
        in_place = reward_utils.tolerance(
            obj_to_target, bounds=(0.0, self.TARGET_RADIUS),
            margin=jnp.linalg.norm(info["obj_init_pos"] - target), sigmoid="long_tail",
        )
        object_grasped = xz_plane_gripper_caging_reward(
            data, action, obj, info["obj_init_pos"], info["init_tcp"],
            info["init_left_pad"], info["init_right_pad"],
            self._leftpad_body_id, self._rightpad_body_id, tcp, obj_radius=self.OBJ_RADIUS,
        )
        in_place_grasped = reward_utils.hamacher_product(object_grasped, in_place)
        reward = 2.0 * object_grasped + 6.0 * in_place_grasped
        reward = jnp.where(obj_to_target < self.TARGET_RADIUS, 10.0, reward)
        tcp_to_obj = jnp.linalg.norm(obj - tcp)
        metrics = {
            "success": (obj_to_target <= 0.05).astype(jnp.float32),
            "near_object": (tcp_to_obj <= 0.03).astype(jnp.float32),
            "grasp_success": jnp.array(0.0),
            "grasp_reward": object_grasped,
            "in_place_reward": in_place,
            "obj_to_target": obj_to_target,
        }
        return reward, metrics
