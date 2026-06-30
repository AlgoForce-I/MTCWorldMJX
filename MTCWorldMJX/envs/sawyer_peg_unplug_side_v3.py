"""Sawyer peg-unplug-side environment (peg-unplug-side-v3)."""

from __future__ import annotations

import jax.numpy as jnp
from mujoco import mjx

from MTCWorldMJX import mjx_env
from MTCWorldMJX.asset_paths import sawyer_xml_path
from MTCWorldMJX.envs._helpers import as_f32, set_free_joint_pos_zero_linear_vel
from MTCWorldMJX.envs.sawyer_xyz import SawyerXYZConfig, SawyerXYZEnv
from MTCWorldMJX.utils import reward as reward_utils


class SawyerPegUnplugSideEnvV3(SawyerXYZEnv):
    def __init__(self, config: SawyerXYZConfig | None = None):
        super().__init__(config)
        self._peg_end_site_id = mjx_env.site_id(self._mj_model, "pegEnd")
        self._plug_body_id = mjx_env.body_id(self._mj_model, "plug1")
        self._box_body_id = mjx_env.body_id(self._mj_model, "box")
        self._goal_site_id = mjx_env.site_id(self._mj_model, "goal")

    def _uses_metaworld_double_reset(self) -> bool:
        return True

    @property
    def xml_path(self):
        return sawyer_xml_path("sawyer_peg_unplug_side.xml")

    def hand_init_pos(self):
        return as_f32([0.0, 0.6, 0.2])

    def mocap_low(self):
        return as_f32([-0.5, 0.40, 0.05])

    def mocap_high(self):
        return as_f32([0.5, 1.0, 0.5])

    def random_reset_bounds(self):
        return as_f32([-0.25, 0.6, -0.001]), as_f32([-0.15, 0.8, 0.001])

    def get_obj_pos(self, data):
        return mjx_env.site_xpos(data, jnp.array(self._peg_end_site_id))

    def get_obj_quat(self, data):
        return mjx_env.body_xquat(data, jnp.array(self._plug_body_id))

    def reset_state_from_rand_vec(self, rand_vec):
        box_pos = rand_vec[:3]
        plug_pos = box_pos + jnp.array([0.044, 0.0, 0.131])
        goal_pos = plug_pos + jnp.array([0.15, 0.0, 0.0])
        return {
            "obj_init_pos": plug_pos,
            "goal_pos": goal_pos,
            "box_pos": box_pos,
            "plug_pos": plug_pos,
            "rand_vec": rand_vec,
        }

    def sample_reset_state(self, rng):
        import jax
        low, high = self.random_reset_bounds()
        vec = jax.random.uniform(rng, shape=low.shape, minval=low, maxval=high)
        return self.reset_state_from_rand_vec(vec)

    def apply_reset_state(self, model, data, reset_state):
        body_pos = model.body_pos.at[self._box_body_id].set(reset_state["box_pos"])
        site_pos = model.site_pos.at[self._goal_site_id].set(reset_state["goal_pos"])
        model = model.replace(body_pos=body_pos, site_pos=site_pos)
        data = set_free_joint_pos_zero_linear_vel(
            data,
            self._obj_qposadr,
            self._obj_qveladr,
            reset_state["plug_pos"],
            quat_wxyz=jnp.array([1.0, 0.0, 0.0, 0.0]),
        )
        return model, data

    def _resolve_goal_pos(self, model, data, reset_state):
        del model, data
        return reset_state["goal_pos"]

    def compute_reward(self, data, action, info):
        obj = self.get_obj_pos(data)
        tcp = self._tcp_center(data)
        tcp_opened = self._gripper_opening(data)
        target = info["goal_pos"]
        obj_to_target = jnp.linalg.norm(obj - target)
        tcp_to_obj = jnp.linalg.norm(obj - tcp)
        object_grasped = self._gripper_caging_reward(
            data, action, obj, info["obj_init_pos"], info["init_tcp"],
            obj_radius=0.025, pad_success_thresh=0.05, object_reach_radius=0.01, xz_thresh=0.005,
            desired_gripper_effort=0.8, high_density=True,
        )
        in_place = reward_utils.tolerance(
            obj_to_target, bounds=(0.0, 0.05),
            margin=jnp.linalg.norm(info["obj_init_pos"] - target), sigmoid="long_tail",
        )
        grasp_success = (tcp_opened > 0.5) & (obj[0] - info["obj_init_pos"][0] > 0.015)
        reward = 2.0 * object_grasped
        reward = jnp.where(
            grasp_success & (tcp_to_obj < 0.035),
            1.0 + 2.0 * object_grasped + 5.0 * in_place,
            reward,
        )
        reward = jnp.where(obj_to_target <= 0.05, 10.0, reward)
        metrics = {
            "success": (obj_to_target <= 0.07).astype(jnp.float32),
            "near_object": (tcp_to_obj <= 0.03).astype(jnp.float32),
            "grasp_success": grasp_success.astype(jnp.float32),
            "grasp_reward": object_grasped,
            "in_place_reward": in_place,
            "obj_to_target": obj_to_target,
        }
        return reward, metrics

    def finalize_reset(self, model, data, reset_state):
        data = mjx.forward(model, data)
        peg_end = mjx_env.site_xpos(data, jnp.array(self._peg_end_site_id))
        reset_state = dict(reset_state)
        reset_state["obj_init_pos"] = peg_end
        return model, data, reset_state
