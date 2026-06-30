"""Sawyer peg-insert-side environment (peg-insert-side-v3)."""

from __future__ import annotations

import jax.numpy as jnp
from mujoco import mjx

from MTCWorldMJX import mjx_env
from MTCWorldMJX.asset_paths import sawyer_xml_path
from MTCWorldMJX.envs._helpers import as_f32, sample_hstack_rand_vec, set_free_joint_xyz, site_quat_xyzw
from MTCWorldMJX.envs.sawyer_xyz import SawyerXYZConfig, SawyerXYZEnv
from MTCWorldMJX.utils import reward as reward_utils


class SawyerPegInsertionSideEnvV3(SawyerXYZEnv):
    TARGET_RADIUS = 0.07

    def __init__(self, config: SawyerXYZConfig | None = None):
        super().__init__(config)
        self._peg_grasp_site_id = mjx_env.site_id(self._mj_model, "pegGrasp")
        self._peg_head_site_id = mjx_env.site_id(self._mj_model, "pegHead")
        self._box_body_id = mjx_env.body_id(self._mj_model, "box")
        self._goal_site_id = mjx_env.site_id(self._mj_model, "goal")
        self._brc1_site_id = mjx_env.site_id(self._mj_model, "bottom_right_corner_collision_box_1")
        self._tlc1_site_id = mjx_env.site_id(self._mj_model, "top_left_corner_collision_box_1")
        self._brc2_site_id = mjx_env.site_id(self._mj_model, "bottom_right_corner_collision_box_2")
        self._tlc2_site_id = mjx_env.site_id(self._mj_model, "top_left_corner_collision_box_2")
        self._peg_geom_id = mjx_env.geom_id(self._mj_model, "peg")

    @property
    def xml_path(self):
        return sawyer_xml_path("sawyer_peg_insertion_side.xml")

    def hand_init_pos(self):
        return as_f32([0.0, 0.6, 0.2])

    def mocap_low(self):
        return as_f32([-0.5, 0.40, 0.05])

    def mocap_high(self):
        return as_f32([0.5, 1.0, 0.5])

    def random_reset_bounds(self):
        obj_low = as_f32([0.0, 0.5, 0.02])
        obj_high = as_f32([0.2, 0.7, 0.02])
        goal_low = as_f32([-0.35, 0.4, -0.001])
        goal_high = as_f32([-0.25, 0.7, 0.001])
        return jnp.concatenate([obj_low, goal_low]), jnp.concatenate([obj_high, goal_high])

    def get_obj_pos(self, data):
        return mjx_env.site_xpos(data, jnp.array(self._peg_grasp_site_id))

    def get_obj_quat(self, data):
        return site_quat_xyzw(data, self._peg_grasp_site_id)

    def reset_state_from_rand_vec(self, rand_vec):
        peg_pos = rand_vec[:3]
        box_pos = rand_vec[3:6]
        goal_pos = box_pos + jnp.array([0.03, 0.0, 0.13])
        return {
            "obj_init_pos": peg_pos,
            "goal_pos": goal_pos,
            "box_pos": box_pos,
            "rand_vec": rand_vec,
        }

    def prepare_reset_state(self, model, data, reset_state):
        peg_head = mjx_env.site_xpos(data, jnp.array(self._peg_head_site_id))
        return model, data, {**reset_state, "peg_head_pos_init": peg_head}

    def sample_reset_state(self, rng):
        low, high = self.random_reset_bounds()
        vec, _ = sample_hstack_rand_vec(rng, low, high, min_xy_dist=0.1)
        return self.reset_state_from_rand_vec(vec)

    def apply_reset_state(self, model, data, reset_state):
        body_pos = model.body_pos.at[self._box_body_id].set(reset_state["box_pos"])
        site_pos = model.site_pos.at[self._goal_site_id].set(reset_state["goal_pos"])
        model = model.replace(body_pos=body_pos, site_pos=site_pos)
        return model, set_free_joint_xyz(
            data, self._obj_qposadr, self._obj_qveladr, reset_state["obj_init_pos"], ndim=3
        )

    def compute_reward(self, data, action, info):
        obj = self.get_obj_pos(data)
        obj_head = mjx_env.site_xpos(data, jnp.array(self._peg_head_site_id))
        tcp = self._tcp_center(data)
        tcp_opened = self._gripper_opening(data)
        scale = jnp.array([1.0, 2.0, 2.0])
        obj_to_target = jnp.linalg.norm((obj_head - info["goal_pos"]) * scale)
        in_place = reward_utils.tolerance(
            obj_to_target, bounds=(0.0, self.TARGET_RADIUS),
            margin=jnp.linalg.norm((info["peg_head_pos_init"] - info["goal_pos"]) * scale),
            sigmoid="long_tail",
        )
        boxes = reward_utils.hamacher_product(
            reward_utils.rect_prism_tolerance(
                obj_head,
                mjx_env.site_xpos(data, jnp.array(self._brc1_site_id)),
                mjx_env.site_xpos(data, jnp.array(self._tlc1_site_id)),
            ),
            reward_utils.rect_prism_tolerance(
                obj_head,
                mjx_env.site_xpos(data, jnp.array(self._brc2_site_id)),
                mjx_env.site_xpos(data, jnp.array(self._tlc2_site_id)),
            ),
        )
        in_place = reward_utils.hamacher_product(in_place, boxes)
        object_grasped = self._gripper_caging_reward(
            data, action, obj, info["obj_init_pos"], info["init_tcp"],
            obj_radius=0.0075, pad_success_thresh=0.03, object_reach_radius=0.01, xz_thresh=0.005,
            high_density=True,
        )
        tcp_to_obj = jnp.linalg.norm(obj - tcp)
        lifted = obj[2] - 0.01 > info["obj_init_pos"][2]
        object_grasped = jnp.where((tcp_to_obj < 0.08) & (tcp_opened > 0) & lifted, 1.0, object_grasped)
        reward = reward_utils.hamacher_product(object_grasped, in_place)
        reward = jnp.where((tcp_to_obj < 0.08) & (tcp_opened > 0) & lifted, reward + 1.0 + 5.0 * in_place, reward)
        reward = jnp.where(obj_to_target <= 0.07, 10.0, reward)
        metrics = {
            "success": (obj_to_target <= 0.07).astype(jnp.float32),
            "near_object": (tcp_to_obj <= 0.03).astype(jnp.float32),
            "grasp_success": ((tcp_to_obj < 0.02) & (tcp_opened > 0) & lifted).astype(jnp.float32),
            "grasp_reward": object_grasped,
            "in_place_reward": in_place,
            "obj_to_target": obj_to_target,
        }
        return reward, metrics
