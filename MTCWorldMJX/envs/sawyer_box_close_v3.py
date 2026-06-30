"""Sawyer box-close environment (box-close-v3)."""

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


class SawyerBoxCloseEnvV3(SawyerXYZEnv):
    def __init__(self, config: SawyerXYZConfig | None = None):
        super().__init__(config)
        self._top_link_body_id = mjx_env.body_id(self._mj_model, "top_link")
        self._boxbody_body_id = mjx_env.body_id(self._mj_model, "boxbody")
        self._goal_site_id = mjx_env.site_id(self._mj_model, "goal")
        self._box_z = jnp.array(0.02, dtype=jnp.float32)

    @property
    def xml_path(self):
        return sawyer_xml_path("sawyer_box.xml")

    def hand_init_pos(self) -> jax.Array:
        return as_f32([0.0, 0.6, 0.2])

    def mocap_low(self) -> jax.Array:
        return as_f32([-0.5, 0.40, 0.05])

    def mocap_high(self) -> jax.Array:
        return as_f32([0.5, 1.0, 0.5])

    def random_reset_bounds(self) -> tuple[jax.Array, jax.Array]:
        obj_low = as_f32([-0.05, 0.5, 0.02])
        obj_high = as_f32([0.05, 0.55, 0.02])
        goal_low = as_f32([-0.1, 0.7, 0.133])
        goal_high = as_f32([0.1, 0.8, 0.133])
        return jnp.concatenate([obj_low, goal_low]), jnp.concatenate([obj_high, goal_high])

    def get_obj_pos(self, data: mjx.Data) -> jax.Array:
        return mjx_env.body_xpos(data, jnp.array(self._top_link_body_id))

    def get_obj_quat(self, data: mjx.Data) -> jax.Array:
        return mjx_env.body_xquat(data, jnp.array(self._top_link_body_id))

    def reset_state_from_rand_vec(self, rand_vec: jax.Array) -> dict[str, jax.Array]:
        obj_init_pos = jnp.concatenate([rand_vec[:2], jnp.array([self._box_z])])
        goal_pos = rand_vec[3:6]
        box_height = self._mj_model.body_pos[self._boxbody_body_id][2]
        return {
            "obj_init_pos": obj_init_pos,
            "goal_pos": goal_pos,
            "box_pos": jnp.concatenate([goal_pos[:2], jnp.array([box_height])]),
            "rand_vec": rand_vec,
        }

    def sample_reset_state(self, rng: jax.Array) -> dict[str, jax.Array]:
        low, high = self.random_reset_bounds()
        vec, _ = sample_hstack_rand_vec(rng, low, high, min_xy_dist=0.25)
        return self.reset_state_from_rand_vec(vec)

    def apply_reset_state(self, model, data, reset_state):
        box_pos = reset_state["box_pos"]
        body_pos = model.body_pos.at[self._boxbody_body_id].set(box_pos)
        site_pos = model.site_pos.at[self._goal_site_id].set(reset_state["goal_pos"])
        model = model.replace(body_pos=body_pos, site_pos=site_pos)
        data = set_free_joint_xyz(
            data, self._obj_qposadr, self._obj_qveladr, reset_state["obj_init_pos"], ndim=3
        )
        return model, data

    def finalize_reset(self, model, data, reset_state):
        data = mjx_env.step(model, data, jnp.array([-1.0, 1.0]), self.n_substeps)
        return model, data, reset_state

    @staticmethod
    def _reward_grab_effort(action: jax.Array) -> jax.Array:
        return jnp.clip((jnp.clip(action[3], -1.0, 1.0) + 1.0) / 2.0, 0.0, 1.0)

    @staticmethod
    def _reward_quat(obj_quat: jax.Array) -> jax.Array:
        ideal = jnp.array([0.707, 0.0, 0.0, 0.707])
        error = jnp.linalg.norm(obj_quat - ideal)
        return jnp.maximum(1.0 - error / 0.2, 0.0)

    @staticmethod
    def _reward_pos(hand: jax.Array, lid: jax.Array, target_pos: jax.Array):
        threshold = 0.02
        radius = jnp.linalg.norm(hand[:2] - lid[:2])
        floor = jnp.where(
            radius <= threshold,
            0.0,
            0.04 * jnp.log(jnp.maximum(radius - threshold, 1e-8)) + 0.4,
        )
        above_floor = jnp.where(
            hand[2] >= floor,
            1.0,
            reward_utils.tolerance(
                floor - hand[2], bounds=(0.0, 0.01), margin=floor / 2.0, sigmoid="long_tail"
            ),
        )
        in_place = reward_utils.tolerance(
            jnp.linalg.norm(hand - lid),
            bounds=(0.0, 0.02),
            margin=0.5,
            sigmoid="long_tail",
        )
        ready = reward_utils.hamacher_product(above_floor, in_place)
        pos_error = (target_pos - lid) * jnp.array([1.0, 1.0, 3.0])
        lifted = 0.2 * (lid[2] > 0.04).astype(jnp.float32) + 0.8 * reward_utils.tolerance(
            jnp.linalg.norm(pos_error), bounds=(0.0, 0.05), margin=0.25, sigmoid="long_tail"
        )
        return ready, lifted

    def compute_reward(self, data, action, info):
        hand = self._hand_pos(data)
        lid = self.get_obj_pos(data) + jnp.array([0.0, 0.0, 0.02])
        target = info["goal_pos"]
        reward_grab = self._reward_grab_effort(action)
        reward_quat = self._reward_quat(self.get_obj_quat(data))
        ready, lifted = self._reward_pos(hand, lid, target)
        reward = 2.0 * reward_utils.hamacher_product(reward_grab, ready) + 8.0 * lifted
        success = jnp.linalg.norm(self.get_obj_pos(data) - target) < 0.08
        reward = jnp.where(success, 10.0, reward) * reward_quat
        metrics = {
            "success": success.astype(jnp.float32),
            "near_object": ready,
            "grasp_success": (reward_grab >= 0.5).astype(jnp.float32),
            "grasp_reward": reward_grab,
            "in_place_reward": lifted,
            "obj_to_target": jnp.array(0.0),
        }
        return reward, metrics
