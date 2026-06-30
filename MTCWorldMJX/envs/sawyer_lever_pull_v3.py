"""Sawyer lever pull environment (lever-pull-v3)."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from mujoco import mjx

from MTCWorldMJX import mjx_env
from MTCWorldMJX.asset_paths import sawyer_xml_path
from MTCWorldMJX.envs._helpers import as_f32, geom_quat_xyzw, sample_obj_only_rand_vec
from MTCWorldMJX.envs.sawyer_xyz import SawyerXYZConfig, SawyerXYZEnv
from MTCWorldMJX.utils import reward as reward_utils


class SawyerLeverPullEnvV3(SawyerXYZEnv):
    """Pull a lever upward to a target (lever-pull-v3)."""

    LEVER_RADIUS = 0.2

    def __init__(self, config: SawyerXYZConfig | None = None):
        super().__init__(config)
        m = self._mj_model
        self._lever_body_id = mjx_env.body_id(m, "lever")
        self._goal_site_id = mjx_env.site_id(m, "goal")
        self._lever_start_site_id = mjx_env.site_id(m, "leverStart")
        self._obj_geom_id = mjx_env.geom_id(m, "objGeom")
        self._lever_joint_qposadr = int(
            m.jnt_qposadr[mjx_env.joint_id(m, "LeverAxis")]
        )

    @property
    def xml_path(self):
        return sawyer_xml_path("sawyer_lever_pull.xml")

    def hand_init_pos(self) -> jax.Array:
        return jnp.array([0.0, 0.4, 0.2], dtype=jnp.float32)

    def mocap_low(self) -> jax.Array:
        return jnp.array([-0.5, 0.40, -0.15], dtype=jnp.float32)

    def mocap_high(self) -> jax.Array:
        return jnp.array([0.5, 1.0, 0.5], dtype=jnp.float32)

    def random_reset_bounds(self) -> tuple[jax.Array, jax.Array]:
        low = as_f32([-0.1, 0.7, 0.0])
        high = as_f32([0.1, 0.8, 0.0])
        return low, high

    def get_obj_pos(self, data: mjx.Data) -> jax.Array:
        return mjx_env.site_xpos(data, jnp.array(self._lever_start_site_id))

    def get_obj_quat(self, data: mjx.Data) -> jax.Array:
        return geom_quat_xyzw(data, self._obj_geom_id)

    def reset_state_from_rand_vec(self, rand_vec: jax.Array) -> dict[str, jax.Array]:
        return self._lever_reset_state(rand_vec[:3], rand_vec)

    def sample_reset_state(self, rng: jax.Array) -> dict[str, jax.Array]:
        low, high = self.random_reset_bounds()
        obj_init_pos = sample_obj_only_rand_vec(rng, low, high)
        return self._lever_reset_state(obj_init_pos, obj_init_pos)

    def _lever_reset_state(
        self,
        obj_init_pos: jax.Array,
        rand_vec: jax.Array,
    ) -> dict[str, jax.Array]:
        lever_pos_init = obj_init_pos + jnp.array(
            [0.12, -self.LEVER_RADIUS, 0.25], dtype=jnp.float32
        )
        goal_pos = obj_init_pos + jnp.array(
            [0.12, 0.0, 0.25 + self.LEVER_RADIUS], dtype=jnp.float32
        )
        return {
            "obj_init_pos": obj_init_pos,
            "goal_pos": goal_pos,
            "lever_pos_init": lever_pos_init,
            "rand_vec": rand_vec,
        }

    def apply_reset_state(
        self,
        model: mjx.Model,
        data: mjx.Data,
        reset_state: dict[str, jax.Array],
    ) -> tuple[mjx.Model, mjx.Data]:
        obj_pos = reset_state["obj_init_pos"]
        body_pos = model.body_pos.at[self._lever_body_id].set(obj_pos)
        site_pos = model.site_pos.at[self._goal_site_id].set(reset_state["goal_pos"])
        model = model.replace(body_pos=body_pos, site_pos=site_pos)
        return model, data

    def compute_reward(
        self,
        data: mjx.Data,
        action: jax.Array,
        info: dict[str, Any],
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        del action
        gripper = self._hand_pos(data)
        lever = self.get_obj_pos(data)
        target = info["goal_pos"]

        scale = jnp.array([4.0, 1.0, 4.0], dtype=jnp.float32)
        offset = jnp.array([0.0, 0.055, 0.07], dtype=jnp.float32)

        shoulder_to_lever = (gripper + offset - lever) * scale
        shoulder_to_lever_init = (info["init_tcp"] + offset - info["lever_pos_init"]) * scale

        ready_to_lift = reward_utils.tolerance(
            jnp.linalg.norm(shoulder_to_lever),
            bounds=(0.0, 0.02),
            margin=jnp.linalg.norm(shoulder_to_lever_init),
            sigmoid="long_tail",
        )

        lever_angle = -data.qpos[self._lever_joint_qposadr]
        lever_error = jnp.abs(lever_angle - (jnp.pi / 2.0))
        lever_engagement = reward_utils.tolerance(
            lever_error,
            bounds=(0.0, jnp.pi / 48.0),
            margin=(jnp.pi / 2.0) - (jnp.pi / 12.0),
            sigmoid="long_tail",
        )

        obj_to_target = jnp.linalg.norm(lever - target)
        in_place_margin = jnp.linalg.norm(info["lever_pos_init"] - target)
        in_place = reward_utils.tolerance(
            obj_to_target,
            bounds=(0.0, 0.04),
            margin=in_place_margin,
            sigmoid="long_tail",
        )

        reward = 10.0 * reward_utils.hamacher_product(ready_to_lift, in_place)
        metrics = {
            "success": (lever_error <= jnp.pi / 24.0).astype(jnp.float32),
            "near_object": (jnp.linalg.norm(shoulder_to_lever) < 0.03).astype(jnp.float32),
            "grasp_success": (ready_to_lift > 0.9).astype(jnp.float32),
            "grasp_reward": ready_to_lift,
            "in_place_reward": lever_engagement,
            "obj_to_target": jnp.linalg.norm(shoulder_to_lever),
        }
        return reward, metrics
