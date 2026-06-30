"""Sawyer door close environment (door-close-v3)."""

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
    sample_obj_only_rand_vec,
    set_joint_qpos,
)
from MTCWorldMJX.envs.sawyer_xyz import SawyerXYZConfig, SawyerXYZEnv
from MTCWorldMJX.utils import reward as reward_utils


class SawyerDoorCloseEnvV3(SawyerXYZEnv):
    """Push a door closed (door-close-v3)."""

    _TARGET_RADIUS = 0.05
    _GOAL_OFFSET = as_f32([0.2, -0.2, 0.0])
    _OPEN_ANGLE = jnp.float32(-1.5708)

    def __init__(self, config: SawyerXYZConfig | None = None):
        super().__init__(config)
        self._door_body_id = mjx_env.body_id(self._mj_model, "door")
        self._handle_geom_id = mjx_env.geom_id(self._mj_model, "handle")
        self._goal_site_id = mjx_env.site_id(self._mj_model, "goal")
        jnt = self._mj_model.joint("doorjoint")
        self._door_qposadr = int(jnt.qposadr[0])
        self._door_qveladr = int(jnt.dofadr[0])

    @property
    def xml_path(self):
        return sawyer_xml_path("door_pull")

    def hand_init_pos(self) -> jax.Array:
        return as_f32([-0.5, 0.6, 0.2])

    def mocap_low(self) -> jax.Array:
        return as_f32([-0.5, 0.40, 0.05])

    def mocap_high(self) -> jax.Array:
        return as_f32([0.5, 1.0, 0.5])

    def random_reset_bounds(self) -> tuple[jax.Array, jax.Array]:
        low = as_f32([0.0, 0.85, 0.15])
        high = as_f32([0.1, 0.95, 0.15])
        return low, high

    def get_obj_pos(self, data: mjx.Data) -> jax.Array:
        return data.geom_xpos[self._handle_geom_id]

    def get_obj_quat(self, data: mjx.Data) -> jax.Array:
        return geom_quat_xyzw(data, self._handle_geom_id)

    def reset_state_from_rand_vec(self, rand_vec: jax.Array) -> dict[str, jax.Array]:
        obj_init_pos = rand_vec[:3]
        goal_pos = obj_init_pos + self._GOAL_OFFSET
        return {
            "obj_init_pos": obj_init_pos,
            "goal_pos": goal_pos,
            "rand_vec": rand_vec,
        }

    def sample_reset_state(self, rng: jax.Array) -> dict[str, jax.Array]:
        low, high = self.random_reset_bounds()
        vec = sample_obj_only_rand_vec(rng, low, high)
        return self.reset_state_from_rand_vec(vec)

    def apply_reset_state(
        self,
        model: mjx.Model,
        data: mjx.Data,
        reset_state: dict[str, jax.Array],
    ) -> tuple[mjx.Model, mjx.Data]:
        obj_pos = reset_state["obj_init_pos"]
        goal_pos = reset_state["goal_pos"]
        body_pos = model.body_pos.at[self._door_body_id].set(obj_pos)
        site_pos = model.site_pos.at[self._goal_site_id].set(goal_pos)
        model = model.replace(body_pos=body_pos, site_pos=site_pos)
        data = set_joint_qpos(
            data, self._door_qposadr, self._door_qveladr, self._OPEN_ANGLE
        )
        return model, data

    def compute_reward(
        self,
        data: mjx.Data,
        action: jax.Array,
        info: dict[str, Any],
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        del action
        tcp = self._tcp_center(data)
        obj = self.get_obj_pos(data)
        target = info["goal_pos"]
        tcp_to_target = jnp.linalg.norm(tcp - target)
        obj_to_target = jnp.linalg.norm(obj - target)
        in_place_margin = jnp.linalg.norm(info["obj_init_pos"] - target)
        in_place = reward_utils.tolerance(
            obj_to_target,
            bounds=(0.0, self._TARGET_RADIUS),
            margin=in_place_margin,
            sigmoid="gaussian",
        )
        hand_margin = jnp.linalg.norm(self.hand_init_pos() - obj) + 0.1
        hand_in_place = reward_utils.tolerance(
            tcp_to_target,
            bounds=(0.0, 0.25 * self._TARGET_RADIUS),
            margin=hand_margin,
            sigmoid="gaussian",
        )
        reward = 3.0 * hand_in_place + 6.0 * in_place
        reward = jnp.where(obj_to_target < self._TARGET_RADIUS, 10.0, reward)
        metrics = {
            "success": (obj_to_target <= 0.08).astype(jnp.float32),
            "near_object": jnp.array(0.0),
            "grasp_success": jnp.array(1.0),
            "grasp_reward": jnp.array(1.0),
            "in_place_reward": hand_in_place,
            "obj_to_target": obj_to_target,
        }
        return reward, metrics
