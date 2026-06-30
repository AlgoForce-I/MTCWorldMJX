"""Sawyer door unlock environment (door-unlock-v3)."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from mujoco import mjx

from MTCWorldMJX import mjx_env
from MTCWorldMJX.asset_paths import sawyer_xml_path
from MTCWorldMJX.envs._helpers import (
    as_f32,
    sample_obj_only_rand_vec,
    set_state_forward,
)
from MTCWorldMJX.envs.sawyer_xyz import SawyerXYZConfig, SawyerXYZEnv
from MTCWorldMJX.utils import reward as reward_utils


class SawyerDoorUnlockEnvV3(SawyerXYZEnv):
    """Push a door lock to unlock (door-unlock-v3)."""

    _LOCK_LENGTH = 0.1
    _GOAL_OFFSET = as_f32([0.1, -0.04, 0.0])
    _FAR_GOAL = as_f32([10.0, 10.0, 10.0])
    _UNLOCK_ANGLE = jnp.float32(1.5708)
    _SHOULDER_OFFSET = as_f32([0.0, 0.055, 0.07])

    def __init__(self, config: SawyerXYZConfig | None = None):
        super().__init__(config)
        self._door_body_id = mjx_env.body_id(self._mj_model, "door")
        self._lock_link_body_id = mjx_env.body_id(self._mj_model, "lock_link")
        self._door_link_body_id = mjx_env.body_id(self._mj_model, "door_link")
        self._lock_start_site_id = mjx_env.site_id(self._mj_model, "lockStartUnlock")
        self._goal_unlock_site_id = mjx_env.site_id(self._mj_model, "goal_unlock")
        self._goal_lock_site_id = mjx_env.site_id(self._mj_model, "goal_lock")
        self._art_qposadr = 9
        self._art_qveladr = 9

    @property
    def xml_path(self):
        return sawyer_xml_path("door_lock")

    def hand_init_pos(self) -> jax.Array:
        return as_f32([0.0, 0.6, 0.2])

    def mocap_low(self) -> jax.Array:
        return as_f32([-0.5, 0.40, -0.15])

    def mocap_high(self) -> jax.Array:
        return as_f32([0.5, 1.0, 0.5])

    def random_reset_bounds(self) -> tuple[jax.Array, jax.Array]:
        low = as_f32([-0.1, 0.8, 0.15])
        high = as_f32([0.1, 0.85, 0.15])
        return low, high

    def get_obj_pos(self, data: mjx.Data) -> jax.Array:
        return data.site_xpos[self._lock_start_site_id]

    def get_obj_quat(self, data: mjx.Data) -> jax.Array:
        return data.xquat[self._door_link_body_id]

    def reset_state_from_rand_vec(self, rand_vec: jax.Array) -> dict[str, jax.Array]:
        return {"door_pos": rand_vec[:3], "rand_vec": rand_vec}

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
        door_pos = reset_state["door_pos"]
        body_pos = model.body_pos.at[self._door_body_id].set(door_pos)
        model = model.replace(body_pos=body_pos)
        data = set_state_forward(
            model, data, self._art_qposadr, self._art_qveladr, self._UNLOCK_ANGLE
        )
        return model, data

    def finalize_reset(
        self,
        model: mjx.Model,
        data: mjx.Data,
        reset_state: dict[str, jax.Array],
    ) -> tuple[mjx.Model, mjx.Data, dict[str, jax.Array]]:
        obj_init_pos = data.xpos[self._lock_link_body_id]
        goal_pos = obj_init_pos + self._GOAL_OFFSET
        site_pos = model.site_pos.at[self._goal_unlock_site_id].set(goal_pos)
        site_pos = site_pos.at[self._goal_lock_site_id].set(self._FAR_GOAL)
        model = model.replace(site_pos=site_pos)
        reset_state = {
            **reset_state,
            "obj_init_pos": obj_init_pos,
            "goal_pos": goal_pos,
        }
        return model, data, reset_state

    def compute_reward(
        self,
        data: mjx.Data,
        action: jax.Array,
        info: dict[str, Any],
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        del action
        gripper = self._hand_pos(data)
        lock = self.get_obj_pos(data)
        scale = as_f32([0.25, 1.0, 0.5])
        shoulder_to_lock = (gripper + self._SHOULDER_OFFSET - lock) * scale
        shoulder_to_lock_init = (
            info["init_tcp"] + self._SHOULDER_OFFSET - info["obj_init_pos"]
        ) * scale
        ready_to_push = reward_utils.tolerance(
            jnp.linalg.norm(shoulder_to_lock),
            bounds=(0.0, 0.02),
            margin=jnp.linalg.norm(shoulder_to_lock_init),
            sigmoid="long_tail",
        )
        obj_to_target = jnp.abs(info["goal_pos"][0] - lock[0])
        pushed = reward_utils.tolerance(
            obj_to_target,
            bounds=(0.0, 0.005),
            margin=self._LOCK_LENGTH,
            sigmoid="long_tail",
        )
        reward = 2.0 * ready_to_push + 8.0 * pushed
        metrics = {
            "success": (obj_to_target <= 0.02).astype(jnp.float32),
            "near_object": (jnp.linalg.norm(shoulder_to_lock) <= 0.05).astype(jnp.float32),
            "grasp_success": (self._gripper_opening(data) > 0.0).astype(jnp.float32),
            "grasp_reward": ready_to_push,
            "in_place_reward": pushed,
            "obj_to_target": obj_to_target,
        }
        return reward, metrics
