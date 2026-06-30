"""Sawyer door open environment (door-open-v3)."""

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


class SawyerDoorEnvV3(SawyerXYZEnv):
    """Pull a door open by its handle (door-open-v3)."""

    _GOAL_OFFSET = as_f32([-0.3, -0.45, 0.0])

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
        return as_f32([0.0, 0.6, 0.2])

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
        data = set_joint_qpos(data, self._door_qposadr, self._door_qveladr, jnp.array(0.0))
        return model, data

    @staticmethod
    def _reward_pos(hand: jax.Array, door: jax.Array, theta: jax.Array) -> tuple[jax.Array, jax.Array]:
        door_adj = door + as_f32([-0.05, 0.0, 0.0])
        threshold = 0.12
        radius = jnp.linalg.norm(hand[:2] - door_adj[:2])
        floor = jnp.where(
            radius <= threshold,
            0.0,
            0.04 * jnp.log(jnp.maximum(radius - threshold, 1e-8)) + 0.4,
        )
        above_floor = jnp.where(
            hand[2] >= floor,
            1.0,
            reward_utils.tolerance(
                floor - hand[2],
                bounds=(0.0, 0.01),
                margin=floor / 2.0,
                sigmoid="long_tail",
            ),
        )
        in_place = reward_utils.tolerance(
            jnp.linalg.norm(hand - door_adj - as_f32([0.05, 0.03, -0.01])),
            bounds=(0.0, threshold / 2.0),
            margin=0.5,
            sigmoid="long_tail",
        )
        ready_to_open = reward_utils.hamacher_product(above_floor, in_place)
        door_angle = -theta
        opened = 0.2 * (theta < -jnp.pi / 90.0).astype(jnp.float32) + 0.8 * reward_utils.tolerance(
            jnp.pi / 2.0 + jnp.pi / 6 - door_angle,
            bounds=(0.0, 0.5),
            margin=jnp.pi / 3.0,
            sigmoid="long_tail",
        )
        return ready_to_open, opened

    def compute_reward(
        self,
        data: mjx.Data,
        action: jax.Array,
        info: dict[str, Any],
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        hand = self._hand_pos(data)
        door = self.get_obj_pos(data)
        theta = data.qpos[self._door_qposadr]
        reward_grab = (jnp.clip(action[3], -1.0, 1.0) + 1.0) / 2.0
        ready_to_open, opened = self._reward_pos(hand, door, theta)
        reward = 2.0 * reward_utils.hamacher_product(ready_to_open, reward_grab) + 8.0 * opened
        success = (jnp.abs(door[0] - info["goal_pos"][0]) <= 0.08).astype(jnp.float32)
        reward = jnp.where(success > 0.0, 10.0, reward)
        metrics = {
            "success": success,
            "near_object": ready_to_open,
            "grasp_success": (reward_grab >= 0.5).astype(jnp.float32),
            "grasp_reward": reward_grab,
            "in_place_reward": opened,
            "obj_to_target": jnp.array(0.0),
        }
        return reward, metrics
