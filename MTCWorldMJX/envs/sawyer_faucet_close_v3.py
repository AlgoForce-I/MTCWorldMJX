"""Sawyer faucet close environment (faucet-close-v3)."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from mujoco import mjx

from MTCWorldMJX import mjx_env
from MTCWorldMJX.asset_paths import sawyer_xml_path
from MTCWorldMJX.envs._helpers import as_f32, sample_obj_only_rand_vec
from MTCWorldMJX.envs.sawyer_xyz import SawyerXYZConfig, SawyerXYZEnv
from MTCWorldMJX.utils import reward as reward_utils


class SawyerFaucetCloseEnvV3(SawyerXYZEnv):
    """Turn a faucet off (faucet-close-v3)."""

    _HANDLE_LENGTH = 0.175
    _TARGET_RADIUS = 0.07
    _FAR_GOAL = as_f32([10.0, 10.0, 10.0])
    _OBJ_OFFSET = as_f32([0.0, 0.0, -0.01])

    def __init__(self, config: SawyerXYZConfig | None = None):
        super().__init__(config)
        self._faucet_body_id = mjx_env.body_id(self._mj_model, "faucetBase")
        self._handle_site_id = mjx_env.site_id(self._mj_model, "handleStartClose")
        self._goal_open_site_id = mjx_env.site_id(self._mj_model, "goal_open")
        self._goal_close_site_id = mjx_env.site_id(self._mj_model, "goal_close")

    @property
    def xml_path(self):
        return sawyer_xml_path("faucet")

    def hand_init_pos(self) -> jax.Array:
        return as_f32([0.0, 0.4, 0.2])

    def mocap_low(self) -> jax.Array:
        return as_f32([-0.5, 0.40, -0.15])

    def mocap_high(self) -> jax.Array:
        return as_f32([0.5, 1.0, 0.5])

    def random_reset_bounds(self) -> tuple[jax.Array, jax.Array]:
        low = as_f32([-0.1, 0.8, 0.0])
        high = as_f32([0.1, 0.85, 0.0])
        return low, high

    def get_obj_pos(self, data: mjx.Data) -> jax.Array:
        return data.site_xpos[self._handle_site_id] + self._OBJ_OFFSET

    def get_obj_quat(self, data: mjx.Data) -> jax.Array:
        return data.xquat[self._faucet_body_id]

    def reset_state_from_rand_vec(self, rand_vec: jax.Array) -> dict[str, jax.Array]:
        obj_init_pos = rand_vec[:3]
        goal_pos = obj_init_pos + as_f32([-self._HANDLE_LENGTH, 0.0, 0.125])
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
        body_pos = model.body_pos.at[self._faucet_body_id].set(obj_pos)
        site_pos = model.site_pos.at[self._goal_close_site_id].set(goal_pos)
        site_pos = site_pos.at[self._goal_open_site_id].set(self._FAR_GOAL)
        return model.replace(body_pos=body_pos, site_pos=site_pos), data

    def compute_reward(
        self,
        data: mjx.Data,
        action: jax.Array,
        info: dict[str, Any],
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        del action
        obj = self.get_obj_pos(data)
        tcp = self._tcp_center(data)
        target = info["goal_pos"]
        target_to_obj = jnp.linalg.norm(obj - target)
        target_to_obj_init = jnp.linalg.norm(info["obj_init_pos"] - target)
        in_place = reward_utils.tolerance(
            target_to_obj,
            bounds=(0.0, self._TARGET_RADIUS),
            margin=jnp.abs(target_to_obj_init - self._TARGET_RADIUS),
            sigmoid="long_tail",
        )
        faucet_reach_radius = 0.01
        tcp_to_obj = jnp.linalg.norm(obj - tcp)
        tcp_to_obj_init = jnp.linalg.norm(info["obj_init_pos"] - info["init_tcp"])
        reach = reward_utils.tolerance(
            tcp_to_obj,
            bounds=(0.0, faucet_reach_radius),
            margin=jnp.abs(tcp_to_obj_init - faucet_reach_radius),
            sigmoid="gaussian",
        )
        reward = (2.0 * reach + 3.0 * in_place) * 2.0
        reward = jnp.where(target_to_obj <= self._TARGET_RADIUS, 10.0, reward)
        metrics = {
            "success": (target_to_obj <= 0.07).astype(jnp.float32),
            "near_object": (tcp_to_obj <= 0.01).astype(jnp.float32),
            "grasp_success": jnp.array(1.0),
            "grasp_reward": reach,
            "in_place_reward": in_place,
            "obj_to_target": target_to_obj,
        }
        return reward, metrics
