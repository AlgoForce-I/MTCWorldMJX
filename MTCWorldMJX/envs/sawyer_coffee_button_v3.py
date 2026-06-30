"""Sawyer coffee button environment (coffee-button-v3)."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from mujoco import mjx

from MTCWorldMJX import mjx_env
from MTCWorldMJX.envs._coffee_base import _CoffeeEnvBase
from MTCWorldMJX.envs._helpers import (
    as_f32,
    coffee_button_reward_v2,
    sample_obj_only_rand_vec,
    set_coffee_mug_pos,
)
from MTCWorldMJX.envs.sawyer_xyz import SawyerXYZConfig


class SawyerCoffeeButtonEnvV3(_CoffeeEnvBase):
    """Press the coffee-machine button (coffee-button-v3)."""

    MAX_DIST = 0.03

    def __init__(self, config: SawyerXYZConfig | None = None):
        super().__init__(config)

    def random_reset_bounds(self) -> tuple[jax.Array, jax.Array]:
        low = as_f32([-0.1, 0.8, -0.001])
        high = as_f32([0.1, 0.9, 0.001])
        return low, high

    def reset_state_from_rand_vec(self, rand_vec: jax.Array) -> dict[str, jax.Array]:
        return self._coffee_button_reset_state(rand_vec[:3], rand_vec)

    def sample_reset_state(self, rng: jax.Array) -> dict[str, jax.Array]:
        low, high = self.random_reset_bounds()
        machine_pos = sample_obj_only_rand_vec(rng, low, high)
        return self._coffee_button_reset_state(machine_pos, machine_pos)

    def _coffee_button_reset_state(
        self,
        machine_pos: jax.Array,
        rand_vec: jax.Array,
    ) -> dict[str, jax.Array]:
        mug_pos = machine_pos + jnp.array([0.0, -0.22, 0.0], dtype=jnp.float32)
        button_pos = machine_pos + jnp.array([0.0, -0.22, 0.3], dtype=jnp.float32)
        goal_pos = button_pos + jnp.array([0.0, self.MAX_DIST, 0.0], dtype=jnp.float32)
        return {
            "machine_pos": machine_pos,
            "mug_pos": mug_pos,
            "obj_init_pos": machine_pos,
            "goal_pos": goal_pos,
            "rand_vec": rand_vec,
        }

    def apply_reset_state(
        self,
        model: mjx.Model,
        data: mjx.Data,
        reset_state: dict[str, jax.Array],
    ) -> tuple[mjx.Model, mjx.Data]:
        body_pos = model.body_pos.at[self._coffee_machine_body_id].set(
            reset_state["machine_pos"]
        )
        model = model.replace(body_pos=body_pos)
        data = set_coffee_mug_pos(
            data,
            self._mug_free_qposadr,
            reset_state["mug_pos"],
            init_qpos=self._init_qpos,
        )
        return model, data

    def get_obj_pos(self, data: mjx.Data) -> jax.Array:
        return mjx_env.site_xpos(data, jnp.array(self._button_start_site_id))

    def get_obj_quat(self, data: mjx.Data) -> jax.Array:
        return jnp.array([1.0, 0.0, 0.0, 0.0], dtype=jnp.float32)

    def compute_reward(
        self,
        data: mjx.Data,
        action: jax.Array,
        info: dict[str, Any],
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        del action
        obj = self.get_obj_pos(data)
        tcp = self._tcp_center(data)
        tcp_opened = self._gripper_opening(data)
        return coffee_button_reward_v2(
            obj,
            tcp,
            info["goal_pos"][1],
            tcp_opened,
            info["init_tcp"],
            jnp.array(self.MAX_DIST, dtype=jnp.float32),
        )
