"""Sawyer coffee pull environment (coffee-pull-v3)."""

from __future__ import annotations

import jax.numpy as jnp

from MTCWorldMJX.envs._coffee_base import _CoffeeMugManipulationEnvBase
from MTCWorldMJX.envs._helpers import as_f32
from MTCWorldMJX.envs.sawyer_xyz import SawyerXYZConfig


class SawyerCoffeePullEnvV3(_CoffeeMugManipulationEnvBase):
    """Pull a mug away from the coffee machine (coffee-pull-v3)."""

    def __init__(self, config: SawyerXYZConfig | None = None):
        super().__init__(config)

    def random_reset_bounds(self) -> tuple[jax.Array, jax.Array]:
        obj_low = as_f32([-0.05, 0.7, -0.001])
        obj_high = as_f32([0.05, 0.75, 0.001])
        goal_low = as_f32([-0.1, 0.55, -0.001])
        goal_high = as_f32([0.1, 0.65, 0.001])
        return jnp.concatenate([obj_low, goal_low]), jnp.concatenate([obj_high, goal_high])

    def _machine_pos(
        self,
        pos_mug_init: jax.Array,
        pos_mug_goal: jax.Array,
    ) -> jax.Array:
        del pos_mug_goal
        return pos_mug_init + jnp.array([0.0, 0.22, 0.0], dtype=jnp.float32)
