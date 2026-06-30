"""Shared coffee-machine Sawyer environments."""

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
    sample_hstack_rand_vec,
    sample_obj_only_rand_vec,
    set_coffee_mug_pos,
)
from MTCWorldMJX.envs.sawyer_xyz import SawyerXYZConfig, SawyerXYZEnv


class _CoffeeEnvBase(SawyerXYZEnv):
    """Base class for tasks using sawyer_coffee.xml."""

    def _post_init_ids(self) -> None:
        super()._post_init_ids()
        m = self._mj_model
        self._mug_free_qposadr = 0
        self._obj_body_id = mjx_env.body_id(m, "obj")
        self._coffee_machine_body_id = mjx_env.body_id(m, "coffee_machine")
        self._mug_geom_id = mjx_env.geom_id(m, "mug")
        self._mug_goal_site_id = mjx_env.site_id(m, "mug_goal")
        self._button_start_site_id = mjx_env.site_id(m, "buttonStart")

    @property
    def xml_path(self):
        return sawyer_xml_path("sawyer_coffee.xml")

    def hand_init_pos(self) -> jax.Array:
        return jnp.array([0.0, 0.4, 0.2], dtype=jnp.float32)

    def mocap_low(self) -> jax.Array:
        return jnp.array([-0.5, 0.40, 0.05], dtype=jnp.float32)

    def mocap_high(self) -> jax.Array:
        return jnp.array([0.5, 1.0, 0.5], dtype=jnp.float32)

    def get_obj_pos(self, data: mjx.Data) -> jax.Array:
        return mjx_env.body_xpos(data, jnp.array(self._obj_body_id))

    def get_obj_quat(self, data: mjx.Data) -> jax.Array:
        return geom_quat_xyzw(data, self._mug_geom_id)


class _CoffeeMugManipulationEnvBase(_CoffeeEnvBase):
    """Shared reset logic for coffee push/pull tasks."""

    def reset_state_from_rand_vec(self, rand_vec: jax.Array) -> dict[str, jax.Array]:
        pos_mug_init = rand_vec[:3]
        pos_mug_goal = rand_vec[3:]
        return self._mug_manipulation_reset_state(pos_mug_init, pos_mug_goal, rand_vec)

    def sample_reset_state(self, rng: jax.Array) -> dict[str, jax.Array]:
        low, high = self.random_reset_bounds()
        vec, _ = sample_hstack_rand_vec(rng, low, high)
        return self._mug_manipulation_reset_state(vec[:3], vec[3:], vec)

    def _mug_manipulation_reset_state(
        self,
        pos_mug_init: jax.Array,
        pos_mug_goal: jax.Array,
        rand_vec: jax.Array,
    ) -> dict[str, jax.Array]:
        machine_pos = self._machine_pos(pos_mug_init, pos_mug_goal)
        return {
            "mug_pos": pos_mug_init,
            "obj_init_pos": pos_mug_init,
            "goal_pos": pos_mug_goal,
            "machine_pos": machine_pos,
            "rand_vec": rand_vec,
        }

    def _machine_pos(
        self,
        pos_mug_init: jax.Array,
        pos_mug_goal: jax.Array,
    ) -> jax.Array:
        raise NotImplementedError

    def apply_reset_state(
        self,
        model: mjx.Model,
        data: mjx.Data,
        reset_state: dict[str, jax.Array],
    ) -> tuple[mjx.Model, mjx.Data]:
        body_pos = model.body_pos.at[self._coffee_machine_body_id].set(
            reset_state["machine_pos"]
        )
        site_pos = model.site_pos.at[self._mug_goal_site_id].set(reset_state["goal_pos"])
        model = model.replace(body_pos=body_pos, site_pos=site_pos)
        data = set_coffee_mug_pos(
            data,
            self._mug_free_qposadr,
            reset_state["mug_pos"],
            init_qpos=self._init_qpos,
        )
        return model, data

    def compute_reward(
        self,
        data: mjx.Data,
        action: jax.Array,
        info: dict[str, Any],
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        from MTCWorldMJX.envs._helpers import coffee_mug_manipulation_reward_v2

        obj = self.get_obj_pos(data)
        tcp_opened = self._gripper_opening(data)
        return coffee_mug_manipulation_reward_v2(
            self,
            data,
            action,
            obj,
            info["goal_pos"],
            info["obj_init_pos"],
            info["init_tcp"],
            tcp_opened,
        )
