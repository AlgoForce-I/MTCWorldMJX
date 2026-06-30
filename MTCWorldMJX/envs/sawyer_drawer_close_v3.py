"""Sawyer drawer close environment (drawer-close-v3)."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from mujoco import mjx

from MTCWorldMJX import mjx_env
from MTCWorldMJX.asset_paths import sawyer_xml_path
from MTCWorldMJX.envs._helpers import as_f32, sample_obj_only_rand_vec, set_state_forward
from MTCWorldMJX.envs.sawyer_xyz import SawyerXYZConfig, SawyerXYZEnv
from MTCWorldMJX.utils import reward as reward_utils


class SawyerDrawerCloseEnvV3(SawyerXYZEnv):
    """Push a drawer closed (drawer-close-v3)."""

    TARGET_RADIUS = 0.05
    _HANDLE_OFFSET = as_f32([0.0, -0.16, 0.05])
    _GOAL_OFFSET = as_f32([0.0, -0.16, 0.09])
    _MAX_DIST = 0.15

    def __init__(self, config: SawyerXYZConfig | None = None):
        super().__init__(config)
        self._drawer_body_id = mjx_env.body_id(self._mj_model, "drawer")
        self._drawer_link_body_id = mjx_env.body_id(self._mj_model, "drawer_link")
        self._goal_site_id = mjx_env.site_id(self._mj_model, "goal")
        jnt = self._mj_model.joint("goal_slidey")
        self._slide_qposadr = int(jnt.qposadr[0])
        self._slide_qveladr = int(jnt.dofadr[0])

    @property
    def xml_path(self):
        return sawyer_xml_path("drawer")

    def hand_init_pos(self) -> jax.Array:
        return as_f32([0.0, 0.6, 0.2])

    def mocap_low(self) -> jax.Array:
        return as_f32([-0.5, 0.40, 0.05])

    def mocap_high(self) -> jax.Array:
        return as_f32([0.5, 1.0, 0.5])

    def random_reset_bounds(self) -> tuple[jax.Array, jax.Array]:
        low = as_f32([-0.1, 0.9, 0.0])
        high = as_f32([0.1, 0.9, 0.0])
        return low, high

    def get_obj_pos(self, data: mjx.Data) -> jax.Array:
        return data.xpos[self._drawer_link_body_id] + self._HANDLE_OFFSET

    def get_obj_quat(self, data: mjx.Data) -> jax.Array:
        return jnp.zeros(4, dtype=jnp.float32)

    def reset_state_from_rand_vec(self, rand_vec: jax.Array) -> dict[str, jax.Array]:
        obj_init_pos = rand_vec[:3]
        goal_pos = obj_init_pos + self._GOAL_OFFSET
        return {
            "drawer_pos": obj_init_pos,
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
        drawer_pos = reset_state["drawer_pos"]
        goal_pos = reset_state["goal_pos"]
        body_pos = model.body_pos.at[self._drawer_body_id].set(drawer_pos)
        site_pos = model.site_pos.at[self._goal_site_id].set(goal_pos)
        model = model.replace(body_pos=body_pos, site_pos=site_pos)
        data = set_state_forward(
            model, data, self._slide_qposadr, self._slide_qveladr, jnp.float32(-self._MAX_DIST)
        )
        return model, data

    def finalize_reset(
        self,
        model: mjx.Model,
        data: mjx.Data,
        reset_state: dict[str, jax.Array],
    ) -> tuple[mjx.Model, mjx.Data, dict[str, jax.Array]]:
        obj_init_pos = self.get_obj_pos(data)
        reset_state = {**reset_state, "obj_init_pos": obj_init_pos}
        return model, data, reset_state

    def compute_reward(
        self,
        data: mjx.Data,
        action: jax.Array,
        info: dict[str, Any],
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        obj = self.get_obj_pos(data)
        tcp = self._tcp_center(data)
        target = info["goal_pos"]
        target_to_obj = jnp.linalg.norm(obj - target)
        target_to_obj_init = jnp.linalg.norm(info["obj_init_pos"] - target)
        in_place = reward_utils.tolerance(
            target_to_obj,
            bounds=(0.0, self.TARGET_RADIUS),
            margin=jnp.abs(target_to_obj_init - self.TARGET_RADIUS),
            sigmoid="long_tail",
        )
        handle_reach_radius = 0.005
        tcp_to_obj = jnp.linalg.norm(obj - tcp)
        tcp_to_obj_init = jnp.linalg.norm(info["obj_init_pos"] - info["init_tcp"])
        reach = reward_utils.tolerance(
            tcp_to_obj,
            bounds=(0.0, handle_reach_radius),
            margin=jnp.abs(tcp_to_obj_init - handle_reach_radius),
            sigmoid="gaussian",
        )
        gripper_closed = jnp.clip(action[-1], 0.0, 1.0)
        reach = reward_utils.hamacher_product(reach, gripper_closed)
        reward = reward_utils.hamacher_product(reach, in_place)
        reward = jnp.where(
            target_to_obj <= self.TARGET_RADIUS + 0.015, 1.0, reward
        )
        reward = reward * 10.0
        metrics = {
            "success": (target_to_obj <= self.TARGET_RADIUS + 0.015).astype(jnp.float32),
            "near_object": (tcp_to_obj <= 0.01).astype(jnp.float32),
            "grasp_success": jnp.array(1.0),
            "grasp_reward": reach,
            "in_place_reward": in_place,
            "obj_to_target": target_to_obj,
        }
        return reward, metrics
