"""Sawyer window open environment (window-open-v3)."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from mujoco import mjx

from MTCWorldMJX import mjx_env
from MTCWorldMJX.asset_paths import sawyer_xml_path
from MTCWorldMJX.envs._helpers import as_f32, sample_obj_only_rand_vec, set_joint_qpos
from MTCWorldMJX.envs.sawyer_xyz import SawyerXYZConfig, SawyerXYZEnv
from MTCWorldMJX.utils import reward as reward_utils


class SawyerWindowOpenEnvV3(SawyerXYZEnv):
    """Slide a window open (window-open-v3)."""

    TARGET_RADIUS = 0.05

    def __init__(self, config: SawyerXYZConfig | None = None):
        super().__init__(config)
        self._window_body_id = mjx_env.body_id(self._mj_model, "window")
        self._handle_site_id = mjx_env.site_id(self._mj_model, "handleOpenStart")
        self._goal_site_id = mjx_env.site_id(self._mj_model, "goal")
        jnt = self._mj_model.joint("window_slide")
        self._slide_qposadr = int(jnt.qposadr[0])
        self._slide_qveladr = int(jnt.dofadr[0])

    @property
    def xml_path(self):
        return sawyer_xml_path("window_horizontal")

    def hand_init_pos(self) -> jax.Array:
        return as_f32([0.0, 0.4, 0.2])

    def mocap_low(self) -> jax.Array:
        return as_f32([-0.5, 0.40, 0.05])

    def mocap_high(self) -> jax.Array:
        return as_f32([0.5, 1.0, 0.5])

    def random_reset_bounds(self) -> tuple[jax.Array, jax.Array]:
        low = as_f32([-0.1, 0.7, 0.16])
        high = as_f32([0.1, 0.9, 0.16])
        return low, high

    def get_obj_pos(self, data: mjx.Data) -> jax.Array:
        return data.site_xpos[self._handle_site_id]

    def get_obj_quat(self, data: mjx.Data) -> jax.Array:
        return jnp.zeros(4, dtype=jnp.float32)

    def reset_state_from_rand_vec(self, rand_vec: jax.Array) -> dict[str, jax.Array]:
        obj_init_pos = rand_vec[:3]
        goal_pos = obj_init_pos + as_f32([0.2, 0.0, 0.0])
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
        body_pos = model.body_pos.at[self._window_body_id].set(obj_pos)
        site_pos = model.site_pos.at[self._goal_site_id].set(goal_pos)
        model = model.replace(body_pos=body_pos, site_pos=site_pos)
        data = set_joint_qpos(data, self._slide_qposadr, self._slide_qveladr, jnp.float32(0.0))
        return model, data

    def finalize_reset(
        self,
        model: mjx.Model,
        data: mjx.Data,
        reset_state: dict[str, jax.Array],
    ) -> tuple[mjx.Model, mjx.Data, dict[str, jax.Array]]:
        data = mjx.forward(model, data)
        window_handle_pos_init = self.get_obj_pos(data)
        reset_state = {
            **reset_state,
            "window_handle_pos_init": window_handle_pos_init,
        }
        return model, data, reset_state

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
        target_to_obj = jnp.abs(obj[0] - target[0])
        target_to_obj_init = jnp.abs(info["obj_init_pos"][0] - target[0])
        in_place = reward_utils.tolerance(
            target_to_obj,
            bounds=(0.0, self.TARGET_RADIUS),
            margin=jnp.abs(target_to_obj_init - self.TARGET_RADIUS),
            sigmoid="long_tail",
        )
        handle_radius = 0.02
        tcp_to_obj = jnp.linalg.norm(obj - tcp)
        tcp_to_obj_init = jnp.linalg.norm(
            info["window_handle_pos_init"] - info["init_tcp"]
        )
        reach = reward_utils.tolerance(
            tcp_to_obj,
            bounds=(0.0, handle_radius),
            margin=jnp.abs(tcp_to_obj_init - handle_radius),
            sigmoid="long_tail",
        )
        reward = 10.0 * reward_utils.hamacher_product(reach, in_place)
        metrics = {
            "success": (target_to_obj <= self.TARGET_RADIUS).astype(jnp.float32),
            "near_object": (tcp_to_obj <= 0.05).astype(jnp.float32),
            "grasp_success": jnp.array(1.0),
            "grasp_reward": reach,
            "in_place_reward": in_place,
            "obj_to_target": target_to_obj,
        }
        return reward, metrics
