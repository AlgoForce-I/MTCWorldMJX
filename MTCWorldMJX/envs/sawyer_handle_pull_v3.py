"""Sawyer handle pull environment (handle-pull-v3)."""

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


class SawyerHandlePullEnvV3(SawyerXYZEnv):
    """Pull a handle upward (handle-pull-v3)."""

    TARGET_RADIUS = 0.05

    def __init__(self, config: SawyerXYZConfig | None = None):
        super().__init__(config)
        self._box_body_id = mjx_env.body_id(self._mj_model, "box")
        self._handle_site_id = mjx_env.site_id(self._mj_model, "handleRight")
        self._goal_pull_site_id = mjx_env.site_id(self._mj_model, "goalPull")
        self._art_qposadr = 9
        self._art_qveladr = 9

    @property
    def xml_path(self):
        return sawyer_xml_path("handle_press")

    def hand_init_pos(self) -> jax.Array:
        return as_f32([0.0, 0.6, 0.2])

    def mocap_low(self) -> jax.Array:
        return as_f32([-0.5, 0.40, 0.05])

    def mocap_high(self) -> jax.Array:
        return as_f32([0.5, 1.0, 0.5])

    def random_reset_bounds(self) -> tuple[jax.Array, jax.Array]:
        low = as_f32([-0.1, 0.8, -0.001])
        high = as_f32([0.1, 0.9, 0.001])
        return low, high

    def get_obj_pos(self, data: mjx.Data) -> jax.Array:
        return data.site_xpos[self._handle_site_id]

    def get_obj_quat(self, data: mjx.Data) -> jax.Array:
        return jnp.zeros(4, dtype=jnp.float32)

    def reset_state_from_rand_vec(self, rand_vec: jax.Array) -> dict[str, jax.Array]:
        return {"box_pos": rand_vec[:3], "rand_vec": rand_vec}

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
        box_pos = reset_state["box_pos"]
        body_pos = model.body_pos.at[self._box_body_id].set(box_pos)
        model = model.replace(body_pos=body_pos)
        data = set_state_forward(
            model, data, self._art_qposadr, self._art_qveladr, jnp.float32(-0.1)
        )
        return model, data

    def finalize_reset(
        self,
        model: mjx.Model,
        data: mjx.Data,
        reset_state: dict[str, jax.Array],
    ) -> tuple[mjx.Model, mjx.Data, dict[str, jax.Array]]:
        goal_pos = data.site_xpos[self._goal_pull_site_id]
        reset_state = {
            **reset_state,
            "obj_init_pos": reset_state["box_pos"],
            "goal_pos": goal_pos,
        }
        return model, data, reset_state

    def compute_reward(
        self,
        data: mjx.Data,
        action: jax.Array,
        info: dict[str, Any],
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        obj = self.get_obj_pos(data)
        target = info["goal_pos"]
        target_to_obj = jnp.abs(target[2] - obj[2])
        target_to_obj_init = jnp.abs(target[2] - info["obj_init_pos"][2])
        in_place = reward_utils.tolerance(
            target_to_obj,
            bounds=(0.0, self.TARGET_RADIUS),
            margin=target_to_obj_init,
            sigmoid="long_tail",
        )
        object_grasped = self._gripper_caging_reward(
            data,
            action,
            obj,
            info["obj_init_pos"],
            info["init_tcp"],
            obj_radius=0.022,
            pad_success_thresh=0.05,
            object_reach_radius=0.01,
            xz_thresh=0.01,
            high_density=True,
        )
        reward = reward_utils.hamacher_product(object_grasped, in_place)
        tcp_opened = self._gripper_opening(data)
        tcp_to_obj = jnp.linalg.norm(obj - self._tcp_center(data))
        near = (
            (tcp_to_obj < 0.035)
            & (tcp_opened > 0.0)
            & (obj[1] - 0.01 > info["obj_init_pos"][2])
        )
        reward = jnp.where(near, reward + 1.0 + 5.0 * in_place, reward)
        reward = jnp.where(target_to_obj < self.TARGET_RADIUS, 10.0, reward)
        metrics = {
            "success": (target_to_obj <= self.TARGET_RADIUS).astype(jnp.float32),
            "near_object": (tcp_to_obj <= 0.05).astype(jnp.float32),
            "grasp_success": (
                (tcp_opened > 0.0) & (obj[2] - 0.03 > info["obj_init_pos"][2])
            ).astype(jnp.float32),
            "grasp_reward": object_grasped,
            "in_place_reward": in_place,
            "obj_to_target": target_to_obj,
        }
        return reward, metrics
