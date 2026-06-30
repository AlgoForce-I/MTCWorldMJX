"""Sawyer dial turn environment (dial-turn-v3)."""

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


class SawyerDialTurnEnvV3(SawyerXYZEnv):
    """Turn a dial to a target orientation (dial-turn-v3)."""

    TARGET_RADIUS = 0.07
    DIAL_RADIUS = 0.05
    DIAL_PUSH_OFFSET = jnp.array([0.05, 0.02, 0.09], dtype=jnp.float32)

    def __init__(self, config: SawyerXYZConfig | None = None):
        super().__init__(config)
        self._dial_body_id = mjx_env.body_id(self._mj_model, "dial")
        self._goal_site_id = mjx_env.site_id(self._mj_model, "goal")
        self._knob_joint_qposadr = int(
            self._mj_model.jnt_qposadr[mjx_env.joint_id(self._mj_model, "knob_Joint_1")]
        )

    @property
    def xml_path(self):
        return sawyer_xml_path("sawyer_dial.xml")

    def hand_init_pos(self) -> jax.Array:
        return jnp.array([0.0, 0.6, 0.2], dtype=jnp.float32)

    def mocap_low(self) -> jax.Array:
        return jnp.array([-0.5, 0.40, 0.05], dtype=jnp.float32)

    def mocap_high(self) -> jax.Array:
        return jnp.array([0.5, 1.0, 0.5], dtype=jnp.float32)

    def random_reset_bounds(self) -> tuple[jax.Array, jax.Array]:
        low = as_f32([-0.1, 0.7, 0.0])
        high = as_f32([0.1, 0.8, 0.0])
        return low, high

    def _dial_tip_pos(self, data: mjx.Data) -> jax.Array:
        dial_center = mjx_env.body_xpos(data, jnp.array(self._dial_body_id))
        dial_angle = data.qpos[self._knob_joint_qposadr]
        offset = jnp.array(
            [jnp.sin(dial_angle), -jnp.cos(dial_angle), 0.0],
            dtype=jnp.float32,
        )
        return dial_center + offset * self.DIAL_RADIUS

    def get_obj_pos(self, data: mjx.Data) -> jax.Array:
        return self._dial_tip_pos(data)

    def get_obj_quat(self, data: mjx.Data) -> jax.Array:
        return mjx_env.body_xquat(data, jnp.array(self._dial_body_id))

    def reset_state_from_rand_vec(self, rand_vec: jax.Array) -> dict[str, jax.Array]:
        return self._dial_reset_state(rand_vec[:3], rand_vec)

    def sample_reset_state(self, rng: jax.Array) -> dict[str, jax.Array]:
        low, high = self.random_reset_bounds()
        obj_init_pos = sample_obj_only_rand_vec(rng, low, high)
        return self._dial_reset_state(obj_init_pos, obj_init_pos)

    def _dial_reset_state(
        self,
        obj_init_pos: jax.Array,
        rand_vec: jax.Array,
    ) -> dict[str, jax.Array]:
        goal_pos = obj_init_pos + jnp.array([0.0, 0.03, 0.03], dtype=jnp.float32)
        return {
            "obj_init_pos": obj_init_pos,
            "goal_pos": goal_pos,
            "rand_vec": rand_vec,
        }

    def apply_reset_state(
        self,
        model: mjx.Model,
        data: mjx.Data,
        reset_state: dict[str, jax.Array],
    ) -> tuple[mjx.Model, mjx.Data]:
        obj_pos = reset_state["obj_init_pos"]
        body_pos = model.body_pos.at[self._dial_body_id].set(obj_pos)
        site_pos = model.site_pos.at[self._goal_site_id].set(reset_state["goal_pos"])
        model = model.replace(body_pos=body_pos, site_pos=site_pos)
        return model, data

    def finalize_reset(
        self,
        model: mjx.Model,
        data: mjx.Data,
        reset_state: dict[str, jax.Array],
    ) -> tuple[mjx.Model, mjx.Data, dict[str, jax.Array]]:
        data = mjx.forward(model, data)
        dial_push_position = self._dial_tip_pos(data) + self.DIAL_PUSH_OFFSET
        reset_state = dict(reset_state)
        reset_state["dial_push_position"] = dial_push_position
        return model, data, reset_state

    def compute_reward(
        self,
        data: mjx.Data,
        action: jax.Array,
        info: dict[str, Any],
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        obj = self._dial_tip_pos(data)
        dial_push_position = self._dial_tip_pos(data) + self.DIAL_PUSH_OFFSET
        tcp = self._tcp_center(data)
        target = info["goal_pos"]

        target_to_obj = jnp.linalg.norm(obj - target)
        target_to_obj_init = jnp.linalg.norm(info["dial_push_position"] - target)

        in_place = reward_utils.tolerance(
            target_to_obj,
            bounds=(0.0, self.TARGET_RADIUS),
            margin=jnp.abs(target_to_obj_init - self.TARGET_RADIUS),
            sigmoid="long_tail",
        )

        dial_reach_radius = 0.005
        tcp_to_obj = jnp.linalg.norm(dial_push_position - tcp)
        tcp_to_obj_init = jnp.linalg.norm(info["dial_push_position"] - info["init_tcp"])
        reach = reward_utils.tolerance(
            tcp_to_obj,
            bounds=(0.0, dial_reach_radius),
            margin=jnp.abs(tcp_to_obj_init - dial_reach_radius),
            sigmoid="gaussian",
        )
        gripper_closed = jnp.clip(jnp.minimum(jnp.maximum(action[-1], 0.0), 1.0), 0.0, 1.0)
        reach = reward_utils.hamacher_product(reach, gripper_closed)

        reward = 10.0 * reward_utils.hamacher_product(reach, in_place)
        metrics = {
            "success": (target_to_obj <= self.TARGET_RADIUS).astype(jnp.float32),
            "near_object": (tcp_to_obj <= 0.01).astype(jnp.float32),
            "grasp_success": jnp.array(1.0, dtype=jnp.float32),
            "grasp_reward": reach,
            "in_place_reward": in_place,
            "obj_to_target": target_to_obj,
        }
        return reward, metrics
