"""Sawyer button press environment (button-press-v3)."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from mujoco import mjx

from MTCWorldMJX import mjx_env
from MTCWorldMJX.asset_paths import sawyer_xml_path
from MTCWorldMJX.envs._helpers import (
    as_f32,
    button_press_reward_v2,
    sample_obj_only_rand_vec,
    set_slide_joint,
)
from MTCWorldMJX.envs.sawyer_xyz import SawyerXYZConfig, SawyerXYZEnv


class _ButtonPressEnvBase(SawyerXYZEnv):
    """Shared logic for button-press Sawyer tasks."""

    _TARGET_AXIS = 1
    _OBJ_OFFSET = jnp.array([0.0, -0.193, 0.0], dtype=jnp.float32)
    _RESET_BUTTON_JOINT = True

    def _post_init_ids(self) -> None:
        super()._post_init_ids()
        m = self._mj_model
        self._box_body_id = mjx_env.body_id(m, "box")
        self._button_body_id = mjx_env.body_id(m, "button")
        self._hole_site_id = mjx_env.site_id(m, "hole")
        self._button_start_site_id = mjx_env.site_id(m, "buttonStart")
        self._btn_joint_qposadr = int(m.jnt_qposadr[mjx_env.joint_id(m, "btnbox_joint")])
        self._btn_joint_qveladr = int(m.jnt_dofadr[mjx_env.joint_id(m, "btnbox_joint")])

    def hand_init_pos(self) -> jax.Array:
        return jnp.array([0.0, 0.4, 0.2], dtype=jnp.float32)

    def mocap_low(self) -> jax.Array:
        return jnp.array([-0.5, 0.40, 0.05], dtype=jnp.float32)

    def mocap_high(self) -> jax.Array:
        return jnp.array([0.5, 1.0, 0.5], dtype=jnp.float32)

    def get_obj_pos(self, data: mjx.Data) -> jax.Array:
        button_pos = mjx_env.body_xpos(data, jnp.array(self._button_body_id))
        return button_pos + self._OBJ_OFFSET

    def get_obj_quat(self, data: mjx.Data) -> jax.Array:
        return mjx_env.body_xquat(data, jnp.array(self._button_body_id))

    def reset_state_from_rand_vec(self, rand_vec: jax.Array) -> dict[str, jax.Array]:
        obj_init_pos = rand_vec[:3]
        return {
            "obj_init_pos": obj_init_pos,
            "goal_pos": jnp.zeros(3, dtype=jnp.float32),
            "rand_vec": rand_vec,
        }

    def sample_reset_state(self, rng: jax.Array) -> dict[str, jax.Array]:
        low, high = self.random_reset_bounds()
        obj_init_pos = sample_obj_only_rand_vec(rng, low, high)
        return {
            "obj_init_pos": obj_init_pos,
            "goal_pos": jnp.zeros(3, dtype=jnp.float32),
            "rand_vec": obj_init_pos,
        }

    def apply_reset_state(
        self,
        model: mjx.Model,
        data: mjx.Data,
        reset_state: dict[str, jax.Array],
    ) -> tuple[mjx.Model, mjx.Data]:
        obj_pos = reset_state["obj_init_pos"]
        body_pos = model.body_pos.at[self._box_body_id].set(obj_pos)
        model = model.replace(body_pos=body_pos)
        if self._RESET_BUTTON_JOINT:
            data = set_slide_joint(
                data,
                self._btn_joint_qposadr,
                self._btn_joint_qveladr,
                jnp.array(0.0),
            )
        return model, data

    def _resolve_goal_pos(
        self,
        model: mjx.Model,
        data: mjx.Data,
        reset_state: dict[str, jax.Array],
    ) -> jax.Array:
        del model, reset_state
        return mjx_env.site_xpos(data, jnp.array(self._hole_site_id))

    def finalize_reset(
        self,
        model: mjx.Model,
        data: mjx.Data,
        reset_state: dict[str, jax.Array],
    ) -> tuple[mjx.Model, mjx.Data, dict[str, jax.Array]]:
        data = mjx.forward(model, data)
        goal_pos = mjx_env.site_xpos(data, jnp.array(self._hole_site_id))
        button_start = mjx_env.site_xpos(data, jnp.array(self._button_start_site_id))
        obj_to_target_init = jnp.abs(goal_pos[self._TARGET_AXIS] - button_start[self._TARGET_AXIS])
        reset_state = dict(reset_state)
        reset_state["obj_to_target_init"] = obj_to_target_init
        return model, data, reset_state


class SawyerButtonPressEnvV3(_ButtonPressEnvBase):
    """Press a button from the side (button-press-v3)."""

    def __init__(self, config: SawyerXYZConfig | None = None):
        super().__init__(config)

    @property
    def xml_path(self):
        return sawyer_xml_path("sawyer_button_press.xml")

    def random_reset_bounds(self) -> tuple[jax.Array, jax.Array]:
        low = as_f32([-0.1, 0.85, 0.115])
        high = as_f32([0.1, 0.9, 0.115])
        return low, high

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
        reward, metrics = button_press_reward_v2(
            obj,
            tcp,
            info["goal_pos"][1],
            tcp_opened,
            info["init_tcp"],
            info["obj_to_target_init"],
        )
        return reward, metrics
