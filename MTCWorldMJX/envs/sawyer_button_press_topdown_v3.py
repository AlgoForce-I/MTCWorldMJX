"""Sawyer top-down button press environment (button-press-topdown-v3)."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
from mujoco import mjx

from MTCWorldMJX.asset_paths import sawyer_xml_path
from MTCWorldMJX.envs._helpers import as_f32, button_press_topdown_reward_v2
from MTCWorldMJX.envs.sawyer_button_press_v3 import _ButtonPressEnvBase
from MTCWorldMJX.envs.sawyer_xyz import SawyerXYZConfig


class SawyerButtonPressTopdownEnvV3(_ButtonPressEnvBase):
    """Press a button from above (button-press-topdown-v3)."""

    _TARGET_AXIS = 2
    _OBJ_OFFSET = jnp.array([0.0, 0.0, 0.193], dtype=jnp.float32)
    _RESET_BUTTON_JOINT = False

    def __init__(self, config: SawyerXYZConfig | None = None):
        super().__init__(config)

    @property
    def xml_path(self):
        return sawyer_xml_path("sawyer_button_press_topdown.xml")

    def random_reset_bounds(self) -> tuple[jax.Array, jax.Array]:
        low = as_f32([-0.1, 0.8, 0.115])
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
        return button_press_topdown_reward_v2(
            obj,
            tcp,
            info["goal_pos"][2],
            tcp_opened,
            info["init_tcp"],
            info["obj_to_target_init"],
        )
