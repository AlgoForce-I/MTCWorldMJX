"""Sawyer top-down wall button press environment (button-press-topdown-wall-v3)."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
from mujoco import mjx

from MTCWorldMJX.asset_paths import sawyer_xml_path
from MTCWorldMJX.envs._helpers import as_f32, button_press_topdown_reward_v2
from MTCWorldMJX.envs.sawyer_button_press_topdown_v3 import SawyerButtonPressTopdownEnvV3
from MTCWorldMJX.envs.sawyer_xyz import SawyerXYZConfig


class SawyerButtonPressTopdownWallEnvV3(SawyerButtonPressTopdownEnvV3):
    """Press a wall-mounted button from above (button-press-topdown-wall-v3)."""

    def __init__(self, config: SawyerXYZConfig | None = None):
        super().__init__(config)

    @property
    def xml_path(self):
        return sawyer_xml_path("sawyer_button_press_topdown_wall.xml")
