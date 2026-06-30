"""Tests for button-press-topdown-wall-v3."""

from MTCWorldMJX.envs.sawyer_button_press_topdown_wall_v3 import (
    SawyerButtonPressTopdownWallEnvV3,
)
from tests.parity_utils import make_env_test

test_button_press_topdown_wall_v3 = make_env_test(
    "button-press-topdown-wall-v3",
    SawyerButtonPressTopdownWallEnvV3,
    "button_press_topdown_wall_v3",
)
