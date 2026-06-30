"""Tests for button-press-wall-v3."""

from MTCWorldMJX.envs.sawyer_button_press_wall_v3 import SawyerButtonPressWallEnvV3
from tests.parity_utils import make_env_test

test_button_press_wall_v3 = make_env_test(
    "button-press-wall-v3", SawyerButtonPressWallEnvV3, "button_press_wall_v3"
)
