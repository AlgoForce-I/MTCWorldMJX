"""Tests for button-press-v3."""

from MTCWorldMJX.envs.sawyer_button_press_v3 import SawyerButtonPressEnvV3
from tests.parity_utils import make_env_test

test_button_press_v3 = make_env_test(
    "button-press-v3", SawyerButtonPressEnvV3, "button_press_v3"
)
