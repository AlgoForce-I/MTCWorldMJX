"""Tests for button-press-topdown-v3."""

from MTCWorldMJX.envs.sawyer_button_press_topdown_v3 import SawyerButtonPressTopdownEnvV3
from tests.parity_utils import make_env_test

test_button_press_topdown_v3 = make_env_test(
    "button-press-topdown-v3", SawyerButtonPressTopdownEnvV3, "button_press_topdown_v3"
)
