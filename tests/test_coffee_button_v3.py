"""Tests for coffee-button-v3."""

from MTCWorldMJX.envs.sawyer_coffee_button_v3 import SawyerCoffeeButtonEnvV3
from tests.parity_utils import make_env_test

test_coffee_button_v3 = make_env_test(
    "coffee-button-v3", SawyerCoffeeButtonEnvV3, "coffee_button_v3"
)
