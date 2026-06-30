"""Tests for dial-turn-v3."""

from MTCWorldMJX.envs.sawyer_dial_turn_v3 import SawyerDialTurnEnvV3
from tests.parity_utils import make_env_test

test_dial_turn_v3 = make_env_test("dial-turn-v3", SawyerDialTurnEnvV3, "dial_turn_v3")
