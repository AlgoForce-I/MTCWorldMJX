"""Tests for lever-pull-v3."""

from MTCWorldMJX.envs.sawyer_lever_pull_v3 import SawyerLeverPullEnvV3
from tests.parity_utils import make_env_test

test_lever_pull_v3 = make_env_test(
    "lever-pull-v3", SawyerLeverPullEnvV3, "lever_pull_v3"
)
