"""Smoke, rollout, and MetaWorld parity for reach-wall-v3."""

from __future__ import annotations

from MTCWorldMJX.envs.sawyer_reach_v3 import SawyerReachWallEnvV3
from tests.parity_utils import make_env_test

test_reach_wall_v3 = make_env_test(
    "reach-wall-v3", SawyerReachWallEnvV3, "reach_wall_v3"
)
