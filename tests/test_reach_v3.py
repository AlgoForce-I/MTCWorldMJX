"""Smoke, rollout, and MetaWorld parity for reach-v3."""

from __future__ import annotations

from MTCWorldMJX.envs.sawyer_reach_v3 import SawyerReachEnvV3
from tests.parity_utils import make_env_test

test_reach_v3 = make_env_test("reach-v3", SawyerReachEnvV3, "reach_v3")
