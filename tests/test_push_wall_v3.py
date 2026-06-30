"""Smoke, rollout, and MetaWorld parity for push-wall-v3."""

from __future__ import annotations

from MTCWorldMJX.envs.sawyer_push_wall_v3 import SawyerPushWallEnvV3
from tests.parity_utils import make_env_test

test_push_wall_v3 = make_env_test("push-wall-v3", SawyerPushWallEnvV3, "push_wall_v3")
