"""Smoke, rollout, and MetaWorld parity for push-v3."""

from __future__ import annotations

from MTCWorldMJX.envs.sawyer_push_v3 import SawyerPushEnvV3
from tests.parity_utils import make_env_test

test_push_v3 = make_env_test("push-v3", SawyerPushEnvV3, "push_v3")
