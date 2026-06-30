"""Smoke, rollout, and MetaWorld parity for push-back-v3."""

from __future__ import annotations

from MTCWorldMJX.envs.sawyer_push_back_v3 import SawyerPushBackEnvV3
from tests.parity_utils import make_env_test

test_push_back_v3 = make_env_test("push-back-v3", SawyerPushBackEnvV3, "push_back_v3")
