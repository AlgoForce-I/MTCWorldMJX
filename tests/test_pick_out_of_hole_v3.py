"""Smoke, rollout, and MetaWorld parity for pick-out-of-hole-v3."""

from __future__ import annotations

from MTCWorldMJX.envs.sawyer_pick_out_of_hole_v3 import SawyerPickOutOfHoleEnvV3
from tests.parity_utils import make_env_test

test_pick_out_of_hole_v3 = make_env_test(
    "pick-out-of-hole-v3", SawyerPickOutOfHoleEnvV3, "pick_out_of_hole_v3"
)
