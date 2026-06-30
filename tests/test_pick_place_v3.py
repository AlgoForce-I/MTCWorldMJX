"""Smoke, rollout, and MetaWorld parity for pick-place-v3."""

from __future__ import annotations

from MTCWorldMJX.envs.sawyer_pick_place_v3 import SawyerPickPlaceEnvV3
from tests.parity_utils import make_env_test

test_pick_place_v3 = make_env_test("pick-place-v3", SawyerPickPlaceEnvV3, "pick_place_v3")
