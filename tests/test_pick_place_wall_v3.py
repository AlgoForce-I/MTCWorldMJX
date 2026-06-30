"""Smoke, rollout, and MetaWorld parity for pick-place-wall-v3."""

from __future__ import annotations

from MTCWorldMJX.envs.sawyer_pick_place_wall_v3 import SawyerPickPlaceWallEnvV3
from tests.parity_utils import make_env_test

test_pick_place_wall_v3 = make_env_test(
    "pick-place-wall-v3", SawyerPickPlaceWallEnvV3, "pick_place_wall_v3"
)
