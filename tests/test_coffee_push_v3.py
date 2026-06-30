"""Tests for coffee-push-v3."""

import numpy as np

from MTCWorldMJX.envs.sawyer_coffee_push_v3 import SawyerCoffeePushEnvV3
from tests.parity_utils import make_env_test

# Midpoint XY distance must exceed MetaWorld's 0.15 while-loop threshold.
FIXED_RAND_VEC = np.array([0.0, 0.55, 0.0, 0.0, 0.75, 0.0], dtype=np.float64)

test_coffee_push_v3 = make_env_test(
    "coffee-push-v3",
    SawyerCoffeePushEnvV3,
    "coffee_push_v3",
    rand_vec=FIXED_RAND_VEC,
    qvel_atol=0.3,
)
