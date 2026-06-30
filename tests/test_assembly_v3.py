"""Smoke, rollout, and MetaWorld parity for assembly-v3."""

from __future__ import annotations

import numpy as np

from MTCWorldMJX.envs.sawyer_assembly_peg_v3 import SawyerNutAssemblyEnvV3
from tests.parity_utils import make_env_test

FIXED_RAND_VEC = np.array([0.0, 0.6, 0.02, 0.05, 0.78, 0.1], dtype=np.float64)

test_assembly_v3 = make_env_test(
    "assembly-v3",
    SawyerNutAssemblyEnvV3,
    "assembly_v3",
    rand_vec=FIXED_RAND_VEC,
)
