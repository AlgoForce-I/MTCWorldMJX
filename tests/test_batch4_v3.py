"""MetaWorld parity tests for batch-4 Sawyer v3 environments."""

from __future__ import annotations

import pytest

from MTCWorldMJX.envs.sawyer_basketball_v3 import SawyerBasketballEnvV3
from MTCWorldMJX.envs.sawyer_bin_picking_v3 import SawyerBinPickingEnvV3
from MTCWorldMJX.envs.sawyer_box_close_v3 import SawyerBoxCloseEnvV3
from MTCWorldMJX.envs.sawyer_disassemble_peg_v3 import SawyerNutDisassembleEnvV3
from MTCWorldMJX.envs.sawyer_hammer_v3 import SawyerHammerEnvV3
from MTCWorldMJX.envs.sawyer_hand_insert_v3 import SawyerHandInsertEnvV3
from MTCWorldMJX.envs.sawyer_peg_insertion_side_v3 import SawyerPegInsertionSideEnvV3
from MTCWorldMJX.envs.sawyer_peg_unplug_side_v3 import SawyerPegUnplugSideEnvV3
from MTCWorldMJX.envs.sawyer_plate_slide_back_side_v3 import SawyerPlateSlideBackSideEnvV3
from MTCWorldMJX.envs.sawyer_plate_slide_back_v3 import SawyerPlateSlideBackEnvV3
from MTCWorldMJX.envs.sawyer_plate_slide_side_v3 import SawyerPlateSlideSideEnvV3
from MTCWorldMJX.envs.sawyer_plate_slide_v3 import SawyerPlateSlideEnvV3
from MTCWorldMJX.envs.sawyer_shelf_place_v3 import SawyerShelfPlaceEnvV3
from MTCWorldMJX.envs.sawyer_soccer_v3 import SawyerSoccerEnvV3
from MTCWorldMJX.envs.sawyer_stick_pull_v3 import SawyerStickPullEnvV3
from MTCWorldMJX.envs.sawyer_stick_push_v3 import SawyerStickPushEnvV3
from MTCWorldMJX.envs.sawyer_sweep_into_goal_v3 import SawyerSweepIntoGoalEnvV3
from MTCWorldMJX.envs.sawyer_sweep_v3 import SawyerSweepEnvV3
from tests.parity_utils import make_env_test

pytest.importorskip("metaworld")

BATCH4_ENVS = [
    ("basketball-v3", SawyerBasketballEnvV3, "basketball_v3"),
    ("bin-picking-v3", SawyerBinPickingEnvV3, "bin_picking_v3"),
    ("box-close-v3", SawyerBoxCloseEnvV3, "box_close_v3"),
    ("hammer-v3", SawyerHammerEnvV3, "hammer_v3"),
    ("hand-insert-v3", SawyerHandInsertEnvV3, "hand_insert_v3"),
    ("shelf-place-v3", SawyerShelfPlaceEnvV3, "shelf_place_v3"),
    ("soccer-v3", SawyerSoccerEnvV3, "soccer_v3"),
    ("sweep-v3", SawyerSweepEnvV3, "sweep_v3"),
    ("sweep-into-v3", SawyerSweepIntoGoalEnvV3, "sweep_into_v3"),
    ("stick-push-v3", SawyerStickPushEnvV3, "stick_push_v3"),
    ("stick-pull-v3", SawyerStickPullEnvV3, "stick_pull_v3"),
    ("plate-slide-v3", SawyerPlateSlideEnvV3, "plate_slide_v3"),
    ("plate-slide-back-v3", SawyerPlateSlideBackEnvV3, "plate_slide_back_v3"),
    ("plate-slide-side-v3", SawyerPlateSlideSideEnvV3, "plate_slide_side_v3"),
    ("plate-slide-back-side-v3", SawyerPlateSlideBackSideEnvV3, "plate_slide_back_side_v3"),
    ("peg-insert-side-v3", SawyerPegInsertionSideEnvV3, "peg_insert_side_v3"),
    ("peg-unplug-side-v3", SawyerPegUnplugSideEnvV3, "peg_unplug_side_v3", {"qvel_atol": 7.0}),
    ("disassemble-v3", SawyerNutDisassembleEnvV3, "disassemble_v3"),
]

for item in BATCH4_ENVS:
    if len(item) == 4:
        env_name, cls, module_name, test_kwargs = item
    else:
        env_name, cls, module_name = item
        test_kwargs = {}
    globals()[f"test_{module_name}"] = make_env_test(
        env_name, cls, module_name, **test_kwargs
    )
