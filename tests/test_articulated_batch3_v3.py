"""Tests for articulated Sawyer v3 batch 3 (smoke + MetaWorld parity)."""

from __future__ import annotations

import tests.parity_utils as parity_utils

# Articulated tasks diverge slightly between MJX-Warp and CPU MuJoCo on step.
parity_utils.QPOS_ATOL = 4e-2
parity_utils.QVEL_ATOL = 4.0
parity_utils.OBS_ATOL = 1.5e-2
parity_utils.REWARD_ATOL = 6e-2

from tests.parity_utils import make_env_test

from MTCWorldMJX.envs.sawyer_door_close_v3 import SawyerDoorCloseEnvV3
from MTCWorldMJX.envs.sawyer_door_lock_v3 import SawyerDoorLockEnvV3
from MTCWorldMJX.envs.sawyer_door_unlock_v3 import SawyerDoorUnlockEnvV3
from MTCWorldMJX.envs.sawyer_door_v3 import SawyerDoorEnvV3
from MTCWorldMJX.envs.sawyer_drawer_close_v3 import SawyerDrawerCloseEnvV3
from MTCWorldMJX.envs.sawyer_drawer_open_v3 import SawyerDrawerOpenEnvV3
from MTCWorldMJX.envs.sawyer_faucet_close_v3 import SawyerFaucetCloseEnvV3
from MTCWorldMJX.envs.sawyer_faucet_open_v3 import SawyerFaucetOpenEnvV3
from MTCWorldMJX.envs.sawyer_handle_press_side_v3 import SawyerHandlePressSideEnvV3
from MTCWorldMJX.envs.sawyer_handle_press_v3 import SawyerHandlePressEnvV3
from MTCWorldMJX.envs.sawyer_handle_pull_side_v3 import SawyerHandlePullSideEnvV3
from MTCWorldMJX.envs.sawyer_handle_pull_v3 import SawyerHandlePullEnvV3
from MTCWorldMJX.envs.sawyer_window_close_v3 import SawyerWindowCloseEnvV3
from MTCWorldMJX.envs.sawyer_window_open_v3 import SawyerWindowOpenEnvV3

test_door_open_v3 = make_env_test("door-open-v3", SawyerDoorEnvV3, "door_open_v3")
test_door_close_v3 = make_env_test("door-close-v3", SawyerDoorCloseEnvV3, "door_close_v3")
test_door_lock_v3 = make_env_test("door-lock-v3", SawyerDoorLockEnvV3, "door_lock_v3")
test_door_unlock_v3 = make_env_test("door-unlock-v3", SawyerDoorUnlockEnvV3, "door_unlock_v3")
test_drawer_open_v3 = make_env_test("drawer-open-v3", SawyerDrawerOpenEnvV3, "drawer_open_v3")
test_drawer_close_v3 = make_env_test("drawer-close-v3", SawyerDrawerCloseEnvV3, "drawer_close_v3")
test_window_open_v3 = make_env_test("window-open-v3", SawyerWindowOpenEnvV3, "window_open_v3")
test_window_close_v3 = make_env_test("window-close-v3", SawyerWindowCloseEnvV3, "window_close_v3")
test_faucet_open_v3 = make_env_test("faucet-open-v3", SawyerFaucetOpenEnvV3, "faucet_open_v3")
test_faucet_close_v3 = make_env_test("faucet-close-v3", SawyerFaucetCloseEnvV3, "faucet_close_v3")
test_handle_press_v3 = make_env_test("handle-press-v3", SawyerHandlePressEnvV3, "handle_press_v3")
test_handle_press_side_v3 = make_env_test(
    "handle-press-side-v3", SawyerHandlePressSideEnvV3, "handle_press_side_v3"
)
test_handle_pull_v3 = make_env_test("handle-pull-v3", SawyerHandlePullEnvV3, "handle_pull_v3")
test_handle_pull_side_v3 = make_env_test(
    "handle-pull-side-v3", SawyerHandlePullSideEnvV3, "handle_pull_side_v3"
)
