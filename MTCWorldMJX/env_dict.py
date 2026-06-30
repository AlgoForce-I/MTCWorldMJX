"""Environment registry for MTCWorldMJX."""

from __future__ import annotations

from typing import Type

from MTCWorldMJX.envs.sawyer_assembly_peg_v3 import SawyerNutAssemblyEnvV3
from MTCWorldMJX.envs.sawyer_basketball_v3 import SawyerBasketballEnvV3
from MTCWorldMJX.envs.sawyer_bin_picking_v3 import SawyerBinPickingEnvV3
from MTCWorldMJX.envs.sawyer_box_close_v3 import SawyerBoxCloseEnvV3
from MTCWorldMJX.envs.sawyer_button_press_topdown_v3 import SawyerButtonPressTopdownEnvV3
from MTCWorldMJX.envs.sawyer_button_press_topdown_wall_v3 import (
    SawyerButtonPressTopdownWallEnvV3,
)
from MTCWorldMJX.envs.sawyer_button_press_v3 import SawyerButtonPressEnvV3
from MTCWorldMJX.envs.sawyer_button_press_wall_v3 import SawyerButtonPressWallEnvV3
from MTCWorldMJX.envs.sawyer_coffee_button_v3 import SawyerCoffeeButtonEnvV3
from MTCWorldMJX.envs.sawyer_coffee_pull_v3 import SawyerCoffeePullEnvV3
from MTCWorldMJX.envs.sawyer_coffee_push_v3 import SawyerCoffeePushEnvV3
from MTCWorldMJX.envs.sawyer_disassemble_peg_v3 import SawyerNutDisassembleEnvV3
from MTCWorldMJX.envs.sawyer_dial_turn_v3 import SawyerDialTurnEnvV3
from MTCWorldMJX.envs.sawyer_door_close_v3 import SawyerDoorCloseEnvV3
from MTCWorldMJX.envs.sawyer_door_lock_v3 import SawyerDoorLockEnvV3
from MTCWorldMJX.envs.sawyer_door_unlock_v3 import SawyerDoorUnlockEnvV3
from MTCWorldMJX.envs.sawyer_door_v3 import SawyerDoorEnvV3
from MTCWorldMJX.envs.sawyer_drawer_close_v3 import SawyerDrawerCloseEnvV3
from MTCWorldMJX.envs.sawyer_drawer_open_v3 import SawyerDrawerOpenEnvV3
from MTCWorldMJX.envs.sawyer_faucet_close_v3 import SawyerFaucetCloseEnvV3
from MTCWorldMJX.envs.sawyer_faucet_open_v3 import SawyerFaucetOpenEnvV3
from MTCWorldMJX.envs.sawyer_hammer_v3 import SawyerHammerEnvV3
from MTCWorldMJX.envs.sawyer_handle_press_side_v3 import SawyerHandlePressSideEnvV3
from MTCWorldMJX.envs.sawyer_handle_press_v3 import SawyerHandlePressEnvV3
from MTCWorldMJX.envs.sawyer_handle_pull_side_v3 import SawyerHandlePullSideEnvV3
from MTCWorldMJX.envs.sawyer_handle_pull_v3 import SawyerHandlePullEnvV3
from MTCWorldMJX.envs.sawyer_hand_insert_v3 import SawyerHandInsertEnvV3
from MTCWorldMJX.envs.sawyer_lever_pull_v3 import SawyerLeverPullEnvV3
from MTCWorldMJX.envs.sawyer_peg_insertion_side_v3 import SawyerPegInsertionSideEnvV3
from MTCWorldMJX.envs.sawyer_peg_unplug_side_v3 import SawyerPegUnplugSideEnvV3
from MTCWorldMJX.envs.sawyer_pick_out_of_hole_v3 import SawyerPickOutOfHoleEnvV3
from MTCWorldMJX.envs.sawyer_pick_place_v3 import SawyerPickPlaceEnvV3
from MTCWorldMJX.envs.sawyer_pick_place_wall_v3 import SawyerPickPlaceWallEnvV3
from MTCWorldMJX.envs.sawyer_plate_slide_back_side_v3 import SawyerPlateSlideBackSideEnvV3
from MTCWorldMJX.envs.sawyer_plate_slide_back_v3 import SawyerPlateSlideBackEnvV3
from MTCWorldMJX.envs.sawyer_plate_slide_side_v3 import SawyerPlateSlideSideEnvV3
from MTCWorldMJX.envs.sawyer_plate_slide_v3 import SawyerPlateSlideEnvV3
from MTCWorldMJX.envs.sawyer_push_back_v3 import SawyerPushBackEnvV3
from MTCWorldMJX.envs.sawyer_push_v3 import SawyerPushEnvV3
from MTCWorldMJX.envs.sawyer_push_wall_v3 import SawyerPushWallEnvV3
from MTCWorldMJX.envs.sawyer_reach_v3 import SawyerReachEnvV3, SawyerReachWallEnvV3
from MTCWorldMJX.envs.sawyer_shelf_place_v3 import SawyerShelfPlaceEnvV3
from MTCWorldMJX.envs.sawyer_soccer_v3 import SawyerSoccerEnvV3
from MTCWorldMJX.envs.sawyer_stick_pull_v3 import SawyerStickPullEnvV3
from MTCWorldMJX.envs.sawyer_stick_push_v3 import SawyerStickPushEnvV3
from MTCWorldMJX.envs.sawyer_sweep_into_goal_v3 import SawyerSweepIntoGoalEnvV3
from MTCWorldMJX.envs.sawyer_sweep_v3 import SawyerSweepEnvV3
from MTCWorldMJX.envs.sawyer_window_close_v3 import SawyerWindowCloseEnvV3
from MTCWorldMJX.envs.sawyer_window_open_v3 import SawyerWindowOpenEnvV3
from MTCWorldMJX.envs.sawyer_xyz import SawyerXYZEnv

ENV_CLS_MAP: dict[str, Type[SawyerXYZEnv]] = {
    "assembly-v3": SawyerNutAssemblyEnvV3,
    "basketball-v3": SawyerBasketballEnvV3,
    "bin-picking-v3": SawyerBinPickingEnvV3,
    "box-close-v3": SawyerBoxCloseEnvV3,
    "button-press-v3": SawyerButtonPressEnvV3,
    "button-press-wall-v3": SawyerButtonPressWallEnvV3,
    "button-press-topdown-v3": SawyerButtonPressTopdownEnvV3,
    "button-press-topdown-wall-v3": SawyerButtonPressTopdownWallEnvV3,
    "coffee-button-v3": SawyerCoffeeButtonEnvV3,
    "coffee-push-v3": SawyerCoffeePushEnvV3,
    "coffee-pull-v3": SawyerCoffeePullEnvV3,
    "disassemble-v3": SawyerNutDisassembleEnvV3,
    "dial-turn-v3": SawyerDialTurnEnvV3,
    "door-close-v3": SawyerDoorCloseEnvV3,
    "door-lock-v3": SawyerDoorLockEnvV3,
    "door-open-v3": SawyerDoorEnvV3,
    "door-unlock-v3": SawyerDoorUnlockEnvV3,
    "drawer-close-v3": SawyerDrawerCloseEnvV3,
    "drawer-open-v3": SawyerDrawerOpenEnvV3,
    "faucet-close-v3": SawyerFaucetCloseEnvV3,
    "faucet-open-v3": SawyerFaucetOpenEnvV3,
    "hammer-v3": SawyerHammerEnvV3,
    "handle-press-side-v3": SawyerHandlePressSideEnvV3,
    "handle-press-v3": SawyerHandlePressEnvV3,
    "handle-pull-side-v3": SawyerHandlePullSideEnvV3,
    "handle-pull-v3": SawyerHandlePullEnvV3,
    "hand-insert-v3": SawyerHandInsertEnvV3,
    "lever-pull-v3": SawyerLeverPullEnvV3,
    "peg-insert-side-v3": SawyerPegInsertionSideEnvV3,
    "peg-unplug-side-v3": SawyerPegUnplugSideEnvV3,
    "pick-out-of-hole-v3": SawyerPickOutOfHoleEnvV3,
    "pick-place-v3": SawyerPickPlaceEnvV3,
    "pick-place-wall-v3": SawyerPickPlaceWallEnvV3,
    "plate-slide-back-side-v3": SawyerPlateSlideBackSideEnvV3,
    "plate-slide-back-v3": SawyerPlateSlideBackEnvV3,
    "plate-slide-side-v3": SawyerPlateSlideSideEnvV3,
    "plate-slide-v3": SawyerPlateSlideEnvV3,
    "push-back-v3": SawyerPushBackEnvV3,
    "push-v3": SawyerPushEnvV3,
    "push-wall-v3": SawyerPushWallEnvV3,
    "reach-v3": SawyerReachEnvV3,
    "reach-wall-v3": SawyerReachWallEnvV3,
    "shelf-place-v3": SawyerShelfPlaceEnvV3,
    "soccer-v3": SawyerSoccerEnvV3,
    "stick-pull-v3": SawyerStickPullEnvV3,
    "stick-push-v3": SawyerStickPushEnvV3,
    "sweep-into-v3": SawyerSweepIntoGoalEnvV3,
    "sweep-v3": SawyerSweepEnvV3,
    "window-close-v3": SawyerWindowCloseEnvV3,
    "window-open-v3": SawyerWindowOpenEnvV3,
}


def make(env_name: str, **kwargs) -> SawyerXYZEnv:
    """Construct an environment by MetaWorld task name."""
    if env_name not in ENV_CLS_MAP:
        known = ", ".join(sorted(ENV_CLS_MAP))
        raise KeyError(f"Unknown environment '{env_name}'. Known environments: {known}")
    return ENV_CLS_MAP[env_name](**kwargs)
