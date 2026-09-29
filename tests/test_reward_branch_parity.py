"""Reward/success parity with MetaWorld on forced reward branches (CPU only).

The single-step parity tests never reach the late reward branches (stick
inserted, container on the goal, block against the shelf front), which is where
the MJX ports had drifted from MetaWorld. MetaWorld's CPU MuJoCo provides the
kinematic state; the same arrays are fed to the MJX ``compute_reward``, so no
device model is needed.
"""

from __future__ import annotations

from types import SimpleNamespace

import jax.numpy as jnp
import mujoco
import numpy as np
import pytest
from mujoco import mjx

metaworld = pytest.importorskip("metaworld")
from metaworld.env_dict import ALL_V3_ENVIRONMENTS  # noqa: E402

from MTCWorldMJX.envs.sawyer_hammer_v3 import SawyerHammerEnvV3  # noqa: E402
from MTCWorldMJX.envs.sawyer_shelf_place_v3 import SawyerShelfPlaceEnvV3  # noqa: E402
from MTCWorldMJX.envs.sawyer_stick_pull_v3 import SawyerStickPullEnvV3  # noqa: E402
from MTCWorldMJX.envs.sawyer_stick_push_v3 import SawyerStickPushEnvV3  # noqa: E402

REWARD_ATOL = 2e-3
NUM_STATES = 200


@pytest.fixture(autouse=True)
def _no_device_model(monkeypatch):
    # compute_reward only reads kinematic arrays; skip the Warp device model.
    monkeypatch.setattr(mjx, "put_model", lambda model, impl=None: None)


def _metaworld_env(name: str):
    env = ALL_V3_ENVIRONMENTS[name]()
    env._set_task_called = True
    low, high = env._random_reset_space.low, env._random_reset_space.high
    env._freeze_rand_vec = True
    env._last_rand_vec = ((low + high) / 2.0).astype(np.float64)
    env.reset()
    env.init_tcp = env.tcp_center.copy()
    return env


def _kinematic_data(mw) -> SimpleNamespace:
    d = mw.data
    return SimpleNamespace(
        xpos=jnp.asarray(d.xpos),
        xquat=jnp.asarray(d.xquat),
        xmat=jnp.asarray(d.xmat),
        site_xpos=jnp.asarray(d.site_xpos),
        geom_xpos=jnp.asarray(d.geom_xpos),
        geom_xmat=jnp.asarray(d.geom_xmat),
        qpos=jnp.asarray(d.qpos),
        qvel=jnp.asarray(d.qvel),
        mocap_pos=jnp.asarray(d.mocap_pos),
        mocap_quat=jnp.asarray(d.mocap_quat),
    )


def _site(mw, name: str) -> int:
    return mujoco.mj_name2id(mw.model, mujoco.mjtObj.mjOBJ_SITE, name)


def _body(mw, name: str) -> int:
    return mujoco.mj_name2id(mw.model, mujoco.mjtObj.mjOBJ_BODY, name)


def _run_parity(mw, mjx_env, info, override, *, check_success: bool = True) -> list[dict]:
    rng = np.random.default_rng(0)
    q0, v0 = mw.data.qpos.copy(), mw.data.qvel.copy()
    rows = []
    for index in range(NUM_STATES):
        noise = rng.normal(0.0, 0.02, q0.shape) if index % 3 else 0.0
        mw.set_state(q0 + noise, v0)
        override(mw, index, rng)
        action = rng.uniform(-1.0, 1.0, 4)
        obs = mw._get_obs()
        mw_reward, mw_info = mw.evaluate_state(obs, action)
        reward, metrics = mjx_env.compute_reward(_kinematic_data(mw), jnp.asarray(action), info)
        np.testing.assert_allclose(float(reward), float(mw_reward), atol=REWARD_ATOL)
        if check_success:
            assert float(metrics["success"]) == float(mw_info["success"])
        rows.append({"obs": obs, "reward": float(mw_reward), "success": float(mw_info["success"])})
    return rows


def test_stick_pull_reward_branches_match_metaworld() -> None:
    mw = _metaworld_env("stick-pull-v3")
    env = SawyerStickPullEnvV3()
    info = {
        "goal_pos": jnp.asarray(mw._target_pos),
        "obj_init_pos": jnp.asarray(mw.obj_init_pos),
        "stick_init_pos": jnp.asarray(mw.stick_init_pos),
        "init_tcp": jnp.asarray(mw.init_tcp),
    }
    insertion, stick_end, stick = _site(mw, "insertion"), _site(mw, "stick_end"), _body(mw, "stick")

    def override(mw, index, rng):
        mode = index % 5
        if mode in (1, 2, 3):  # stick held above its initial height
            mw.data.xpos[stick] = mw.tcp_center + rng.normal(0.0, 0.006, 3)
        if mode in (2, 3):  # stick end inside the handle
            mw.data.site_xpos[stick_end] = mw.data.site_xpos[insertion] + np.array(
                [rng.uniform(0.0, 0.02), rng.normal(0.0, 0.02), rng.normal(0.0, 0.03)]
            )
        if mode in (3, 4):  # container handle on the goal; mode 4 has no stick
            shift = mw._target_pos + rng.normal(0.0, 0.05, 3) - mw.data.site_xpos[insertion]
            mw.data.site_xpos[insertion] += shift
            if mode == 3:
                mw.data.site_xpos[stick_end] += shift

    rows = _run_parity(mw, env, info, override)
    exploit = [
        row
        for index, row in enumerate(rows)
        if index % 5 == 4 and np.linalg.norm(row["obs"][11:14] - mw._target_pos) <= 0.12
    ]
    assert exploit, "no container-on-goal-without-stick states sampled"
    assert all(row["reward"] < 10.0 and row["success"] == 0.0 for row in exploit)
    assert any(row["reward"] == 10.0 and row["success"] == 1.0 for row in rows)


def test_stick_push_reward_branches_match_metaworld() -> None:
    mw = _metaworld_env("stick-push-v3")
    env = SawyerStickPushEnvV3()
    info = {
        "goal_pos": jnp.asarray(mw._target_pos),
        "obj_init_pos": jnp.asarray(mw.obj_init_pos),
        "stick_init_pos": jnp.asarray(mw.stick_init_pos),
        "init_tcp": jnp.asarray(mw.init_tcp),
    }
    insertion, stick = _site(mw, "insertion"), _body(mw, "stick")

    def override(mw, index, rng):
        mode = index % 4
        if mode in (1, 2):
            mw.data.xpos[stick] = mw.tcp_center - np.array([0.015, 0.0, 0.0]) + rng.normal(0.0, 0.006, 3)
        if mode in (2, 3):  # container on the goal; mode 3 has no stick
            mw.data.site_xpos[insertion] = (
                mw._target_pos - np.array([0.0, 0.09, 0.0]) + rng.normal(0.0, 0.05, 3)
            )

    # MetaWorld's stick-push success also needs a contact-based grasp check that
    # MJX approximates by distance, so only the reward is compared here.
    rows = _run_parity(mw, env, info, override, check_success=False)
    no_stick_on_goal = [
        row
        for index, row in enumerate(rows)
        if index % 4 == 3 and np.linalg.norm(row["obs"][11:14] - mw._target_pos) <= 0.12
    ]
    assert no_stick_on_goal
    assert all(row["reward"] < 10.0 for row in no_stick_on_goal)


def test_shelf_place_reward_branches_match_metaworld() -> None:
    mw = _metaworld_env("shelf-place-v3")
    env = SawyerShelfPlaceEnvV3()
    info = {
        "goal_pos": jnp.asarray(mw._target_pos),
        "obj_init_pos": jnp.asarray(mw.obj_init_pos),
        "init_tcp": jnp.asarray(mw.init_tcp),
    }
    obj = _body(mw, "obj")
    ee_sites = (_site(mw, "rightEndEffector"), _site(mw, "leftEndEffector"))

    def override(mw, index, rng):
        target = mw._target_pos
        mode = index % 4
        if mode == 0:  # in front of the shelf, below its top (bound-loss band)
            pos = np.array(
                [
                    target[0] + rng.uniform(-0.2, 0.2),
                    rng.uniform(target[1] - 0.2, target[1]),
                    rng.uniform(-0.02, 0.3),
                ]
            )
        elif mode == 1:  # behind the goal
            pos = np.array(
                [
                    target[0] + rng.uniform(-0.2, 0.2),
                    rng.uniform(target[1], target[1] + 0.1),
                    rng.uniform(-0.02, 0.3),
                ]
            )
        elif mode == 2:
            pos = target + rng.normal(0.0, 0.05, 3)
        else:
            return
        mw.data.xpos[obj] = pos
        if rng.random() < 0.5:  # tcp on the block so the grasp bonus uses in_place
            for site in ee_sites:
                mw.data.site_xpos[site] = pos + rng.normal(0.0, 0.005, 3)

    _run_parity(mw, env, info, override)


def test_hammer_success_is_nail_depth_only() -> None:
    mw = _metaworld_env("hammer-v3")
    env = SawyerHammerEnvV3()
    info = {
        "goal_pos": jnp.asarray(mw._target_pos),
        "obj_init_pos": jnp.asarray(mw.obj_init_pos),
        "init_tcp": jnp.asarray(mw.init_tcp),
    }
    joint = mujoco.mj_name2id(mw.model, mujoco.mjtObj.mjOBJ_JOINT, "NailSlideJoint")
    nail_adr = mw.model.jnt_qposadr[joint]

    def override(mw, index, rng):
        if index % 2:
            qpos = mw.data.qpos.copy()
            qpos[nail_adr] = rng.uniform(0.0, 0.2)
            mw.set_state(qpos, mw.data.qvel.copy())

    rows = _run_parity(mw, env, info, override)
    assert any(row["success"] == 1.0 and row["reward"] < 10.0 for row in rows)
