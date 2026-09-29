"""Goal observation parity with MetaWorld when the goal is visible (Continual World).

MetaWorld clips every step observation to ``sawyer_observation_space``, whose
goal part is the env's ``goal_space``. Clipping the goal to ``[0, inf)`` instead
hid every negative goal coordinate after the first step (7 of the 10 CW10
tasks). The existing parity tests run partially observable, so they never saw
the goal.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from MTCWorldMJX.env_dict import ENV_CLS_MAP, make
from MTCWorldMJX.envs.sawyer_xyz import SawyerXYZConfig

OBS_ATOL = 5e-3
FULLY_OBSERVABLE = SawyerXYZConfig(
    partially_observable=False,
    solver_iterations=50,
    ls_iterations=50,
)

# CW10 tasks whose goals can have a negative coordinate.
NEGATIVE_GOAL_ENVS = [
    "push-wall-v3",
    "faucet-close-v3",
    "push-back-v3",
    "handle-press-side-v3",
    "push-v3",
    "shelf-place-v3",
    "peg-unplug-side-v3",
]


@pytest.mark.parametrize("env_name", sorted(ENV_CLS_MAP))
def test_goal_space_matches_metaworld(env_name: str) -> None:
    env_dict = pytest.importorskip("metaworld.env_dict")
    upstream = env_dict.ALL_V3_ENVIRONMENTS[env_name]().goal_space
    low, high = ENV_CLS_MAP[env_name].goal_space_bounds()
    np.testing.assert_allclose(low, upstream.low, atol=1e-6)
    np.testing.assert_allclose(high, upstream.high, atol=1e-6)


@pytest.mark.parametrize("env_name", NEGATIVE_GOAL_ENVS)
def test_step_goal_observation_matches_metaworld(env_name: str) -> None:
    env_dict = pytest.importorskip("metaworld.env_dict")
    env = make(env_name, config=FULLY_OBSERVABLE)
    keys = jax.random.split(jax.random.PRNGKey(0), 16)
    states = jax.jit(jax.vmap(env.reset))(keys)
    goals = np.asarray(states.info["goal_pos"])
    lanes = np.nonzero((goals < 0).any(axis=1))[0]
    assert lanes.size, f"no sampled {env_name} goal has a negative coordinate"
    lane = int(lanes[0])
    rand_vec = np.asarray(states.info["rand_vec"])[lane]
    state = jax.jit(lambda k, v: env.reset(k, rand_vec=v))(keys[lane], jnp.asarray(rand_vec))
    np.testing.assert_allclose(state.info["goal_pos"], goals[lane], atol=1e-6)

    mw = env_dict.ALL_V3_ENVIRONMENTS[env_name]()
    mw._set_task_called = True
    mw._partially_observable = False
    mw.__dict__.pop("sawyer_observation_space", None)
    mw._freeze_rand_vec = True
    mw._last_rand_vec = np.asarray(rand_vec, dtype=np.float64)
    mw_obs, _ = mw.reset()
    np.testing.assert_allclose(state.obs[-3:], mw_obs[-3:], atol=OBS_ATOL)

    step = jax.jit(env.step)
    action = np.zeros(4, dtype=np.float32)
    for _ in range(3):
        state = step(state, jnp.asarray(action))
        mw_obs, *_ = mw.step(action)
    np.testing.assert_allclose(state.obs[-3:], mw_obs[-3:], atol=OBS_ATOL)
    np.testing.assert_allclose(state.obs[-3:], goals[lane], atol=OBS_ATOL)
    np.testing.assert_allclose(state.obs, mw_obs, atol=OBS_ATOL)
