"""Shared MetaWorld parity test utilities."""

from __future__ import annotations

import contextlib
import time
from typing import Any, Type

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from MTCWorldMJX.envs.sawyer_xyz import SawyerXYZConfig, SawyerXYZEnv

metaworld = pytest.importorskip("metaworld")
from metaworld.env_dict import ALL_V3_ENVIRONMENTS  # noqa: E402

QPOS_ATOL = 5e-3
QVEL_ATOL = 0.15
STEP_QVEL_ATOL = 0.4
OBS_ATOL = 5e-3
REWARD_ATOL = 2e-2
PARITY_ACTION = np.array([0.1, -0.2, 0.05, 0.5], dtype=np.float32)

PARITY_CONFIG = SawyerXYZConfig(
    solver_iterations=50,
    ls_iterations=50,
    zero_geom_margins=True,
)

METRIC_KEYS = (
    "success",
    "near_object",
    "grasp_success",
    "grasp_reward",
    "in_place_reward",
    "obj_to_target",
)

# Minimum obj-goal XY separation that an env's MetaWorld ``reset_model``
# enforces via a rejection loop (``while dist < threshold: resample``). With a
# frozen ``rand_vec`` that loop cannot resample, so a midpoint vector below the
# threshold makes ``mw.reset()`` spin forever.
DEFAULT_MIN_XY_DIST = 0.15
MIN_XY_DIST = {
    "box-close-v3": 0.25,
    "disassemble-v3": 0.1,
}


@contextlib.contextmanager
def _timed(label: str):
    """Print a flushed start/end + duration so it streams under ``pytest -s``."""
    print(f"  [start] {label}", flush=True)
    t0 = time.perf_counter()
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        print(f"  [done ] {label}: {dt:.3f}s", flush=True)


def _ensure_min_xy_dist(
    vec: np.ndarray, low: np.ndarray, high: np.ndarray, min_dist: float
) -> np.ndarray:
    """Push obj/goal XY apart until ``min_dist`` is met, staying inside bounds."""
    vec = vec.copy()
    if vec.size != 6:
        return vec

    obj_xy = vec[:2]
    goal_xy = vec[-3:-1]

    def xy_dist() -> float:
        return float(np.linalg.norm(obj_xy - goal_xy))

    if xy_dist() >= min_dist:
        return vec

    target = min_dist + 0.001

    # First try separating along Y via the goal.
    if goal_xy[1] >= obj_xy[1]:
        goal_xy[1] = min(high[4], goal_xy[1] + (target - xy_dist()))
    else:
        goal_xy[1] = max(low[4], goal_xy[1] - (target - xy_dist()))
    vec[-2] = goal_xy[1]

    if xy_dist() >= min_dist:
        vec[-3:-1] = goal_xy
        return vec

    # Y-only separation can hit a bound (disassemble: shared midpoint Y and a
    # tight goal-Y range). Pick the goal XY corner farthest from the object.
    best_goal = goal_xy.copy()
    best_dist = xy_dist()
    for gx in (low[3], high[3]):
        for gy in (low[4], high[4]):
            candidate = np.array([gx, gy], dtype=np.float64)
            dist = float(np.linalg.norm(obj_xy - candidate))
            if dist > best_dist:
                best_dist = dist
                best_goal = candidate

    if best_dist >= min_dist:
        vec[-3] = best_goal[0]
        vec[-2] = best_goal[1]
        return vec

    # Last resort: also search obj XY corners with the best goal found so far.
    for ox in (low[0], high[0]):
        for oy in (low[1], high[1]):
            candidate_obj = np.array([ox, oy], dtype=np.float64)
            dist = float(np.linalg.norm(candidate_obj - best_goal))
            if dist >= min_dist:
                vec[0] = candidate_obj[0]
                vec[1] = candidate_obj[1]
                vec[-3] = best_goal[0]
                vec[-2] = best_goal[1]
                return vec

    raise ValueError(
        f"cannot satisfy min_xy_dist={min_dist} within bounds for {vec}"
    )


def fixed_rand_vec(env_name: str) -> np.ndarray:
    """Midpoint of MetaWorld reset bounds for deterministic parity.

    Honors the env's real obj-goal separation constraint so the frozen vector
    is accepted by MetaWorld's ``reset_model`` rejection loop on the first try.
    The same vector is fed to both the MetaWorld and MJX envs by the caller.
    """
    mw_cls = ALL_V3_ENVIRONMENTS[env_name]
    env = mw_cls()
    env._set_task_called = True
    low = env._random_reset_space.low
    high = env._random_reset_space.high
    vec = ((low + high) / 2.0).astype(np.float64)
    min_dist = MIN_XY_DIST.get(env_name, DEFAULT_MIN_XY_DIST)
    return _ensure_min_xy_dist(vec, low, high, min_dist)


def make_metaworld_env(env_name: str, rand_vec: np.ndarray):
    env = ALL_V3_ENVIRONMENTS[env_name]()
    env._set_task_called = True
    env._freeze_rand_vec = True
    env._last_rand_vec = rand_vec.copy()
    return env


def make_mjx_env(mjx_cls: Type[SawyerXYZEnv]) -> SawyerXYZEnv:
    return mjx_cls(config=PARITY_CONFIG)


class _DefaultEnv:
    """A default-config env with jitted reset/step, shared within one test.

    Scoped per test (built in ``make_env_test``) rather than module-global so
    that the per-test cache clear in ``conftest`` actually frees device memory
    and the ~50 models don't accumulate in VRAM. The persistent compilation
    cache still makes the (re)compile a cheap disk load on warm runs.
    """

    def __init__(self, env: SawyerXYZEnv):
        self.env = env
        self.reset = jax.jit(env.reset)
        self.step = jax.jit(env.step)


def _make_default_env(env_name: str) -> _DefaultEnv:
    from MTCWorldMJX import make

    with _timed(f"make({env_name}) [load_mj_model + mjx.put_model warp kernels]"):
        env = make(env_name)
    with _timed("_DefaultEnv wrap (jax.jit is lazy, should be ~0s)"):
        denv = _DefaultEnv(env)
    return denv


def run_smoke_test(env_name: str, denv: _DefaultEnv | None = None) -> None:
    denv = denv or _make_default_env(env_name)
    rng = jax.random.PRNGKey(0)
    with _timed("smoke: denv.reset (FIRST compile of reset graph)"):
        state = denv.reset(rng)
        state.obs.block_until_ready()

    assert state.obs.shape == (39,)
    assert jnp.isfinite(state.obs).all()
    assert state.data.qpos.shape == (denv.env.mj_model.nq,)
    assert state.data.qvel.shape == (denv.env.mj_model.nv,)

    with _timed("smoke: denv.step (FIRST compile of step graph)"):
        next_state = denv.step(state, jnp.zeros(4))
        next_state.obs.block_until_ready()
    assert next_state.obs.shape == (39,)
    assert jnp.isfinite(next_state.reward)
    assert next_state.info["path_length"] == 1


def run_rollout_test(
    env_name: str, steps: int = 10, denv: _DefaultEnv | None = None
) -> None:
    denv = denv or _make_default_env(env_name)
    rng = jax.random.PRNGKey(42)
    state = denv.reset(rng)
    for _ in range(steps):
        rng, key = jax.random.split(rng)
        action = jax.random.uniform(key, (4,), minval=-1.0, maxval=1.0)
        state = denv.step(state, action)
        assert jnp.isfinite(state.obs).all()


def run_parity_test(
    env_name: str,
    mjx_cls: Type[SawyerXYZEnv],
    *,
    rand_vec: np.ndarray | None = None,
    qvel_atol: float | None = None,
    step_qvel_atol: float | None = None,
    extra_reset_checks: Any | None = None,
) -> None:
    if rand_vec is None:
        rand_vec = fixed_rand_vec(env_name)
    if qvel_atol is None:
        qvel_atol = QVEL_ATOL
    if step_qvel_atol is None:
        step_qvel_atol = max(qvel_atol, STEP_QVEL_ATOL)
    with _timed("parity: make_metaworld_env"):
        mw = make_metaworld_env(env_name, rand_vec)
    with _timed("parity: make_mjx_env (PARITY_CONFIG, mjx.put_model warp kernels)"):
        env = make_mjx_env(mjx_cls)
    jit_reset = jax.jit(lambda key, rv: env.reset(key, rand_vec=rv))
    jit_step = jax.jit(env.step)

    with _timed("parity: mw.reset (CPU MuJoCo)"):
        mw_obs, _ = mw.reset()
    with _timed("parity: jit_reset (FIRST compile, PARITY_CONFIG graph)"):
        mjx_state = jit_reset(jax.random.PRNGKey(0), jnp.asarray(rand_vec))
        mjx_state.obs.block_until_ready()

    if extra_reset_checks is not None:
        extra_reset_checks(mw, mjx_state)

    if hasattr(mw, "_target_pos") and mw._target_pos is not None:
        np.testing.assert_allclose(
            mjx_state.info["goal_pos"], mw._target_pos, atol=1e-5
        )
    if getattr(mw, "obj_init_pos", None) is not None:
        np.testing.assert_allclose(
            mjx_state.info["obj_init_pos"],
            np.asarray(mw.obj_init_pos),
            atol=1e-4,
        )

    np.testing.assert_allclose(mjx_state.obs, mw_obs, atol=OBS_ATOL)
    np.testing.assert_allclose(mjx_state.data.qpos, mw.data.qpos, atol=QPOS_ATOL)
    np.testing.assert_allclose(mjx_state.data.qvel, mw.data.qvel, atol=qvel_atol)

    mw_obs, mw_reward, _, _, mw_info = mw.step(PARITY_ACTION)
    with _timed("parity: jit_step (FIRST compile, PARITY_CONFIG graph)"):
        mjx_state = jit_step(mjx_state, jnp.asarray(PARITY_ACTION))
        mjx_state.obs.block_until_ready()

    np.testing.assert_allclose(mjx_state.obs, mw_obs, atol=OBS_ATOL)
    np.testing.assert_allclose(mjx_state.reward, mw_reward, atol=REWARD_ATOL)
    np.testing.assert_allclose(mjx_state.data.qpos, mw.data.qpos, atol=QPOS_ATOL)
    np.testing.assert_allclose(mjx_state.data.qvel, mw.data.qvel, atol=step_qvel_atol)

    for key in METRIC_KEYS:
        if key in mjx_state.metrics and key in mw_info:
            np.testing.assert_allclose(
                mjx_state.metrics[key],
                mw_info[key],
                atol=REWARD_ATOL if "reward" in key or key == "success" else OBS_ATOL,
            )


def make_env_test(
    env_name: str,
    mjx_cls: Type[SawyerXYZEnv],
    module_name: str,
    *,
    rand_vec: np.ndarray | None = None,
    qvel_atol: float | None = None,
    step_qvel_atol: float | None = None,
):
    """Create a consolidated test function for one environment."""

    def test_env() -> None:
        print(f"\n=== {env_name}: building default env ===", flush=True)
        denv = _make_default_env(env_name)
        print(f"=== {env_name}: smoke test ===", flush=True)
        run_smoke_test(env_name, denv=denv)
        print(f"=== {env_name}: rollout test ===", flush=True)
        run_rollout_test(env_name, denv=denv)
        print(f"=== {env_name}: parity test ===", flush=True)
        run_parity_test(
            env_name,
            mjx_cls,
            rand_vec=rand_vec,
            qvel_atol=qvel_atol,
            step_qvel_atol=step_qvel_atol,
        )

    test_env.__name__ = f"test_{module_name}"
    test_env.__doc__ = f"Smoke, rollout, and MetaWorld parity for {env_name}."
    return test_env
