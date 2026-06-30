"""Smoke test for the benchmark API (MT1/MT10, MT/ML env factories, rollout)."""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp

from MTCWorldMJX import (
    MT1,
    make_mt_envs,
    make_ml_envs_train,
    rollout,
)

NUM_ENVS = 512
ROLLOUT_STEPS = 200


def test_mt1_metadata() -> None:
    bench = MT1("reach-v3", seed=0)
    assert len(bench.train_tasks) == 50
    assert bench.train_tasks[0].env_name == "reach-v3"
    assert bench.train_tasks[0].partially_observable is False


def test_make_mt_envs_single_task() -> None:
    env = make_mt_envs("reach-v3", seed=0, num_envs=NUM_ENVS)
    state = env.reset(jax.random.PRNGKey(0))
    assert state.obs.shape == (NUM_ENVS, 39)

    actions = jnp.zeros((NUM_ENVS, 4))
    state = env.step(state, actions)
    assert state.obs.shape == (NUM_ENVS, 39)


def test_make_mt_envs_mt10() -> None:
    mt10 = make_mt_envs("MT10", seed=0, num_envs=NUM_ENVS)
    assert len(mt10) == 10
    _, subenv = next(iter(mt10.items()))
    sub_state = subenv.reset(jax.random.PRNGKey(1))
    assert sub_state.obs.shape == (NUM_ENVS, 39)


def test_make_ml_envs_train_hidden_goal() -> None:
    ml = make_ml_envs_train("reach-v3", seed=0, num_envs=NUM_ENVS)
    assert ml.partially_observable is True
    ml_state = ml.reset(jax.random.PRNGKey(2))
    assert ml_state.obs.shape == (NUM_ENVS, 39)


def test_rollout_throughput() -> None:
    env = make_mt_envs("reach-v3", seed=0, num_envs=NUM_ENVS)

    def zero_policy(obs, key):
        del key
        return jnp.zeros((obs.shape[0], 4))

    state = env.reset(jax.random.PRNGKey(3))
    state, traj = rollout(env, state, zero_policy, jax.random.PRNGKey(4), ROLLOUT_STEPS)
    traj["reward"].block_until_ready()  # warm compile

    t0 = time.time()
    state, traj = rollout(env, state, zero_policy, jax.random.PRNGKey(5), ROLLOUT_STEPS)
    traj["reward"].block_until_ready()
    dt = time.time() - t0

    assert traj["reward"].shape == (ROLLOUT_STEPS, NUM_ENVS)
    print(f"rollout: {NUM_ENVS} envs x {ROLLOUT_STEPS} steps in {dt:.3f}s")
    print(f"throughput: {NUM_ENVS * ROLLOUT_STEPS / dt:,.0f} env-steps/s")
