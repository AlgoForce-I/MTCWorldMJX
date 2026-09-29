"""Smoke tests for Continual World benchmark scaffolding."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from MTCWorldMJX import (
    CW10,
    CWConfig,
    TASK_SEQS,
    cw_obs_dim,
    make_cl_test_envs,
    make_cl_train_env,
)
from MTCWorldMJX.cw_eval import evaluate_cl_policy, evaluate_train_env_smoke


def test_cw10_metadata() -> None:
    bench = CW10(seed=1)
    assert bench.num_tasks == 10
    assert bench.task_names == TASK_SEQS["CW10"]
    assert len(bench.goals_for("hammer-v3")) == 50


def test_cw20_length() -> None:
    from MTCWorldMJX.cw_benchmarks import CW20

    bench = CW20(seed=1)
    assert bench.num_tasks == 20


def test_cl_test_env_obs_shape() -> None:
    test_envs = make_cl_test_envs("CW10", seed=1)
    assert len(test_envs) == 10
    state = test_envs[0].reset(jax.random.PRNGKey(0))
    state.obs.block_until_ready()
    assert state.obs.shape == (cw_obs_dim(10),)
    assert int(state.info["task_idx"]) == 0


def test_cl_test_envs_follow_config_randomization() -> None:
    # Continual World evaluates with get_single_env(...) at its default
    # randomization, random_init_all: a fresh goal every test episode.
    test_envs = make_cl_test_envs("CW10", seed=1)
    assert {env.randomization for env in test_envs} == {"random_init_all"}
    fixed = make_cl_test_envs("CW10", config=CWConfig(randomization="deterministic"), seed=1)
    assert {env.randomization for env in fixed} == {"deterministic"}
    explicit = make_cl_test_envs("CW10", seed=1, randomization="random_init_fixed20")
    assert {env.randomization for env in explicit} == {"random_init_fixed20"}


def test_cl_train_env_task_switch() -> None:
    config = CWConfig(seed=1, steps_per_task=5, episode_horizon=200)
    env = make_cl_train_env("CW10", config=config)

    def zero_policy(obs, key):
        del key
        return jnp.zeros((4,))

    diag = evaluate_train_env_smoke(env, zero_policy, jax.random.PRNGKey(1), num_steps=12)
    diag["mean_reward"].block_until_ready()
    assert int(diag["global_step"]) == 12
    assert int(diag["seq_idx"]) == 2  # after 5 and 10 steps, on task index 2


def test_evaluate_cl_policy_smoke() -> None:
    test_envs = make_cl_test_envs("CW10", seed=1)

    def zero_policy(obs, key):
        del key
        return jnp.zeros((4,))

    log = evaluate_cl_policy(test_envs, zero_policy, jax.random.PRNGKey(2), num_episodes=1)
    assert "test/stochastic/average_success" in log
    assert len(log["per_task"]) == 10  # type: ignore[arg-type]
