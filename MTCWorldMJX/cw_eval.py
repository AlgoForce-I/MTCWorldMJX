"""Continual World policy evaluation helpers."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import jax
import jax.numpy as jnp

from MTCWorldMJX.cw_env import CWTaskEnv, ContinualLearningEnv, run_episode

PolicyFn = Callable[[jax.Array, jax.Array], jax.Array]


def evaluate_task_env(
    env: CWTaskEnv,
    policy: PolicyFn,
    rng: jax.Array,
    *,
    num_episodes: int = 10,
) -> dict[str, jax.Array | float | str]:
    """Evaluate ``policy`` on a single Continual World task environment."""
    successes = []
    returns = []
    for _ in range(num_episodes):
        rng, key = jax.random.split(rng)
        _, ep_return, ep_success = run_episode(env, policy, key)
        successes.append(float(ep_success))
        returns.append(float(ep_return))

    return {
        "env_name": env.env_name,
        "task_idx": env.task_idx,
        "success_rate": jnp.mean(jnp.asarray(successes)),
        "mean_return": jnp.mean(jnp.asarray(returns)),
        "num_episodes": num_episodes,
    }


def evaluate_cl_policy(
    test_envs: Mapping[str, CWTaskEnv] | Sequence[CWTaskEnv],
    policy: PolicyFn,
    rng: jax.Array,
    *,
    num_episodes: int = 10,
    mode: str = "stochastic",
) -> dict[str, dict[str, jax.Array | float | str | int]]:
    """Evaluate on every Continual World test task.

    Returns a nested dict keyed by ``test/{mode}/{seq_idx}/{env_name}/...`` to
    mirror the reference Continual World logging layout.
    """
    del mode  # reserved for deterministic vs stochastic policy variants
    if isinstance(test_envs, Mapping):
        env_list = list(test_envs.values())
    else:
        env_list = list(test_envs)

    per_task: dict[str, dict[str, jax.Array | float | str | int]] = {}
    log: dict[str, dict[str, jax.Array | float | str | int]] = {}

    for env in env_list:
        rng, key = jax.random.split(rng)
        result = evaluate_task_env(env, policy, key, num_episodes=num_episodes)
        per_task[env.env_name] = result
        prefix = f"test/stochastic/{env.task_idx}/{env.env_name}"
        log[f"{prefix}/success"] = result["success_rate"]
        log[f"{prefix}/return"] = result["mean_return"]

    avg_success = jnp.mean(jnp.asarray([r["success_rate"] for r in per_task.values()]))
    log["test/stochastic/average_success"] = avg_success
    log["per_task"] = per_task  # type: ignore[assignment]
    return log


def evaluate_train_env_smoke(
    env: ContinualLearningEnv,
    policy: PolicyFn,
    rng: jax.Array,
    *,
    num_steps: int,
) -> dict[str, jax.Array | int]:
    """Run a short training-env rollout and return basic diagnostics."""
    state = env.reset(rng)
    returns = []
    for _ in range(num_steps):
        rng, key = jax.random.split(rng)
        action = policy(state.obs, key)
        state = env.step(state, action)
        returns.append(state.reward)
        if float(state.truncated + state.terminated) > 0:
            rng, key = jax.random.split(rng)
            state = env.reset_from_state(key, state)

    return {
        "seq_idx": state.info["seq_idx"],
        "global_step": state.info["global_step"],
        "mean_reward": jnp.mean(jnp.stack(returns)),
        "num_steps": num_steps,
    }
