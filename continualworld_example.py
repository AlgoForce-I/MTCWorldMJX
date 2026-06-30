"""Minimal Continual World (CW10 / CW20) training loop example.

Run:

    .venv/bin/python continualworld_example.py
    .venv/bin/python continualworld_example.py --benchmark CW20
    .venv/bin/python continualworld_example.py --mode single --steps-per-task 2000

Default **saturated** mode vectorizes each CW task with ``VectorEnv`` (512 lanes by
default) and JIT ``lax.scan`` rollouts to demonstrate GPU throughput. **single**
mode uses one lane per task (sequential CL env) for protocol debugging.

Paper-scale runs use ``steps_per_task=1_000_000`` via ``CWConfig``.
"""

from __future__ import annotations

import argparse
import os
import time
from collections.abc import Callable
from dataclasses import dataclass

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp

from MTCWorldMJX import CWConfig, cw_obs_dim, make_cl_test_envs, make_cl_train_env, rollout
from MTCWorldMJX.cw_benchmarks import CWBenchmark, TASK_SEQS, cw_sawyer_config
from MTCWorldMJX.cw_env import CWTaskEnv, ContinualLearningEnv
from MTCWorldMJX import mjx_env
from MTCWorldMJX.mt_benchmarks import VectorEnv, rand_vecs_for_env

SEED = 1
NUM_ENVS = 512
STEPS_PER_TASK_SATURATED = 200
STEPS_PER_TASK_SINGLE = 2000
EVAL_EPISODES = 1

SegmentRolloutFn = Callable[
    [mjx_env.State, jax.Array],
    tuple[mjx_env.State, jax.Array],
]
VectorRolloutFn = Callable[
    [mjx_env.State, jax.Array],
    tuple[mjx_env.State, jax.Array],
]


@dataclass(frozen=True)
class SaturatedTaskPlan:
    """One CW sequence slot: vectorized env + task one-hot index."""

    env_name: str
    task_idx: int
    venv: VectorEnv


@jax.jit
def random_policy(obs: jax.Array, key: jax.Array) -> jax.Array:
    del obs
    return jax.random.uniform(key, (4,), minval=-1.0, maxval=1.0)


@jax.jit
def batched_random_policy(obs: jax.Array, key: jax.Array) -> jax.Array:
    return jax.random.uniform(key, (obs.shape[0], 4), minval=-1.0, maxval=1.0)


def build_saturated_plan(
    benchmark: CWBenchmark,
    *,
    num_envs: int,
) -> list[SaturatedTaskPlan]:
    """VectorEnv per CW sequence slot (``random_init_all`` goal sampling)."""
    sawyer_config = cw_sawyer_config(benchmark.config)
    plans: list[SaturatedTaskPlan] = []
    for task_idx, env_name in enumerate(benchmark.task_names):
        rand_vecs = rand_vecs_for_env(benchmark.tasks, env_name)
        venv = VectorEnv(
            env_name,
            rand_vecs,
            num_envs,
            config=sawyer_config,
            partially_observable=False,
            task_select="random",
            seed=SEED + task_idx,
        )
        plans.append(SaturatedTaskPlan(env_name, task_idx, venv))
    return plans


def make_jitted_vectorized_rollout(venv: VectorEnv, num_steps: int) -> VectorRolloutFn:
    """JIT vectorized ``rollout`` for one task (``num_steps`` fixed at compile time)."""

    @jax.jit
    def rollout_segment(
        state: mjx_env.State,
        rollout_key: jax.Array,
    ) -> tuple[mjx_env.State, jax.Array]:
        state, traj = rollout(venv, state, batched_random_policy, rollout_key, num_steps)
        return state, traj["reward"]

    return rollout_segment


def build_vectorized_rollouts(
    plan: list[SaturatedTaskPlan],
    *,
    num_steps: int,
) -> list[VectorRolloutFn]:
    return [make_jitted_vectorized_rollout(slot.venv, num_steps) for slot in plan]


def saturated_cl_train(
    plan: list[SaturatedTaskPlan],
    rollout_fns: list[VectorRolloutFn],
    rng: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Run CW sequence with batched env lanes per task."""
    rewards_chunks: list[jax.Array] = []
    for slot, rollout_fn in zip(plan, rollout_fns):
        rng, reset_key, rollout_key = jax.random.split(rng, 3)
        state = slot.venv.reset(reset_key)
        _, task_rewards = rollout_fn(state, rollout_key)
        rewards_chunks.append(task_rewards.reshape(-1))
    return rng, jnp.concatenate(rewards_chunks)


def jitted_vectorized_eval(
    eval_rollouts: list[VectorRolloutFn],
    plan: list[SaturatedTaskPlan],
    rng: jax.Array,
) -> jax.Array:
    """Per-task mean success across vectorized lanes (one rollout each)."""
    rates = []
    for slot, rollout_fn in zip(plan, eval_rollouts):
        rng, reset_key, rollout_key = jax.random.split(rng, 3)
        state = slot.venv.reset(reset_key)
        state, _ = rollout_fn(state, rollout_key)
        success = state.metrics.get("success", jnp.zeros((slot.venv.num_envs,)))
        rates.append(jnp.mean((success >= 1.0).astype(jnp.float32)))
    return jnp.asarray(rates)


def make_jitted_segment_rollout(task_env: CWTaskEnv) -> SegmentRolloutFn:
    """JIT single-lane ``steps`` with functional episode reset on done."""

    @jax.jit
    def rollout_segment(
        state: mjx_env.State,
        step_keys: jax.Array,
    ) -> tuple[mjx_env.State, jax.Array]:
        state = task_env._strip_cw_info(state)

        def body(carry: mjx_env.State, step_key: jax.Array):
            carry = task_env._strip_cw_info(carry)
            act_key, reset_key = jax.random.split(step_key)
            action = random_policy(carry.obs, act_key)
            nxt = task_env.step(carry, action)
            done = (nxt.truncated + nxt.terminated) > 0
            nxt = jax.lax.cond(
                done,
                lambda _: task_env.reset(reset_key),
                lambda s: s,
                nxt,
            )
            nxt = task_env._strip_cw_info(nxt)
            return nxt, nxt.reward

        return jax.lax.scan(body, state, step_keys)

    return rollout_segment


def make_jitted_episode_rollout(task_env: CWTaskEnv) -> SegmentRolloutFn:
    segment = make_jitted_segment_rollout(task_env)

    @jax.jit
    def one_episode(
        state: mjx_env.State,
        step_keys: jax.Array,
    ) -> tuple[mjx_env.State, jax.Array]:
        return segment(state, step_keys)

    return one_episode


def build_segment_rollouts(
    train_env: ContinualLearningEnv,
) -> list[SegmentRolloutFn]:
    return [make_jitted_segment_rollout(te) for te in train_env.task_envs]


def single_lane_cl_train(
    train_env: ContinualLearningEnv,
    rng: jax.Array,
    segment_rollouts: list[SegmentRolloutFn],
    *,
    steps_per_task: int,
) -> tuple[jax.Array, jax.Array]:
    rewards_chunks: list[jax.Array] = []
    for task_env, rollout_fn in zip(train_env.task_envs, segment_rollouts):
        rng, reset_key, keys_key = jax.random.split(rng, 3)
        state = task_env._strip_cw_info(task_env.reset(reset_key))
        step_keys = jax.random.split(keys_key, steps_per_task)
        _, rewards = rollout_fn(state, step_keys)
        rewards_chunks.append(rewards)
    return rng, jnp.concatenate(rewards_chunks)


def jitted_eval_success_rates(
    test_envs: list[CWTaskEnv],
    episode_fns: list[SegmentRolloutFn],
    rng: jax.Array,
    *,
    num_episodes: int,
    horizon: int,
) -> jax.Array:
    rates = []
    for task_env, episode_fn in zip(test_envs, episode_fns):
        ep_successes = []
        for _ in range(num_episodes):
            rng, reset_key, keys_key = jax.random.split(rng, 3)
            state = task_env._strip_cw_info(task_env.reset(reset_key))
            step_keys = jax.random.split(keys_key, horizon)
            state, _ = episode_fn(state, step_keys)
            ep_successes.append(state.metrics.get("success", jnp.array(0.0)) >= 1.0)
        rates.append(jnp.mean(jnp.asarray(ep_successes, dtype=jnp.float32)))
    return jnp.asarray(rates)


def _default_steps_per_task(mode: str) -> int:
    return STEPS_PER_TASK_SATURATED if mode == "saturated" else STEPS_PER_TASK_SINGLE


def run_training(
    *,
    mode: str,
    name: str,
    steps_per_task: int,
    num_envs: int,
) -> None:
    benchmark = CWBenchmark(name, config=CWConfig(seed=SEED, steps_per_task=steps_per_task))
    num_tasks = benchmark.num_tasks
    horizon = benchmark.config.episode_horizon

    if mode == "saturated":
        lane_steps = steps_per_task
        plan = build_saturated_plan(benchmark, num_envs=num_envs)
        rollout_fns = build_vectorized_rollouts(plan, num_steps=lane_steps)
        eval_rollouts = build_vectorized_rollouts(plan, num_steps=horizon)
        total_env_steps = lane_steps * num_envs * num_tasks

        print(f"mode: saturated ({num_envs} vectorized lanes per task)")
        print(
            f"{name}: {num_tasks} tasks, {lane_steps} steps/lane/task "
            f"({total_env_steps:,} total env-steps)"
        )
        print(f"obs dim (with one-hot): {cw_obs_dim(num_tasks)}")
        print(f"episode horizon: {horizon}")

        rng = jax.random.PRNGKey(SEED)
        t0 = time.perf_counter()
        rng, rewards = saturated_cl_train(plan, rollout_fns, rng)
        rewards.block_until_ready()
        warmup_s = time.perf_counter() - t0
        print(f"warmup (compile + run): {warmup_s:.3f}s")

        t0 = time.perf_counter()
        rng, rewards = saturated_cl_train(plan, rollout_fns, rng)
        rewards.block_until_ready()
        elapsed = time.perf_counter() - t0
        final_task = plan[-1].env_name

        rng, eval_key = jax.random.split(rng)
        t0_eval = time.perf_counter()
        success_rates = jitted_vectorized_eval(eval_rollouts, plan, eval_key)
        success_rates.block_until_ready()
        eval_s = time.perf_counter() - t0_eval
        eval_env_steps = horizon * num_envs * num_tasks

    else:
        config = CWConfig(seed=SEED, steps_per_task=steps_per_task)
        train_env = make_cl_train_env(name, config=config)
        test_envs = make_cl_test_envs(name, seed=SEED)
        total_env_steps = steps_per_task * num_tasks

        print("mode: single (1 lane, sequential ContinualLearningEnv)")
        print(f"{name}: {num_tasks} tasks, {steps_per_task} steps/task ({total_env_steps:,} total)")
        print(f"obs dim (with one-hot): {cw_obs_dim(num_tasks)}")
        print(f"episode horizon: {horizon}")

        rng = jax.random.PRNGKey(SEED)
        segment_rollouts = build_segment_rollouts(train_env)
        episode_rollouts = [make_jitted_episode_rollout(env) for env in test_envs]

        t0 = time.perf_counter()
        rng, rewards = single_lane_cl_train(
            train_env, rng, segment_rollouts, steps_per_task=steps_per_task
        )
        rewards.block_until_ready()
        warmup_s = time.perf_counter() - t0
        print(f"warmup (compile + run): {warmup_s:.3f}s")

        t0 = time.perf_counter()
        rng, rewards = single_lane_cl_train(
            train_env, rng, segment_rollouts, steps_per_task=steps_per_task
        )
        rewards.block_until_ready()
        elapsed = time.perf_counter() - t0
        final_task = train_env.task_envs[-1].env_name

        rng, eval_key = jax.random.split(rng)
        t0_eval = time.perf_counter()
        success_rates = jitted_eval_success_rates(
            test_envs,
            episode_rollouts,
            eval_key,
            num_episodes=EVAL_EPISODES,
            horizon=horizon,
        )
        success_rates.block_until_ready()
        eval_s = time.perf_counter() - t0_eval
        eval_env_steps = horizon * EVAL_EPISODES * num_tasks

    throughput = total_env_steps / elapsed
    print(f"training: mean reward = {float(jnp.mean(rewards)):.4f}, final task = {final_task}")
    print(f"jit rollout: {total_env_steps:,} env-steps in {elapsed:.3f}s")
    print(f"throughput: {throughput:,.0f} env-steps/s")

    avg_success = float(jnp.mean(success_rates))
    eval_throughput = eval_env_steps / eval_s
    print(
        f"eval: avg success = {avg_success:.4f} "
        f"({eval_env_steps:,} env-steps in {eval_s:.3f}s, "
        f"{eval_throughput:,.0f} env-steps/s)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Continual World JIT training demo")
    parser.add_argument(
        "--benchmark",
        choices=sorted(TASK_SEQS),
        default="CW10",
        help="Continual World task sequence (default: CW10)",
    )
    parser.add_argument(
        "--mode",
        choices=("saturated", "single"),
        default="saturated",
        help="saturated: vectorized GPU rollout (default); single: 1-lane CL env",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=NUM_ENVS,
        help=f"Vectorized lanes per task in saturated mode (default: {NUM_ENVS})",
    )
    parser.add_argument(
        "--steps-per-task",
        type=int,
        default=None,
        help="Steps per lane per task (default: 200 saturated, 2000 single)",
    )
    args = parser.parse_args()

    steps_per_task = (
        args.steps_per_task
        if args.steps_per_task is not None
        else _default_steps_per_task(args.mode)
    )
    if args.num_envs < 1:
        raise SystemExit("--num-envs must be >= 1")

    run_training(
        mode=args.mode,
        name=args.benchmark,
        steps_per_task=steps_per_task,
        num_envs=args.num_envs,
    )


if __name__ == "__main__":
    main()
