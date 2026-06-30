"""Minimal MT50 usage example.

Run:

    .venv/bin/python metaworld_example.py

Builds all fifty MetaWorld v3 tasks as batched ``VectorEnv`` instances (observable
goals), resets one task, and runs a short random rollout.
"""

from __future__ import annotations

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp

from MTCWorldMJX import MT50, make_mt_envs, rollout

NUM_ENVS = 8
ROLLOUT_STEPS = 20
SEED = 0
DEMO_TASK = "reach-v3"


def main() -> None:
    benchmark = MT50(seed=SEED)
    print(f"MT50: {len(benchmark.train_classes)} tasks, {len(benchmark.train_tasks)} goals")

    envs = make_mt_envs("MT50", seed=SEED, num_envs=NUM_ENVS)
    print(f"vectorized envs: {len(envs)} tasks x {NUM_ENVS} lanes each")

    venv = envs[DEMO_TASK]
    rng = jax.random.PRNGKey(SEED)
    rng, reset_key, rollout_key = jax.random.split(rng, 3)

    state = venv.reset(reset_key)
    state.obs.block_until_ready()
    print(f"{DEMO_TASK} obs shape: {state.obs.shape}")

    def random_policy(obs, key):
        return jax.random.uniform(key, (obs.shape[0], 4), minval=-1.0, maxval=1.0)

    state, traj = rollout(venv, state, random_policy, rollout_key, ROLLOUT_STEPS)
    traj["reward"].block_until_ready()

    mean_reward = float(jnp.mean(traj["reward"]))
    print(f"{DEMO_TASK} rollout ({ROLLOUT_STEPS} steps): mean reward = {mean_reward:.4f}")


if __name__ == "__main__":
    main()
