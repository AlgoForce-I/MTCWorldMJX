"""Continual World environment orchestration on top of MJX Sawyer envs."""

from __future__ import annotations

from typing import Literal

import jax
import jax.numpy as jnp

from MTCWorldMJX import mjx_env
from MTCWorldMJX.env_dict import ENV_CLS_MAP, make
from MTCWorldMJX.envs.sawyer_xyz import OBS_DIM, SawyerXYZConfig, SawyerXYZEnv
from MTCWorldMJX.mt_benchmarks import Task

RandomizationKind = Literal[
    "deterministic",
    "random_init_all",
    "random_init_fixed20",
    "random_init_small_box",
]

ALLOWED_RANDOMIZATION: tuple[RandomizationKind, ...] = (
    "deterministic",
    "random_init_all",
    "random_init_fixed20",
    "random_init_small_box",
)


def append_task_one_hot(obs: jax.Array, task_idx: int, num_tasks: int) -> jax.Array:
    """Append a task-index one-hot to observations (Continual World layout)."""
    one_hot = jnp.zeros(num_tasks, dtype=obs.dtype)
    one_hot = one_hot.at[task_idx].set(1.0)
    return jnp.concatenate([obs, one_hot])


def append_task_one_hot_batched(
    obs: jax.Array, task_idx: int, num_tasks: int
) -> jax.Array:
    """Batched observations ``(num_envs, obs_dim)`` with task one-hot appended."""
    one_hot = jnp.zeros((obs.shape[0], num_tasks), dtype=obs.dtype)
    one_hot = one_hot.at[:, task_idx].set(1.0)
    return jnp.concatenate([obs, one_hot], axis=-1)


def cw_obs_dim(num_tasks: int) -> int:
    """Observation dimension with Continual World task one-hot appended."""
    return OBS_DIM + num_tasks


def _sample_rand_vec(
    rng: jax.Array,
    env: SawyerXYZEnv,
    tasks: list[Task],
    kind: RandomizationKind,
) -> jax.Array | None:
    """Return a ``rand_vec`` for ``reset``, or ``None`` to sample freely in ``reset``."""
    if kind == "random_init_all":
        return None
    if kind == "deterministic":
        return jnp.asarray(tasks[0].rand_vec, dtype=jnp.float32)
    if kind == "random_init_fixed20":
        pool = min(20, len(tasks))
        idx = jax.random.randint(rng, (), 0, pool)
        return jnp.asarray(tasks[int(idx)].rand_vec, dtype=jnp.float32)
    low, high = env.random_reset_bounds()
    mid_low = low + 0.45 * (high - low)
    mid_high = low + 0.55 * (high - low)
    return jax.random.uniform(rng, shape=low.shape, minval=mid_low, maxval=mid_high)


class CWTaskEnv:
    """Single Continual World task with optional goal pool and one-hot observations."""

    def __init__(
        self,
        env_name: str,
        task_idx: int,
        num_tasks: int,
        tasks: tuple[Task, ...],
        *,
        sawyer_config: SawyerXYZConfig,
        randomization: RandomizationKind,
    ):
        if env_name not in ENV_CLS_MAP:
            raise ValueError(f"Unknown environment '{env_name}'")
        if randomization not in ALLOWED_RANDOMIZATION:
            raise ValueError(f"Unknown randomization '{randomization}'")
        env_tasks = [t for t in tasks if t.env_name == env_name]
        if not env_tasks:
            raise ValueError(f"No goals found for environment '{env_name}'")
        if randomization == "random_init_fixed20" and len(env_tasks) < 20:
            raise ValueError(
                f"random_init_fixed20 requires at least 20 goals for {env_name}, "
                f"got {len(env_tasks)}"
            )

        self.env_name = env_name
        self.task_idx = task_idx
        self.num_tasks = num_tasks
        self.tasks = tuple(env_tasks)
        self.randomization = randomization
        self.env = make(env_name, config=sawyer_config)
        self._jit_reset = jax.jit(self._reset_impl)
        self._jit_step = jax.jit(self.env.step)

    @classmethod
    def create(
        cls,
        env_name: str,
        task_idx: int,
        num_tasks: int,
        tasks: list[Task],
        *,
        sawyer_config: SawyerXYZConfig,
        randomization: RandomizationKind,
    ) -> CWTaskEnv:
        return cls(
            env_name,
            task_idx,
            num_tasks,
            tuple(tasks),
            sawyer_config=sawyer_config,
            randomization=randomization,
        )

    def _reset_impl(self, rng: jax.Array, rand_vec: jax.Array | None) -> mjx_env.State:
        if rand_vec is None:
            return self.env.reset(rng)
        return self.env.reset(rng, rand_vec=rand_vec)

    def _strip_cw_info(self, state: mjx_env.State) -> mjx_env.State:
        drop = {"global_step", "forced_task_change", "task_changed"}
        info = {k: v for k, v in state.info.items() if k not in drop}
        return state.replace(info=info)

    def _with_one_hot(self, state: mjx_env.State) -> mjx_env.State:
        obs = append_task_one_hot(state.obs, self.task_idx, self.num_tasks)
        info = dict(state.info)
        info["task_idx"] = jnp.asarray(self.task_idx, dtype=jnp.int32)
        return state.replace(obs=obs, info=info)

    def reset(self, rng: jax.Array) -> mjx_env.State:
        rng, vec_rng = jax.random.split(rng)
        rand_vec = _sample_rand_vec(vec_rng, self.env, list(self.tasks), self.randomization)
        state = self._jit_reset(rng, rand_vec)
        return self._with_one_hot(state)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        nxt = self._jit_step(self._strip_cw_info(state), action)
        return self._with_one_hot(nxt)


class ContinualLearningEnv:
    """Sequential Continual World training environment.

    Mirrors ``continualworld.envs.ContinualLearningEnv``: one task at a time,
    ``steps_per_task`` environment steps per task, episode horizon from
    ``max_path_length`` on the underlying Sawyer config (default 200).
    """

    def __init__(self, task_envs: tuple[CWTaskEnv, ...], steps_per_task: int):
        if steps_per_task < 1:
            raise ValueError("steps_per_task must be >= 1")
        self.task_envs = task_envs
        self.steps_per_task = steps_per_task
        self.num_tasks = len(task_envs)
        self.steps_limit = self.num_tasks * steps_per_task
        self.name = "ContinualLearningEnv"

    def reset(self, rng: jax.Array) -> mjx_env.State:
        state = self.task_envs[0].reset(rng)
        info = dict(state.info)
        info["seq_idx"] = jnp.asarray(0, dtype=jnp.int32)
        info["global_step"] = jnp.asarray(0, dtype=jnp.int32)
        return state.replace(info=info)

    def reset_from_state(self, rng: jax.Array, state: mjx_env.State) -> mjx_env.State:
        """Reset the active task after an episode ends (same ``seq_idx``)."""
        seq_idx = int(state.info["seq_idx"])
        if int(state.info["global_step"]) >= self.steps_limit:
            raise RuntimeError("Steps limit exceeded for ContinualLearningEnv")
        new_state = self.task_envs[seq_idx].reset(rng)
        info = dict(new_state.info)
        info["seq_idx"] = jnp.asarray(seq_idx, dtype=jnp.int32)
        info["global_step"] = state.info["global_step"]
        return new_state.replace(info=info)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        global_step = int(state.info["global_step"]) + 1
        if global_step > self.steps_limit:
            raise RuntimeError("Steps limit exceeded for ContinualLearningEnv")

        seq_idx = int(state.info["seq_idx"])
        nxt = self.task_envs[seq_idx].step(state, action)

        done = bool((nxt.truncated + nxt.terminated) > 0)
        forced_task_change = global_step % self.steps_per_task == 0
        if forced_task_change:
            done = True
            if seq_idx < self.num_tasks - 1:
                seq_idx += 1

        info = dict(nxt.info)
        info["seq_idx"] = jnp.asarray(seq_idx, dtype=jnp.int32)
        info["global_step"] = jnp.asarray(global_step, dtype=jnp.int32)
        info["forced_task_change"] = jnp.asarray(forced_task_change, dtype=jnp.bool_)
        info["task_changed"] = jnp.asarray(
            forced_task_change and seq_idx > int(state.info["seq_idx"]),
            dtype=jnp.bool_,
        )

        truncated = jnp.asarray(float(done), dtype=jnp.float32)
        return nxt.replace(truncated=truncated, info=info)


def run_episode(
    env: CWTaskEnv | ContinualLearningEnv,
    policy,
    rng: jax.Array,
    *,
    initial_state: mjx_env.State | None = None,
    max_steps: int | None = None,
) -> tuple[mjx_env.State, jax.Array, bool]:
    """Run one episode; ``policy(obs, key) -> action``.

    Returns final state, episode return, and episode-level success.
    """
    if initial_state is None:
        state = env.reset(rng)
    else:
        state = initial_state

    ep_return = jnp.array(0.0)
    ep_success = False
    horizon = max_steps
    if horizon is None and isinstance(env, CWTaskEnv):
        horizon = env.env.config.max_path_length
    if horizon is None and isinstance(env, ContinualLearningEnv):
        horizon = env.task_envs[0].env.config.max_path_length

    for _ in range(int(horizon)):
        rng, key = jax.random.split(rng)
        action = policy(state.obs, key)
        state = env.step(state, action)
        ep_return = ep_return + state.reward
        if float(state.metrics.get("success", 0.0)) >= 1.0:
            ep_success = True
        if float(state.truncated + state.terminated) > 0:
            break

    return state, ep_return, ep_success
