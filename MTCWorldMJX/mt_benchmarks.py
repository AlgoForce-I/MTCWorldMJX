"""MetaWorld-style benchmarks for MTCWorldMJX."""

from __future__ import annotations

import abc
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass, replace
from functools import partial
from typing import Literal, Type

import jax
import jax.numpy as jnp
import numpy as np

from MTCWorldMJX import mjx_env
from MTCWorldMJX.env_dict import ENV_CLS_MAP, make
from MTCWorldMJX.envs.sawyer_xyz import SawyerXYZConfig, SawyerXYZEnv

EnvDict = OrderedDict[str, Type[SawyerXYZEnv]]
TaskSelect = Literal["random", "pseudorandom"]

# MJX Warp shares one collision workspace across vmap lanes; naconmax must scale
# with num_envs (worst MetaWorld tasks need ~32 contact slots per lane at 512 lanes).
_NACONMAX_PER_LANE = 32

_N_GOALS = 50


def vector_env_naconmax(num_envs: int, base: int = 2000) -> int:
    """Contact-array size for batched ``VectorEnv`` rollouts."""
    if num_envs <= 1:
        return base
    return max(base, num_envs * _NACONMAX_PER_LANE)

_MT_OVERRIDE = dict(partially_observable=False)
_ML_OVERRIDE = dict(partially_observable=True)

MT10_ENV_NAMES = (
    "reach-v3",
    "push-v3",
    "pick-place-v3",
    "door-open-v3",
    "drawer-open-v3",
    "drawer-close-v3",
    "button-press-topdown-v3",
    "peg-insert-side-v3",
    "window-open-v3",
    "window-close-v3",
)

MT25_ENV_NAMES = MT10_ENV_NAMES + (
    "coffee-pull-v3",
    "pick-out-of-hole-v3",
    "disassemble-v3",
    "pick-place-wall-v3",
    "basketball-v3",
    "stick-pull-v3",
    "button-press-wall-v3",
    "faucet-open-v3",
    "door-lock-v3",
    "lever-pull-v3",
    "sweep-into-v3",
    "faucet-close-v3",
    "coffee-button-v3",
    "button-press-topdown-wall-v3",
    "dial-turn-v3",
)

ML10_TRAIN_ENV_NAMES = (
    "reach-v3",
    "push-v3",
    "pick-place-v3",
    "door-open-v3",
    "drawer-close-v3",
    "button-press-topdown-v3",
    "peg-insert-side-v3",
    "window-open-v3",
    "sweep-v3",
    "basketball-v3",
)

ML10_TEST_ENV_NAMES = (
    "drawer-open-v3",
    "door-close-v3",
    "shelf-place-v3",
    "sweep-into-v3",
    "lever-pull-v3",
)

ML25_TRAIN_ENV_NAMES = MT25_ENV_NAMES

ML25_TEST_ENV_NAMES = (
    "basketball-v3",
    "door-close-v3",
    "shelf-place-v3",
    "sweep-v3",
    "button-press-v3",
)

ML45_TRAIN_ENV_NAMES = (
    "assembly-v3",
    "basketball-v3",
    "button-press-topdown-v3",
    "button-press-topdown-wall-v3",
    "button-press-v3",
    "button-press-wall-v3",
    "coffee-button-v3",
    "coffee-pull-v3",
    "coffee-push-v3",
    "dial-turn-v3",
    "disassemble-v3",
    "door-close-v3",
    "door-open-v3",
    "drawer-close-v3",
    "drawer-open-v3",
    "faucet-open-v3",
    "faucet-close-v3",
    "hammer-v3",
    "handle-press-side-v3",
    "handle-press-v3",
    "handle-pull-side-v3",
    "handle-pull-v3",
    "lever-pull-v3",
    "pick-place-wall-v3",
    "pick-out-of-hole-v3",
    "push-back-v3",
    "pick-place-v3",
    "plate-slide-v3",
    "plate-slide-side-v3",
    "plate-slide-back-v3",
    "plate-slide-back-side-v3",
    "peg-insert-side-v3",
    "peg-unplug-side-v3",
    "soccer-v3",
    "stick-push-v3",
    "stick-pull-v3",
    "push-wall-v3",
    "push-v3",
    "reach-wall-v3",
    "reach-v3",
    "shelf-place-v3",
    "sweep-into-v3",
    "sweep-v3",
    "window-open-v3",
    "window-close-v3",
)

ML45_TEST_ENV_NAMES = (
    "bin-picking-v3",
    "box-close-v3",
    "hand-insert-v3",
    "door-lock-v3",
    "door-unlock-v3",
)


@dataclass(frozen=True)
class Task:
    """A single MDP instance: env name, frozen reset vector, and observation mode."""

    env_name: str
    rand_vec: np.ndarray
    partially_observable: bool


class Benchmark(abc.ABC):
    """Collection of environment classes and pre-generated tasks."""

    _train_classes: EnvDict
    _test_classes: EnvDict
    _train_tasks: list[Task]
    _test_tasks: list[Task]

    @property
    def train_classes(self) -> EnvDict:
        return self._train_classes

    @property
    def test_classes(self) -> EnvDict:
        return self._test_classes

    @property
    def train_tasks(self) -> list[Task]:
        return self._train_tasks

    @property
    def test_tasks(self) -> list[Task]:
        return self._test_tasks


def _env_dict(env_names: Sequence[str]) -> EnvDict:
    return OrderedDict((name, ENV_CLS_MAP[name]) for name in env_names)


def _config_for_task(
    task: Task,
    config: SawyerXYZConfig | None = None,
    **config_overrides,
) -> SawyerXYZConfig:
    if config is None:
        return SawyerXYZConfig(
            partially_observable=task.partially_observable,
            **config_overrides,
        )
    return replace(
        config,
        partially_observable=task.partially_observable,
        **config_overrides,
    )


def make_env_for_task(
    task: Task,
    config: SawyerXYZConfig | None = None,
    **config_overrides,
) -> SawyerXYZEnv:
    """Construct an environment configured for a benchmark task."""
    return make(task.env_name, config=_config_for_task(task, config, **config_overrides))


def reset_for_task(
    env: SawyerXYZEnv,
    rng: jax.Array,
    task: Task,
) -> mjx_env.State:
    """Reset ``env`` with the task's frozen ``rand_vec``."""
    return env.reset(rng, rand_vec=jnp.asarray(task.rand_vec))


def tasks_for_env(tasks: Sequence[Task], env_name: str) -> list[Task]:
    """Filter a task list to one environment name."""
    return [task for task in tasks if task.env_name == env_name]


def _make_tasks(
    env_names: Sequence[str],
    *,
    partially_observable: bool,
    seed: int | None = None,
    n_goals: int = _N_GOALS,
    config: SawyerXYZConfig | None = None,
) -> list[Task]:
    """Sample ``n_goals`` unique reset vectors per environment."""
    rng = jax.random.PRNGKey(0 if seed is None else seed)
    base_config = replace(
        config or SawyerXYZConfig(),
        partially_observable=partially_observable,
    )

    tasks: list[Task] = []
    for env_name in env_names:
        env = make(env_name, config=base_config)

        def _sample_rand_vec(key, env=env):
            """Reproduce ``reset()``'s ``rand_vec`` without running physics."""
            _, reset_rng, vec_rng = jax.random.split(key, 3)
            reset_state = env.sample_reset_state(reset_rng)
            return env._sample_rand_vec(vec_rng, reset_state.get("rand_vec"))

        rng, batch_rng = jax.random.split(rng)
        keys = jax.random.split(batch_rng, n_goals)
        rand_vecs = np.asarray(
            jax.jit(jax.vmap(_sample_rand_vec))(keys), dtype=np.float64
        )

        unique_task_rand_vecs = np.unique(rand_vecs, axis=0)
        if unique_task_rand_vecs.shape[0] != n_goals:
            raise ValueError(
                f"Only generated {unique_task_rand_vecs.shape[0]} unique goals for "
                f"{env_name}, expected {n_goals}"
            )

        for rand_vec in rand_vecs:
            tasks.append(
                Task(
                    env_name=env_name,
                    rand_vec=rand_vec,
                    partially_observable=partially_observable,
                )
            )
    return tasks


def rand_vecs_for_env(tasks: Sequence[Task], env_name: str) -> jax.Array:
    """Stack the reset vectors of all tasks belonging to ``env_name``."""
    vecs = [task.rand_vec for task in tasks if task.env_name == env_name]
    if not vecs:
        raise ValueError(f"No tasks found for environment '{env_name}'")
    return jnp.asarray(np.stack(vecs), dtype=jnp.float32)


class VectorEnv:
    """Batched, JAX-native environment over a pool of tasks (one MuJoCo model).

    The model is loaded once. ``reset`` assigns one task (``rand_vec``) per lane
    and ``step`` is vmapped across lanes with functional, on-device autoreset:
    when a lane is done it is restored to its episode's initial state. With
    ``rand_vecs=None`` every reset samples a fresh ``rand_vec`` per lane instead
    (Continual World's ``random_init_all``); ``task_idx`` is then ``-1``.
    """

    def __init__(
        self,
        env_name: str,
        rand_vecs: jax.Array | None,
        num_envs: int,
        config: SawyerXYZConfig | None = None,
        *,
        partially_observable: bool = False,
        task_select: TaskSelect = "random",
        terminate_on_success: bool = False,
        autoreset: bool = True,
        seed: int | None = None,
        **config_overrides,
    ):
        if num_envs < 1:
            raise ValueError("num_envs must be >= 1")
        self.env_name = env_name
        self.num_envs = num_envs
        self.partially_observable = partially_observable
        self.task_select = task_select
        self.terminate_on_success = terminate_on_success
        self.autoreset = autoreset

        self.rand_vecs = None if rand_vecs is None else jnp.asarray(rand_vecs, dtype=jnp.float32)
        self.n_tasks = 0 if self.rand_vecs is None else int(self.rand_vecs.shape[0])

        self._config = (
            replace(config, partially_observable=partially_observable, **config_overrides)
            if config is not None
            else SawyerXYZConfig(partially_observable=partially_observable, **config_overrides)
        )
        scaled_naconmax = vector_env_naconmax(num_envs, self._config.naconmax)
        if scaled_naconmax != self._config.naconmax:
            self._config = replace(self._config, naconmax=scaled_naconmax)
        self.env = make(env_name, config=self._config)

        # Static task ordering for pseudorandom (cycling) selection.
        order = np.arange(self.n_tasks)
        if seed is not None:
            np.random.default_rng(seed).shuffle(order)
        self._task_order = jnp.asarray(order)

        self._reset_jit = jax.jit(self._reset_impl)
        self._step_jit = jax.jit(self._step_impl)

    def _select_task_indices(self, key: jax.Array, start: jax.Array) -> jax.Array:
        if self.task_select == "pseudorandom":
            idx = (start + jnp.arange(self.num_envs)) % self.n_tasks
            return self._task_order[idx]
        return jax.random.randint(key, (self.num_envs,), 0, self.n_tasks)

    def _reset_impl(self, key: jax.Array, start: jax.Array) -> mjx_env.State:
        key, idx_key = jax.random.split(key)
        keys = jax.random.split(key, self.num_envs)
        if self.rand_vecs is None:
            task_idx = jnp.full((self.num_envs,), -1, dtype=jnp.int32)
            state = jax.vmap(self.env.reset)(keys)
        else:
            task_idx = self._select_task_indices(idx_key, start)
            rvs = self.rand_vecs[task_idx]
            state = jax.vmap(lambda k, rv: self.env.reset(k, rand_vec=rv))(keys, rvs)

        info = dict(state.info)
        info["first_data"] = state.data
        info["first_obs"] = state.obs
        info["first_prev_obs"] = state.info["prev_obs"]
        info["task_idx"] = task_idx
        return state.replace(info=info)

    def _maybe_terminate_on_success(self, state: mjx_env.State) -> mjx_env.State:
        if not self.terminate_on_success:
            return state
        success = state.metrics.get("success", jnp.zeros((self.num_envs,)))
        terminated = jnp.where(success >= 1.0, 1.0, 0.0).astype(jnp.float32)
        return state.replace(terminated=jnp.maximum(state.terminated, terminated))

    def _step_impl(self, state: mjx_env.State, actions: jax.Array) -> mjx_env.State:
        nxt = jax.vmap(self.env.step)(state, actions)
        nxt = self._maybe_terminate_on_success(nxt)
        if not self.autoreset:
            return nxt

        done = (nxt.truncated + nxt.terminated) > 0

        def where_done(initial, current):
            # Some MJX (warp backend) Data leaves are a global contact workspace
            # shared across lanes (no leading num_envs axis). Those are transient
            # and recomputed by the next step's forward pass, so leave them as-is
            # and only restore per-env (batched) state for done lanes.
            if jnp.ndim(current) == 0 or current.shape[0] != self.num_envs:
                return current
            d = done.reshape((self.num_envs,) + (1,) * (jnp.ndim(current) - 1))
            return jnp.where(d, initial, current)

        data = jax.tree_util.tree_map(where_done, nxt.info["first_data"], nxt.data)
        obs = where_done(nxt.info["first_obs"], nxt.obs)
        info = dict(nxt.info)
        info["prev_obs"] = where_done(nxt.info["first_prev_obs"], nxt.info["prev_obs"])
        info["path_length"] = jnp.where(done, 0, nxt.info["path_length"])
        return nxt.replace(data=data, obs=obs, info=info)

    def reset(self, key: jax.Array, start: int = 0) -> mjx_env.State:
        """Reset all lanes; ``obs`` has shape ``(num_envs, obs_dim)``."""
        return self._reset_jit(key, jnp.asarray(start, dtype=jnp.int32))

    def step(self, state: mjx_env.State, actions: jax.Array) -> mjx_env.State:
        """Vmapped step with functional autoreset for done lanes."""
        return self._step_jit(state, actions)


def rollout(
    venv: VectorEnv,
    state: mjx_env.State,
    policy,
    key: jax.Array,
    num_steps: int,
) -> tuple[mjx_env.State, dict[str, jax.Array]]:
    """Scan ``num_steps`` of ``venv`` on-device.

    ``policy(obs, key) -> actions`` of shape ``(num_envs, action_dim)``.
    Returns the final state and stacked per-step ``reward``/``done``.
    """
    def body(carry, step_key):
        state = carry
        actions = policy(state.obs, step_key)
        state = venv._step_impl(state, actions)
        done = (state.truncated + state.terminated) > 0
        return state, {"reward": state.reward, "done": done}

    keys = jax.random.split(key, num_steps)
    return jax.lax.scan(body, state, keys)


class MT1(Benchmark):
    """Single-environment multi-task benchmark with observable goals."""

    ENV_NAMES = tuple(ENV_CLS_MAP.keys())

    def __init__(
        self,
        env_name: str,
        seed: int | None = None,
        *,
        n_goals: int = _N_GOALS,
        config: SawyerXYZConfig | None = None,
    ):
        if env_name not in ENV_CLS_MAP:
            raise ValueError(f"{env_name} is not a V3 environment")
        env_dict = _env_dict([env_name])
        self._train_classes = env_dict
        self._test_classes = env_dict
        self._train_tasks = _make_tasks(
            [env_name],
            partially_observable=_MT_OVERRIDE["partially_observable"],
            seed=seed,
            n_goals=n_goals,
            config=config,
        )
        self._test_tasks = []


class MT10(Benchmark):
    """Ten-task multi-task benchmark with observable goals."""

    def __init__(
        self,
        seed: int | None = None,
        *,
        n_goals: int = _N_GOALS,
        config: SawyerXYZConfig | None = None,
    ):
        self._train_classes = _env_dict(MT10_ENV_NAMES)
        self._test_classes = OrderedDict()
        self._train_tasks = _make_tasks(
            MT10_ENV_NAMES,
            partially_observable=_MT_OVERRIDE["partially_observable"],
            seed=seed,
            n_goals=n_goals,
            config=config,
        )
        self._test_tasks = []


class MT25(Benchmark):
    """Twenty-five-task multi-task benchmark with observable goals."""

    def __init__(
        self,
        seed: int | None = None,
        *,
        n_goals: int = _N_GOALS,
        config: SawyerXYZConfig | None = None,
    ):
        self._train_classes = _env_dict(MT25_ENV_NAMES)
        self._test_classes = OrderedDict()
        self._train_tasks = _make_tasks(
            MT25_ENV_NAMES,
            partially_observable=_MT_OVERRIDE["partially_observable"],
            seed=seed,
            n_goals=n_goals,
            config=config,
        )
        self._test_tasks = []


class MT50(Benchmark):
    """Full fifty-task multi-task benchmark with observable goals."""

    def __init__(
        self,
        seed: int | None = None,
        *,
        n_goals: int = _N_GOALS,
        config: SawyerXYZConfig | None = None,
    ):
        env_names = tuple(ENV_CLS_MAP.keys())
        self._train_classes = _env_dict(env_names)
        self._test_classes = OrderedDict()
        self._train_tasks = _make_tasks(
            env_names,
            partially_observable=_MT_OVERRIDE["partially_observable"],
            seed=seed,
            n_goals=n_goals,
            config=config,
        )
        self._test_tasks = []


class ML1(Benchmark):
    """Single-environment meta-learning benchmark with hidden goals."""

    ENV_NAMES = tuple(ENV_CLS_MAP.keys())

    def __init__(
        self,
        env_name: str,
        seed: int | None = None,
        *,
        n_goals: int = _N_GOALS,
        config: SawyerXYZConfig | None = None,
    ):
        if env_name not in ENV_CLS_MAP:
            raise ValueError(f"{env_name} is not a V3 environment")
        env_dict = _env_dict([env_name])
        self._train_classes = env_dict
        self._test_classes = env_dict
        self._train_tasks = _make_tasks(
            [env_name],
            partially_observable=_ML_OVERRIDE["partially_observable"],
            seed=seed,
            n_goals=n_goals,
            config=config,
        )
        test_seed = None if seed is None else seed + 1
        self._test_tasks = _make_tasks(
            [env_name],
            partially_observable=_ML_OVERRIDE["partially_observable"],
            seed=test_seed,
            n_goals=n_goals,
            config=config,
        )


class ML10(Benchmark):
    """Ten train / five test meta-learning benchmark."""

    def __init__(
        self,
        seed: int | None = None,
        *,
        n_goals: int = _N_GOALS,
        config: SawyerXYZConfig | None = None,
    ):
        self._train_classes = _env_dict(ML10_TRAIN_ENV_NAMES)
        self._test_classes = _env_dict(ML10_TEST_ENV_NAMES)
        self._train_tasks = _make_tasks(
            ML10_TRAIN_ENV_NAMES,
            partially_observable=_ML_OVERRIDE["partially_observable"],
            seed=seed,
            n_goals=n_goals,
            config=config,
        )
        self._test_tasks = _make_tasks(
            ML10_TEST_ENV_NAMES,
            partially_observable=_ML_OVERRIDE["partially_observable"],
            seed=seed,
            n_goals=n_goals,
            config=config,
        )


class ML25(Benchmark):
    """Twenty-five train / five test meta-learning benchmark."""

    def __init__(
        self,
        seed: int | None = None,
        *,
        n_goals: int = _N_GOALS,
        config: SawyerXYZConfig | None = None,
    ):
        self._train_classes = _env_dict(ML25_TRAIN_ENV_NAMES)
        self._test_classes = _env_dict(ML25_TEST_ENV_NAMES)
        self._train_tasks = _make_tasks(
            ML25_TRAIN_ENV_NAMES,
            partially_observable=_ML_OVERRIDE["partially_observable"],
            seed=seed,
            n_goals=n_goals,
            config=config,
        )
        self._test_tasks = _make_tasks(
            ML25_TEST_ENV_NAMES,
            partially_observable=_ML_OVERRIDE["partially_observable"],
            seed=seed,
            n_goals=n_goals,
            config=config,
        )


class ML45(Benchmark):
    """Forty-five train / five test meta-learning benchmark."""

    def __init__(
        self,
        seed: int | None = None,
        *,
        n_goals: int = _N_GOALS,
        config: SawyerXYZConfig | None = None,
    ):
        self._train_classes = _env_dict(ML45_TRAIN_ENV_NAMES)
        self._test_classes = _env_dict(ML45_TEST_ENV_NAMES)
        self._train_tasks = _make_tasks(
            ML45_TRAIN_ENV_NAMES,
            partially_observable=_ML_OVERRIDE["partially_observable"],
            seed=seed,
            n_goals=n_goals,
            config=config,
        )
        self._test_tasks = _make_tasks(
            ML45_TEST_ENV_NAMES,
            partially_observable=_ML_OVERRIDE["partially_observable"],
            seed=seed,
            n_goals=n_goals,
            config=config,
        )


class CustomML(Benchmark):
    """Custom train/test meta-learning split."""

    def __init__(
        self,
        train_envs: Sequence[str],
        test_envs: Sequence[str],
        seed: int | None = None,
        *,
        n_goals: int = _N_GOALS,
        config: SawyerXYZConfig | None = None,
    ):
        if set(train_envs).intersection(test_envs):
            raise ValueError("The test tasks cannot contain any of the train tasks.")
        self._train_classes = _env_dict(train_envs)
        self._test_classes = _env_dict(test_envs)
        self._train_tasks = _make_tasks(
            train_envs,
            partially_observable=_ML_OVERRIDE["partially_observable"],
            seed=seed,
            n_goals=n_goals,
            config=config,
        )
        self._test_tasks = _make_tasks(
            test_envs,
            partially_observable=_ML_OVERRIDE["partially_observable"],
            seed=seed,
            n_goals=n_goals,
            config=config,
        )


def _vector_envs_from_tasks(
    classes: EnvDict,
    tasks: Sequence[Task],
    *,
    num_envs: int,
    partially_observable: bool,
    config: SawyerXYZConfig | None,
    task_select: TaskSelect,
    terminate_on_success: bool,
    autoreset: bool,
    seed: int | None,
    **config_overrides,
) -> OrderedDict[str, VectorEnv]:
    envs: OrderedDict[str, VectorEnv] = OrderedDict()
    for env_name in classes:
        envs[env_name] = VectorEnv(
            env_name,
            rand_vecs_for_env(tasks, env_name),
            num_envs,
            config=config,
            partially_observable=partially_observable,
            task_select=task_select,
            terminate_on_success=terminate_on_success,
            autoreset=autoreset,
            seed=seed,
            **config_overrides,
        )
    return envs


def make_mt_envs(
    name: str,
    seed: int | None = None,
    *,
    num_envs: int = 1,
    config: SawyerXYZConfig | None = None,
    task_select: TaskSelect = "random",
    terminate_on_success: bool = False,
    autoreset: bool = True,
    **config_overrides,
) -> VectorEnv | OrderedDict[str, VectorEnv]:
    """Build MT benchmark environments (observable goals).

    For a single task name (e.g. ``"reach-v3"``), returns one batched
    ``VectorEnv`` with ``num_envs`` lanes. For ``"MT10"``/``"MT25"``/``"MT50"``,
    returns an ``OrderedDict`` mapping each task name to its ``VectorEnv``.
    """
    po = _MT_OVERRIDE["partially_observable"]
    if name in ENV_CLS_MAP:
        benchmark = MT1(name, seed=seed, config=config)
        return VectorEnv(
            name,
            rand_vecs_for_env(benchmark.train_tasks, name),
            num_envs,
            config=config,
            partially_observable=po,
            task_select=task_select,
            terminate_on_success=terminate_on_success,
            autoreset=autoreset,
            seed=seed,
            **config_overrides,
        )
    if name == "MT10":
        benchmark = MT10(seed=seed, config=config)
    elif name == "MT25":
        benchmark = MT25(seed=seed, config=config)
    elif name == "MT50":
        benchmark = MT50(seed=seed, config=config)
    else:
        raise ValueError(
            "Invalid MT env name. Must be a V3 task name, 'MT10', 'MT25', or 'MT50'."
        )
    return _vector_envs_from_tasks(
        benchmark.train_classes,
        benchmark.train_tasks,
        num_envs=num_envs,
        partially_observable=po,
        config=config,
        task_select=task_select,
        terminate_on_success=terminate_on_success,
        autoreset=autoreset,
        seed=seed,
        **config_overrides,
    )


def make_ml_envs(
    name: str,
    seed: int | None = None,
    *,
    split: Literal["train", "test"] = "train",
    num_envs: int = 1,
    config: SawyerXYZConfig | None = None,
    task_select: TaskSelect = "random",
    terminate_on_success: bool = False,
    autoreset: bool = True,
    **config_overrides,
) -> VectorEnv | OrderedDict[str, VectorEnv]:
    """Build ML benchmark environments (hidden goals) for a train/test split."""
    po = _ML_OVERRIDE["partially_observable"]
    if name in ENV_CLS_MAP:
        benchmark = ML1(name, seed=seed, config=config)
    elif name == "ML10":
        benchmark = ML10(seed=seed, config=config)
    elif name == "ML25":
        benchmark = ML25(seed=seed, config=config)
    elif name == "ML45":
        benchmark = ML45(seed=seed, config=config)
    else:
        raise ValueError(
            "Invalid ML env name. Must be a V3 task name, 'ML10', 'ML25', or 'ML45'."
        )

    tasks = benchmark.train_tasks if split == "train" else benchmark.test_tasks
    if name in ENV_CLS_MAP:
        return VectorEnv(
            name,
            rand_vecs_for_env(tasks, name),
            num_envs,
            config=config,
            partially_observable=po,
            task_select=task_select,
            terminate_on_success=terminate_on_success,
            autoreset=autoreset,
            seed=seed,
            **config_overrides,
        )

    classes = benchmark.train_classes if split == "train" else benchmark.test_classes
    return _vector_envs_from_tasks(
        classes,
        tasks,
        num_envs=num_envs,
        partially_observable=po,
        config=config,
        task_select=task_select,
        terminate_on_success=terminate_on_success,
        autoreset=autoreset,
        seed=seed,
        **config_overrides,
    )


make_ml_envs_train = partial(
    make_ml_envs,
    terminate_on_success=False,
    task_select="pseudorandom",
    split="train",
)
make_ml_envs_test = partial(
    make_ml_envs,
    terminate_on_success=True,
    task_select="pseudorandom",
    split="test",
)

__all__ = [
    "Benchmark",
    "CustomML",
    "ML1",
    "ML10",
    "ML25",
    "ML45",
    "MT1",
    "MT10",
    "MT25",
    "MT50",
    "Task",
    "VectorEnv",
    "make_env_for_task",
    "make_ml_envs",
    "make_ml_envs_test",
    "make_ml_envs_train",
    "make_mt_envs",
    "rand_vecs_for_env",
    "reset_for_task",
    "rollout",
    "tasks_for_env",
]
