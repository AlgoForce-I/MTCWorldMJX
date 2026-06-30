"""Continual World benchmark definitions and factories."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

from MTCWorldMJX.cw_env import (
    ALLOWED_RANDOMIZATION,
    CWTaskEnv,
    ContinualLearningEnv,
    RandomizationKind,
    cw_obs_dim,
)
from MTCWorldMJX.env_dict import ENV_CLS_MAP
from MTCWorldMJX.envs.sawyer_xyz import SawyerXYZConfig
from MTCWorldMJX.mt_benchmarks import Task, _make_tasks, tasks_for_env

TASK_SEQS: dict[str, list[str]] = {
    "CW10": [
        "hammer-v3",
        "push-wall-v3",
        "faucet-close-v3",
        "push-back-v3",
        "stick-pull-v3",
        "handle-press-side-v3",
        "push-v3",
        "shelf-place-v3",
        "window-close-v3",
        "peg-unplug-side-v3",
    ],
}

TASK_SEQS["CW20"] = TASK_SEQS["CW10"] + TASK_SEQS["CW10"]

CW_DEFAULT_SEED = 1
CW_N_GOALS = 50
CW_EPISODE_HORIZON = 200
CW_STEPS_PER_TASK = 1_000_000


@dataclass(frozen=True)
class CWConfig:
    """Continual World protocol settings aligned with the reference benchmark."""

    steps_per_task: int = CW_STEPS_PER_TASK
    episode_horizon: int = CW_EPISODE_HORIZON
    randomization: RandomizationKind = "random_init_all"
    n_goals: int = CW_N_GOALS
    seed: int = CW_DEFAULT_SEED
    partially_observable: bool = False


def cw_sawyer_config(config: CWConfig | None = None) -> SawyerXYZConfig:
    """Sawyer config for Continual World (visible goals, 200-step episodes)."""
    config = config or CWConfig()
    return SawyerXYZConfig(
        max_path_length=config.episode_horizon,
        partially_observable=config.partially_observable,
    )


def _validate_task_names(task_names: Sequence[str]) -> None:
    unknown = [name for name in task_names if name not in ENV_CLS_MAP]
    if unknown:
        raise ValueError(f"Unknown Continual World task(s): {unknown}")


class CWBenchmark:
    """Continual World task sequence with pre-generated goal pools per environment."""

    def __init__(
        self,
        name: str,
        *,
        config: CWConfig | None = None,
        seed: int | None = None,
    ):
        if name not in TASK_SEQS:
            known = ", ".join(sorted(TASK_SEQS))
            raise ValueError(f"Unknown Continual World benchmark '{name}'. Known: {known}")
        self.name = name
        self.config = config or CWConfig()
        if seed is not None:
            self.config = replace(self.config, seed=seed)

        self.task_names = list(TASK_SEQS[name])
        _validate_task_names(self.task_names)

        unique_names = list(dict.fromkeys(self.task_names))
        sawyer_config = cw_sawyer_config(self.config)
        self._tasks = _make_tasks(
            unique_names,
            partially_observable=self.config.partially_observable,
            seed=self.config.seed,
            n_goals=self.config.n_goals,
            config=sawyer_config,
        )

    @property
    def num_tasks(self) -> int:
        return len(self.task_names)

    @property
    def tasks(self) -> list[Task]:
        return list(self._tasks)

    def goals_for(self, env_name: str) -> list[Task]:
        return tasks_for_env(self._tasks, env_name)


class CW10(CWBenchmark):
    def __init__(self, *, config: CWConfig | None = None, seed: int | None = None):
        super().__init__("CW10", config=config, seed=seed)


class CW20(CWBenchmark):
    def __init__(self, *, config: CWConfig | None = None, seed: int | None = None):
        super().__init__("CW20", config=config, seed=seed)


def _task_envs_for_benchmark(
    benchmark: CWBenchmark,
    *,
    randomization: RandomizationKind | None = None,
) -> tuple[CWTaskEnv, ...]:
    kind = randomization if randomization is not None else benchmark.config.randomization
    sawyer_config = cw_sawyer_config(benchmark.config)
    runners: list[CWTaskEnv] = []
    for i, env_name in enumerate(benchmark.task_names):
        runners.append(
            CWTaskEnv.create(
                env_name,
                task_idx=i,
                num_tasks=benchmark.num_tasks,
                tasks=benchmark.tasks,
                sawyer_config=sawyer_config,
                randomization=kind,
            )
        )
    return tuple(runners)


def make_cl_train_env(
    name: str = "CW10",
    *,
    config: CWConfig | None = None,
    seed: int | None = None,
    steps_per_task: int | None = None,
    randomization: RandomizationKind | None = None,
) -> ContinualLearningEnv:
    """Build a sequential Continual World training environment."""
    benchmark = CWBenchmark(name, config=config, seed=seed)
    cfg = benchmark.config
    if steps_per_task is not None:
        cfg = replace(cfg, steps_per_task=steps_per_task)
    task_envs = _task_envs_for_benchmark(benchmark, randomization=randomization)
    return ContinualLearningEnv(task_envs=task_envs, steps_per_task=cfg.steps_per_task)


def make_cl_test_envs(
    name: str = "CW10",
    *,
    config: CWConfig | None = None,
    seed: int | None = None,
    randomization: RandomizationKind = "deterministic",
) -> list[CWTaskEnv]:
    """Per-task evaluation environments in sequence order (deterministic goals by default)."""
    benchmark = CWBenchmark(name, config=config, seed=seed)
    return list(_task_envs_for_benchmark(benchmark, randomization=randomization))


__all__ = [
    "ALLOWED_RANDOMIZATION",
    "CW10",
    "CW20",
    "CWBenchmark",
    "CWConfig",
    "CW_DEFAULT_SEED",
    "CW_EPISODE_HORIZON",
    "CW_N_GOALS",
    "CW_STEPS_PER_TASK",
    "TASK_SEQS",
    "cw_obs_dim",
    "cw_sawyer_config",
    "make_cl_test_envs",
    "make_cl_train_env",
]
