# MTCWorldMJX

JAX-native MetaWorld v3 manipulation environments built on [MuJoCo MJX](https://github.com/google-deepmind/mujoco) with the **Warp** GPU backend. Environments expose `jax.jit`-compatible `reset` and `step` for high-throughput RL and continual-learning research, while a parity suite validates behavior against the original [MetaWorld](https://github.com/Farama-Foundation/Metaworld) (CPU MuJoCo) reference.

## Status

| Milestone | Status |
|-----------|--------|
| 50 MetaWorld v3 Sawyer tasks in MJX | Done |
| Parity validation vs MetaWorld | **55/55 env tests passing** |
| MetaWorld-style MT/ML benchmarks | Done (`mt_benchmarks.py`, `VectorEnv`, `rollout`) |
| Continual World protocol scaffolding | **Partial** (`cw_benchmarks.py`, `cw_env.py`, JIT examples) |
| CL metrics (FT / forgetting) & JAX learners | Next step |

## Features

- **Fully JAX-native hot path** — physics via MJX/Warp; no CPU MuJoCo in `reset`/`step` rollouts.
- **50 environments** — all standard MetaWorld v3 Sawyer XYZ tasks (`reach-v3` … `window-close-v3`).
- **Validated against MetaWorld** — automated parity checks for observations, rewards, joint state, and metrics.
- **Vectorized training API** — `VectorEnv`, `make_mt_envs`, `make_ml_envs_*`, and `rollout` for batched simulation.
- **Continual World (CW10 / CW20)** — task sequences, sequential training env, per-task eval hooks, GPU-saturated rollouts.
- **Persistent compilation caches** — JAX and Warp disk caches make repeated test/training runs much faster after the first compile.

## Requirements

- Python ≥ 3.10
- NVIDIA GPU with CUDA (Warp backend)
- Linux recommended (developed on Ubuntu)

## Installation

```bash
git clone <repo-url> MTCWorldMJX
cd MTCWorldMJX
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

For **parity tests**, install MetaWorld separately (not a runtime dependency of the library itself):

```bash
pip install metaworld gymnasium scipy
```

## Quick start

### Single environment

```python
import jax
import jax.numpy as jnp
from MTCWorldMJX import make

env = make("reach-v3")
jit_reset = jax.jit(env.reset)
jit_step = jax.jit(env.step)

rng = jax.random.PRNGKey(0)
state = jit_reset(rng)
state = jit_step(state, jnp.zeros(4))

print(state.obs.shape)  # (39,)
print(float(state.reward))
```

### MetaWorld vectorized rollout (MT)

```python
import jax
import jax.numpy as jnp
from MTCWorldMJX import make_mt_envs, rollout

env = make_mt_envs("reach-v3", seed=0, num_envs=512)
state = env.reset(jax.random.PRNGKey(0))


def policy(obs, key):
    del key
    return jnp.zeros((obs.shape[0], 4))


state, traj = rollout(env, state, policy, jax.random.PRNGKey(1), num_steps=200)
# traj["reward"].shape == (200, 512)
```

### Continual World (CW10)

```python
from MTCWorldMJX import CW10, CWConfig, make_cl_train_env, make_cl_test_envs, cw_obs_dim

bench = CW10(seed=1)
train_env = make_cl_train_env("CW10", config=CWConfig(seed=1, steps_per_task=1_000_000))
test_envs = make_cl_test_envs("CW10", seed=1)

print(bench.task_names)       # 10-task CW sequence
print(cw_obs_dim(10))         # 49 = 39-dim MetaWorld obs + 10-dim task one-hot
print(len(test_envs))         # one eval env per sequence slot
```

### Examples (repo root)

| Script | Purpose |
|--------|---------|
| [`metaworld_example.py`](metaworld_example.py) | MT50-style vectorized rollout demo |
| [`continualworld_example.py`](continualworld_example.py) | **CW10/CW20 JIT rollouts** (GPU-saturated by default) |

```bash
# MetaWorld: vectorized MT50 sample
.venv/bin/python metaworld_example.py

# Continual World: 512 lanes × 200 steps × 10 tasks (default saturated mode)
.venv/bin/python continualworld_example.py
.venv/bin/python continualworld_example.py --benchmark CW20
.venv/bin/python continualworld_example.py --mode single --steps-per-task 2000
```

### Available environments

All tasks are registered in `MTCWorldMJX.env_dict.ENV_CLS_MAP` and constructible via `make(name)`:

```
assembly-v3, basketball-v3, bin-picking-v3, box-close-v3,
button-press-v3, button-press-wall-v3, button-press-topdown-v3,
button-press-topdown-wall-v3, coffee-button-v3, coffee-push-v3,
coffee-pull-v3, dial-turn-v3, disassemble-v3, door-close-v3,
door-lock-v3, door-open-v3, door-unlock-v3, drawer-close-v3,
drawer-open-v3, faucet-close-v3, faucet-open-v3, hammer-v3,
handle-press-side-v3, handle-pull-v3, handle-pull-side-v3,
hand-insert-v3, lever-pull-v3, peg-insert-side-v3, peg-unplug-side-v3,
pick-out-of-hole-v3, pick-place-v3, pick-place-wall-v3, plate-slide-v3,
plate-slide-back-v3, plate-slide-side-v3, plate-slide-back-side-v3,
push-v3, push-back-v3, push-wall-v3, reach-v3, reach-wall-v3,
shelf-place-v3, soccer-v3, stick-push-v3, stick-pull-v3,
sweep-v3, sweep-into-v3, window-open-v3, window-close-v3
```

## Architecture

```
MTCWorldMJX/
├── mjx_env.py          # State dataclass, model loading, MJX step helpers
├── env_dict.py         # ENV_CLS_MAP + make()
├── mt_benchmarks.py    # MetaWorld MT/ML suites, VectorEnv, rollout
├── cw_benchmarks.py    # CW10/CW20 task sequences, CWConfig, CL factories
├── cw_env.py           # CWTaskEnv, ContinualLearningEnv, one-hot obs
├── cw_eval.py          # Per-task eval helpers for CL policies
├── envs/
│   ├── sawyer_xyz.py   # Base class: reset, step, obs, rewards
│   ├── _helpers.py     # Shared reset/reward/obs utilities
│   └── sawyer_*_v3.py  # Per-task implementations
├── assets/sawyer_xyz/  # MJCF models (from MetaWorld)
└── utils/reward.py     # MetaWorld-compatible reward helpers

metaworld_example.py      # MT vectorized demo
continualworld_example.py # CW JIT rollout demo (saturated / single-lane)
```

Each environment subclasses `SawyerXYZEnv` and implements task-specific reset bounds, object placement, observations, and reward logic. The base class handles hand settling, mocap control, observation packing (39-dim MetaWorld layout), and episode limits.

**Observation layout (39-dim):** hand position (3) + gripper (1) + interleaved object blocks (pos/quat per object) + previous observation (18) + goal (3).

**Continual World observations:** 39-dim layout + task-index one-hot (`cw_obs_dim(num_tasks)` → 49 for CW10).

**Action space:** 4-dim continuous — end-effector delta (x, y, z) and gripper effort, scaled by `action_scale` (default 0.01).

## Testing

### Full test suite

```bash
.venv/bin/python -m pytest tests/ -q
```

Expected result: **60 passed** — 50 environment parity tests, 5 MetaWorld benchmark API tests, 5 Continual World smoke tests.

### Single environment (streaming output)

```bash
.venv/bin/python -m pytest tests/test_reach_v3.py::test_reach_v3 -s
```

### What parity checks

For each environment, against installed MetaWorld with a **fixed** `rand_vec` and **fixed** action:

| Check | Tolerance (typical) |
|-------|---------------------|
| Reset observation | `5e-3` |
| Reset qpos | `5e-3` |
| Reset qvel | `0.15` (`7.0` for `peg-unplug-side-v3`) |
| Step observation | `5e-3` |
| Step reward | `2e-2` |
| Step qpos / qvel | `5e-3` / `0.4` |
| Step metrics | reward-like keys `2e-2`, others `5e-3` |

Parity tests use `PARITY_CONFIG` (50 solver iterations, `zero_geom_margins=True`) to align MJX with MetaWorld. Default `make()` uses lighter solver settings tuned for speed.

### Test infrastructure notes

- **Do not use `pytest-forked` / `pytest-isolate`** — Warp initializes CUDA in the parent process; forking breaks the GPU context.
- JAX persistent cache: `$TMPDIR/mtcworldmjx_jax_cache` (override with `MTCWMJX_JAX_CACHE_DIR`; legacy `CWMJX_JAX_CACHE_DIR` still works).
- Warp kernel cache: `~/.cache/warp/`.
- First run per environment compiles Warp kernels (can take seconds); subsequent runs load from cache.

## Benchmark APIs

### MetaWorld MT / ML (`mt_benchmarks`)

Imported from the top-level package (`from MTCWorldMJX import …`):

- `MT1`, `MT10`, `MT25`, `MT50`
- `ML1`, `ML10`, `ML25`, `ML45`
- `make_mt_envs`, `make_ml_envs_train`, `make_ml_envs_test`
- `VectorEnv` — batched lanes over frozen task `rand_vec`s
- `rollout` — JIT-friendly trajectory collection

See `tests/test_benchmarks.py` and `metaworld_example.py`.

### Continual World (`cw_benchmarks`)

Aligned with the [Continual World](https://github.com/ContinualAI/continualworld) protocol (v3 task names):

| Export | Description |
|--------|-------------|
| `TASK_SEQS`, `CW10`, `CW20` | CW10 and CW20 task order |
| `CWConfig` | `steps_per_task`, `episode_horizon=200`, goal pools (`seed=1`), randomization |
| `make_cl_train_env` | Sequential `ContinualLearningEnv` (single-lane CL training) |
| `make_cl_test_envs` | Per-sequence-slot `CWTaskEnv` list for evaluation |
| `cw_obs_dim` | Observation size with task one-hot |

**Randomization modes** (via `CWConfig.randomization`): `deterministic`, `random_init_all` (default), `random_init_fixed20`, `random_init_small_box`.

`continualworld_example.py` demonstrates two rollout modes:

- **`saturated` (default)** — `VectorEnv` with 512 lanes per task; JIT `rollout` per CW slot (~80k+ env-steps/s on a warm GPU run).
- **`single`** — one-lane `ContinualLearningEnv` matching the reference sequential CL env (~300–400 env-steps/s).

Not yet implemented: forward-transfer / forgetting metrics (`cw_metrics`), JAX SAC, and CL regularizers from the reference repo.

## Known limitations

Parity is **tolerance-based**, not bit-exact:

1. **MJX/Warp vs CPU MuJoCo** — small numerical differences are expected; tolerances account for this.
2. **`peg-unplug-side-v3`** — MetaWorld's double `reset_model()` leaves a large plug angular velocity that Warp does not reproduce; reset qvel uses a relaxed tolerance (`7.0`). Observations and step dynamics still match within normal bounds.
3. **Parity depth** — one fixed reset vector and one fixed action per env; not long-horizon statistical equivalence.
4. **GPU required** — the default `impl="warp"` backend needs CUDA.
5. **Continual World** — protocol scaffolding and rollouts are in place; paper-matched training baselines (SAC + EWC / PackNet / …) are not bundled yet.

## Roadmap

### Next

- [ ] Continual World metrics (forward transfer, forgetting, backward transfer)
- [ ] JAX training baseline (e.g. SAC) on `ContinualLearningEnv` / saturated `VectorEnv`
- [ ] Reproducible experiment configs aligned with the original Continual World paper

### Future

- Longer-horizon parity sampling
- Optional CPU MJX backend for debugging
- End-to-end continual-learning experiment scripts

## Citation

If you use this codebase, please cite **MTCWorldMJX** and the underlying benchmarks and simulators as appropriate:

- **MTCWorldMJX** — if you use this JAX/MJX port, vectorized environments, or parity-validated implementations.
- **MetaWorld** — the underlying manipulation task suite.
- **Continual World** — continual RL benchmarks and evaluation protocols built on MetaWorld.
- **MuJoCo** — the physics engine; MJX is a JAX re-implementation of MuJoCo.

```bibtex
@misc{mtcworldmjx2026,
  title        = {{MTCWorldMJX}: {JAX}-Native {MetaWorld} v3 Environments on {MuJoCo} {MJX}},
  author       = {{MTCWorldMJX contributors}},
  year         = {2026},
  howpublished = {Software},
  note         = {Apache-2.0 licensed JAX port of MetaWorld v3 with MJX/Warp backend}
}

@inproceedings{yu2019meta,
  title     = {{Meta-World}: A Benchmark and Evaluation for Multi-Task and Meta Reinforcement Learning},
  author    = {Yu, Tianhe and Quillen, Deirdre and He, Zhanpeng and Julian, Ryan and Hausman, Karol and Finn, Chelsea and Levine, Sergey},
  booktitle = {Conference on Robot Learning},
  year      = {2019},
  pages     = {1094--1100},
  volume    = {100},
  url       = {https://mlanthology.org/corl/2019/yu2019corl-metaworld/}
}

@inproceedings{wolczyk2021continual,
  title     = {Continual World: A Robotic Benchmark For Continual Reinforcement Learning},
  author    = {Wo{\l}czyk, Maciej and Zaj{\k{a}}c, Micha{\l} and Pascanu, Razvan and Kuci{\'n}ski, {\L}ukasz and Mi{\l}o{\'s}, Piotr},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2021},
  volume    = {34},
  pages     = {28496--28510},
  url       = {https://proceedings.neurips.cc/paper_files/paper/2021/file/ef8446f35513a8d6aa2308357a268a7e-Paper.pdf}
}

@inproceedings{todorov2012mujoco,
  title     = {{MuJoCo}: A Physics Engine for Model-Based Control},
  author    = {Todorov, Emanuel and Erez, Tom and Tassa, Yuval},
  booktitle = {2012 IEEE/RSJ International Conference on Intelligent Robots and Systems},
  pages     = {5026--5033},
  year      = {2012},
  organization = {IEEE},
  doi       = {10.1109/IROS.2012.6386109}
}
```

## License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE).

MuJoCo assets follow the MetaWorld / MuJoCo asset licenses bundled under `MTCWorldMJX/assets/`.
