# MTCWorldMJX

JAX-native MetaWorld v3 manipulation environments built on [MuJoCo MJX](https://github.com/google-deepmind/mujoco) with the **Warp** GPU backend. Environments expose `jax.jit`-compatible `reset` and `step` for high-throughput RL and continual-learning research, while a parity suite validates behavior against the original [MetaWorld](https://github.com/Farama-Foundation/Metaworld) (CPU MuJoCo) reference.

## Status

| Milestone | Status |
|-----------|--------|
| 50 MetaWorld v3 Sawyer tasks in MJX | Done |
| Parity validation vs MetaWorld | **55/55 tests passing** |
| MetaWorld-style MT/ML benchmark scaffolding | Partial (`benchmarks.py`, vectorized rollouts) |
| **Continual World benchmarks** | **Next step** |

## Features

- **Fully JAX-native hot path** — physics via MJX/Warp; no CPU MuJoCo in `reset`/`step` rollouts.
- **50 environments** — all standard MetaWorld v3 Sawyer XYZ tasks (`reach-v3` … `window-close-v3`).
- **Validated against MetaWorld** — automated parity checks for observations, rewards, joint state, and metrics.
- **Vectorized training API** — `VectorEnv`, `make_mt_envs`, `make_ml_envs_*`, and `rollout` for batched simulation.
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

### Vectorized benchmark rollout

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

### Available environments

All tasks are registered in `MTCWorldMJX.env_dict.ENV_CLS_MAP` and constructible via `make(name)`:

```
assembly-v3, basketball-v3, bin-picking-v3, box-close-v3,
button-press-v3, button-press-wall-v3, button-press-topdown-v3,
button-press-topdown-wall-v3, coffee-button-v3, coffee-push-v3,
coffee-pull-v3, dial-turn-v3, disassemble-v3, door-close-v3,
door-lock-v3, door-open-v3, door-unlock-v3, drawer-close-v3,
drawer-open-v3, faucet-close-v3, faucet-open-v3, hammer-v3,
handle-press-v3, handle-press-side-v3, handle-pull-v3,
handle-pull-side-v3, hand-insert-v3, lever-pull-v3,
peg-insert-side-v3, peg-unplug-side-v3, pick-out-of-hole-v3,
pick-place-v3, pick-place-wall-v3, plate-slide-v3,
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
├── benchmarks.py       # MT/ML task suites, VectorEnv, rollout
├── envs/
│   ├── sawyer_xyz.py   # Base class: reset, step, obs, rewards
│   ├── _helpers.py     # Shared reset/reward/obs utilities
│   └── sawyer_*_v3.py  # Per-task implementations
├── assets/sawyer_xyz/  # MJCF models (from MetaWorld)
└── utils/reward.py     # MetaWorld-compatible reward helpers
```

Each environment subclasses `SawyerXYZEnv` and implements task-specific reset bounds, object placement, observations, and reward logic. The base class handles hand settling, mocap control, observation packing (39-dim MetaWorld layout), and episode limits.

**Observation layout (39-dim):** hand position (3) + gripper (1) + interleaved object blocks (pos/quat per object) + previous observation (18) + goal (3).

**Action space:** 4-dim continuous — end-effector delta (x, y, z) and gripper effort, scaled by `action_scale` (default 0.01).

## Testing

### Full parity suite

Runs smoke, rollout, and MetaWorld parity for every environment:

```bash
.venv/bin/python -m pytest tests/ -q
```

Expected result: **55 passed** (50 environment parity tests + 5 benchmark API tests).

### Single environment (streaming output)

Use the standalone MT50 demo to see per-stage timings (useful when debugging compilation hangs):

```bash
.venv/bin/python main.py
```

Equivalent pytest invocation:

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

## Benchmark API (current)

`MTCWorldMJX.benchmarks` provides MetaWorld-style **MT** (multi-task) and **ML** (meta-learning) suites:

- `MT1`, `MT10`, `MT25`, `MT50`
- `ML1`, `ML10`, `ML25`, `ML45`
- `make_mt_envs`, `make_ml_envs_train`, `make_ml_envs_test`
- `VectorEnv` — batched lanes over frozen task `rand_vec`s
- `rollout` — JIT-friendly trajectory collection

This covers task sampling, observation mode (partially observable for ML), and vectorized execution. See `tests/test_benchmarks.py` for usage examples.

## Known limitations

Parity is **tolerance-based**, not bit-exact:

1. **MJX/Warp vs CPU MuJoCo** — small numerical differences are expected; tolerances account for this.
2. **`peg-unplug-side-v3`** — MetaWorld's double `reset_model()` leaves a large plug angular velocity that Warp does not reproduce; reset qvel uses a relaxed tolerance (`7.0`). Observations and step dynamics still match within normal bounds.
3. **Parity depth** — one fixed reset vector and one fixed action per env; not long-horizon statistical equivalence.
4. **GPU required** — the default `impl="warp"` backend needs CUDA.

## Roadmap

### Next: Continual World benchmarks

The immediate next milestone is integrating the **[Continual World](https://github.com/ContinualAI/continualworld)** benchmark protocols on top of this JAX-native MetaWorld port:

- [ ] Continual World task sequences and evaluation splits
- [ ] Standard continual-learning metrics (forward transfer, backward transfer, forgetting)
- [ ] Reproducible experiment configs aligned with the original Continual World paper/baselines
- [ ] End-to-end JAX training examples using `VectorEnv` + `rollout`

MetaWorld MT/ML scaffolding in `benchmarks.py` is a foundation; Continual World adds the continual-learning experimental protocol.

### Future

- Longer-horizon parity sampling
- Optional CPU MJX backend for debugging
- Reference RL / continual-learning training scripts

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
