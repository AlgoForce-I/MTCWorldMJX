"""Goal sampling of the batched ``VectorEnv``.

Continual World's ``random_init_all`` draws a fresh random reset vector every
episode (MetaWorld with ``_freeze_rand_vec = False``). A pool of pre-generated
tasks is a different, narrower protocol; ``rand_vecs=None`` selects the
reference one.
"""

from __future__ import annotations

import jax
import numpy as np

from MTCWorldMJX.mt_benchmarks import VectorEnv

NUM_ENVS = 4


def test_free_sampling_draws_fresh_goals_each_reset() -> None:
    venv = VectorEnv("push-v3", None, NUM_ENVS, partially_observable=False)
    low, high = (np.asarray(b) for b in venv.env.random_reset_bounds())

    first = np.asarray(venv.reset(jax.random.PRNGKey(0)).info["rand_vec"])
    second = np.asarray(venv.reset(jax.random.PRNGKey(1)).info["rand_vec"])

    assert first.shape == (NUM_ENVS, low.size)
    for vecs in (first, second):
        assert np.all(vecs >= low - 1e-6) and np.all(vecs <= high + 1e-6)
        assert len({tuple(np.round(v, 6)) for v in vecs}) == NUM_ENVS, "lanes share a goal"
    assert not np.allclose(first, second), "a new reset must draw new goals"


def test_pool_sampling_stays_within_the_pool() -> None:
    pool = np.array([[0.0, 0.6, 0.02, 0.05, 0.85, 0.01], [0.05, 0.62, 0.02, -0.05, 0.82, 0.01]], dtype=np.float32)
    venv = VectorEnv("push-v3", pool, NUM_ENVS, partially_observable=False)
    vecs = np.asarray(venv.reset(jax.random.PRNGKey(0)).info["rand_vec"])
    for v in vecs:
        assert np.isclose(pool, v, atol=1e-6).all(axis=1).any()
