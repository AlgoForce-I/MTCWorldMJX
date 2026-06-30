"""Pytest hooks for MTCWorldMJX tests.

The dominant cost of these tests is *compilation*, which has two independent,
disk-cached layers:

* XLA compilation of the MJX pipeline. A persistent on-disk JAX compilation
  cache (configured below) makes identical MJX graphs compile once and reload
  cheaply on subsequent tests/runs.
* Warp's native (nvrtc) collision/solver kernels. Warp keeps its own on-disk
  cache (``~/.cache/warp``). The first time an env's collision kernels are
  built (e.g. box-close-v3's ``capsule_box`` / ``plane_convex`` kernels) the
  cold nvrtc compile is slow; once cached, every later build is ~1s regardless
  of how many other envs ran in the same process.

Note on isolation: do NOT run these under ``pytest-forked``/``pytest-isolate``.
Collection imports the env modules, which import Warp and initialize the CUDA
driver in the pytest process. ``os.fork()`` then hands the child a dead CUDA
context, so the first ``jax.device_put`` aborts with
``CUDA_ERROR_NOT_INITIALIZED``. Fork-based isolation is fundamentally unsafe
here; the disk caches above are what make the suite fast.
"""

from __future__ import annotations

import gc
import os
import tempfile

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import pytest

_CACHE_DIR = os.environ.get(
    "CWMJX_JAX_CACHE_DIR", os.path.join(tempfile.gettempdir(), "cwmjx_jax_cache")
)
jax.config.update("jax_compilation_cache_dir", _CACHE_DIR)
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)


@pytest.fixture(autouse=True)
def _free_device_memory():
    """Drop in-memory JAX caches after each test.

    The suite builds ~50 envs in one process; without this, compiled
    executables and device buffers accumulate in VRAM. Clearing the in-memory
    cache forces a reload from the persistent on-disk cache (cheap), so we trade
    a little speed for bounded memory.
    """
    yield
    jax.clear_caches()
    gc.collect()
