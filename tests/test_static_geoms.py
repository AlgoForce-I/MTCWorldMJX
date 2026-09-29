"""Collision geometry of world-fixed bodies follows the per-episode layout.

MuJoCo Warp computes the world pose of geoms on bodies welded to the world only
once, in ``make_data`` (``mujoco_warp`` ``smooth._geom_local_to_global`` skips
them afterwards). Envs that move such bodies at reset through
``model.body_pos`` must refresh those poses, otherwise the scene collides
against the XML-default layout while observations and rewards use the sampled
one (a window frame 14 cm away jams the sash, a shelf floats elsewhere).

The reference is MuJoCo C kinematics on the same model fields and qpos.
"""

from __future__ import annotations

import copy

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import pytest

from MTCWorldMJX.env_dict import make

# Envs whose reset writes ``model.body_pos``.
BODY_MOVING_ENVS = [
    "assembly-v3", "basketball-v3", "box-close-v3", "button-press-topdown-v3",
    "button-press-topdown-wall-v3", "button-press-v3", "button-press-wall-v3",
    "coffee-button-v3", "coffee-pull-v3", "coffee-push-v3", "dial-turn-v3",
    "disassemble-v3", "door-close-v3", "door-lock-v3", "door-open-v3",
    "door-unlock-v3", "drawer-close-v3", "drawer-open-v3", "faucet-close-v3",
    "faucet-open-v3", "hammer-v3", "handle-press-side-v3", "handle-press-v3",
    "handle-pull-side-v3", "handle-pull-v3", "lever-pull-v3", "peg-insert-side-v3",
    "peg-unplug-side-v3", "plate-slide-back-side-v3", "plate-slide-back-v3",
    "plate-slide-side-v3", "plate-slide-v3", "shelf-place-v3", "soccer-v3",
    "window-close-v3", "window-open-v3",
]

# Continual World tasks known to move a world-fixed body; guards against the
# test passing only because nothing static moved.
MOVES_STATIC_BODY = {
    "faucet-close-v3", "handle-press-side-v3", "shelf-place-v3",
    "window-close-v3", "peg-unplug-side-v3",
}

POS_ATOL = 1e-4
MAT_ATOL = 1e-4


def _static_geom_mask(m: mujoco.MjModel) -> np.ndarray:
    body = m.geom_bodyid
    return (m.body_weldid[body] == 0) & (m.body_mocapid[m.body_rootid[body]] == -1)


def _reference_geom_poses(mj_model: mujoco.MjModel, state) -> tuple[np.ndarray, np.ndarray]:
    m = copy.deepcopy(mj_model)
    m.body_pos[:] = np.asarray(state.model.body_pos)
    m.body_quat[:] = np.asarray(state.model.body_quat)
    m.geom_pos[:] = np.asarray(state.model.geom_pos)
    m.geom_quat[:] = np.asarray(state.model.geom_quat)
    d = mujoco.MjData(m)
    d.qpos[:] = np.asarray(state.data.qpos)
    d.mocap_pos[:] = np.asarray(state.data.mocap_pos)
    d.mocap_quat[:] = np.asarray(state.data.mocap_quat)
    mujoco.mj_kinematics(m, d)
    return d.geom_xpos.copy(), d.geom_xmat.copy()


def _assert_static_geoms_match(env, state, label: str) -> None:
    m = env.mj_model
    mask = _static_geom_mask(m)
    ref_pos, ref_mat = _reference_geom_poses(m, state)
    pos = np.asarray(state.data.geom_xpos)
    mat = np.asarray(state.data.geom_xmat).reshape(m.ngeom, 9)
    pos_err = np.linalg.norm(pos - ref_pos, axis=1)[mask]
    mat_err = np.abs(mat - ref_mat).max(axis=1)[mask]
    names = np.array([
        mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or f"#{g}"
        for g in np.where(mask)[0]
    ])
    bad = pos_err > POS_ATOL
    assert not bad.any(), (
        f"{label}: static geoms off their reset layout: "
        f"{dict(zip(names[bad].tolist(), np.round(pos_err[bad], 4).tolist()))}"
    )
    assert mat_err.max(initial=0.0) < MAT_ATOL, f"{label}: static geom orientation mismatch"


@pytest.mark.parametrize("env_name", BODY_MOVING_ENVS)
def test_static_geoms_follow_reset_layout(env_name: str) -> None:
    env = make(env_name)
    state = jax.jit(env.reset)(jax.random.PRNGKey(3))
    m = env.mj_model

    static_bodies = np.where(m.body_weldid == 0)[0]
    moved = np.abs(np.asarray(state.model.body_pos)[static_bodies] - m.body_pos[static_bodies]).max(axis=1) > 1e-6
    if env_name in MOVES_STATIC_BODY:
        assert moved.any(), f"{env_name} was expected to move a world-fixed body at reset"

    _assert_static_geoms_match(env, state, f"{env_name} after reset")
    state = jax.jit(env.step)(state, jnp.zeros(env.action_size))
    _assert_static_geoms_match(env, state, f"{env_name} after one step")
