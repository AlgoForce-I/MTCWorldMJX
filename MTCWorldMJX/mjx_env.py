"""Core MJX environment primitives for MTCWorldMJX."""

from __future__ import annotations

from typing import Any, Dict

import jax
import jax.numpy as jnp
import mujoco
from flax import struct
from mujoco import mjx


@struct.dataclass
class State:
    """Environment state for training and inference."""

    model: mjx.Model
    data: mjx.Data
    obs: jax.Array
    reward: jax.Array
    truncated: jax.Array
    terminated: jax.Array
    metrics: Dict[str, jax.Array] = struct.field(default_factory=dict)
    info: Dict[str, Any] = struct.field(default_factory=dict)


def load_mj_model(
    xml_path: str,
    *,
    timestep: float | None = None,
    solver_iterations: int = 6,
    ls_iterations: int = 6,
    zero_geom_margins: bool = True,
) -> mujoco.MjModel:
    """Load and patch a MuJoCo model for MJX-Warp compatibility."""
    model = mujoco.MjModel.from_xml_path(xml_path)
    if timestep is not None:
        model.opt.timestep = timestep
    model.opt.iterations = solver_iterations
    model.opt.ls_iterations = ls_iterations
    if zero_geom_margins:
        model.geom_margin[:] = 0.0
    return model


def make_data(
    mj_model: mujoco.MjModel,
    mjx_model: mjx.Model,
    *,
    qpos: jax.Array | None = None,
    qvel: jax.Array | None = None,
    ctrl: jax.Array | None = None,
    mocap_pos: jax.Array | None = None,
    mocap_quat: jax.Array | None = None,
    naconmax: int = 2000,
    njmax: int = 1000,
) -> mjx.Data:
    """Initialize MJX data from a MuJoCo model."""
    impl = mjx_model.impl.value
    data = mjx.make_data(
        mj_model,
        impl=impl,
        naconmax=naconmax,
        njmax=njmax,
    )
    if qpos is not None:
        data = data.replace(qpos=qpos)
    if qvel is not None:
        data = data.replace(qvel=qvel)
    if ctrl is not None:
        data = data.replace(ctrl=ctrl)
    if mocap_pos is not None:
        data = data.replace(mocap_pos=mocap_pos.reshape(mj_model.nmocap, -1))
    if mocap_quat is not None:
        data = data.replace(mocap_quat=mocap_quat.reshape(mj_model.nmocap, -1))
    return data


def step(
    model: mjx.Model,
    data: mjx.Data,
    ctrl: jax.Array,
    n_substeps: int = 1,
) -> mjx.Data:
    """Apply controls and step physics for ``n_substeps``."""
    def single_step(carry, _):
        carry = carry.replace(ctrl=ctrl)
        return mjx.step(model, carry), None

    return jax.lax.scan(single_step, data, None, length=n_substeps)[0]


def body_id(mj_model: mujoco.MjModel, name: str) -> int:
    return mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, name)


def site_id(mj_model: mujoco.MjModel, name: str) -> int:
    return mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, name)


def geom_id(mj_model: mujoco.MjModel, name: str) -> int:
    return mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, name)


def joint_id(mj_model: mujoco.MjModel, name: str) -> int:
    return mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, name)


def site_xpos(data: mjx.Data, site_ids: jax.Array) -> jax.Array:
    site_ids = jnp.asarray(site_ids)
    if site_ids.ndim == 0:
        return data.site_xpos[site_ids]
    return data.site_xpos[site_ids]


def body_xpos(data: mjx.Data, body_ids: jax.Array) -> jax.Array:
    body_ids = jnp.asarray(body_ids)
    if body_ids.ndim == 0:
        return data.xpos[body_ids]
    return data.xpos[body_ids]


def body_xquat(data: mjx.Data, body_ids: jax.Array) -> jax.Array:
    body_ids = jnp.asarray(body_ids)
    if body_ids.ndim == 0:
        return data.xquat[body_ids]
    return data.xquat[body_ids]


def geom_xpos(data: mjx.Data, geom_ids: jax.Array) -> jax.Array:
    geom_ids = jnp.asarray(geom_ids)
    if geom_ids.ndim == 0:
        return data.geom_xpos[geom_ids]
    return data.geom_xpos[geom_ids]


def geom_xquat(data: mjx.Data, geom_ids: jax.Array) -> jax.Array:
    """Quaternion from geom orientation matrix (w, x, y, z)."""
    xmat = data.geom_xmat[geom_ids].reshape(-1, 3, 3)
    w = jnp.sqrt(jnp.maximum(1.0 + xmat[..., 0, 0] + xmat[..., 1, 1] + xmat[..., 2, 2], 0.0)) / 2.0
    x = jnp.sign(xmat[..., 2, 1] - xmat[..., 1, 2]) * jnp.sqrt(
        jnp.maximum(1.0 + xmat[..., 0, 0] - xmat[..., 1, 1] - xmat[..., 2, 2], 0.0)
    ) / 2.0
    y = jnp.sign(xmat[..., 0, 2] - xmat[..., 2, 0]) * jnp.sqrt(
        jnp.maximum(1.0 - xmat[..., 0, 0] + xmat[..., 1, 1] - xmat[..., 2, 2], 0.0)
    ) / 2.0
    z = jnp.sign(xmat[..., 1, 0] - xmat[..., 0, 1]) * jnp.sqrt(
        jnp.maximum(1.0 - xmat[..., 0, 0] - xmat[..., 1, 1] + xmat[..., 2, 2], 0.0)
    ) / 2.0
    return jnp.stack([w, x, y, z], axis=-1)
