"""Core MJX environment primitives for MTCWorldMJX."""

from __future__ import annotations

from typing import Any, Dict

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
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
    # MuJoCo Warp sends box-box pairs to its convex collider unless native CCD
    # is disabled, and that path gives every point of a contact patch the same
    # (deepest) depth; the primitive collider keeps per-point depths like
    # MuJoCo C. Warp ignores this flag otherwise.
    model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_NATIVECCD
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


def _quat_mul(a: jax.Array, b: jax.Array) -> jax.Array:
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return jnp.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        axis=-1,
    )


def _quat_to_mat(q: jax.Array) -> jax.Array:
    q = q / jnp.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return jnp.stack(
        [
            jnp.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], axis=-1),
            jnp.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], axis=-1),
            jnp.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], axis=-1),
        ],
        axis=-2,
    )


class StaticGeoms:
    """World poses of geoms on bodies welded to the world.

    MuJoCo Warp computes these poses once in ``make_data`` and skips them in
    kinematics, so after ``model.body_pos`` of a world-fixed body changes they
    keep the layout the data was created with and collisions happen there.
    ``refresh`` recomputes them from the model and writes them into ``data``.
    """

    def __init__(self, mj_model: mujoco.MjModel):
        static_body = (mj_model.body_weldid == 0) & (
            mj_model.body_mocapid[mj_model.body_rootid] == -1
        )
        # Body ids are topologically sorted, so parents are placed before children.
        self._body_ids = [int(b) for b in np.nonzero(static_body)[0] if b > 0]
        self._parent_ids = [int(mj_model.body_parentid[b]) for b in self._body_ids]
        self._geom_ids = np.nonzero(static_body[mj_model.geom_bodyid])[0]
        self._geom_body_ids = mj_model.geom_bodyid[self._geom_ids]
        self._nbody = mj_model.nbody

    def refresh(self, model: mjx.Model, data: mjx.Data) -> mjx.Data:
        if self._geom_ids.size == 0:
            return data
        pos = jnp.zeros((self._nbody, 3), dtype=data.geom_xpos.dtype)
        quat = jnp.zeros((self._nbody, 4), dtype=data.geom_xpos.dtype).at[:, 0].set(1.0)
        for body, parent in zip(self._body_ids, self._parent_ids):
            parent_mat = _quat_to_mat(quat[parent])
            pos = pos.at[body].set(pos[parent] + parent_mat @ model.body_pos[body])
            quat = quat.at[body].set(_quat_mul(quat[parent], model.body_quat[body]))

        geoms = jnp.asarray(self._geom_ids)
        body_quat = quat[self._geom_body_ids]
        geom_xpos = pos[self._geom_body_ids] + jnp.einsum(
            "gij,gj->gi", _quat_to_mat(body_quat), model.geom_pos[geoms]
        )
        geom_xmat = _quat_to_mat(_quat_mul(body_quat, model.geom_quat[geoms]))
        return data.replace(
            geom_xpos=data.geom_xpos.at[geoms].set(geom_xpos),
            geom_xmat=data.geom_xmat.at[geoms].set(
                geom_xmat.reshape(data.geom_xmat[geoms].shape)
            ),
        )


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
