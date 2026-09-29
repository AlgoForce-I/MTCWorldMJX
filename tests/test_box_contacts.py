"""Box-box contact depths match MuJoCo C.

MuJoCo Warp routes box-box pairs to its convex (CCD) collider unless native CCD
is disabled, and that path gives every point of the contact patch the single
deepest penetration. When a grasped box is slightly tilted between the Sawyer
pads, the shallow side then pushes back as hard as the deep side, the contact
problem changes and the fingers close through the object (stick-pull's grasp
gate fails). The primitive box-box collider (a port of MuJoCo C's
``mjc_BoxBox``) keeps per-point depths.
"""

from __future__ import annotations

import jax
import mujoco
import numpy as np
import pytest

from MTCWorldMJX import mjx_env
from MTCWorldMJX.env_dict import make

metaworld = pytest.importorskip("metaworld")
from metaworld import policies  # noqa: E402
from metaworld.env_dict import ALL_V3_ENVIRONMENTS  # noqa: E402

# First goal of the CW10 (seed 42) stick-pull pool; the stick sits tilted in the grasp.
STICK_PULL_VEC = np.array([
    -0.07974991947412491, 0.6286692023277283, 0.0005166720366105437,
    0.42698511481285095, 0.4799478352069855, 0.019931785762310028,
])


def _metaworld_grasp_state(steps: int = 50):
    env = ALL_V3_ENVIRONMENTS["stick-pull-v3"]()
    env._set_task_called = True
    env._partially_observable = False
    env.__dict__.pop("sawyer_observation_space", None)
    env._freeze_rand_vec = True
    env._last_rand_vec = STICK_PULL_VEC.copy()
    obs, _ = env.reset()
    policy = policies.SawyerStickPullV3Policy()
    for _ in range(steps):
        obs, *_ = env.step(np.clip(policy.get_action(obs), -1, 1))
    env.model.geom_margin[:] = 0.0  # as MTCWorldMJX loads its models
    mujoco.mj_forward(env.model, env.data)
    return env.model, env.data


def _depths_by_pair(names, geom_pairs, dists, obj: str) -> dict[str, np.ndarray]:
    out: dict[str, list[float]] = {}
    for (g1, g2), dist in zip(geom_pairs, dists):
        n1, n2 = names(g1), names(g2)
        if obj in (n1, n2):
            out.setdefault(n2 if n1 == obj else n1, []).append(float(dist))
    return {k: np.sort(v) for k, v in out.items()}


def test_box_box_depths_match_mujoco_c() -> None:
    mw_model, mw_data = _metaworld_grasp_state()
    env = make("stick-pull-v3")
    m = env.mj_model

    def name(g):
        return mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, int(g)) or f"#{int(g)}"

    data = mjx_env.make_data(
        m, env.mjx_model,
        qpos=mw_data.qpos.copy(), qvel=mw_data.qvel.copy(), ctrl=mw_data.ctrl.copy(),
        mocap_pos=mw_data.mocap_pos.copy(), mocap_quat=mw_data.mocap_quat.copy(),
    )
    data = jax.jit(mujoco.mjx.forward)(env.mjx_model, data)
    n = int(np.asarray(data._impl.nacon).ravel()[0])
    warp = _depths_by_pair(name, np.asarray(data._impl.contact__geom)[:n], np.asarray(data._impl.contact__dist)[:n], "objGeom")
    ref = _depths_by_pair(
        name, [(c.geom1, c.geom2) for c in mw_data.contact[:mw_data.ncon]],
        [c.dist for c in mw_data.contact[:mw_data.ncon]], "objGeom",
    )

    assert set(warp) == set(ref), f"contacting geoms differ: {sorted(warp)} vs {sorted(ref)}"
    for geom, depths in ref.items():
        # Deepest and shallowest points of each patch: a single shared depth
        # makes the shallow end as deep as the deepest one.
        np.testing.assert_allclose(warp[geom].min(), depths.min(), atol=2e-4, err_msg=geom)
        np.testing.assert_allclose(warp[geom].max(), depths.max(), atol=2e-4, err_msg=geom)
