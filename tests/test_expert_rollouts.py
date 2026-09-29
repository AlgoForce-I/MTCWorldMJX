"""Multi-step behavior against MetaWorld, driven by MetaWorld's scripted policies.

The single-step parity tests cannot see errors that only show once objects are
touched, grasped or pushed. These rollouts compare what matters for learning:
whether the expert succeeds and what the reward pays along the way.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

metaworld = pytest.importorskip("metaworld")
from metaworld import policies  # noqa: E402
from metaworld.env_dict import ALL_V3_ENVIRONMENTS  # noqa: E402

from MTCWorldMJX.env_dict import make  # noqa: E402
from MTCWorldMJX.envs.sawyer_xyz import SawyerXYZConfig  # noqa: E402

HORIZON = 200
CW_CONFIG = SawyerXYZConfig(max_path_length=HORIZON, partially_observable=False)


def _metaworld_env(env_name: str, rand_vec: np.ndarray):
    env = ALL_V3_ENVIRONMENTS[env_name]()
    env._set_task_called = True
    env._partially_observable = False
    # set_task drops the cached observation space when observability changes;
    # without this the goal observation is clipped to zero after the first step.
    env.__dict__.pop("sawyer_observation_space", None)
    env._freeze_rand_vec = True
    env._last_rand_vec = np.asarray(rand_vec, dtype=np.float64).copy()
    env.max_path_length = HORIZON
    return env


def _mjx_reset(env_name: str, num: int, seed: int = 0, config: SawyerXYZConfig = CW_CONFIG):
    env = make(env_name, config=config)
    keys = jax.random.split(jax.random.PRNGKey(seed), num)
    state = jax.jit(jax.vmap(env.reset))(keys)
    return env, state, jax.jit(jax.vmap(env.step))


def _run_metaworld(env_name: str, rand_vec: np.ndarray, policy_cls, actions=None):
    env = _metaworld_env(env_name, rand_vec)
    obs, _ = env.reset()
    policy = policy_cls() if policy_cls is not None else None
    out = {"obs": [], "reward": [], "success": [], "action": []}
    for t in range(HORIZON):
        action = actions[t] if actions is not None else np.clip(policy.get_action(obs), -1, 1)
        obs, reward, _, _, info = env.step(np.asarray(action, dtype=np.float32))
        out["obs"].append(obs.copy())
        out["reward"].append(reward)
        out["success"].append(float(info["success"]))
        out["action"].append(np.asarray(action, dtype=np.float32))
    return {k: np.asarray(v) for k, v in out.items()}


def _run_mjx(state, step, policy_cls=None, actions=None):
    num = state.obs.shape[0]
    pols = [policy_cls() for _ in range(num)] if policy_cls is not None else None
    out = {"obs": [], "reward": [], "success": []}
    for t in range(HORIZON):
        if actions is not None:
            action = actions[:, t]
        else:
            obs = np.asarray(state.obs, dtype=np.float64)
            action = np.stack([np.clip(p.get_action(o), -1, 1) for p, o in zip(pols, obs)])
        state = step(state, jnp.asarray(action, dtype=jnp.float32))
        out["obs"].append(np.asarray(state.obs))
        out["reward"].append(np.asarray(state.reward))
        out["success"].append(np.asarray(state.metrics["success"]))
    return {k: np.stack(v, axis=1) for k, v in out.items()}


def test_window_close_expert_matches_metaworld() -> None:
    """The sash must slide when pushed; a stale frame collider jammed it (0/N)."""
    env, state, step = _mjx_reset("window-close-v3", num=3)
    rand_vecs = np.asarray(state.info["rand_vec"])
    mjx_success = _run_mjx(state, step, policies.SawyerWindowCloseV3Policy)["success"].max(axis=1)
    mw_success = np.array([
        _run_metaworld("window-close-v3", v, policies.SawyerWindowCloseV3Policy)["success"].max()
        for v in rand_vecs
    ])
    assert mw_success.all(), "MetaWorld expert is expected to solve these goals"
    np.testing.assert_array_equal(mjx_success > 0.5, mw_success > 0.5)


def test_peg_rests_in_box_like_metaworld() -> None:
    """With no action the plug must stay seated in its box, as in MetaWorld."""
    env, state, step = _mjx_reset("peg-unplug-side-v3", num=2)
    rand_vecs = np.asarray(state.info["rand_vec"])
    zero = np.zeros((2, HORIZON, 4), dtype=np.float32)
    mjx_obj = _run_mjx(state, step, actions=zero)["obs"][:, :20, 4:7]
    for i, vec in enumerate(rand_vecs):
        mw_obj = _run_metaworld("peg-unplug-side-v3", vec, None, actions=zero[i])["obs"][:20, 4:7]
        drift = np.linalg.norm(mjx_obj[i] - mw_obj, axis=-1).max()
        assert drift < 2e-3, f"plug drifted {drift * 1000:.1f} mm from MetaWorld while at rest"


def test_stick_pull_grasp_reward_matches_metaworld() -> None:
    """The stick must stay in the 2 cm grasp gate while it is inserted and pulled.

    With box-box contacts at one shared depth (Warp's CCD path) the fingers
    closed through the stick and this stage paid ~1/3 of MetaWorld's reward.
    """
    # Grasp geometry matters; these are the first four goals of the Continual
    # World (CW10, seed 42) stick-pull training pool, where the mismatch showed.
    rand_vecs = np.array([
        [-0.07974991947412491, 0.6286692023277283, 0.0005166720366105437, 0.42698511481285095, 0.4799478352069855, 0.019931785762310028],
        [-0.06154647096991539, 0.6403729319572449, 0.00013385368220042437, 0.3806197941303253, 0.5408675670623779, 0.0200518649071455],
        [-0.03426678106188774, 0.6270967721939087, 0.0006511147366836667, 0.4393388032913208, 0.5232818126678467, 0.020024873316287994],
        [-0.04434707388281822, 0.574880838394165, 0.0007327962084673345, 0.44183290004730225, 0.5053320527076721, 0.019906044006347656],
    ])
    env = make("stick-pull-v3", config=CW_CONFIG)
    keys = jax.random.split(jax.random.PRNGKey(0), len(rand_vecs))
    state = jax.jit(jax.vmap(lambda k, v: env.reset(k, rand_vec=v)))(keys, jnp.asarray(rand_vecs, jnp.float32))
    step = jax.jit(jax.vmap(env.step))
    mw = [_run_metaworld("stick-pull-v3", v, policies.SawyerStickPullV3Policy) for v in rand_vecs]
    actions = np.stack([r["action"] for r in mw])
    mjx_reward = _run_mjx(state, step, actions=actions)["reward"]
    # The 40 steps after insertion (MetaWorld's grasped+inserted branch pays >= 6),
    # i.e. the stage that leads to success.
    mw_stage, mjx_stage = [], []
    for i, r in enumerate(mw):
        inserted = np.nonzero(r["reward"] >= 6.0)[0]
        assert inserted.size, "MetaWorld expert is expected to insert the stick"
        window = slice(inserted[0], inserted[0] + 40)
        mw_stage.append(r["reward"][window].mean())
        mjx_stage.append(mjx_reward[i, window].mean())
    mw_stage, mjx_stage = float(np.mean(mw_stage)), float(np.mean(mjx_stage))
    assert 0.8 * mw_stage < mjx_stage < 1.25 * mw_stage, (
        f"insert/pull reward {mjx_stage:.2f} vs MetaWorld {mw_stage:.2f}"
    )
