"""Base Sawyer XYZ MetaWorld environments implemented with MJX."""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from mujoco import mjx

from MTCWorldMJX import mjx_env
from MTCWorldMJX.utils import reward as reward_utils


MOCAP_QUAT = jnp.array([1.0, 0.0, 1.0, 0.0])
HAND_SPACE_LOW = jnp.array([-0.525, 0.348, -0.0525])
HAND_SPACE_HIGH = jnp.array([0.525, 1.025, 0.7])
OBS_OBJ_MAX_LEN = 14
CURR_OBS_DIM = 18
OBS_DIM = 39


@dataclass(frozen=True)
class SawyerXYZConfig:
    ctrl_dt: float = 0.0125
    sim_dt: float = 0.0025
    frame_skip: int = 5
    max_path_length: int = 500
    action_scale: float = 0.01
    partially_observable: bool = True
    impl: str = "warp"
    naconmax: int = 2000
    njmax: int = 1000
    hand_reset_steps: int = 50
    solver_iterations: int = 6
    ls_iterations: int = 6
    zero_geom_margins: bool = True


class SawyerXYZEnv(abc.ABC):
    """JAX-compatible base class for Sawyer XYZ MetaWorld environments."""

    def __init__(self, config: SawyerXYZConfig | None = None):
        self.config = config or SawyerXYZConfig()
        self._mj_model = mjx_env.load_mj_model(
            str(self.xml_path),
            timestep=self.config.sim_dt,
            solver_iterations=self.config.solver_iterations,
            ls_iterations=self.config.ls_iterations,
            zero_geom_margins=self.config.zero_geom_margins,
        )
        self._reset_mocap_welds(self._mj_model)
        self._mjx_model = mjx.put_model(self._mj_model, impl=self.config.impl)
        self._static_geoms = mjx_env.StaticGeoms(self._mj_model)
        self._post_init_ids()
        self._init_qpos = jnp.array(self._mj_model.qpos0)
        self._init_qvel = jnp.zeros(self._mj_model.nv)

    @property
    @abc.abstractmethod
    def xml_path(self):
        """Path to the task MJCF file."""

    def _post_init_ids(self) -> None:
        m = self._mj_model
        self._mocap_id = m.body_mocapid[mjx_env.body_id(m, "mocap")]
        self._hand_body_id = mjx_env.body_id(m, "hand")
        self._rightclaw_body_id = mjx_env.body_id(m, "rightclaw")
        self._leftclaw_body_id = mjx_env.body_id(m, "leftclaw")
        self._rightpad_body_id = mjx_env.body_id(m, "rightpad")
        self._leftpad_body_id = mjx_env.body_id(m, "leftpad")
        self._right_ee_site_id = mjx_env.site_id(m, "rightEndEffector")
        self._left_ee_site_id = mjx_env.site_id(m, "leftEndEffector")
        self._obj_qposadr = int(m.jnt_qposadr[-1])
        self._obj_qveladr = int(m.jnt_dofadr[-1])

    @staticmethod
    def _reset_mocap_welds(model: mujoco.MjModel) -> None:
        if model.nmocap > 0 and model.eq_data is not None:
            for i in range(model.eq_data.shape[0]):
                if model.eq_type[i] == mujoco.mjtEq.mjEQ_WELD:
                    model.eq_data[i] = np.array(
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 5.0]
                    )

    @property
    def mj_model(self) -> mujoco.MjModel:
        return self._mj_model

    @property
    def mjx_model(self) -> mjx.Model:
        return self._mjx_model

    @property
    def action_size(self) -> int:
        return 4

    @property
    def observation_size(self) -> int:
        return OBS_DIM

    @property
    def dt(self) -> float:
        return self.config.ctrl_dt

    @property
    def n_substeps(self) -> int:
        return self.config.frame_skip

    @abc.abstractmethod
    def hand_init_pos(self) -> jax.Array:
        """Default reset position for the mocap body."""

    @abc.abstractmethod
    def mocap_low(self) -> jax.Array:
        """Lower bounds for mocap XYZ control."""

    @abc.abstractmethod
    def mocap_high(self) -> jax.Array:
        """Upper bounds for mocap XYZ control."""

    @classmethod
    def goal_space_bounds(cls) -> tuple[np.ndarray, np.ndarray]:
        """MetaWorld's ``goal_space``: the clip range of the visible goal observation."""
        from MTCWorldMJX.env_dict import ENV_CLS_MAP
        from MTCWorldMJX.envs._goal_spaces import GOAL_SPACES

        names = {klass: name for name, klass in ENV_CLS_MAP.items()}
        for klass in cls.__mro__:
            if klass in names:
                low, high = GOAL_SPACES[names[klass]]
                return np.asarray(low), np.asarray(high)
        return np.full(3, -np.inf), np.full(3, np.inf)

    @abc.abstractmethod
    def random_reset_bounds(self) -> tuple[jax.Array, jax.Array]:
        """Inclusive bounds for sampling ``[obj_init_pos, goal_pos]``."""

    @abc.abstractmethod
    def sample_reset_state(self, rng: jax.Array) -> dict[str, jax.Array]:
        """Sample task-specific reset variables."""

    @abc.abstractmethod
    def apply_reset_state(
        self,
        model: mjx.Model,
        data: mjx.Data,
        reset_state: dict[str, jax.Array],
    ) -> tuple[mjx.Model, mjx.Data]:
        """Apply sampled reset state to model and data."""

    @abc.abstractmethod
    def get_obj_pos(self, data: mjx.Data) -> jax.Array:
        """Object position used in observations."""

    @abc.abstractmethod
    def get_obj_quat(self, data: mjx.Data) -> jax.Array:
        """Object quaternion used in observations."""

    @abc.abstractmethod
    def compute_reward(
        self,
        data: mjx.Data,
        action: jax.Array,
        info: dict[str, Any],
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        """Return scalar reward and metric dict."""

    def _tcp_center(self, data: mjx.Data) -> jax.Array:
        sites = jnp.array([self._right_ee_site_id, self._left_ee_site_id])
        ee_pos = mjx_env.site_xpos(data, sites)
        return ee_pos.mean(axis=0)

    def _gripper_opening(self, data: mjx.Data) -> jax.Array:
        bodies = jnp.array([self._rightclaw_body_id, self._leftclaw_body_id])
        claw_pos = mjx_env.body_xpos(data, bodies)
        dist = jnp.linalg.norm(claw_pos[0] - claw_pos[1])
        return jnp.clip(dist / 0.1, 0.0, 1.0)

    def _hand_pos(self, data: mjx.Data) -> jax.Array:
        return data.xpos[self._hand_body_id]

    def _get_curr_obs_no_goal(self, data: mjx.Data) -> jax.Array:
        pos_hand = self._hand_pos(data)
        gripper = self._gripper_opening(data)[None]
        obj_pos = self.get_obj_pos(data)
        obj_quat = self.get_obj_quat(data)
        obs_obj = jnp.zeros(OBS_OBJ_MAX_LEN)
        obs_obj = obs_obj.at[:7].set(jnp.concatenate([obj_pos, obj_quat]))
        return jnp.concatenate([pos_hand, gripper, obs_obj])

    def _get_obs(
        self,
        data: mjx.Data,
        prev_obs: jax.Array,
        goal_pos: jax.Array,
    ) -> jax.Array:
        curr_obs = self._get_curr_obs_no_goal(data)
        if self.config.partially_observable:
            goal = jnp.zeros_like(goal_pos)
        else:
            goal = goal_pos
        return jnp.concatenate([curr_obs, prev_obs, goal])

    def _set_mocap_pos(self, data: mjx.Data, pos: jax.Array) -> mjx.Data:
        mocap_pos = data.mocap_pos.at[self._mocap_id].set(pos)
        mocap_quat = data.mocap_quat.at[self._mocap_id].set(MOCAP_QUAT)
        return data.replace(mocap_pos=mocap_pos, mocap_quat=mocap_quat)

    def _apply_xyz_action(self, data: mjx.Data, action: jax.Array) -> mjx.Data:
        action = jnp.clip(action[:3], -1.0, 1.0)
        delta = action * self.config.action_scale
        new_pos = jnp.clip(
            data.mocap_pos[self._mocap_id] + delta,
            self.mocap_low(),
            self.mocap_high(),
        )
        return self._set_mocap_pos(data, new_pos)

    def _gripper_ctrl(self, action: jax.Array) -> jax.Array:
        g = jnp.clip(action[-1], -1.0, 1.0)
        return jnp.array([g, -g])

    def _settle_hand(self, model: mjx.Model, data: mjx.Data) -> tuple[mjx.Data, jax.Array]:
        hand_pos = self.hand_init_pos()
        ctrl = jnp.array([-1.0, 1.0])

        def body_fn(carry, _):
            carry = self._set_mocap_pos(carry, hand_pos)
            carry = mjx_env.step(model, carry, ctrl, self.n_substeps)
            return carry, None

        data, _ = jax.lax.scan(body_fn, data, None, length=self.config.hand_reset_steps)
        init_tcp = self._tcp_center(data)
        return data, init_tcp

    def _sample_rand_vec(self, rng: jax.Array, fixed: jax.Array | None) -> jax.Array:
        if fixed is not None:
            return fixed
        low, high = self.random_reset_bounds()
        return jax.random.uniform(rng, shape=low.shape, minval=low, maxval=high)

    def reset_state_from_rand_vec(self, rand_vec: jax.Array) -> dict[str, jax.Array]:
        """Build reset kwargs from a MetaWorld-compatible rand vec."""
        del rand_vec
        raise NotImplementedError

    def prepare_reset_state(
        self,
        model: mjx.Model,
        data: mjx.Data,
        reset_state: dict[str, jax.Array],
    ) -> tuple[mjx.Model, mjx.Data, dict[str, jax.Array]]:
        """Optional hook after hand settle, before ``apply_reset_state``."""
        return model, data, reset_state

    def finalize_reset(
        self,
        model: mjx.Model,
        data: mjx.Data,
        reset_state: dict[str, jax.Array],
    ) -> tuple[mjx.Model, mjx.Data, dict[str, jax.Array]]:
        """Optional hook after ``apply_reset_state`` (forward / sim steps)."""
        return model, data, reset_state

    def _resolve_goal_pos(
        self,
        model: mjx.Model,
        data: mjx.Data,
        reset_state: dict[str, jax.Array],
    ) -> jax.Array:
        """Return goal position after forward kinematics (override for site-based goals)."""
        del model, data
        return reset_state["goal_pos"]

    def _uses_metaworld_double_reset(self) -> bool:
        """MetaWorld ``reset()`` calls ``reset_model``, ``mj_resetData``, ``reset_model``."""
        return False

    def reset(
        self,
        rng: jax.Array,
        rand_vec: jax.Array | None = None,
    ) -> mjx_env.State:
        rng, reset_rng, vec_rng = jax.random.split(rng, 3)
        if rand_vec is not None:
            reset_state = self.reset_state_from_rand_vec(rand_vec)
        else:
            reset_state = self.sample_reset_state(reset_rng)

        qpos = self._init_qpos
        qvel = self._init_qvel
        data = mjx_env.make_data(
            self._mj_model,
            self._mjx_model,
            qpos=qpos,
            qvel=qvel,
            naconmax=self.config.naconmax,
            njmax=self.config.njmax,
        )
        model = self._mjx_model
        if self._uses_metaworld_double_reset():
            data, _ = self._settle_hand(model, data)
            model, data = self.apply_reset_state(model, data, reset_state)
            data = self._static_geoms.refresh(model, data)
            data = data.replace(qpos=self._init_qpos, qvel=self._init_qvel)
            data, init_tcp = self._settle_hand(model, data)
        else:
            data, init_tcp = self._settle_hand(model, data)
        init_left_pad = data.xpos[self._leftpad_body_id]
        init_right_pad = data.xpos[self._rightpad_body_id]
        model, data, reset_state = self.prepare_reset_state(
            model, data, reset_state
        )
        model, data = self.apply_reset_state(model, data, reset_state)
        data = self._static_geoms.refresh(model, data)
        model, data, reset_state = self.finalize_reset(model, data, reset_state)
        data = self._static_geoms.refresh(model, data)
        if not reset_state.get("skip_final_forward", False):
            data = mjx.forward(model, data)

        goal_pos = self._resolve_goal_pos(model, data, reset_state)
        prev_obs = self._get_curr_obs_no_goal(data)
        obs = self._get_obs(data, prev_obs, goal_pos)

        info = {
            "rng": rng,
            "prev_obs": prev_obs,
            "goal_pos": goal_pos,
            "obj_init_pos": reset_state["obj_init_pos"],
            "init_tcp": init_tcp,
            "init_left_pad": init_left_pad,
            "init_right_pad": init_right_pad,
            "path_length": jnp.array(0, dtype=jnp.int32),
            "rand_vec": self._sample_rand_vec(vec_rng, reset_state.get("rand_vec")),
        }
        info.update(
            {
                k: v
                for k, v in reset_state.items()
                if k not in ("obj_init_pos", "goal_pos", "rand_vec", "skip_final_forward")
            }
        )
        metrics = {
            "success": jnp.array(0.0),
            "near_object": jnp.array(0.0),
            "grasp_success": jnp.array(0.0),
            "grasp_reward": jnp.array(0.0),
            "in_place_reward": jnp.array(0.0),
            "obj_to_target": jnp.array(0.0),
            "unscaled_reward": jnp.array(0.0),
        }
        return mjx_env.State(
            model=model,
            data=data,
            obs=obs,
            reward=jnp.array(0.0),
            truncated=jnp.array(0.0),
            terminated=jnp.array(0.0),
            metrics=metrics,
            info=info,
        )

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        data = self._apply_xyz_action(state.data, action)
        ctrl = self._gripper_ctrl(action)
        data = mjx_env.step(state.model, data, ctrl, self.n_substeps)
        data = mjx.forward(state.model, data)

        curr_obs = self._get_curr_obs_no_goal(data)
        obs = self._get_obs(data, state.info["prev_obs"], state.info["goal_pos"])
        if self.config.partially_observable:
            goal_low = goal_high = jnp.zeros(3)
        else:
            goal_low, goal_high = (jnp.asarray(b, dtype=obs.dtype) for b in self.goal_space_bounds())
        obs = jnp.clip(
            obs,
            jnp.concatenate(
                [
                    HAND_SPACE_LOW,
                    jnp.array([-1.0]),
                    jnp.full(OBS_OBJ_MAX_LEN, -jnp.inf),
                    HAND_SPACE_LOW,
                    jnp.array([-1.0]),
                    jnp.full(OBS_OBJ_MAX_LEN, -jnp.inf),
                    goal_low,
                ]
            ),
            jnp.concatenate(
                [
                    HAND_SPACE_HIGH,
                    jnp.array([1.0]),
                    jnp.full(OBS_OBJ_MAX_LEN, jnp.inf),
                    HAND_SPACE_HIGH,
                    jnp.array([1.0]),
                    jnp.full(OBS_OBJ_MAX_LEN, jnp.inf),
                    goal_high,
                ]
            ),
        )

        reward, step_metrics = self.compute_reward(data, action, state.info)
        path_length = state.info["path_length"] + 1
        truncated = path_length >= self.config.max_path_length

        metrics = state.metrics.copy()
        metrics.update(step_metrics)
        metrics["unscaled_reward"] = reward

        info = dict(state.info)
        info["prev_obs"] = curr_obs
        info["path_length"] = path_length

        return mjx_env.State(
            model=state.model,
            data=data,
            obs=obs,
            reward=reward,
            truncated=truncated.astype(jnp.float32),
            terminated=jnp.array(0.0, dtype=jnp.float32),
            metrics=metrics,
            info=info,
        )

    def _gripper_caging_reward(
        self,
        data: mjx.Data,
        action: jax.Array,
        obj_pos: jax.Array,
        obj_init_pos: jax.Array,
        init_tcp: jax.Array,
        obj_radius: float,
        pad_success_thresh: float,
        object_reach_radius: float,
        xz_thresh: float,
        high_density: bool = False,
        medium_density: bool = False,
        desired_gripper_effort: float = 1.0,
    ) -> jax.Array:
        left_pad = data.xpos[self._leftpad_body_id]
        right_pad = data.xpos[self._rightpad_body_id]
        tcp = self._tcp_center(data)

        pad_y = jnp.array([left_pad[1], right_pad[1]])
        pad_to_obj = jnp.abs(pad_y - obj_pos[1])
        pad_to_obj_init = jnp.abs(pad_y - obj_init_pos[1])
        caging_lr_margin = jnp.abs(pad_to_obj_init - pad_success_thresh)
        caging_lr = jnp.array(
            [
                reward_utils.tolerance(
                    pad_to_obj[0],
                    bounds=(obj_radius, pad_success_thresh),
                    margin=caging_lr_margin[0],
                    sigmoid="long_tail",
                ),
                reward_utils.tolerance(
                    pad_to_obj[1],
                    bounds=(obj_radius, pad_success_thresh),
                    margin=caging_lr_margin[1],
                    sigmoid="long_tail",
                ),
            ]
        )
        caging_y = reward_utils.hamacher_product(caging_lr[0], caging_lr[1])

        xz = jnp.array([0, 2])
        caging_xz_margin = jnp.linalg.norm(obj_init_pos[xz] - init_tcp[xz]) - xz_thresh
        caging_xz = reward_utils.tolerance(
            jnp.linalg.norm(tcp[xz] - obj_pos[xz]),
            bounds=(0.0, xz_thresh),
            margin=caging_xz_margin,
            sigmoid="long_tail",
        )

        gripper_closed = jnp.clip(action[-1], 0.0, desired_gripper_effort) / desired_gripper_effort
        caging = reward_utils.hamacher_product(caging_y, caging_xz)
        gripping = jnp.where(caging > 0.97, gripper_closed, 0.0)
        caging_and_gripping = reward_utils.hamacher_product(caging, gripping)

        if high_density:
            caging_and_gripping = (caging_and_gripping + caging) / 2.0
        if medium_density:
            tcp_to_obj = jnp.linalg.norm(obj_pos - tcp)
            tcp_to_obj_init = jnp.linalg.norm(obj_init_pos - init_tcp)
            reach_margin = jnp.abs(tcp_to_obj_init - object_reach_radius)
            reach = reward_utils.tolerance(
                tcp_to_obj,
                bounds=(0.0, object_reach_radius),
                margin=reach_margin,
                sigmoid="long_tail",
            )
            caging_and_gripping = (caging_and_gripping + reach) / 2.0

        return caging_and_gripping
