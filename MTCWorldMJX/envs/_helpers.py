"""Shared helpers for Sawyer XYZ MJX environments."""

from __future__ import annotations

from collections.abc import Callable
import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from mujoco import mjx

from MTCWorldMJX.utils import reward as reward_utils


def as_f32(arr) -> jax.Array:
    return jnp.asarray(arr, dtype=jnp.float32)


def quat_wxyz_to_xyzw(quat: jax.Array) -> jax.Array:
    """Convert MuJoCo wxyz quaternion to scipy/MetaWorld xyzw layout."""
    return jnp.array([quat[1], quat[2], quat[3], quat[0]], dtype=jnp.float32)


def push_xy_positions_from_rand_vec(
    rand_vec: jax.Array,
    *,
    obj_z: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Build push-style XY-sampled object and goal positions."""
    goal_pos = jnp.concatenate([rand_vec[-3:-1], jnp.array([obj_z])])
    obj_init_pos = jnp.concatenate([rand_vec[:2], jnp.array([obj_z])])
    return obj_init_pos, goal_pos


def geom_quat_xyzw(data, geom_id: int) -> jax.Array:
    """Object quaternion in MetaWorld xyzw layout from geom orientation."""
    from MTCWorldMJX import mjx_env

    quat_wxyz = mjx_env.geom_xquat(data, jnp.array([geom_id]))[0]
    return quat_wxyz_to_xyzw(quat_wxyz)


def xmat_to_quat_wxyz(xmat: jax.Array) -> jax.Array:
    """Quaternion (w, x, y, z) from a 3x3 rotation matrix."""
    xmat = xmat.reshape(3, 3)
    w = jnp.sqrt(jnp.maximum(1.0 + xmat[0, 0] + xmat[1, 1] + xmat[2, 2], 0.0)) / 2.0
    x = jnp.sign(xmat[2, 1] - xmat[1, 2]) * jnp.sqrt(
        jnp.maximum(1.0 + xmat[0, 0] - xmat[1, 1] - xmat[2, 2], 0.0)
    ) / 2.0
    y = jnp.sign(xmat[0, 2] - xmat[2, 0]) * jnp.sqrt(
        jnp.maximum(1.0 - xmat[0, 0] + xmat[1, 1] - xmat[2, 2], 0.0)
    ) / 2.0
    z = jnp.sign(xmat[1, 0] - xmat[0, 1]) * jnp.sqrt(
        jnp.maximum(1.0 - xmat[0, 0] - xmat[1, 1] + xmat[2, 2], 0.0)
    ) / 2.0
    return jnp.array([w, x, y, z], dtype=jnp.float32)


def site_quat_xyzw(data: mjx.Data, site_id: int) -> jax.Array:
    """Site orientation quaternion in MetaWorld xyzw layout."""
    return quat_wxyz_to_xyzw(xmat_to_quat_wxyz(data.site_xmat[site_id]))


def pack_interleaved_obj_obs(
    pos: jax.Array,
    quat: jax.Array,
    *,
    max_len: int = 14,
) -> jax.Array:
    """Interleave per-object (pos, quat) blocks like MetaWorld."""
    n_objs = pos.size // 3
    pos_rows = pos.reshape(n_objs, 3)
    quat_rows = quat.reshape(-1, 4)
    packed = jnp.concatenate(
        [jnp.concatenate([pos_rows[i], quat_rows[i]]) for i in range(n_objs)]
    )
    out = jnp.zeros(max_len, dtype=jnp.float32)
    return out.at[: packed.size].set(packed)


def set_joint_qpos(
    data: mjx.Data,
    qposadr: int,
    qveladr: int,
    value: jax.Array,
) -> mjx.Data:
    """Set a scalar joint position and zero its velocity."""
    qpos = data.qpos.at[qposadr].set(value)
    qvel = data.qvel.at[qveladr].set(0.0)
    return data.replace(qpos=qpos, qvel=qvel)


def set_state_forward(
    model: mjx.Model,
    data: mjx.Data,
    qposadr: int,
    qveladr: int,
    value: jax.Array,
) -> mjx.Data:
    """Set a joint position and forward, matching MetaWorld ``set_state``."""
    data = set_joint_qpos(data, qposadr, qveladr, value)
    return mjx.forward(model, data)


def run_cpu_hand_settle(
    mj_model: mujoco.MjModel,
    qpos: np.ndarray,
    qvel: np.ndarray,
    hand_init_pos: np.ndarray,
    *,
    steps: int,
    frame_skip: int,
) -> mujoco.MjData:
    """Replicate MetaWorld ``_reset_hand`` settling on CPU MuJoCo."""
    mj_data = mujoco.MjData(mj_model)
    mj_data.qpos[:] = np.asarray(qpos, dtype=np.float64)
    mj_data.qvel[:] = np.asarray(qvel, dtype=np.float64)
    mocap_id = int(mj_model.body_mocapid[mj_model.body("mocap").id])
    ctrl = np.array([-1.0, 1.0], dtype=np.float64)
    hand_pos = np.asarray(hand_init_pos, dtype=np.float64)
    mocap_quat = np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float64)
    for _ in range(steps):
        mj_data.mocap_pos[mocap_id] = hand_pos
        mj_data.mocap_quat[mocap_id] = mocap_quat
        mj_data.ctrl[:] = ctrl
        for _ in range(frame_skip):
            mujoco.mj_step(mj_model, mj_data)
    return mj_data


def run_cpu_physics(
    mj_model: mujoco.MjModel,
    qpos: np.ndarray,
    qvel: np.ndarray,
    configure: Callable[[mujoco.MjModel, mujoco.MjData], None],
    *,
    steps: int,
    ctrl: np.ndarray | None = None,
) -> mujoco.MjData:
    """Run CPU MuJoCo steps after a reset configure hook."""
    mj_data = mujoco.MjData(mj_model)
    mj_data.qpos[:] = np.asarray(qpos, dtype=np.float64)
    mj_data.qvel[:] = np.asarray(qvel, dtype=np.float64)
    configure(mj_model, mj_data)
    if ctrl is None:
        ctrl = np.array([-1.0, 1.0], dtype=np.float64)
    for _ in range(steps):
        mj_data.ctrl[:] = ctrl
        mujoco.mj_step(mj_model, mj_data)
    return mj_data


def sync_mjx_kinematics(mjx_data: mjx.Data, mj_data: mujoco.MjData) -> mjx.Data:
    """Copy CPU MuJoCo kinematics into an MJX data object."""
    mocap_pos = None
    if mj_data.mocap_pos.size:
        mocap_pos = jnp.asarray(mj_data.mocap_pos, dtype=jnp.float32)
    out = mjx_data.replace(
        qpos=jnp.asarray(mj_data.qpos, dtype=jnp.float32),
        qvel=jnp.asarray(mj_data.qvel, dtype=jnp.float32),
        xpos=jnp.asarray(mj_data.xpos, dtype=jnp.float32),
        xquat=jnp.asarray(mj_data.xquat, dtype=jnp.float32),
        site_xpos=jnp.asarray(mj_data.site_xpos, dtype=jnp.float32),
    )
    if mocap_pos is not None:
        out = out.replace(mocap_pos=mocap_pos)
    return out


def read_mj_site_xpos(mj_data: mujoco.MjData, site_id: int) -> np.ndarray:
    """Read a site position from CPU data without an explicit forward pass."""
    return np.asarray(mj_data.site_xpos[site_id], dtype=np.float32)


def read_mj_body_xpos(mj_data: mujoco.MjData, body_id: int) -> np.ndarray:
    """Read a body position from CPU data without an explicit forward pass."""
    return np.asarray(mj_data.xpos[body_id], dtype=np.float32)


def physics_steps(
    model: mjx.Model,
    data: mjx.Data,
    *,
    steps: int,
    ctrl: jax.Array | None = None,
) -> mjx.Data:
    """Step physics with fixed control."""
    if ctrl is None:
        ctrl = jnp.array([-1.0, 1.0])

    def body_fn(carry, _):
        carry = carry.replace(ctrl=ctrl)
        return mjx.step(model, carry), None

    data, _ = jax.lax.scan(body_fn, data, None, length=steps)
    return data


def set_free_joint_xyz(
    data: mjx.Data,
    qposadr: int,
    qveladr: int,
    pos: jax.Array,
    *,
    ndim: int = 3,
) -> mjx.Data:
    """Set free-joint (or slide) position and zero velocity."""
    qpos = data.qpos.at[qposadr : qposadr + ndim].set(pos[:ndim])
    qvel = data.qvel.at[qveladr : qveladr + ndim + 3].set(0.0)
    return data.replace(qpos=qpos, qvel=qvel)


def set_free_joint_pos_zero_linear_vel(
    data: mjx.Data,
    qposadr: int,
    qveladr: int,
    pos: jax.Array,
    quat_wxyz: jax.Array | None = None,
) -> mjx.Data:
    """Match MetaWorld peg-unplug ``_set_obj_xyz`` (linear qvel only)."""
    qpos = data.qpos.at[qposadr : qposadr + 3].set(pos[:3])
    if quat_wxyz is not None:
        qpos = qpos.at[qposadr + 3 : qposadr + 7].set(quat_wxyz)
    qvel = data.qvel.at[qveladr : qveladr + 3].set(0.0)
    return data.replace(qpos=qpos, qvel=qvel)


def calibrate_hand_settle_geom_z_offset(
    mj_model: mujoco.MjModel,
    mjx_model: mjx.Model,
    init_qpos: jax.Array,
    init_qvel: jax.Array,
    hand_init_pos: jax.Array,
    geom_id: int,
    *,
    hand_reset_steps: int,
    frame_skip: int,
    settle_hand,
) -> jnp.ndarray:
    """CPU-vs-warp geom-z offset after hand settle (init-time calibration)."""
    from MTCWorldMJX import mjx_env

    mj_data = run_cpu_hand_settle(
        mj_model,
        np.asarray(init_qpos),
        np.asarray(init_qvel),
        np.asarray(hand_init_pos),
        steps=hand_reset_steps,
        frame_skip=frame_skip,
    )
    cpu_z = float(mj_data.geom_xpos[geom_id, 2])
    data = mjx_env.make_data(mj_model, mjx_model, qpos=init_qpos, qvel=init_qvel)
    data, _ = settle_hand(mjx_model, data)
    warp_z = float(np.asarray(data.geom_xpos[geom_id, 2]))
    return jnp.asarray(cpu_z - warp_z, dtype=jnp.float32)


def set_slide_joint_xy(data: mjx.Data, qposadr: int, qveladr: int, xy: jax.Array) -> mjx.Data:
    """Set a 2D slide joint and zero its velocities."""
    qpos = data.qpos.at[qposadr : qposadr + 2].set(xy[:2])
    qvel = data.qvel.at[qveladr : qveladr + 2].set(0.0)
    return data.replace(qpos=qpos, qvel=qvel)


def sample_hstack_rand_vec(
    rng: jax.Array,
    low: jax.Array,
    high: jax.Array,
    *,
    min_xy_dist: float = 0.15,
) -> tuple[jax.Array, jax.Array]:
    """Sample 6D [obj, goal] vector with optional XY separation constraint."""
    key, subkey = jax.random.split(rng)
    vec = jax.random.uniform(subkey, shape=low.shape, minval=low, maxval=high)

    def cond_fn(carry):
        v, _ = carry
        return jnp.linalg.norm(v[:2] - v[-3:-1]) < min_xy_dist

    def body_fn(carry):
        v, key = carry
        key, subkey = jax.random.split(key)
        v = jax.random.uniform(subkey, shape=low.shape, minval=low, maxval=high)
        return v, key

    vec, _ = jax.lax.while_loop(cond_fn, body_fn, (vec, key))
    return vec, key


def sample_obj_only_rand_vec(
    rng: jax.Array,
    low: jax.Array,
    high: jax.Array,
) -> jax.Array:
    """Sample a 3D object-only reset vector."""
    return jax.random.uniform(rng, shape=low.shape, minval=low, maxval=high)


def set_slide_joint(
    data: mjx.Data,
    qposadr: int,
    qveladr: int,
    pos: jax.Array,
) -> mjx.Data:
    """Set a 1-DoF slide/hinge joint position and zero its velocity."""
    qpos = data.qpos.at[qposadr].set(pos)
    qvel = data.qvel.at[qveladr].set(0.0)
    return data.replace(qpos=qpos, qvel=qvel)


def button_press_reward_v2(
    obj: jax.Array,
    tcp: jax.Array,
    target_y: jax.Array,
    tcp_opened: jax.Array,
    init_tcp: jax.Array,
    obj_to_target_init: jax.Array,
    *,
    near_bounds: tuple[float, float] = (0.0, 0.05),
    tcp_closed_from_open: bool = True,
    near_tcp_mult: float = 2.0,
    press_tcp_thresh: float = 0.05,
    press_mult: float = 8.0,
    success_thresh: float = 0.02,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Shared v2 reward for side-facing button press tasks."""
    tcp_to_obj = jnp.linalg.norm(obj - tcp)
    tcp_to_obj_init = jnp.linalg.norm(obj - init_tcp)
    obj_to_target = jnp.abs(target_y - obj[1])

    tcp_closed = jnp.maximum(tcp_opened, 0.0) if tcp_closed_from_open else 1.0 - tcp_opened
    near_button = reward_utils.tolerance(
        tcp_to_obj,
        bounds=near_bounds,
        margin=tcp_to_obj_init,
        sigmoid="long_tail",
    )
    button_pressed = reward_utils.tolerance(
        obj_to_target,
        bounds=(0.0, 0.005),
        margin=obj_to_target_init,
        sigmoid="long_tail",
    )

    reward = near_tcp_mult * reward_utils.hamacher_product(tcp_closed, near_button)
    reward = jnp.where(
        tcp_to_obj <= press_tcp_thresh,
        reward + press_mult * button_pressed,
        reward,
    )

    metrics = {
        "success": (obj_to_target <= success_thresh).astype(jnp.float32),
        "near_object": (tcp_to_obj <= 0.05).astype(jnp.float32),
        "grasp_success": (tcp_opened > 0.0).astype(jnp.float32),
        "grasp_reward": near_button,
        "in_place_reward": button_pressed,
        "obj_to_target": obj_to_target,
    }
    return reward, metrics


def button_press_wall_reward_v2(
    obj: jax.Array,
    tcp: jax.Array,
    target_y: jax.Array,
    tcp_opened: jax.Array,
    init_tcp: jax.Array,
    obj_to_target_init: jax.Array,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """v2 reward for button-press-wall-v3."""
    tcp_to_obj = jnp.linalg.norm(obj - tcp)
    tcp_to_obj_init = jnp.linalg.norm(obj - init_tcp)
    obj_to_target = jnp.abs(target_y - obj[1])

    near_button = reward_utils.tolerance(
        tcp_to_obj,
        bounds=(0.0, 0.01),
        margin=tcp_to_obj_init,
        sigmoid="long_tail",
    )
    button_pressed = reward_utils.tolerance(
        obj_to_target,
        bounds=(0.0, 0.005),
        margin=obj_to_target_init,
        sigmoid="long_tail",
    )

    tcp_status = (1.0 - tcp_opened) / 2.0
    reward = jnp.where(
        tcp_to_obj > 0.07,
        2.0 * reward_utils.hamacher_product(tcp_status, near_button),
        2.0 + 2.0 * (1.0 + tcp_opened) + 4.0 * button_pressed**2,
    )

    metrics = {
        "success": (obj_to_target <= 0.03).astype(jnp.float32),
        "near_object": (tcp_to_obj <= 0.05).astype(jnp.float32),
        "grasp_success": (tcp_opened > 0.0).astype(jnp.float32),
        "grasp_reward": near_button,
        "in_place_reward": button_pressed,
        "obj_to_target": obj_to_target,
    }
    return reward, metrics


def button_press_topdown_reward_v2(
    obj: jax.Array,
    tcp: jax.Array,
    target_z: jax.Array,
    tcp_opened: jax.Array,
    init_tcp: jax.Array,
    obj_to_target_init: jax.Array,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """v2 reward for top-down button press tasks."""
    tcp_to_obj = jnp.linalg.norm(obj - tcp)
    tcp_to_obj_init = jnp.linalg.norm(obj - init_tcp)
    obj_to_target = jnp.abs(target_z - obj[2])

    tcp_closed = 1.0 - tcp_opened
    near_button = reward_utils.tolerance(
        tcp_to_obj,
        bounds=(0.0, 0.01),
        margin=tcp_to_obj_init,
        sigmoid="long_tail",
    )
    button_pressed = reward_utils.tolerance(
        obj_to_target,
        bounds=(0.0, 0.005),
        margin=obj_to_target_init,
        sigmoid="long_tail",
    )

    reward = 5.0 * reward_utils.hamacher_product(tcp_closed, near_button)
    reward = jnp.where(tcp_to_obj <= 0.03, reward + 5.0 * button_pressed, reward)

    metrics = {
        "success": (obj_to_target <= 0.024).astype(jnp.float32),
        "near_object": (tcp_to_obj <= 0.05).astype(jnp.float32),
        "grasp_success": (tcp_opened > 0.0).astype(jnp.float32),
        "grasp_reward": near_button,
        "in_place_reward": button_pressed,
        "obj_to_target": obj_to_target,
    }
    return reward, metrics


def set_coffee_mug_pos(
    data: mjx.Data,
    qposadr: int,
    pos: jax.Array,
    *,
    init_qpos: jax.Array | None = None,
) -> mjx.Data:
    """Set coffee mug free-joint XYZ matching MetaWorld ``_set_obj_xyz``."""
    qpos = data.qpos.at[qposadr : qposadr + 3].set(pos[:3])
    if init_qpos is not None:
        qpos = qpos.at[qposadr + 3 : qposadr + 7].set(init_qpos[qposadr + 3 : qposadr + 7])
    qvel = data.qvel.at[0:6].set(0.0)
    qvel = qvel.at[9:15].set(0.0)
    return data.replace(qpos=qpos, qvel=qvel)


def coffee_button_reward_v2(
    obj: jax.Array,
    tcp: jax.Array,
    target_y: jax.Array,
    tcp_opened: jax.Array,
    init_tcp: jax.Array,
    max_dist: jax.Array,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """v2 reward for coffee-button-v3."""
    tcp_to_obj = jnp.linalg.norm(obj - tcp)
    tcp_to_obj_init = jnp.linalg.norm(obj - init_tcp)
    obj_to_target = jnp.abs(target_y - obj[1])

    tcp_closed = jnp.maximum(tcp_opened, 0.0)
    near_button = reward_utils.tolerance(
        tcp_to_obj,
        bounds=(0.0, 0.05),
        margin=tcp_to_obj_init,
        sigmoid="long_tail",
    )
    button_pressed = reward_utils.tolerance(
        obj_to_target,
        bounds=(0.0, 0.005),
        margin=max_dist,
        sigmoid="long_tail",
    )

    reward = 2.0 * reward_utils.hamacher_product(tcp_closed, near_button)
    reward = jnp.where(
        tcp_to_obj <= 0.05,
        reward + 8.0 * button_pressed,
        reward,
    )

    metrics = {
        "success": (obj_to_target <= 0.02).astype(jnp.float32),
        "near_object": (tcp_to_obj <= 0.05).astype(jnp.float32),
        "grasp_success": (tcp_opened > 0.0).astype(jnp.float32),
        "grasp_reward": near_button,
        "in_place_reward": button_pressed,
        "obj_to_target": obj_to_target,
    }
    return reward, metrics


def coffee_mug_manipulation_reward_v2(
    env,
    data: mjx.Data,
    action: jax.Array,
    obj: jax.Array,
    target: jax.Array,
    obj_init_pos: jax.Array,
    init_tcp: jax.Array,
    tcp_opened: jax.Array,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """v2 reward shared by coffee-push-v3 and coffee-pull-v3."""
    scale = jnp.array([2.0, 2.0, 1.0], dtype=jnp.float32)
    target_to_obj = jnp.linalg.norm((obj - target) * scale)
    target_to_obj_init = jnp.linalg.norm((obj_init_pos - target) * scale)

    in_place = reward_utils.tolerance(
        target_to_obj,
        bounds=(0.0, 0.05),
        margin=target_to_obj_init,
        sigmoid="long_tail",
    )
    tcp = env._tcp_center(data)
    tcp_to_obj = jnp.linalg.norm(obj - tcp)

    object_grasped = env._gripper_caging_reward(
        data,
        action,
        obj,
        obj_init_pos,
        init_tcp,
        obj_radius=0.02,
        pad_success_thresh=0.05,
        object_reach_radius=0.04,
        xz_thresh=0.05,
        desired_gripper_effort=0.7,
        medium_density=True,
    )

    reward = reward_utils.hamacher_product(object_grasped, in_place)
    reward = jnp.where(
        (tcp_to_obj < 0.04) & (tcp_opened > 0.0),
        reward + 1.0 + 5.0 * in_place,
        reward,
    )
    reward = jnp.where(target_to_obj < 0.05, 10.0, reward)

    obj_to_target = jnp.linalg.norm(obj - target)
    metrics = {
        "success": (obj_to_target <= 0.07).astype(jnp.float32),
        "near_object": (tcp_to_obj <= 0.03).astype(jnp.float32),
        "grasp_success": (object_grasped >= 0.5).astype(jnp.float32),
        "grasp_reward": object_grasped,
        "in_place_reward": in_place,
        "obj_to_target": obj_to_target,
    }
    return reward, metrics


def reach_reward(
    tcp: jax.Array,
    target: jax.Array,
    hand_init: jax.Array,
    *,
    target_radius: float = 0.05,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    tcp_to_target = jnp.linalg.norm(tcp - target)
    in_place_margin = jnp.linalg.norm(hand_init - target)
    in_place = reward_utils.tolerance(
        tcp_to_target,
        bounds=(0.0, target_radius),
        margin=in_place_margin,
        sigmoid="long_tail",
    )
    reward = 10.0 * in_place
    success = (tcp_to_target <= target_radius).astype(jnp.float32)
    return reward, tcp_to_target, in_place


def xz_plane_gripper_caging_reward(
    data,
    action: jax.Array,
    obj_pos: jax.Array,
    obj_init_pos: jax.Array,
    init_tcp: jax.Array,
    init_left_pad: jax.Array,
    init_right_pad: jax.Array,
    leftpad_body_id: int,
    rightpad_body_id: int,
    tcp: jax.Array,
    *,
    obj_radius: float,
    pad_success_margin: float = 0.05,
    grip_success_extra: float = 0.01,
    xz_success_margin: float = 0.005,
) -> jax.Array:
    """Gripper caging for soccer/sweep tasks using XZ-plane projection."""
    from MTCWorldMJX import mjx_env

    del action
    left_pad = mjx_env.body_xpos(data, jnp.array(leftpad_body_id))
    right_pad = mjx_env.body_xpos(data, jnp.array(rightpad_body_id))
    grip_margin = obj_radius + grip_success_extra

    right_caging = reward_utils.tolerance(
        obj_pos[1] - right_pad[1],
        bounds=(obj_radius, pad_success_margin),
        margin=jnp.abs(jnp.abs(obj_pos[1] - init_right_pad[1]) - pad_success_margin),
        sigmoid="long_tail",
    )
    left_caging = reward_utils.tolerance(
        left_pad[1] - obj_pos[1],
        bounds=(obj_radius, pad_success_margin),
        margin=jnp.abs(jnp.abs(obj_pos[1] - init_left_pad[1]) - pad_success_margin),
        sigmoid="long_tail",
    )
    right_grip = reward_utils.tolerance(
        obj_pos[1] - right_pad[1],
        bounds=(obj_radius, grip_margin),
        margin=jnp.abs(jnp.abs(obj_pos[1] - init_right_pad[1]) - pad_success_margin),
        sigmoid="long_tail",
    )
    left_grip = reward_utils.tolerance(
        left_pad[1] - obj_pos[1],
        bounds=(obj_radius, grip_margin),
        margin=jnp.abs(jnp.abs(obj_pos[1] - init_left_pad[1]) - pad_success_margin),
        sigmoid="long_tail",
    )
    y_caging = reward_utils.hamacher_product(right_caging, left_caging)
    y_grip = reward_utils.hamacher_product(right_grip, left_grip)

    tcp_xz = tcp + jnp.array([0.0, -tcp[1], 0.0])
    obj_xz = obj_pos + jnp.array([0.0, -obj_pos[1], 0.0])
    init_obj_xz = obj_init_pos + jnp.array([0.0, -obj_init_pos[1], 0.0])
    init_tcp_xz = init_tcp + jnp.array([0.0, -init_tcp[1], 0.0])
    xz_margin = jnp.linalg.norm(init_obj_xz - init_tcp_xz) - xz_success_margin
    x_z_caging = reward_utils.tolerance(
        jnp.linalg.norm(tcp_xz - obj_xz),
        bounds=(0.0, xz_success_margin),
        margin=xz_margin,
        sigmoid="long_tail",
    )
    caging = reward_utils.hamacher_product(y_caging, x_z_caging)
    gripping = jnp.where(caging > 0.95, y_grip, 0.0)
    return (caging + gripping) / 2.0
