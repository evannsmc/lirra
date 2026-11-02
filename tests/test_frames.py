"""Frame and quaternion conventions checked against MuJoCo at known poses."""

import math

import mujoco
import numpy as np
import pytest

from rl_stl.envs.skydio_x2 import frames
from rl_stl.envs.skydio_x2.model import load_x2_model

POSES = {
    "identity": np.array([1.0, 0.0, 0.0, 0.0]),
    "yaw90": np.array([math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4)]),
    "roll90": np.array([math.cos(math.pi / 4), math.sin(math.pi / 4), 0.0, 0.0]),
    "pitch90": np.array([math.cos(math.pi / 4), 0.0, math.sin(math.pi / 4), 0.0]),
    "arbitrary": np.array([0.8, 0.1, -0.3, 0.5]) / np.linalg.norm([0.8, 0.1, -0.3, 0.5]),
}


def mj_mat(q):
    out = np.zeros(9)
    mujoco.mju_quat2Mat(out, q)
    return out.reshape(3, 3)


@pytest.mark.parametrize("name", list(POSES))
def test_quat_to_rotmat_matches_mujoco(name):
    q = POSES[name]
    np.testing.assert_allclose(np.asarray(frames.quat_to_rotmat(q)), mj_mat(q), atol=1e-6)


def test_known_rotations():
    ex, ey, ez = np.eye(3)
    # 90 deg yaw: body x points along world y.
    np.testing.assert_allclose(np.asarray(frames.rotate(POSES["yaw90"], ex)), ey, atol=1e-6)
    # 90 deg roll: body y points along world z, body z along world -y.
    np.testing.assert_allclose(np.asarray(frames.rotate(POSES["roll90"], ey)), ez, atol=1e-6)
    np.testing.assert_allclose(np.asarray(frames.rotate(POSES["roll90"], ez)), -ey, atol=1e-6)
    # 90 deg pitch: body z points along world x.
    np.testing.assert_allclose(np.asarray(frames.rotate(POSES["pitch90"], ez)), ex, atol=1e-6)
    assert abs(float(frames.tilt_cosine(POSES["roll90"]))) < 1e-6


@pytest.mark.parametrize("name", list(POSES))
def test_rotmat_quat_roundtrip(name):
    q = POSES[name]
    q_back = np.asarray(frames.rotmat_to_quat(frames.quat_to_rotmat(q)))
    # q and -q are the same rotation.
    assert min(np.abs(q_back - q).max(), np.abs(q_back + q).max()) < 1e-5


def test_quat_mul_matches_mujoco():
    a, b = POSES["arbitrary"], POSES["yaw90"]
    out = np.zeros(4)
    mujoco.mju_mulQuat(out, a, b)
    np.testing.assert_allclose(np.asarray(frames.quat_mul(a, b)), out, atol=1e-6)
    v = np.array([0.3, -1.2, 2.0])
    rv = np.zeros(3)
    mujoco.mju_rotVecQuat(rv, v, a)
    np.testing.assert_allclose(np.asarray(frames.rotate(a, v)), rv, atol=1e-5)
    np.testing.assert_allclose(np.asarray(frames.rotate_inv(a, frames.rotate(a, v))), v, atol=1e-5)


def test_euler_zyx_roundtrip():
    rpy = np.array([0.3, -0.4, 1.1])
    q = frames.quat_from_euler_zyx(*rpy)
    np.testing.assert_allclose(np.asarray(frames.rotmat_to_euler_zyx(frames.quat_to_rotmat(q))), rpy, atol=1e-5)
    np.testing.assert_allclose(np.asarray(frames.quat_to_rotmat(frames.quat_from_euler_zyx(0.0, 0.0, 0.5))),
                               mj_mat(np.array([math.cos(0.25), 0, 0, math.sin(0.25)])), atol=1e-6)


def test_skew_vee():
    v = np.array([1.0, -2.0, 0.5])
    w = np.array([0.2, 0.1, -3.0])
    np.testing.assert_allclose(np.asarray(frames.skew(v) @ w), np.cross(v, w), atol=1e-6)
    np.testing.assert_allclose(np.asarray(frames.vee(frames.skew(v))), v, atol=1e-6)


@pytest.mark.parametrize("name", ["yaw90", "roll90", "arbitrary"])
def test_free_joint_velocity_frames(name):
    """qvel[0:3] is world-frame linear velocity; qvel[3:6] is body-frame angular velocity."""
    m = load_x2_model()
    d = mujoco.MjData(m)
    q = POSES[name]
    body = m.body("x2").id
    d.qpos[:] = np.concatenate([[0.0, 0.0, 2.0], q])
    lin = np.array([0.4, -0.7, 0.2])
    ang = np.array([0.5, 1.5, -0.8])
    d.qvel[:] = np.concatenate([lin, ang])
    mujoco.mj_forward(m, d)
    vel = np.zeros(6)  # [angular; linear] in world frame at the body frame origin
    mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_XBODY, body, vel, 0)
    np.testing.assert_allclose(vel[3:], lin, atol=1e-9)
    np.testing.assert_allclose(vel[:3], mj_mat(q) @ ang, atol=1e-9)
    # Gyro sensor (site aligned with body) reports body-frame angular velocity.
    np.testing.assert_allclose(d.sensor("body_gyro").data, ang, atol=1e-9)
    # Body <-> world transform helpers agree.
    np.testing.assert_allclose(np.asarray(frames.rotate(q, ang)), vel[:3], atol=1e-5)
    # COM velocity from helper equals MuJoCo's subtree COM velocity.
    r_com = m.body_ipos[body]
    v_com = np.asarray(frames.point_velocity_world(q, lin, ang, r_com))
    mujoco.mj_subtreeVel(m, d)
    np.testing.assert_allclose(v_com, d.subtree_linvel[body], atol=1e-5)
