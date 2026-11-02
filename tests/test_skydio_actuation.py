"""Skydio X2 actuation: hover equilibrium and action-to-wrench map against MuJoCo."""

import mujoco
import numpy as np
import pytest

from rl_stl.envs.skydio_x2.model import (
    MOTOR_NAMES,
    load_x2_model,
    load_x2_spec,
    x2_params_from_model,
)


@pytest.fixture(scope="module")
def model():
    return load_x2_model()


@pytest.fixture(scope="module")
def params(model):
    return x2_params_from_model(model)


def test_model_constants(model, params):
    assert model.nq == 7 and model.nv == 6 and model.nu == 4
    assert model.opt.timestep == pytest.approx(0.01)
    assert params.mass == pytest.approx(1.325)
    assert (params.motor_min, params.motor_max) == (0.0, 13.0)
    np.testing.assert_allclose(model.actuator_gear[:, :3], np.tile([0, 0, 1], (4, 1)))
    # Gear fix (upstream Menagerie 9bac002): diagonal rotors share yaw-drag sign.
    np.testing.assert_allclose(model.actuator_gear[:, 5], [-0.0201, 0.0201, -0.0201, 0.0201])
    assert np.linalg.matrix_rank(params.allocation) == 4
    np.testing.assert_allclose(params.inertia_com_body, params.inertia_com_body.T, atol=1e-12)
    assert np.all(np.linalg.eigvalsh(params.inertia_com_body) > 0)


def test_unfixed_pinned_gear_is_singular():
    spec = load_x2_spec(apply_gear_fix=False)
    gears = [spec.actuator(n).gear[5] for n in MOTOR_NAMES]
    np.testing.assert_allclose(gears, [-0.0201, 0.0201, 0.0201, -0.0201])
    with pytest.raises(ValueError, match="rank-deficient"):
        x2_params_from_model(spec.compile())


@pytest.mark.parametrize("fluid", [True, False])
def test_hover_keyframe_equilibrium(fluid):
    m = load_x2_model(fluid=fluid)
    p = x2_params_from_model(m)
    key_ctrl = m.key_ctrl[0]
    assert key_ctrl.sum() == pytest.approx(p.hover_thrust_total, rel=1e-6)
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    start = d.qpos.copy()
    for _ in range(300):  # 3 s
        mujoco.mj_step(m, d)
    np.testing.assert_allclose(d.qpos, start, atol=1e-6)
    np.testing.assert_allclose(d.qvel, 0.0, atol=1e-6)


def test_hover_with_allocation_inverse(params):
    f = params.allocation_inv @ np.array([params.hover_thrust_total, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(f, params.hover_thrust_per_motor, rtol=1e-9)


def test_fluid_drag_opposes_motion():
    m = load_x2_model(fluid=True)
    d = mujoco.MjData(m)
    d.qpos[:] = [0, 0, 2.0, 1, 0, 0, 0]
    d.qvel[:] = [2.0, 0, 0, 0, 0, 3.0]
    mujoco.mj_forward(m, d)
    assert d.qfrc_passive[0] < 0 and d.qfrc_passive[5] < 0
    m_nofluid = load_x2_model(fluid=False)
    d2 = mujoco.MjData(m_nofluid)
    d2.qpos[:], d2.qvel[:] = d.qpos, d.qvel
    mujoco.mj_forward(m_nofluid, d2)
    np.testing.assert_allclose(d2.qfrc_passive, 0.0, atol=1e-12)


def test_action_to_wrench_matches_mujoco(model, params):
    rng = np.random.default_rng(0)
    d = mujoco.MjData(model)
    body = model.body("x2").id
    for _ in range(25):
        quat = rng.normal(size=4)
        quat /= np.linalg.norm(quat)
        d.qpos[:] = np.concatenate([rng.normal(size=3) + [0, 0, 3.0], quat])
        d.qvel[:] = 0.0
        d.ctrl[:] = rng.uniform(params.motor_min, params.motor_max, size=4)
        mujoco.mj_forward(model, d)
        rot = d.xmat[body].reshape(3, 3)
        # Free joint generalized force: world force on translational dofs,
        # body-frame torque about the body origin on rotational dofs.
        force_body = rot.T @ d.qfrc_actuator[:3]
        torque_origin_body = d.qfrc_actuator[3:6]
        torque_com_body = torque_origin_body - np.cross(params.com_body, force_body)
        wrench = params.allocation @ d.ctrl
        np.testing.assert_allclose(force_body, [0.0, 0.0, wrench[0]], atol=1e-9)
        np.testing.assert_allclose(torque_com_body, wrench[1:], atol=1e-9)


def test_mj_fwd_actuation_force_matches_ctrl(model):
    d = mujoco.MjData(model)
    d.qpos[:] = [0, 0, 1.0, 1, 0, 0, 0]
    d.ctrl[:] = [1.0, 2.0, 3.0, 4.0]
    mujoco.mj_fwdPosition(model, d)
    mujoco.mj_fwdVelocity(model, d)
    mujoco.mj_fwdActuation(model, d)
    np.testing.assert_allclose(d.actuator_force, d.ctrl)
