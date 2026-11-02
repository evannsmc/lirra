"""CTBR action interface: mapping endpoints, allocation, desaturation priority, closed-loop rates."""

import math

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import pytest

from rl_stl.envs.skydio_x2.ctbr import (
    CTBRConfig,
    command_to_normalized,
    ctbr_motor_thrusts,
    make_ctbr_params,
    mix,
    normalized_to_command,
    rate_control_torque,
)
from rl_stl.envs.skydio_x2.model import load_x2_model, x2_params_from_model


@pytest.fixture(scope="module")
def model():
    return load_x2_model()


@pytest.fixture(scope="module")
def x2(model):
    return x2_params_from_model(model)


@pytest.fixture(scope="module")
def p(x2):
    return make_ctbr_params(x2)


def test_thrust_and_rate_endpoints(p, x2):
    for a_t, expected in [(-1.0, 0.0), (0.0, x2.hover_thrust_total), (1.0, 52.0), (-3.0, 0.0), (5.0, 52.0)]:
        thrust, _ = normalized_to_command(jnp.array([a_t, 0, 0, 0]), p)
        assert float(thrust) == pytest.approx(expected, rel=1e-5, abs=1e-5)
    # Piecewise-linear on each side of hover.
    thrust, _ = normalized_to_command(jnp.array([0.5, 0, 0, 0]), p)
    assert float(thrust) == pytest.approx(x2.hover_thrust_total + 0.5 * (52.0 - x2.hover_thrust_total), rel=1e-5)
    thrust, _ = normalized_to_command(jnp.array([-0.5, 0, 0, 0]), p)
    assert float(thrust) == pytest.approx(0.5 * x2.hover_thrust_total, rel=1e-5)
    _, rates = normalized_to_command(jnp.array([0.0, 1.0, -1.0, 1.0]), p)
    np.testing.assert_allclose(np.asarray(rates), [2 * math.pi, -2 * math.pi, math.pi], rtol=1e-6)


def test_collective_margins(x2):
    p = make_ctbr_params(x2, CTBRConfig(collective_min_frac=0.1, collective_max_frac=0.8))
    lo, _ = normalized_to_command(jnp.array([-1.0, 0, 0, 0]), p)
    hi, _ = normalized_to_command(jnp.array([1.0, 0, 0, 0]), p)
    assert float(lo) == pytest.approx(5.2, rel=1e-5) and float(hi) == pytest.approx(41.6, rel=1e-5)
    with pytest.raises(ValueError):
        make_ctbr_params(x2, CTBRConfig(collective_max_frac=0.2))


def test_normalized_roundtrip(p):
    rng = np.random.default_rng(1)
    for _ in range(20):
        a = rng.uniform(-1, 1, size=4)
        thrust, rates = normalized_to_command(jnp.asarray(a), p)
        np.testing.assert_allclose(np.asarray(command_to_normalized(thrust, rates, p)), a, atol=1e-5)


def test_allocation_inverse(x2):
    np.testing.assert_allclose(x2.allocation @ x2.allocation_inv, np.eye(4), atol=1e-12)
    # X configuration: equal thrust share, symmetric roll/pitch/yaw columns.
    np.testing.assert_allclose(x2.allocation_inv[:, 0], 0.25)
    np.testing.assert_allclose(np.abs(x2.allocation_inv[:, 1:]).std(axis=0), 0.0, atol=1e-12)


def test_rate_torque_zero_error_is_gyroscopic(p, x2):
    w = jnp.array([0.4, -1.0, 2.0])
    tau = rate_control_torque(w, w, p)
    np.testing.assert_allclose(np.asarray(tau), np.cross(w, x2.inertia_com_body @ np.asarray(w)), atol=1e-6)


def test_unsaturated_mix_is_exact(p, x2):
    wrench = np.array([x2.hover_thrust_total, 0.05, -0.08, 0.02])
    motors, info = mix(jnp.asarray(wrench[0]), jnp.asarray(wrench[1:]), p)
    np.testing.assert_allclose(np.asarray(info.wrench_realized), wrench, atol=1e-5)
    np.testing.assert_allclose(np.asarray(motors), x2.allocation_inv @ wrench, atol=1e-5)
    assert not bool(info.saturated)
    assert float(info.rp_scale) == 1.0 and float(info.yaw_scale) == 1.0


def test_yaw_is_sacrificed_before_thrust_and_roll_pitch(p, x2):
    thrust = x2.hover_thrust_total
    torque = np.array([0.3, -0.2, 1.0])  # yaw far beyond authority
    motors, info = mix(jnp.asarray(thrust), jnp.asarray(torque), p)
    realized = np.asarray(info.wrench_realized)
    assert 0.0 < float(info.yaw_scale) < 1.0
    assert realized[0] == pytest.approx(thrust, rel=1e-5)
    np.testing.assert_allclose(realized[1:3], torque[:2], atol=1e-5)
    assert realized[3] == pytest.approx(float(info.yaw_scale) * torque[2], rel=1e-4)
    assert np.all(np.asarray(motors) >= -1e-6) and np.all(np.asarray(motors) <= 13 + 1e-6)


def test_thrust_shifted_to_keep_roll_pitch(p):
    thrust = 50.0  # close to 4 * 13 N
    torque = np.array([1.5, 0.0, 0.0])
    _, info = mix(jnp.asarray(thrust), jnp.asarray(torque), p)
    realized = np.asarray(info.wrench_realized)
    assert float(info.rp_scale) == 1.0
    assert realized[1] == pytest.approx(1.5, rel=1e-5)
    assert realized[0] < thrust and float(info.thrust_shift) < 0
    assert bool(info.saturated)


def test_infeasible_roll_pitch_scaled(p):
    torque = np.array([5.0, 3.0, 0.0])  # needs > 13 N motor spread
    motors, info = mix(jnp.asarray(26.0), jnp.asarray(torque), p)
    realized = np.asarray(info.wrench_realized)
    assert float(info.rp_scale) < 1.0
    # Direction of the roll/pitch torque is preserved.
    np.testing.assert_allclose(realized[1] / realized[2], torque[0] / torque[1], rtol=1e-4)
    assert np.all(np.asarray(motors) >= -1e-6) and np.all(np.asarray(motors) <= 13 + 1e-6)


def test_motor_bounds_random_and_vmap(p):
    rng = np.random.default_rng(2)
    thrust = jnp.asarray(rng.uniform(-10, 70, size=256))
    torque = jnp.asarray(rng.normal(scale=2.0, size=(256, 3)))
    motors, info = jax.jit(jax.vmap(mix, in_axes=(0, 0, None)))(thrust, torque, p)
    motors = np.asarray(motors)
    assert motors.shape == (256, 4)
    assert motors.min() >= -1e-6 and motors.max() <= 13 + 1e-6
    assert np.all(np.isfinite(np.asarray(info.wrench_realized)))


def _simulate_rate_step(model, p, action, duration):
    step = jax.jit(ctbr_motor_thrusts)
    d = mujoco.MjData(model)
    d.qpos[:] = [0, 0, 5.0, 1, 0, 0, 0]
    mujoco.mj_forward(model, d)
    rates = []
    for _ in range(round(duration / model.opt.timestep)):
        motors, _ = step(jnp.asarray(action), jnp.asarray(d.qvel[3:6]), p)
        d.ctrl[:] = np.asarray(motors)
        mujoco.mj_step(model, d)
        rates.append(d.qvel[3:6].copy())
    return np.array(rates)


@pytest.mark.parametrize("axis,a", [(0, 0.2), (1, -0.2)])
def test_closed_loop_roll_pitch_rate_step(model, p, axis, a):
    action = np.zeros(4)
    action[1 + axis] = a
    w = _simulate_rate_step(model, p, action, 0.5)
    target = a * float(p.rate_max[axis])
    k = 1.0 / float(p.rate_gain[axis])  # time constant
    idx_tau = round(k / model.opt.timestep) - 1
    # First-order response: ~63 % at one time constant (discrete Euler, 1 step delay tolerance).
    assert 0.55 < w[idx_tau, axis] / target < 0.75
    assert abs(w[-1, axis] - target) < 0.01 * abs(target)
    assert np.abs(np.delete(w, axis, axis=1)).max() < 0.01 * abs(target)


def test_closed_loop_yaw_rate_step(model, p):
    action = np.array([0.0, 0.0, 0.0, 0.2])
    w = _simulate_rate_step(model, p, action, 1.0)
    target = 0.2 * float(p.rate_max[2])
    assert abs(w[-1, 2] - target) < 0.02 * target
    assert np.abs(w[:, :2]).max() < 0.02 * target
