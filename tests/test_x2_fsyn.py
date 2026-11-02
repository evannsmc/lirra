"""f_syn v1 (3-D double integrator), MuJoCo <-> f_syn mapping, and the outer-loop tracker."""

import math

import numpy as np
import pytest

from rl_stl.dynamics.quadrotor_fsyn import (
    FsynConfig,
    accel_box_from_limits,
    fsyn_state_from_mujoco,
    make_fsyn,
    mujoco_state_from_fsyn,
)
from rl_stl.envs.skydio_x2.host_sim import X2HostSim


@pytest.fixture(scope="module")
def sim():
    return X2HostSim()


@pytest.fixture(scope="module")
def fsyn(sim):
    return make_fsyn(sim.x2)


def test_zoh_matches_constant_acceleration(fsyn):
    z0 = np.array([0.1, -0.2, 1.0, 0.5, 0.0, -0.3])
    u = np.array([0.4, -1.0, 0.7])
    zs = fsyn.rollout(z0, np.tile(u, (10, 1)))
    t = 10 * fsyn.dt
    np.testing.assert_allclose(zs[-1, :3], z0[:3] + z0[3:] * t + 0.5 * u * t**2, atol=1e-12)
    np.testing.assert_allclose(zs[-1, 3:], z0[3:] + u * t, atol=1e-12)
    p, v = fsyn.intra_step_reference(z0, u, fsyn.dt)
    np.testing.assert_allclose(np.concatenate([p, v]), fsyn.step(z0, u), atol=1e-12)


def test_stlpy_system_shapes(fsyn):
    sys = fsyn.stlpy_system()
    assert (sys.n, sys.m, sys.p) == (6, 3, 9)
    z, u = np.arange(6.0), np.array([1.0, 2.0, 3.0])
    np.testing.assert_allclose(sys.output_fcn(z, u), np.concatenate([z, u]))
    np.testing.assert_allclose(sys.dynamics_fcn(z, u), fsyn.step(z, u))


def test_accel_box_respects_tilt_and_thrust(sim):
    cfg = FsynConfig()
    a_min, a_max = accel_box_from_limits(sim.x2, cfg)
    g, m = sim.x2.gravity, sim.x2.mass
    for ax in (a_min[0], a_max[0]):
        for ay in (a_min[1], a_max[1]):
            for az in (a_min[2], a_max[2]):
                f = np.array([ax, ay, az + g])
                tilt = math.degrees(math.atan2(np.linalg.norm(f[:2]), f[2]))
                assert tilt <= cfg.tilt_max_deg + 1e-9
                assert np.linalg.norm(f) * m <= cfg.thrust_headroom_frac * 52.0 + 1e-9
    with pytest.raises(ValueError):
        accel_box_from_limits(sim.x2, FsynConfig(accel_down_max=10.0))


def test_mapping_roundtrip(sim):
    rng = np.random.default_rng(3)
    for _ in range(10):
        z = rng.normal(size=6)
        quat = rng.normal(size=4)
        quat /= np.linalg.norm(quat)
        w = rng.normal(size=3)
        qpos, qvel = mujoco_state_from_fsyn(sim.x2, z, quat, w)
        np.testing.assert_allclose(fsyn_state_from_mujoco(sim.x2, qpos, qvel), z, atol=1e-12)


def test_tracker_holds_hover(sim):
    z0 = np.array([0.3, -0.2, 2.0, 0.0, 0.0, 0.0])
    sim.reset(*mujoco_state_from_fsyn(sim.x2, z0))
    for _ in range(300):
        sim.step_tracking(z0[:3], z0[3:], np.zeros(3))
    np.testing.assert_allclose(fsyn_state_from_mujoco(sim.x2, sim.data.qpos, sim.data.qvel), z0, atol=1e-4)


def test_tracker_follows_fsyn_plan(sim, fsyn):
    rng = np.random.default_rng(4)
    z0 = np.array([0.0, 0.0, 2.0, 0.0, 0.0, 0.0])
    us = 0.5 * rng.uniform(fsyn.a_min, fsyn.a_max, size=(15, 3))
    plan = fsyn.rollout(z0, us)
    sim.reset(*mujoco_state_from_fsyn(sim.x2, z0))
    n_sub = round(fsyn.dt / sim.dt)
    for k, u in enumerate(us):
        for j in range(n_sub):
            p, v = fsyn.intra_step_reference(plan[k], u, j * sim.dt)
            sim.step_tracking(p, v, u)
    err = fsyn_state_from_mujoco(sim.x2, sim.data.qpos, sim.data.qvel) - plan[-1]
    assert np.linalg.norm(err[:3]) < 0.05
    assert np.linalg.norm(err[3:]) < 0.3
