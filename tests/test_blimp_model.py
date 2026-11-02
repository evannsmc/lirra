"""MuJoCo blimp f_sim vs the NAHS BlimpSim constants, and the f_syn blocks."""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from rl_stl.envs.blimp.model import (
    BlimpParams,
    allocation,
    blimp_fsyn,
    blimp_linear,
    buoyancy_wrench,
    load_blimp_model,
    motor_command,
    state_from_mujoco,
)
from rl_stl.synthesis.terminal import axis_blocks

P = BlimpParams()


def _sim(model, data, ctrl, seconds):
    bid = model.body("blimp").id
    for _ in range(int(round(seconds / P.timestep))):
        data.ctrl[:] = ctrl
        data.xfrc_applied[bid] = buoyancy_wrench(data, model, P)
        mujoco.mj_step(model, data)


def test_linearisation_matches_nahs_numbers():
    # Values from BlimpSim.py (A_lin/B_lin), computed in docs/blimp_model.md.
    L = blimp_linear(P)
    assert L.a_v == pytest.approx(0.07289, abs=1e-5)
    assert L.b_v == pytest.approx(5.8309, abs=1e-4)
    assert L.a_vz == pytest.approx(0.26756, abs=1e-5)
    assert L.b_vz == pytest.approx(5.5741, abs=1e-4)
    assert L.a_w == pytest.approx(0.14055, abs=1e-5)
    assert L.k_restore == pytest.approx(9.9786, abs=1e-3)
    assert L.b_tilt == pytest.approx(39.946, abs=1e-3)
    assert L.a_wz == pytest.approx(0.14808, abs=1e-5)
    assert L.b_wz == pytest.approx(171.79, abs=1e-2)


def test_neutral_buoyancy_hover():
    m = load_blimp_model(P)
    d = mujoco.MjData(m)
    _sim(m, d, np.zeros(m.nu), 10.0)
    assert np.allclose(d.qpos[:3], [0, 0, 1.5], atol=1e-9)
    assert np.allclose(d.qvel, 0.0, atol=1e-9)


def test_lateral_step_response_matches_decoupled_model():
    m = load_blimp_model(P)
    d = mujoco.MjData(m)
    L = blimp_linear(P)
    f = 0.019
    _sim(m, d, motor_command(np.array([f, 0, 0, 0]), P), 5.0)
    v_pred = L.b_v * f / L.a_v * (1 - np.exp(-L.a_v * 5.0))
    assert d.qvel[0] == pytest.approx(v_pred, rel=0.01)


def test_pendulum_swing_frequency():
    m = load_blimp_model(P)
    d = mujoco.MjData(m)
    d.qpos[3:7] = [np.cos(0.05), np.sin(0.05), 0, 0]  # 0.1 rad roll
    bid = m.body("blimp").id
    roll = []
    for _ in range(int(6.0 / P.timestep)):
        d.xfrc_applied[bid] = buoyancy_wrench(d, m, P)
        mujoco.mj_step(m, d)
        roll.append(d.qvel[3])
    roll = np.array(roll)
    crossings = np.nonzero(np.diff(np.sign(roll)) != 0)[0]
    period = 2 * np.mean(np.diff(crossings)) * P.timestep
    L = blimp_linear(P)
    # NAHS linear model: omega_n = sqrt(k_restore) (2.0 s period); MuJoCo includes the added-mass
    # coupling of the translational dofs, which lowers the effective inertia about the CG.
    assert 1.85 < period < 2.1  # measured 1.953 s
    assert 2 * np.pi / np.sqrt(L.k_restore) == pytest.approx(1.989, abs=0.01)


def test_allocation_full_rank_and_motor_limits():
    M = allocation(P)
    assert np.linalg.matrix_rank(M) == 4
    u = np.array([0.019, -0.019, 0.03, 0.0002])
    assert np.allclose(M @ motor_command(u, P), u, atol=1e-12)


def test_fsyn_blocks_decoupled_and_state_map():
    model = blimp_fsyn(0.5)
    axis_blocks(model, {"x": ((0, 4), 0), "y": ((1, 5), 1), "z": ((2, 6), 2), "yaw": ((3, 7), 3)})
    # ZOH numbers quoted from the NAHS discretisation (dt = 0.5): p_x <- 0.720 f_x, v_x <- 2.863 f_x
    assert model.B[0, 0] == pytest.approx(0.720, abs=2e-3)
    assert model.B[4, 0] == pytest.approx(2.863, abs=2e-3)
    q = np.array([1.0, 2.0, 1.5, np.cos(0.2), 0, 0, np.sin(0.2)])
    v = np.array([0.1, 0.2, 0.0, 0, 0, 0.3])
    x = state_from_mujoco(q, v, z_ref=1.5)
    assert np.allclose(x, [1.0, 2.0, 0.0, 0.4, 0.1, 0.2, 0.0, 0.3])
