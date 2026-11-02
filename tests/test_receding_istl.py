"""Receding I-STL stack: specs, tube, terminal sets, MILP vs independent check, supervisor."""

from __future__ import annotations

import numpy as np
import pytest

from rl_stl.dynamics.linear import dlqr, zoh_double_integrator
from rl_stl.evaluation.patrol import (
    ExpertProposer,
    PatrolDIConfig,
    build_di,
    episode_summary,
    nominal_proposer,
    run_di_episode,
)
from rl_stl.stl.nahs import PatrolTask, first_enforced_window, seconds_to_steps, window_spec
from rl_stl.stl.robustness import formula_horizon, robustness
from rl_stl.supervisor.receding import RecedingConfig, RecedingSupervisor
from rl_stl.synthesis.check import check_plan
from rl_stl.synthesis.istl import History, ISTLSettings, RecedingISTL
from rl_stl.synthesis.terminal import (
    Polygon,
    axis_blocks,
    max_control_invariant,
    verify_invariant,
)
from rl_stl.synthesis.tube import constant_tube, disturbance_tube

gurobipy = pytest.importorskip("gurobipy")


# ------------------------------------------------------------------------ specs
def test_nahs_step_convention_matches_reference_code():
    # rta_fixed.py at dT = 0.5: F_[0,7] for theta, F_[0,5] G_[0,2] for phi.
    assert seconds_to_steps(4.0, 0.5) == 7
    assert PatrolTask("theta", dt=0.5).horizon == 7
    assert PatrolTask("phi", dt=0.5).horizon == 7
    assert PatrolTask("phi", dt=0.5).terminal_block == 3
    assert PatrolTask("theta", dt=0.25).horizon == 15


def test_theta_semantics_on_hand_traces():
    task = PatrolTask("theta", dt=0.5)
    psi = task.formula()
    inside = np.zeros((8, 2))
    assert robustness(psi, inside) < 0  # 8 samples (4 s) inside: violated
    trace = inside.copy()
    trace[7] = [1.61, 0.0]
    assert robustness(psi, trace) == pytest.approx(0.2)


def test_window_spec_layout():
    task = PatrolTask("theta", dt=0.5)
    h, N = task.horizon, 12
    f = window_spec(task.formula(), N)
    assert formula_horizon(f) == h + N - 1  # signal of h + N samples
    assert first_enforced_window(0, h) == h - 1
    assert first_enforced_window(h - 1, h) == 0


# ------------------------------------------------------------------------- tube
def test_disturbance_tube_matches_monte_carlo():
    m = zoh_double_integrator(1, 0.25)
    K = -dlqr(m.A, m.B, np.diag([10.0, 1.0]), np.eye(1))
    w = np.array([0.1])
    tube = disturbance_tube(m.A, m.B, K, m.B, w, n_max=60)
    rng = np.random.default_rng(0)
    A_K = m.A + m.B @ K
    e = np.zeros((2000, 2))
    for i in range(1, 30):
        e = e @ A_K.T + rng.choice([-1, 1], size=(2000, 1)) * w * m.B[:, 0]
        assert np.all(np.abs(e) <= tube.state(i) + 1e-12)
    assert np.all(tube.z_inf >= tube.state(60) - 1e-12)
    assert np.all(np.diff(tube.z, axis=0) >= -1e-15)  # monotone


def test_constant_tube():
    t = constant_tube(4, 2, np.array([0.1, 0.1, 0.0, 0.0]))
    assert np.all(t.state(0) == 0) and t.state(5)[0] == 0.1


# --------------------------------------------------------------------- terminal
def test_invariant_polygon_verified_and_controller_keeps_state():
    m = zoh_double_integrator(1, 0.25)
    region = Polygon.from_box(np.array([1.5, -1.0]), np.array([2.5, 1.0]))
    P = max_control_invariant(m.A, m.B[:, 0], region, -0.5, 0.5)
    verify_invariant(m.A, m.B[:, 0], P, -0.5, 0.5)
    assert P.area < region.area  # the stopping constraint removed something
    assert not P.contains(np.array([1.5, -0.5]))  # moving into S at the face: not invariant


def test_axis_block_decoupling_check():
    m = zoh_double_integrator(2, 0.25)
    axis_blocks(m, {"x": ((0, 2), 0)})
    with pytest.raises(ValueError):
        axis_blocks(m, {"x": ((0, 2), 1)})


def test_face_terminal_controller_is_invariant():
    st = build_di(PatrolDIConfig())
    T = st.problem.terminal
    m = st.model
    rng = np.random.default_rng(1)
    for f, face in enumerate(T.faces):
        # random points of the face set: sample vertices' convex combinations per block
        for _ in range(20):
            x = np.zeros(4)
            for bname, poly in face.polygons.items():
                wts = rng.dirichlet(np.ones(len(poly.vertices)))
                x[list(T.blocks[bname].states)] = wts @ poly.vertices
            for _ in range(40):
                u = T.controller(x, f)
                assert np.all(u >= T.u_lo - 1e-9) and np.all(u <= T.u_hi + 1e-9)
                x = m.step(x, u)
                assert face.contains(T.blocks, x, 1e-6)


# ------------------------------------------------------------------------- MILP
@pytest.mark.parametrize("spec", ["theta", "phi"])
def test_milp_plan_passes_independent_check(spec):
    st = build_di(PatrolDIConfig(spec=spec, kp=0.25, kd=1.0))
    p = st.problem
    x0 = np.array([1.5, 1.5, 0.0, 0.0])
    st.nominal.reset(x0)
    Xn, Un = st.nominal.rollout(p.model, x0, p.n_plan)
    fw = first_enforced_window(0, p.h)
    mdl = RecedingISTL(p, fw, ISTLSettings(time_limit=10))
    hist = History(np.zeros((p.h - 1, 2)), [None] * (p.h - 1))
    mdl.update(x0, hist, Xn, Un)
    res = mdl.solve()
    assert res.has_solution
    chk = check_plan(p, x0, res.X, res.U, hist, None, fw)
    assert chk.ok, chk.reason
    assert chk.face == res.face
    # Perturbing the plan into S for longer than allowed must fail the check.
    Ubad = np.tile(-p.u_hi, (p.n_plan, 1))
    assert not check_plan(p, x0, p.model.rollout(x0, Ubad), Ubad, hist, None, fw).ok


def test_milp_reuse_equals_fresh_build():
    st = build_di(PatrolDIConfig(kp=0.25, kd=1.0))
    p = st.problem
    hist = History(np.zeros((p.h - 1, 2)), [None] * (p.h - 1))
    mdl = RecedingISTL(p, 0, ISTLSettings(time_limit=10))
    objs = []
    for x0 in (np.array([1.6, 1.6, 0.0, 0.0]), np.array([-1.7, 0.2, 0.0, 0.3])):
        st.nominal.reset(x0)
        Xn, Un = st.nominal.rollout(p.model, x0, p.n_plan)
        mdl.update(x0, hist, Xn, Un)
        r1 = mdl.solve()
        fresh = RecedingISTL(p, 0, ISTLSettings(time_limit=10))
        fresh.update(x0, hist, Xn, Un)
        r2 = fresh.solve()
        assert r1.status == r2.status
        objs.append((r1.objective, r2.objective))
    for a, b in objs:
        assert a == pytest.approx(b, rel=1e-3, abs=1e-6)


def test_tube_age_tightens_thresholds():
    st = build_di(PatrolDIConfig(kp=0.25, kd=1.0))
    p = st.problem
    x0 = np.array([1.6, 1.6, 0.0, 0.0])
    hist = History(np.zeros((p.h - 1, 2)), [None] * (p.h - 1))
    st.nominal.reset(x0)
    Xn, Un = st.nominal.rollout(p.model, x0, p.n_plan)
    mdl = RecedingISTL(p, 0, ISTLSettings(time_limit=10))
    mdl.update(x0, hist, Xn, Un, root_age=None)
    r0 = mdl.solve()
    mdl.update(x0, hist, Xn, Un, root_age=50)
    r50 = mdl.solve()
    assert r50.objective >= r0.objective - 1e-6
    assert check_plan(p, x0, r50.X, r50.U, hist, 50, 0).ok
    # the age-0 plan need not satisfy the older (wider) tube
    chk = check_plan(p, x0, r0.X, r0.U, hist, 50, 0)
    assert chk.rho_lower <= check_plan(p, x0, r0.X, r0.U, hist, None, 0).rho_lower + 1e-12


# ------------------------------------------------------------------- supervisor
@pytest.mark.parametrize("spec", ["theta", "phi"])
def test_shield_satisfies_and_warm_start_always_feasible(spec):
    cfg = PatrolDIConfig(spec=spec, kp=0.25, kd=1.0, episode_seconds=20.0, disturbance="persistent")
    st = build_di(cfg)
    sup = RecedingSupervisor(st.problem, st.K, st.nominal, nominal_proposer,
                             RecedingConfig(variant="shield", latency_steps=2))
    ep = run_di_episode(st, cfg, sup, seed=3)
    s = episode_summary(st, ep)
    assert s["satisfied"]
    assert s["warm_start_infeasible"] == 0
    assert s["n_runtime_solves"] > 0
    assert all(r["plan_certified"] for r in ep["steps"])
    # every release happens exactly d steps after the request (injected latency)
    for sl in ep["solves"][1:]:
        assert sl["release_step"] - sl["request_step"] == 2
    # the error stays inside the tube
    assert s["max_tracking_error"] <= float(np.max(st.tube.z_inf)) + 1e-9


def test_policy_only_nominal_violates_and_expert_needs_no_solves():
    cfg = PatrolDIConfig(spec="theta", kp=0.25, kd=1.0, episode_seconds=20.0)
    st = build_di(cfg)
    sup = RecedingSupervisor(st.problem, st.K, st.nominal, nominal_proposer,
                             RecedingConfig(variant="policy_only"))
    assert not episode_summary(st, run_di_episode(st, cfg, sup, seed=0))["satisfied"]
    st = build_di(cfg)
    sup = RecedingSupervisor(st.problem, st.K, st.nominal, ExpertProposer(st.problem),
                             RecedingConfig(variant="shield"))
    s = episode_summary(st, run_di_episode(st, cfg, sup, seed=0))
    assert s["satisfied"] and s["n_runtime_solves"] == 0 and s["frac_proposal"] == 1.0


def test_compiled_check_matches_reference_recursion():
    from rl_stl.synthesis.check import compiled_windows, interval_robustness

    rng = np.random.default_rng(5)
    for spec in ("theta", "phi"):
        st = build_di(PatrolDIConfig(spec=spec, kp=0.25, kd=1.0))
        p = st.problem
        for _ in range(30):
            fw = int(rng.integers(0, p.h))
            x0 = np.concatenate([rng.uniform(-2, 2, 2), rng.uniform(-0.5, 0.5, 2)])
            U = rng.uniform(p.u_lo, p.u_hi, (p.n_plan, 2))
            X = p.model.rollout(x0, U)
            ages = [None if rng.random() < 0.5 else int(rng.integers(0, 20)) for _ in range(p.h - 1)]
            hist = History(rng.uniform(-2, 2, (p.h - 1, 2)), ages)
            age = None if rng.random() < 0.5 else int(rng.integers(0, 20))
            ref = interval_robustness(p, X, hist, age, fw)
            fast = compiled_windows(p, fw)(X, hist, age)
            assert fast == pytest.approx(ref, abs=1e-12) or (np.isinf(ref) and fast == ref)


# --------------------------------------------------------------- clock reference
def test_clock_reference_reproduces_rollout_and_shifts():
    from rl_stl.policies.nominal import ClockReference

    st = build_di(PatrolDIConfig(spec="theta", kp=0.25, kd=1.0))
    p = st.problem
    x0 = np.array([1.6, 1.4, 0.1, -0.2])
    ref = ClockReference(st.nominal, p.model)
    ref.reset(x0)
    X0, U0 = ref.window(0, p.n_plan)
    st.nominal.reset(x0)
    Xr, Ur = st.nominal.rollout(p.model, x0, p.n_plan)
    assert np.allclose(X0, Xr) and np.allclose(U0, Ur)
    X5, U5 = ref.window(5, p.n_plan)
    assert np.allclose(X5[:-5], X0[5:]) and np.allclose(U5[:-5], U0[5:])
    assert X5.shape == (p.n_plan + 1, p.model.n) and U5.shape == (p.n_plan, p.model.m)
    full = ref.window(0, 400)[0]  # the patrol keeps cycling through the waypoints
    assert len({ref.index_at(k) for k in range(400)}) == 4
    with pytest.raises(ValueError):
        ref.window(-1, 3)
    assert np.all(np.isfinite(full))


@pytest.mark.parametrize("spec", ["theta", "phi"])
def test_clock_reference_shield_satisfies_and_warm_start_always_feasible(spec):
    """The objective changes, the certificate does not: same guarantees as the rerooted loop."""
    cfg = PatrolDIConfig(spec=spec, kp=0.25, kd=1.0, episode_seconds=20.0, disturbance="persistent")
    st = build_di(cfg)
    sup = RecedingSupervisor(st.problem, st.K, st.nominal, nominal_proposer,
                             RecedingConfig(variant="shield", latency_steps=2, reference="clock"))
    ep = run_di_episode(st, cfg, sup, seed=3)
    s = episode_summary(st, ep)
    assert s["satisfied"]
    assert s["warm_start_infeasible"] == 0
    assert s["n_runtime_solves"] > 0
    assert all(r["plan_certified"] for r in ep["steps"])
    # each request tracks the reference at its own (delay-compensated) root step
    p = st.problem
    for sl in ep["solves"][1:]:
        Xn, _ = sup.reference.window(sl["root_step"], p.n_plan)
        assert sl["root_step"] == sl["request_step"] + 2 and np.all(np.isfinite(Xn))
