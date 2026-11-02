"""3-D cube specifications theta^+ / phi^+: semantics, waypoints, terminal sets, synthesis, check,
JAX certificate and features, PPO env (the 2-D counterparts live in test_receding_istl.py and
test_segment_policy.py)."""

from __future__ import annotations

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "patrol"))

from common import load_system  # noqa: E402

from rl_stl.envs.patrol_plan import PatrolPlanEnv  # noqa: E402
from rl_stl.evaluation.patrol import (  # noqa: E402
    ExpertProposer,
    PatrolDIConfig,
    build_di,
    episode_summary,
    nominal_proposer,
    run_di_episode,
)
from rl_stl.policies.segment import features, outside_margin, patrol_certificate  # noqa: E402
from rl_stl.stl.nahs import PatrolTask, first_enforced_window  # noqa: E402
from rl_stl.stl.robustness import robustness  # noqa: E402
from rl_stl.supervisor.receding import RecedingConfig, RecedingSupervisor  # noqa: E402
from rl_stl.synthesis.check import check_plan, compiled_windows, interval_robustness  # noqa: E402
from rl_stl.synthesis.istl import History, ISTLSettings, RecedingISTL  # noqa: E402

pytest.importorskip("gurobipy")
PLUS = ["theta_plus", "phi_plus"]


# ------------------------------------------------------------------------ specs
def test_cube_spec_shares_the_temporal_structure():
    for base in ("theta", "phi"):
        for dt in (0.25, 0.5):
            a, b = PatrolTask(base, dt=dt), PatrolTask(base + "_plus", dt=dt)
            assert (a.dim, b.dim) == (2, 3)
            assert a.horizon == b.horizon and a.terminal_block == b.terminal_block
            assert b.base_spec == base
    with pytest.raises(ValueError):
        PatrolTask("psi")


def test_theta_plus_semantics_on_hand_traces():
    psi = PatrolTask("theta_plus", dt=0.5).formula()
    inside = np.zeros((8, 3))
    assert robustness(psi, inside) < 0  # 4 s inside the cube: violated
    for axis in range(3):
        for sign in (1, -1):
            trace = inside.copy()
            trace[7, axis] = sign * 1.61
            assert robustness(psi, trace) == pytest.approx(0.2)
    # above the square but inside the planar footprint: outside the cube, inside the 2-D S
    trace = inside.copy()
    trace[3] = [0.0, 0.0, 1.61]
    assert robustness(psi, trace) == pytest.approx(0.2)
    assert robustness(PatrolTask("theta", dt=0.5).formula(), trace[:, :2]) < 0


def test_phi_plus_semantics_on_hand_traces():
    task = PatrolTask("phi_plus", dt=0.5)  # F_[0,5] G_[0,2] notS (or notS now)
    psi = task.formula()
    tr = np.zeros((task.horizon + 1, 3))
    tr[0] = [0.0, 0.0, 0.5]
    assert robustness(psi, tr) < 0
    tr[3:6] = [0.0, 0.0, -1.71]  # three outside samples (1 s) below the cube
    assert robustness(psi, tr) == pytest.approx(0.3)
    tr[4] = [0.0, 0.0, 0.0]  # broken dwell
    assert robustness(psi, tr) < 0


def test_six_waypoints_on_face_centres_with_adjacent_legs():
    task = PatrolTask("theta_plus")
    W = task.waypoints()
    assert W.shape == (6, 3)
    off = task.waypoint_radius - task.half_width
    assert off == pytest.approx(0.11)
    for w in W:  # one non-zero coordinate: the face centre moved outward by off
        assert np.count_nonzero(w) == 1 and np.max(np.abs(w)) == pytest.approx(task.half_width + off)
    assert {tuple(np.sign(w).astype(int)) for w in W} == {tuple(s * e) for s in (1, -1) for e in np.eye(3, dtype=int)}
    for a, b in zip(W, np.roll(W, -1, axis=0)):
        assert a @ b == 0.0  # adjacent faces (never opposite)
        assert outside_margin((a + b) / 2, task.half_width) < 0  # each leg cuts through S+
    # the 2-D order is the restriction of the cube cycle to the plane
    planar = [w[:2] for w in W if w[2] == 0]
    assert np.allclose(planar, PatrolTask("theta").waypoints())


def test_outside_margin_matches_formula_robustness():
    rng = np.random.default_rng(0)
    from rl_stl.stl.nahs import outside_box

    f = outside_box(1.41, 3)
    for _ in range(50):
        y = rng.uniform(-2.5, 2.5, 3)
        assert outside_margin(y, 1.41) == pytest.approx(robustness(f, y[None]))


# --------------------------------------------------------------------- terminal
def test_six_face_terminal_sets_are_invariant():
    st = build_di(PatrolDIConfig(spec="theta_plus"))
    T = st.problem.terminal
    assert [f.name for f in T.faces] == ["+x", "-x", "+y", "-y", "+z", "-z"]
    m = st.model
    c_out = st.task.half_width + st.problem.delta + float(np.max(st.tube.z_inf[:3]))
    rng = np.random.default_rng(1)
    for f, face in enumerate(T.faces):
        axis, sign = "xyz".index(face.name[1]), 1 if face.name[0] == "+" else -1
        for _ in range(10):
            x = np.zeros(6)
            for bname, poly in face.polygons.items():
                x[list(T.blocks[bname].states)] = rng.dirichlet(np.ones(len(poly.vertices))) @ poly.vertices
            for _ in range(40):
                u = T.controller(x, f)
                assert np.all(u >= T.u_lo - 1e-9) and np.all(u <= T.u_hi + 1e-9)
                x = m.step(x, u)
                assert face.contains(T.blocks, x, 1e-6)
                assert sign * x[axis] >= c_out - 1e-6  # stays outside the tightened cube


# ------------------------------------------------------------------------- MILP
@pytest.mark.parametrize("spec", PLUS)
def test_cube_milp_plan_passes_independent_check(spec):
    st = build_di(PatrolDIConfig(spec=spec, kp=0.25, kd=1.0))
    p = st.problem
    x0 = np.array([1.5, 1.5, 0.3, 0.0, 0.0, 0.0])
    st.nominal.reset(x0)
    Xn, Un = st.nominal.rollout(p.model, x0, p.n_plan)
    fw = first_enforced_window(0, p.h)
    hist = History(np.zeros((p.h - 1, 3)), [None] * (p.h - 1))
    mdl = RecedingISTL(p, fw, ISTLSettings(time_limit=20))
    mdl.update(x0, hist, Xn, Un)
    res = mdl.solve()
    assert res.has_solution
    chk = check_plan(p, x0, res.X, res.U, hist, None, fw)
    assert chk.ok, chk.reason
    assert chk.face == res.face
    Ubad = np.tile(-p.u_hi, (p.n_plan, 1))
    assert not check_plan(p, x0, p.model.rollout(x0, Ubad), Ubad, hist, None, fw).ok


def test_cube_compiled_check_matches_reference_recursion():
    rng = np.random.default_rng(5)
    for spec in PLUS:
        p = build_di(PatrolDIConfig(spec=spec, kp=0.25, kd=1.0)).problem
        for _ in range(20):
            fw = int(rng.integers(0, p.h))
            x0 = np.concatenate([rng.uniform(-2, 2, 3), rng.uniform(-0.5, 0.5, 3)])
            U = rng.uniform(p.u_lo, p.u_hi, (p.n_plan, 3))
            X = p.model.rollout(x0, U)
            ages = [None if rng.random() < 0.5 else int(rng.integers(0, 20)) for _ in range(p.h - 1)]
            hist = History(rng.uniform(-2, 2, (p.h - 1, 3)), ages)
            age = None if rng.random() < 0.5 else int(rng.integers(0, 20))
            ref = interval_robustness(p, X, hist, age, fw)
            fast = compiled_windows(p, fw)(X, hist, age)
            assert fast == pytest.approx(ref, abs=1e-12) or (np.isinf(ref) and fast == ref)


# ------------------------------------------------------------------- supervisor
@pytest.mark.parametrize("spec", PLUS)
def test_cube_shield_satisfies_and_warm_start_always_feasible(spec):
    cfg = PatrolDIConfig(spec=spec, kp=0.25, kd=1.0, episode_seconds=20.0, disturbance="persistent")
    st = build_di(cfg)
    sup = RecedingSupervisor(st.problem, st.K, st.nominal, nominal_proposer,
                             RecedingConfig(variant="shield", latency_steps=2, reference="clock"))
    ep = run_di_episode(st, cfg, sup, seed=3)
    s = episode_summary(st, ep)
    assert ep["X"].shape[1] == 6
    assert s["satisfied"] and s["warm_start_infeasible"] == 0 and s["n_runtime_solves"] > 0
    assert all(r["plan_certified"] for r in ep["steps"])
    assert s["max_tracking_error"] <= float(np.max(st.tube.z_inf)) + 1e-9


def test_cube_nominal_violates_and_expert_needs_no_solves():
    cfg = PatrolDIConfig(spec="theta_plus", kp=0.25, kd=1.0, episode_seconds=20.0)
    st = build_di(cfg)
    sup = RecedingSupervisor(st.problem, st.K, st.nominal, nominal_proposer, RecedingConfig(variant="policy_only"))
    assert not episode_summary(st, run_di_episode(st, cfg, sup, seed=0))["satisfied"]
    st = build_di(cfg)
    sup = RecedingSupervisor(st.problem, st.K, st.nominal, ExpertProposer(st.problem), RecedingConfig(variant="shield"))
    s = episode_summary(st, run_di_episode(st, cfg, sup, seed=0))
    assert s["satisfied"] and s["n_runtime_solves"] == 0 and s["frac_proposal"] == 1.0


def test_cube_clock_reference_visits_all_six_waypoints():
    from rl_stl.policies.nominal import ClockReference

    st = build_di(PatrolDIConfig(spec="theta_plus"))
    ref = ClockReference(st.nominal, st.problem.model)
    ref.reset(np.array([1.6, 0.0, 0.2, 0.0, 0.0, 0.0]))
    ref.window(0, 600)
    assert len({ref.index_at(k) for k in range(600)}) == 6


# ------------------------------------------------------------ learned proposer
@pytest.mark.parametrize("system", ["di", "x2", "blimp"])
def test_cube_level_tables_and_feature_parity(system):
    _, _, T = load_system(system, "phi_plus", "clock")
    assert T.p == 3 and T.n_waypoints == 6 and T.H.shape[1] == 6
    rng = np.random.default_rng(0)
    x = rng.normal(size=T.n)
    hy = rng.uniform(-2, 2, (T.h - 1, 3))
    mask = (np.arange(T.h - 1) >= 3).astype(float)
    Xn = rng.normal(size=(T.n_plan + 1, T.n))
    a = features(T, x, hy, mask, 5, 1, x_ref=Xn)
    f32 = lambda v: jnp.asarray(v, jnp.float32)  # noqa: E731
    b = np.asarray(features(T, f32(x), f32(hy), f32(mask), 5, 1, jnp, x_ref=f32(Xn)))
    assert a.shape == (T.obs_size,) and np.allclose(a, b, atol=1e-5)


@pytest.mark.parametrize("system", ["di", "x2", "blimp"])
@pytest.mark.parametrize("spec", PLUS)
def test_cube_jax_certificate_agrees_with_check(system, spec):
    sysm, setups, T = load_system(system, spec)
    cert = jax.jit(lambda x0, U, hy, hm, fw, lv: patrol_certificate(T, x0, U, hy, hm, fw, lv)[0],
                   static_argnums=(5,))
    rng = np.random.default_rng(2)
    n_ok = n = 0
    for trial in range(8):
        lv = trial % len(setups)
        st = setups[lv]
        p = st.problem
        x0 = sysm.random_start(rng, st)
        k = int(rng.integers(0, 40))
        fw = first_enforced_window(k, p.h)
        hy = np.tile(p.C_out @ x0, (p.h - 1, 1)) + rng.normal(0, 0.05, (p.h - 1, 3))
        hist = History(hy, [None] * (p.h - 1))
        hm = (np.arange(p.h - 1) >= fw).astype(float)
        st.nominal.reset(x0)
        Xn, Un = st.nominal.rollout(p.model, x0, p.n_plan)
        mdl = RecedingISTL(p, fw, ISTLSettings(time_limit=10))
        mdl.update(x0, hist, Xn, Un)
        res = mdl.solve()
        if not res.has_solution:
            continue
        scale = (p.u_hi - p.u_lo) / 2
        for U in (res.U, np.clip(res.U + rng.normal(0, 0.05, res.U.shape) * scale,
                                 p.u_lo + p.tube.kz[:p.n_plan], p.u_hi - p.tube.kz[:p.n_plan])):
            X = p.model.rollout(x0, U)
            ok = check_plan(p, x0, X, U, hist, None, fw).ok
            c = float(cert(jnp.asarray(x0), jnp.asarray(U), jnp.asarray(hy), jnp.asarray(hm), fw, lv))
            # float32 JAX rollout vs float64 check: plans at the margin differ by up to ~5e-4
            # (exact agreement with jax_enable_x64); the training hinge uses eps_cert = 0.01.
            if abs(c) > 1e-3:
                assert ok == (c >= 0), (ok, c)
            n += 1
            n_ok += ok
    assert n >= 6 and n_ok > 0


@pytest.mark.parametrize("system", ["di", "x2"])
def test_cube_plan_env_matches_numpy_and_steps(system):
    from rl_stl.policies.nominal import ClockReference

    _, setups, T = load_system(system, "theta_plus", "clock")
    st = setups[1]
    p = st.problem
    env = PatrolPlanEnv(T, st.nominal, p.track_weight, p.gamma0, p.n_cost, p.effort_weight,
                        np.zeros(T.obs_size), np.ones(T.obs_size))
    x0 = np.array([1.6, 0.0, 0.3, 0.0, 0.0, -0.1])
    st.nominal.reset(x0)
    Xn, Un = st.nominal.rollout(p.model, x0, p.n_plan)
    Xj, Uj = env.nominal_rollout(jnp.asarray(x0, jnp.float32), st.nominal.index)
    assert np.allclose(Xn, np.asarray(Xj), atol=1e-4) and np.allclose(Un, np.asarray(Uj), atol=1e-4)
    ref = ClockReference(st.nominal, p.model, T.ref_input_scale)
    ref.reset(x0)
    X, _ = ref.window(0, 80)
    Xc, _, Ic = env.clock_reference(jnp.asarray(x0, jnp.float32), st.nominal.index, 80)
    assert np.allclose(X, np.asarray(Xc), atol=1e-3)
    assert [ref.index_at(k) for k in range(80)] == [int(i) for i in Ic]
    s = jax.jit(env.reset)(jax.random.PRNGKey(3))
    assert s.obs.shape == (T.obs_size,)
    assert abs(float(jnp.linalg.norm(s.data["x"][:3])) - 1.9) <= 0.41  # spherical shell start
    step = jax.jit(env.step)
    for _ in range(T.h + 2):
        s = step(s, jnp.zeros(T.action_size))
    assert np.isfinite(float(s.reward)) and int(s.data["first_window"]) == 0
