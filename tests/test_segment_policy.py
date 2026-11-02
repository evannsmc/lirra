"""Learned-proposer plumbing: numpy/JAX feature parity, JAX certificate vs check, PPO env."""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "patrol"))

from common import load_system  # noqa: E402

from rl_stl.envs.patrol_plan import PatrolPlanEnv  # noqa: E402
from rl_stl.policies.segment import (  # noqa: E402
    SegmentPolicy,
    features,
    init_mlp,
    patrol_certificate,
    squash,
    unsquash,
)
from rl_stl.stl.nahs import first_enforced_window  # noqa: E402
from rl_stl.synthesis.check import check_plan  # noqa: E402
from rl_stl.synthesis.istl import History, ISTLSettings, RecedingISTL  # noqa: E402

pytest.importorskip("gurobipy")


@pytest.mark.parametrize("system", ["di", "blimp"])
def test_numpy_and_jax_features_agree(system):
    _, setups, T = load_system(system, "phi")
    rng = np.random.default_rng(0)
    for level in range(len(setups)):
        x = rng.normal(size=T.n)
        hy = rng.uniform(-2, 2, (T.h - 1, 2))
        mask = (np.arange(T.h - 1) >= 3).astype(float)
        a = features(T, x, hy, mask, 2, level)
        b = np.asarray(features(T, jnp.asarray(x, jnp.float32), jnp.asarray(hy, jnp.float32),
                                jnp.asarray(mask, jnp.float32), 2, level, jnp))
        assert a.shape == (T.obs_size,)
        assert np.allclose(a, b, atol=1e-5)


@pytest.mark.parametrize("system", ["di", "x2"])
def test_clock_features_agree_and_widen_the_observation(system):
    _, setups, T0 = load_system(system, "theta")
    _, _, T = load_system(system, "theta", "clock")
    assert T.reference == "clock" and T.obs_size == T0.obs_size + 3 * T.n
    rng = np.random.default_rng(4)
    x = rng.normal(size=T.n)
    hy = rng.uniform(-2, 2, (T.h - 1, 2))
    mask = (np.arange(T.h - 1) >= 3).astype(float)
    Xn = rng.normal(size=(T.n_plan + 1, T.n))
    a = features(T, x, hy, mask, 1, 0, x_ref=Xn)
    f32 = lambda v: jnp.asarray(v, jnp.float32)  # noqa: E731
    b = np.asarray(features(T, f32(x), f32(hy), f32(mask), 1, 0, jnp, x_ref=f32(Xn)))
    assert a.shape == (T.obs_size,) and np.allclose(a, b, atol=1e-5)
    assert np.allclose(a[:T0.obs_size], features(T0, x, hy, mask, 1, 0))
    with pytest.raises(ValueError):
        features(T, x, hy, mask, 1, 0)


def test_squash_roundtrip_and_bounds():
    _, setups, T = load_system("di", "theta")
    rng = np.random.default_rng(1)
    for level in range(len(setups)):
        U = squash(T, rng.normal(size=T.action_size) * 3, level)
        lo, hi = T.u_lo + T.kz[level], T.u_hi - T.kz[level]
        assert np.all(U >= lo - 1e-12) and np.all(U <= hi + 1e-12)
        assert np.allclose(squash(T, unsquash(T, U, level), level), U, atol=1e-6)


@pytest.mark.parametrize("system", ["di", "x2", "blimp"])
def test_jax_certificate_agrees_with_check_on_expert_and_perturbed_plans(system):
    sysm, setups, T = load_system(system, "theta")
    cert = jax.jit(lambda x0, U, hy, hm, fw, lv: patrol_certificate(T, x0, U, hy, hm, fw, lv)[0],
                   static_argnums=(5,))
    rng = np.random.default_rng(2)
    n_ok = n = 0
    for trial in range(12):
        lv = trial % len(setups)
        st = setups[lv]
        p = st.problem
        x0 = sysm.random_start(rng, st)
        k = int(rng.integers(0, 40))
        fw = first_enforced_window(k, p.h)
        hy = np.tile(x0[:2], (p.h - 1, 1)) + rng.normal(0, 0.05, (p.h - 1, 2))
        hist = History(hy, [None] * (p.h - 1))
        hm = (np.arange(p.h - 1) >= fw).astype(float)
        st.nominal.reset(x0)
        Xn, Un = st.nominal.rollout(p.model, x0, p.n_plan)
        mdl = RecedingISTL(p, fw, ISTLSettings(time_limit=5))
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
    assert n >= 10 and 0 < n_ok


def test_policy_save_load_roundtrip(tmp_path):
    params = init_mlp(jax.random.PRNGKey(0), [5, 8, 3])
    pol = SegmentPolicy(params, np.zeros(5, np.float32), np.ones(5, np.float32), (8,))
    pol.save(tmp_path / "p.npz")
    back = SegmentPolicy.load(tmp_path / "p.npz")
    o = jnp.ones(5)
    assert np.allclose(pol.raw(o), back.raw(o))
    assert back.reference == "rerooted"
    dataclasses.replace(pol, reference="clock").save(tmp_path / "c.npz")
    assert SegmentPolicy.load(tmp_path / "c.npz").reference == "clock"


def test_learned_proposer_rejects_a_reference_mismatch():
    from rl_stl.policies.segment import LearnedProposer

    _, setups, T = load_system("di", "theta")
    params = init_mlp(jax.random.PRNGKey(0), [T.obs_size, 8, T.action_size])
    pol = SegmentPolicy(params, np.zeros(T.obs_size), np.ones(T.obs_size), (8,), "clock")
    with pytest.raises(ValueError):
        LearnedProposer(pol, T, 0, setups[0].nominal)


def test_plan_env_nominal_matches_numpy_and_steps():
    _, setups, T = load_system("di", "theta")
    st = setups[1]
    p = st.problem
    env = PatrolPlanEnv(T, st.nominal, p.track_weight, p.gamma0, p.n_cost, p.effort_weight,
                        np.zeros(T.obs_size), np.ones(T.obs_size))
    x0 = np.array([1.6, 1.4, 0.1, -0.2])
    st.nominal.reset(x0)
    Xn, Un = st.nominal.rollout(p.model, x0, p.n_plan)
    Xj, Uj = env.nominal_rollout(jnp.asarray(x0, jnp.float32), st.nominal.index)
    assert np.allclose(Xn, np.asarray(Xj), atol=1e-4) and np.allclose(Un, np.asarray(Uj), atol=1e-4)
    s = jax.jit(env.reset)(jax.random.PRNGKey(3))
    step = jax.jit(env.step)
    for _ in range(T.h + 2):
        s = step(s, jnp.zeros(T.action_size))
    assert np.isfinite(float(s.reward))
    assert int(s.data["first_window"]) == 0
    assert dataclasses.is_dataclass(env.cfg)


def test_plan_env_clock_reference_matches_numpy_and_steps():
    from rl_stl.policies.nominal import ClockReference

    _, setups, T = load_system("di", "theta", "clock")
    st = setups[1]
    p = st.problem
    env = PatrolPlanEnv(T, st.nominal, p.track_weight, p.gamma0, p.n_cost, p.effort_weight,
                        np.zeros(T.obs_size), np.ones(T.obs_size))
    x0 = np.array([1.6, 1.4, 0.1, -0.2])
    ref = ClockReference(st.nominal, p.model)
    ref.reset(x0)
    L = 60
    Xj, Uj, Ij = env.clock_reference(jnp.asarray(x0, jnp.float32), st.nominal.index, L)
    X, U = ref.window(0, L)
    assert np.allclose(X, np.asarray(Xj), atol=1e-3) and np.allclose(U, np.asarray(Uj), atol=1e-3)
    assert [ref.index_at(k) for k in range(L)] == [int(i) for i in Ij]
    s = jax.jit(env.reset)(jax.random.PRNGKey(3))
    assert s.obs.shape == (T.obs_size,)
    step = jax.jit(env.step)
    for k in range(T.h + 2):
        s = step(s, jnp.zeros(T.action_size))
    assert np.isfinite(float(s.reward)) and int(s.data["k"]) == T.h + 2
    assert "waypoints" in s.metrics


def test_clock_reference_input_scale_binds_and_env_matches_numpy(tmp_path):
    """The blimp's clock reference saturates at 0.8 x bounds, identically in numpy and JAX;
    the scale is saved with the policy and a mismatch is refused."""
    from rl_stl.policies.nominal import ClockReference
    from rl_stl.policies.segment import LearnedProposer

    sysm, setups, T = load_system("blimp", "theta", "clock")
    assert T.ref_input_scale == 0.8 and load_system("blimp", "theta")[2].ref_input_scale == 1.0
    st = setups[0]
    p = st.problem
    env = PatrolPlanEnv(T, st.nominal, p.track_weight, p.gamma0, p.n_cost, p.effort_weight,
                        np.zeros(T.obs_size), np.ones(T.obs_size))
    x0 = sysm.eval_start(np.random.default_rng(12345), st)
    ref = ClockReference(st.nominal, p.model, T.ref_input_scale)
    ref.reset(x0)
    L = 80
    X, U = ref.window(0, L)
    Xj, Uj, _ = env.clock_reference(jnp.asarray(x0, jnp.float32), st.nominal.index, L)
    assert np.allclose(X, np.asarray(Xj), atol=1e-3) and np.allclose(U, np.asarray(Uj), atol=1e-5)
    assert np.all(U <= 0.8 * p.u_hi + 1e-12) and np.all(U >= 0.8 * p.u_lo - 1e-12)
    full = ClockReference(st.nominal, p.model)
    full.reset(x0)
    assert np.max(np.abs(full.window(0, L)[1])) > np.max(np.abs(U)) + 1e-4  # the scale binds
    with pytest.raises(ValueError):
        ClockReference(st.nominal, p.model, 0.0)

    params = init_mlp(jax.random.PRNGKey(0), [T.obs_size, 8, T.action_size])
    pol = SegmentPolicy(params, np.zeros(T.obs_size), np.ones(T.obs_size), (8,), "clock", T.ref_input_scale)
    pol.save(tmp_path / "p.npz")
    assert SegmentPolicy.load(tmp_path / "p.npz").ref_input_scale == 0.8
    LearnedProposer(pol, T, 0, st.nominal)
    pol.ref_input_scale = 1.0
    with pytest.raises(ValueError):
        LearnedProposer(pol, T, 0, st.nominal)
