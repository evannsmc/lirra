"""Playground X2 hover environment: shapes, frames, MJX-vs-C consistency, STL term, termination."""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rl_stl.envs.skydio_x2.host_sim import X2HostSim
from rl_stl.envs.skydio_x2.hover import (
    X2Hover,
    default_config,
    hover_box_formula,
    periodic_exit_formula,
)
from rl_stl.stl.jax_robustness import compile_robustness


def _deterministic_config(**stl):
    cfg = default_config()
    cfg.reset.pos_range = [0.0, 0.0, 0.0]
    cfg.reset.roll_pitch_deg = 0.0
    cfg.reset.yaw_range = 0.0
    cfg.reset.lin_vel = 0.0
    cfg.reset.ang_vel = 0.0
    for k, v in stl.items():
        cfg.stl[k] = v
    return cfg


@pytest.fixture(scope="module")
def env():
    return X2Hover(_deterministic_config())


@pytest.fixture(scope="module")
def jit_fns(env):
    return jax.jit(env.reset), jax.jit(env.step)


def test_reset_observation_layout(env, jit_fns):
    reset, _ = jit_fns
    s = reset(jax.random.PRNGKey(0))
    obs = np.asarray(s.obs)
    assert obs.shape == (25,) and env.observation_size == 25 and env.action_size == 4
    np.testing.assert_allclose(obs[0:3], 0.0, atol=1e-6)  # at target
    np.testing.assert_allclose(obs[3:6], 0.0, atol=1e-6)
    np.testing.assert_allclose(obs[6:15], np.eye(3).reshape(9), atol=1e-6)
    np.testing.assert_allclose(obs[22:25], [1.0, 1.0, 0.0], atol=1e-6)  # deadline clock, in box, time frac
    assert env.n_substeps == 2


def test_zero_action_hovers(jit_fns):
    reset, step = jit_fns
    s = reset(jax.random.PRNGKey(0))
    for _ in range(50):
        s = step(s, jnp.zeros(4))
    assert float(s.done) == 0.0
    assert float(s.metrics["task/pos_err"]) < 1e-3
    assert float(s.reward) > 1.8  # pos + vel + upright + alive at their maxima


def test_mjx_matches_host_c_mujoco(env, jit_fns):
    reset, step = jit_fns
    s = reset(jax.random.PRNGKey(0))
    host = X2HostSim()
    host.reset(np.asarray(s.data.qpos, dtype=float), np.asarray(s.data.qvel, dtype=float))
    rng = np.random.default_rng(0)
    actions = np.clip(rng.normal(scale=0.3, size=(40, 4)), -1, 1)
    for a in actions:
        s = step(s, jnp.asarray(a, dtype=jnp.float32))
        for _ in range(env.n_substeps):
            host.step_action(a)
    np.testing.assert_allclose(np.asarray(s.data.qpos), host.data.qpos, atol=2e-3)
    np.testing.assert_allclose(np.asarray(s.data.qvel), host.data.qvel, atol=2e-2)


def test_body_rate_action_frame(jit_fns):
    reset, step = jit_fns
    s = reset(jax.random.PRNGKey(0))
    for _ in range(15):  # 0.3 s of +p command
        s = step(s, jnp.array([0.0, 0.2, 0.0, 0.0]))
    obs = np.asarray(s.obs)
    assert obs[15] == pytest.approx(0.2 * 2 * math.pi, rel=0.02)  # ang_vel_body[0]
    assert abs(obs[16]) < 0.02 and abs(obs[17]) < 0.02
    np.testing.assert_allclose(obs[18:22], [0.0, 0.2, 0.0, 0.0], atol=1e-7)  # last_action


def test_info_reinitialized_after_data_reset(jit_fns):
    reset, step = jit_fns
    s0 = reset(jax.random.PRNGKey(0))
    s = s0
    for _ in range(5):
        s = step(s, jnp.array([0.1, 0.0, 0.0, 0.0]))
    assert int(s.info["x2_step"]) == 5
    # Emulate BraxAutoResetWrapper(full_reset=False): data/obs restored, info kept.
    s = s.replace(data=s0.data, obs=s0.obs)
    s = step(s, jnp.zeros(4))
    assert int(s.info["x2_step"]) == 1
    assert float(s.metrics["reward/action_rate"]) == pytest.approx(0.0, abs=1e-9)


def test_stl_hinge_penalty_timing():
    cfg = _deterministic_config(enabled=True, window=5, deadline_s=0.0, delta=0.05)
    cfg.stl.box_halfwidth = [0.2, 0.2, 0.2]
    env = X2Hover(cfg)
    reset, step = jax.jit(env.reset), jax.jit(env.step)
    s = reset(jax.random.PRNGKey(0))
    assert s.obs.shape == (25 + 15,)
    penalties = []
    for _ in range(6):
        s = step(s, jnp.zeros(4))
        penalties.append(float(s.metrics["stl/penalty"]))
    # Hovering at the target: rho = 0.2 >= delta, no penalty; nothing scored before window fills.
    assert penalties == [0.0] * 6
    assert float(s.metrics["stl/rho_window"]) == pytest.approx(0.2, abs=1e-3)

    cfg.target = [0.0, 0.0, 1.2]  # spawn at target, then evaluate against a box 0.3 m away
    env_far = X2Hover(cfg)
    s = jax.jit(env_far.reset)(jax.random.PRNGKey(0))
    far_step = jax.jit(env_far.step)
    s = s.replace(data=s.data.replace(qpos=s.data.qpos.at[2].add(0.3)))
    vals = []
    for _ in range(6):
        s = far_step(s, jnp.zeros(4))
        vals.append(float(s.metrics["stl/penalty"]))
    assert vals[:3] == [0.0, 0.0, 0.0]  # window of 5 not yet full
    assert vals[4] > 0.05 and vals[5] > 0.05  # rho ~ -0.1 -> penalty ~ 0.15


@pytest.mark.parametrize(
    "build, closed_form",
    [
        (hover_box_formula, lambda y, h: np.min(h - np.abs(y))),
        (periodic_exit_formula, lambda y, h: np.max(np.abs(y) - h)),
    ],
)
def test_x2_spec_robustness_matches_stlpy_and_closed_form(build, closed_form):
    h = np.array([0.2, 0.15, 0.25])
    formula = build(h, 7)
    fn = compile_robustness(formula)
    rng = np.random.default_rng(3)
    for _ in range(20):
        y = rng.normal(scale=0.2, size=(7, 3))
        rho = float(fn(jnp.asarray(y)))
        ref = float(np.asarray(formula.robustness(np.ascontiguousarray(y.T), 0)).reshape(-1)[0])
        assert rho == pytest.approx(ref, abs=1e-5)
        assert rho == pytest.approx(closed_form(y, h), abs=1e-5)


def test_stl_periodic_exit_penalty():
    cfg = _deterministic_config(enabled=True, spec="periodic_exit", window=5, deadline_s=0.0, delta=0.05)
    cfg.stl.box_halfwidth = [0.2, 0.2, 0.2]
    env = X2Hover(cfg)
    reset, step = jax.jit(env.reset), jax.jit(env.step)
    s = reset(jax.random.PRNGKey(0))
    penalties = []
    for _ in range(6):
        s = step(s, jnp.zeros(4))
        penalties.append(float(s.metrics["stl/penalty"]))
    # Parked at the target, the drone never leaves the box: rho = -0.2, penalty = delta + 0.2.
    assert penalties[:3] == [0.0, 0.0, 0.0]
    np.testing.assert_allclose(penalties[3:], 0.25, atol=2e-3)
    assert float(s.metrics["stl/rho_window"]) == pytest.approx(-0.2, abs=1e-3)

    cfg.target = [0.0, 0.0, 1.2]  # spawn at target, then sit 0.3 m above it (outside the box)
    env_out = X2Hover(cfg)
    s = jax.jit(env_out.reset)(jax.random.PRNGKey(0))
    out_step = jax.jit(env_out.step)
    s = s.replace(data=s.data.replace(qpos=s.data.qpos.at[2].add(0.3)))
    vals = []
    for _ in range(6):
        s = out_step(s, jnp.zeros(4))
        vals.append(float(s.metrics["stl/penalty"]))
    assert vals == [0.0] * 6  # rho ~ 0.1 >= delta
    assert float(s.metrics["stl/rho_window"]) == pytest.approx(0.1, abs=5e-3)


def test_unknown_stl_spec_rejected():
    with pytest.raises(ValueError, match="unknown stl.spec"):
        X2Hover(_deterministic_config(spec="nope"))


def test_waypoint_target_randomization():
    cfg = _deterministic_config()
    cfg.target_range = [1.0, 1.0, 0.5]
    env = X2Hover(cfg)
    reset, step = jax.jit(env.reset), jax.jit(env.step)
    targets, errs = [], []
    for seed in range(3):
        s = reset(jax.random.PRNGKey(seed))
        targets.append(np.asarray(s.info["x2_target"]))
        s = step(s, jnp.zeros(4))
        # Spawned at the target (pos_range = 0) and hovering: error stays ~0 w.r.t. the sampled target.
        errs.append(float(s.metrics["task/pos_err"]))
        np.testing.assert_allclose(np.asarray(s.obs[0:3]), 0.0, atol=1e-3)
    targets = np.array(targets)
    assert np.all(np.abs(targets - [0.0, 0.0, 1.5]) <= [1.0, 1.0, 0.5])
    assert np.ptp(targets, axis=0).max() > 0.05
    assert max(errs) < 1e-3


def test_tilt_termination(env, jit_fns):
    reset, step = jit_fns
    s = reset(jax.random.PRNGKey(0))
    q = np.array([math.cos(math.pi / 4), math.sin(math.pi / 4), 0.0, 0.0])  # 90 deg roll
    s = s.replace(data=s.data.replace(qpos=s.data.qpos.at[3:7].set(jnp.asarray(q, dtype=jnp.float32))))
    s = step(s, jnp.zeros(4))
    assert float(s.done) == 1.0


@pytest.mark.slow
def test_tracker_policy_survives_random_resets():
    """Scripted CTBR tracker holds hover from randomized resets (incl. yaw near +-pi) in MJX."""
    from rl_stl.envs.skydio_x2 import frames
    from rl_stl.envs.skydio_x2.tracker import make_tracker_params, tracker_action

    env = X2Hover()
    tp = make_tracker_params(env.x2.mass, env.x2.gravity)
    com, target = jnp.asarray(env.x2.com_body), jnp.asarray(env.unwrapped._config.target)

    def policy(state):
        d = state.data
        quat, w = d.qpos[3:7], d.qvel[3:6]
        p = d.qpos[:3] + frames.rotate(quat, com)
        v = frames.point_velocity_world(quat, d.qvel[:3], w, com)
        return tracker_action(quat, p, v, target, jnp.zeros(3), jnp.zeros(3), tp, env.ctbr_params)

    batch = 64
    reset = jax.jit(jax.vmap(env.reset))
    step = jax.jit(jax.vmap(lambda s: env.step(s, policy(s))))
    s = reset(jax.random.split(jax.random.PRNGKey(0), batch))
    alive = np.ones(batch)
    for _ in range(400):
        s = step(s)
        alive *= 1.0 - np.asarray(s.done)
    assert alive.all()
    assert float(np.asarray(s.metrics["task/pos_err"]).max()) < 0.01


@pytest.mark.slow
def test_vmap_batch_smoke():
    env = X2Hover()
    batch = 32
    vreset, vstep = jax.jit(jax.vmap(env.reset)), jax.jit(jax.vmap(env.step))
    s = vreset(jax.random.split(jax.random.PRNGKey(1), batch))
    for _ in range(20):
        s = vstep(s, jnp.zeros((batch, 4)))
    assert s.obs.shape == (batch, 25)
    assert bool(jnp.isfinite(s.obs).all()) and bool(jnp.isfinite(s.reward).all())
