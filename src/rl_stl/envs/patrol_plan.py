"""Planning-level patrol environment for PPO fine-tuning of the segment proposer (pure JAX).

One environment step is one planning step ``dt``. The action is a whole proposal: ``N``
synthesis-model inputs in ``(-1, 1)`` (Brax's tanh-Gaussian policy), mapped into the tightened
input box exactly as at runtime (:func:`rl_stl.policies.segment.squash` semantics). The reward
scores the *proposal*, so PPO optimises what the runtime check and the synthesis objective
reward, and no solver is ever called inside the JIT-compiled batch:

    r = - c_cost * J(plan; nominal)  -  c_cert * max(0, eps - cert(plan))  +  c_ok * 1[cert >= eps]
        - c_syn * 1[cert < 0]

``J`` is the synthesis objective (discounted L1 deviation from the nominal patrol reference plus
input effort), ``cert`` the differentiable runtime check (:func:`patrol_certificate`). The
first planned input is then applied to ``f_syn`` with uncertainty inside the level's bound:

* double integrator: ``x+ = A x + B (u + w)``, ``w`` uniform in ``[-w_bar, w_bar]``;
* constant-tube vehicles (X2, blimp): the measured state is the planned one plus a position
  tracking error uniform in ``[-eps, eps]`` on every signal axis (planar, or x, y, z for the cube
  specs; the tube assumption; MuJoCo is not in the loop).

Initial states: positions on a ring (2-D) or a spherical shell (cube specs) of radius ``start_r``
around the centre of ``S``, uniform velocities on the patrol axes.

The reference is the one the tables were built for (``LevelTables.reference``): ``rerooted``
rolls the nominal out from the current state, with the waypoint index advanced on the realised
trajectory; ``clock`` rolls the nominal patrol out once at reset, from the episode's initial
state (inputs saturated at ``LevelTables.ref_input_scale`` times the bounds), and the plan
rooted at step ``k`` tracks ``x_nom(k..k+N)``. The ``waypoints`` metric
counts waypoints the vehicle itself reaches (tolerance as in the evaluation), whichever
reference the reward uses.

Optional path-accuracy term (``c_wp > 0``; off by default): the executed state's progress along the
closed patrol polyline through the waypoints (the projection of :func:`rl_stl.evaluation.patrol.
patrol_progress`) is tracked, and a step on which it passes a waypoint is scored as in
:func:`~rl_stl.evaluation.patrol.waypoint_miss`: with ``miss`` the distance to the waypoint at the
(linearly interpolated) crossing,

    r_wp = c_wp * ( 1[forward pass] * max(0, 1 - miss / wp_radius)  -  1[backward pass] )

so a precise pass earns up to ``c_wp``, a pass ``wp_radius`` away nothing, and crossing a waypoint
backwards costs the full bonus (no gain from oscillating across a waypoint). With
``wp_target = "plan"`` the same score is summed over the steps of the proposed segment ``X`` (its
``f_syn`` rollout from the current state) instead of the executed step, so the term rates the action
itself at every planning step whose segment passes a waypoint. It only shapes the
reward; the runtime check, the certificate terms and the fallback dynamics are unchanged.

With ``fallback`` the env models the checked loop: a proposal that fails the runtime check
(``cert < 0``) is not flown; the vehicle follows the cached plan (the last accepted proposal),
reaching its next nominal state up to the tube error as the tracker does (constant tube) or
applying its next input plus ``w`` (disturbance), and once the cache is exhausted a stopping
input (PD on velocity, regulated axes to their set-points). Before the first proposal passes
there is no certified plan (the real loop solves at ``t = 0-``), so proposals are flown.
The solver is not modelled: its plan is approximated by the shifted cached plan, which is a
feasible point of the request. ``c_syn`` prices the synthesis call that each rejection triggers
in the real loop. Without ``fallback`` every proposal's first input is flown, as before.

Observations are the runtime features (:func:`features`) normalised with fixed statistics
(those of the behaviour-cloning data) so a BC network maps onto Brax's policy unchanged.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
from mujoco_playground._src import mjx_env

from rl_stl.policies.nominal import PatrolPD
from rl_stl.policies.segment import BIG, LevelTables, features, patrol_certificate, signal_margin


@dataclasses.dataclass(frozen=True)
class PlanEnvConfig:
    episode_steps: int = 120
    c_cost: float = 0.1
    c_cert: float = 20.0
    c_ok: float = 1.0
    eps_cert: float = 0.005
    uncertainty: str = "disturbance"  # "disturbance" (B w) or "tube" (planar error)
    start_r: tuple[float, float] = (1.5, 2.3)
    start_v: float = 0.3
    waypoint_metric_tol: float = 0.25  # as experiments/patrol/evaluate.py::summarize
    fallback: bool = False  # rejected proposals are replaced by the cached plan (checked loop)
    c_syn: float = 0.0  # price of the synthesis call triggered by a rejected proposal
    c_wp: float = 0.0  # path-accuracy bonus per waypoint pass (0: off, reward unchanged)
    wp_radius: float = 0.5  # pass distance [m] at which the path-accuracy bonus reaches 0
    wp_target: str = "executed"  # passes scored: "executed" (the flown step) or "plan" (the proposed segment)


class PatrolPlanEnv:
    """Minimal Playground-compatible env (``reset``/``step``, ``observation_size``/``action_size``)."""

    def __init__(self, T: LevelTables, nominal: PatrolPD, track_weight: np.ndarray, gamma0: float,
                 n_cost: int | None, effort_weight: float, obs_mean: np.ndarray, obs_std: np.ndarray,
                 cfg: PlanEnvConfig = PlanEnvConfig()):
        self.T, self.cfg = T, cfg
        self.obs_mean, self.obs_std = jnp.asarray(obs_mean), jnp.asarray(obs_std)
        self.A, self.B, self.C = jnp.asarray(T.A), jnp.asarray(T.B), jnp.asarray(T.C_out)
        N = T.n_plan
        Nc = N if n_cost is None else min(n_cost, N)
        self.cost_w = jnp.asarray([gamma0**i if 1 <= i <= Nc else 0.0 for i in range(N + 1)])
        self.track_w = jnp.asarray(track_weight)
        self.effort_w = float(effort_weight)
        ax = nominal.planar
        self.pos = jnp.array([a.pos for a in ax])
        self.vel = jnp.array([a.vel for a in ax])
        self.inp = jnp.array([a.input for a in ax])
        self.kp = jnp.array([a.kp for a in ax])
        self.kd = jnp.array([a.kd for a in ax])
        self.n_ax = len(ax)
        self.n_wp = int(np.asarray(nominal.waypoints).shape[0])
        if self.n_ax != T.p or self.n_wp != T.n_waypoints:
            raise ValueError("nominal patrol axes / waypoints do not match the level tables")
        self.reg = [(a.pos, a.vel, a.input, a.kp, a.kd, sp) for a, sp in nominal.regulated]
        self.wps = jnp.asarray(nominal.waypoints)
        self.wp_tol = float(nominal.waypoint_tol)
        self.leg_a, self.leg_ab = self.wps, jnp.roll(self.wps, -1, axis=0) - self.wps
        self.u_lo, self.u_hi = jnp.asarray(T.u_lo), jnp.asarray(T.u_hi)
        self.clock = T.reference == "clock"

    @property
    def observation_size(self) -> int:
        return self.T.obs_size

    @property
    def action_size(self) -> int:
        return self.T.action_size

    # ----------------------------------------------------------------- nominal
    def _advance(self, x, idx):
        d = jnp.linalg.norm(self.wps[idx] - x[self.pos])
        return jnp.where(d < self.wp_tol, (idx + 1) % self.n_wp, idx)

    def _u_nom(self, x, idx):
        u = jnp.zeros(self.T.m)
        u = u.at[self.inp].set(self.kp * (self.wps[idx] - x[self.pos]) - self.kd * x[self.vel])
        for p, v, i, kp, kd, sp in self.reg:
            u = u.at[i].set(kp * (sp - x[p]) - kd * x[v])
        return jnp.clip(u, self.u_lo, self.u_hi)

    def _u_stop(self, x):
        """Stopping input past the end of the cached plan: damp the planar velocity, hold set-points."""
        u = jnp.zeros(self.T.m)
        u = u.at[self.inp].set(-self.kd * x[self.vel])
        for p, v, i, kp, kd, sp in self.reg:
            u = u.at[i].set(kp * (sp - x[p]) - kd * x[v])
        return jnp.clip(u, self.u_lo, self.u_hi)

    def nominal_rollout(self, x0, idx):
        def f(carry, _):
            x, i = carry
            i = self._advance(x, i)
            u = self._u_nom(x, i)
            xn = self.A @ x + self.B @ u
            return (xn, i), (xn, u)
        _, (X, U) = jax.lax.scan(f, (x0, idx), None, length=self.T.n_plan)
        return jnp.concatenate([x0[None], X]), U

    def clock_reference(self, x0, idx, length: int):
        """Clock reference from the episode start: ``X (length+1, n)``, ``U (length, m)``, and the
        waypoint index ``I (length,)`` used at each step (as :class:`ClockReference`)."""
        s = self.T.ref_input_scale

        def f(carry, _):
            x, i = carry
            i = self._advance(x, i)
            u = jnp.clip(self._u_nom(x, i), s * self.u_lo, s * self.u_hi)
            xn = self.A @ x + self.B @ u
            return (xn, i), (xn, u, i)
        _, (X, U, I) = jax.lax.scan(f, (x0, idx), None, length=length)
        return jnp.concatenate([x0[None], X]), U, I

    def _reference(self, s):
        """``(x_nom (N+1, n), u_nom (N, m), waypoint index)`` for the plan rooted at ``s``."""
        if not self.clock:
            Xn, Un = self.nominal_rollout(s["x"], s["wp"])
            return Xn, Un, s["wp"]
        N, (n, m) = self.T.n_plan, (self.T.n, self.T.m)
        k = jnp.minimum(s["k"], s["Uref"].shape[0] - N)
        Xn = jax.lax.dynamic_slice(s["Xref"], (k, 0), (N + 1, n))
        Un = jax.lax.dynamic_slice(s["Uref"], (k, 0), (N, m))
        return Xn, Un, s["Iref"][k]

    def _patrol_s(self, p):
        """Path parameter of ``p`` on the closed polyline through the waypoints, in ``[0, n_wp]``
        (as :func:`rl_stl.evaluation.patrol._patrol_projection`)."""
        A, AB = self.leg_a, self.leg_ab
        t = jnp.clip(jnp.sum((p[None] - A) * AB, axis=1) / jnp.sum(AB * AB, axis=1), 0.0, 1.0)
        d = jnp.linalg.norm(p[None] - (A + t[:, None] * AB), axis=1)
        i = jnp.argmin(d)
        return i + t[i]

    def _waypoint_pass(self, p0, p1):
        """``(forward pass, backward pass, miss [m])`` of the step ``p0 -> p1`` across a waypoint
        (at most one per step; the waypoint legs are much longer than one step)."""
        n = self.n_wp
        s0, s1 = self._patrol_s(p0), self._patrol_s(p1)
        ds = jnp.mod(s1 - s0 + n / 2, n) - n / 2
        s1 = s1 + n * jnp.round((s0 + ds - s1) / n)  # unwrapped next to s0, exact (no s0 + ds rounding)
        fwd = (ds > 0) & (jnp.floor(s1) > jnp.floor(s0))
        bwd = (ds < 0) & (jnp.floor(s0) > jnp.floor(s1))
        m = jnp.floor(jnp.maximum(s0, s1))
        f = jnp.clip((m - s0) / jnp.where(ds == 0, 1.0, ds), 0.0, 1.0)
        miss = jnp.linalg.norm(p0 + f * (p1 - p0) - self.wps[jnp.mod(m, n).astype(jnp.int32)])
        return fwd, bwd, miss

    # ----------------------------------------------------------------- helpers
    def _obs(self, s):
        fw = s["first_window"]
        mask = (jnp.arange(self.T.h - 1) >= fw).astype(jnp.float32)
        Xn, _, wp = self._reference(s) if self.clock else (None, None, s["wp"])
        o = features(self.T, s["x"], s["hist_y"], mask, wp, s["level"], jnp, x_ref=Xn)
        return (o - self.obs_mean) / self.obs_std

    def _plan(self, s, action):
        kz = jnp.asarray(self.T.kz)[s["level"]]
        lo, hi = self.u_lo + kz, self.u_hi - kz
        a = jnp.clip(action, -1.0, 1.0).reshape(self.T.n_plan, self.T.m)
        return (lo + hi) / 2 + (hi - lo) / 2 * a

    # --------------------------------------------------------------------- API
    def reset(self, rng: jax.Array) -> mjx_env.State:
        T, cfg = self.T, self.cfg
        k1, k2, k3, k4, k5 = jax.random.split(rng, 5)
        level = jax.random.randint(k1, (), 0, T.w_levels.shape[0])
        r = jax.random.uniform(k3, (), minval=cfg.start_r[0], maxval=cfg.start_r[1])
        if self.n_ax == 2:
            ang = jax.random.uniform(k2, (), minval=0.0, maxval=2 * jnp.pi)
            direction = jnp.array([jnp.cos(ang), jnp.sin(ang)])
        else:  # uniform on the sphere
            g = jax.random.normal(k2, (self.n_ax,))
            direction = g / jnp.maximum(jnp.linalg.norm(g), 1e-9)
        x = jnp.zeros(T.n)
        x = x.at[self.pos].set(r * direction)
        x = x.at[self.vel].set(jax.random.uniform(k4, (self.n_ax,), minval=-cfg.start_v, maxval=cfg.start_v))
        p = x[self.pos]
        wp = (jnp.argmin(jnp.linalg.norm(self.wps - p, axis=1)) + 1) % self.n_wp
        data = {"x": x, "hist_y": jnp.zeros((T.h - 1, self.C.shape[0])), "first_window": T.h - 1,
                "wp": wp, "wp_real": wp, "level": level, "k": 0, "rng": k5}
        if self.clock:
            data["Xref"], data["Uref"], data["Iref"] = self.clock_reference(x, wp, cfg.episode_steps + T.n_plan)
        # cached plan (states, inputs) and its next step; none at reset (jc = N): stop until a proposal passes
        data["Xc"], data["Uc"], data["jc"] = jnp.zeros((T.n_plan + 1, T.n)), jnp.zeros((T.n_plan, T.m)), T.n_plan
        data["hc"] = False  # a certified plan has been cached
        metrics = {"cert_ok": jnp.zeros(()), "cert": jnp.zeros(()), "plan_cost": jnp.zeros(()),
                   "realised_violation": jnp.zeros(()), "waypoints": jnp.zeros(()), "reward": jnp.zeros(()),
                   "rejected": jnp.zeros(()), "wp_pass": jnp.zeros(()), "wp_pass_miss": jnp.zeros(()),
                   "wp_bonus": jnp.zeros(())}
        return mjx_env.State(data, self._obs(data), jnp.zeros(()), jnp.zeros(()), metrics, {})

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        T, cfg, s = self.T, self.cfg, state.data
        U = self._plan(s, action)
        mask = (jnp.arange(T.h - 1) >= s["first_window"]).astype(jnp.float32)
        cert, _ = patrol_certificate(T, s["x"], U, s["hist_y"], mask, s["first_window"], s["level"])
        X = jax.lax.scan(lambda x, u: (self.A @ x + self.B @ u, self.A @ x + self.B @ u), s["x"], U)[1]
        X = jnp.concatenate([s["x"][None], X])
        Xn, Un, _ = self._reference(s)
        J = jnp.sum(self.cost_w[:, None] * self.track_w[None] * jnp.abs(X - Xn)) \
            + self.effort_w * jnp.sum(jnp.abs(U - Un))
        ok = cert >= cfg.eps_cert
        accepted = cert >= 0.0  # the runtime check
        reward = -cfg.c_cost * J - cfg.c_cert * jnp.maximum(0.0, cfg.eps_cert - jnp.maximum(cert, -1.0)) \
            + cfg.c_ok * ok - cfg.c_syn * (1.0 - accepted)
        # apply the first planned input under bounded uncertainty
        rng, kw = jax.random.split(s["rng"])
        bound = jnp.asarray(T.w_levels)[s["level"]]
        if cfg.uncertainty == "disturbance":
            w = jax.random.uniform(kw, (T.m,), minval=-1.0, maxval=1.0) * jnp.resize(bound, (T.m,))
            x_next = self.A @ s["x"] + self.B @ (U[0] + w)
        else:
            e = jax.random.uniform(kw, (self.n_ax,), minval=-1.0, maxval=1.0) * bound[:self.n_ax]
            x_next = (self.A @ s["x"] + self.B @ U[0]).at[self.pos].add(e)
        Xc, Uc, jc = s["Xc"], s["Uc"], s["jc"]
        if cfg.fallback:
            # rejected: fly the cached plan; under the tube assumption the tracker keeps the state within
            # the tube of the plan's next nominal state (constant tube) or adds w to its input (disturbance)
            N = T.n_plan
            j = jnp.minimum(jc, N - 1)
            if cfg.uncertainty == "disturbance":
                x_cache = self.A @ s["x"] + self.B @ (Uc[j] + w)
                x_stop = self.A @ s["x"] + self.B @ (self._u_stop(s["x"]) + w)
            else:
                x_cache = Xc[j + 1].at[self.pos].add(e)
                x_stop = (self.A @ s["x"] + self.B @ self._u_stop(s["x"])).at[self.pos].add(e)
            # before the first accepted proposal there is no certified plan (the real loop has the t = 0-
            # solve): fly the proposal, as without fallback
            x_next = jnp.where(accepted | ~s["hc"], x_next, jnp.where(jc < N, x_cache, x_stop))
            Xc, Uc = jnp.where(accepted, X, Xc), jnp.where(accepted, U, Uc)
            jc = jnp.where(accepted, 1, jc + 1)
            hc = s["hc"] | accepted
        else:
            hc = s["hc"]
        y_now = self.C @ s["x"]
        hist = jnp.concatenate([s["hist_y"][1:], y_now[None]])
        fw = jnp.maximum(s["first_window"] - 1, 0)
        # realised violation of the window [k-h+1, k+1] that just closed (all samples measured;
        # it exists once k >= h-1, i.e. first_window == 0 before the update)
        Ymeas = jnp.concatenate([s["hist_y"], y_now[None], (self.C @ x_next)[None]])  # h+1 samples
        m = jnp.where(signal_margin(T, Ymeas, jnp) >= 0, 1.0, 0.0)
        if T.base_spec == "theta":
            sat = jnp.max(m)
        else:
            G = jnp.stack([jnp.min(m[j:j + T.hG + 1]) for j in range(T.hF + 1)])
            sat = jnp.maximum(m[0], jnp.max(G))
        violated = jnp.where(s["first_window"] == 0, 1.0 - sat, 0.0)
        reward = reward - 5.0 * violated
        fwd, bwd, miss = self._waypoint_pass(s["x"][self.pos], x_next[self.pos])
        if cfg.wp_target == "plan":  # passes of the proposed segment (scores the action directly)
            pf, pb, pm = jax.vmap(self._waypoint_pass)(X[:-1, self.pos], X[1:, self.pos])
            wp_bonus = cfg.c_wp * jnp.sum(pf * jnp.maximum(0.0, 1.0 - pm / cfg.wp_radius) - pb)
        else:
            wp_bonus = cfg.c_wp * (fwd * jnp.maximum(0.0, 1.0 - miss / cfg.wp_radius) - bwd)
        if cfg.c_wp != 0.0:  # off by default: the reward graph is then exactly the previous one
            reward = reward + wp_bonus
        wp = self._advance(x_next, s["wp"])
        reached = jnp.linalg.norm(self.wps[s["wp_real"]] - x_next[self.pos]) < cfg.waypoint_metric_tol
        wp_real = jnp.where(reached, (s["wp_real"] + 1) % self.n_wp, s["wp_real"])
        data = {**s, "x": x_next, "hist_y": hist, "first_window": fw, "wp": wp, "wp_real": wp_real,
                "k": s["k"] + 1, "rng": rng, "Xc": Xc, "Uc": Uc, "jc": jc, "hc": hc}
        metrics = {"cert_ok": ok.astype(jnp.float32), "cert": jnp.clip(cert, -BIG, BIG),
                   "plan_cost": J, "realised_violation": violated, "waypoints": reached.astype(jnp.float32),
                   "reward": reward, "rejected": 1.0 - accepted.astype(jnp.float32),
                   "wp_pass": fwd.astype(jnp.float32), "wp_pass_miss": jnp.where(fwd, miss, 0.0),
                   "wp_bonus": wp_bonus}
        return mjx_env.State(data, self._obs(data), reward, jnp.zeros(()), metrics, state.info)
