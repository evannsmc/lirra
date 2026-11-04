"""NAHS patrol experiment builders and the double-integrator closed loop.

:class:`PatrolDIConfig` fully determines the double-integrator benchmark: ``f_syn`` is the
exact ZOH model, ``f_sim = f_syn + B w`` with ``w`` sampled inside ``W`` (so the tube
assumption holds by construction), and the vehicle flies the supervisor's command directly.
The X2 and blimp experiments reuse :func:`build_patrol_problem` with their own ``f_syn`` and an
empirical constant tube.

The cube specs (``theta_plus``, ``phi_plus``) use the 3-D double integrator: the same per-axis
bounds, PD, LQR gains and disturbance bound on x, y and z, the signal ``y = (px, py, pz)``, six
terminal face sets and six waypoints.

Uncertain predicates (``*_u`` specs, :mod:`rl_stl.stl.nahs`): the face sets must satisfy the interval
predicate of their face with margin ``delta`` and the widest tube for every state in them. The face
offset ``c_out`` is the worst case over the face set's extent (tangential positions anywhere in the
tightened state box), computed by :func:`interval_face_offset` and verified on the built polygons by
:func:`verify_interval_faces`. :func:`uncertain_scores` scores an executed trace against the interval
predicates and against a drawn true region.
"""

from __future__ import annotations

import dataclasses
from typing import Literal

import numpy as np

from rl_stl.dynamics.linear import DiscreteLinearModel, dlqr, zoh_double_integrator
from rl_stl.policies.nominal import AxisPD, PatrolPD
from rl_stl.stl.interval import IntervalPredicate, interval_term_min
from rl_stl.stl.nahs import PatrolTask
from rl_stl.stl.robustness import iter_predicates, robustness
from rl_stl.supervisor.receding import (
    ProposalContext,
    RecedingSupervisor,
    realised_robustness,
)
from rl_stl.synthesis.istl import ISTLSettings, RecedingISTL, RecedingProblem
from rl_stl.synthesis.terminal import axis_blocks, patrol_terminal_set
from rl_stl.synthesis.tube import Tube, disturbance_tube


@dataclasses.dataclass(frozen=True)
class PatrolDIConfig:
    spec: Literal["theta", "phi", "theta_plus", "phi_plus", "theta_u", "phi_u", "theta_plus_u", "phi_plus_u"] = "theta"
    dt: float = 0.25
    n_plan: int = 24
    delta: float = 0.05
    v_max: float = 1.0
    a_max: float = 0.5
    w_bar: float = 0.02  # |w_i| <= w_bar [m/s^2], acceleration disturbance
    lqr_q: tuple[float, float] = (10.0, 1.0)  # position, velocity weights of K
    lqr_r: float = 1.0
    kp: float = 0.6  # nominal PD
    kd: float = 1.2
    track_pos: float = 1.0
    track_vel: float = 0.2
    gamma0: float = 0.95
    effort_weight: float = 0.01
    n_cost: int | None = None
    episode_seconds: float = 60.0
    x0: tuple[float, float, float, float] = (1.5, 1.5, 0.0, 0.0)
    disturbance: Literal["uniform", "persistent", "none"] = "uniform"
    pred_offset: float = 0.05  # _u specs: half-width uncertainty [m] (PatrolTask.pred_offset)
    pred_rotation: float = 0.05  # _u specs: rotation uncertainty [rad]
    nominal_predicates: bool = False  # _u specs ablation: plan/check with the exact nominal predicates


@dataclasses.dataclass
class PatrolSetup:
    task: PatrolTask
    model: DiscreteLinearModel
    problem: RecedingProblem
    K: np.ndarray
    tube: Tube
    nominal: PatrolPD
    w_bar: np.ndarray


def build_patrol_problem(task: PatrolTask, model: DiscreteLinearModel, C_out: np.ndarray,
                         x_lo: np.ndarray, x_hi: np.ndarray, u_lo: np.ndarray, u_hi: np.ndarray,
                         tube: Tube, delta: float, n_plan: int,
                         blocks_spec: dict[str, tuple[tuple[int, int], int]],
                         plane: tuple[str, str], position_index: dict[str, int],
                         track_weight: np.ndarray, gamma0: float, effort_weight: float,
                         n_cost: int | None = None) -> RecedingProblem:
    """Receding problem with the verified face terminal set for the patrol task."""
    blocks = axis_blocks(model, blocks_spec)
    psi = task.formula()
    if task.interval_predicates:
        c_out = interval_face_offset(task.outside_formula(), C_out, x_lo + tube.z_inf, x_hi - tube.z_inf,
                                     np.abs(C_out) @ tube.z_inf, delta)
    else:
        c_out = task.half_width + delta + float(np.max(np.abs(C_out) @ tube.z_inf))
    terminal = patrol_terminal_set(model, blocks, plane, c_out, x_lo + tube.z_inf,
                                   x_hi - tube.z_inf, u_lo + tube.kz_inf, u_hi - tube.kz_inf,
                                   position_index)
    if task.interval_predicates:
        verify_interval_faces(terminal, task.outside_formula(), C_out, position_index,
                              np.abs(C_out) @ tube.z_inf, delta)
    return RecedingProblem(model, C_out, psi, n_plan, x_lo, x_hi, u_lo, u_hi, tube,
                           delta, terminal, task.terminal_block, track_weight, gamma0, n_cost,
                           effort_weight, name=f"patrol_{task.spec}")


def _face_axis(pred: IntervalPredicate) -> tuple[int, float]:
    """Signal coordinate and sign of a face predicate's nominal normal ``sign * e_j``."""
    a = pred.a[:, 0]
    j = int(np.argmax(np.abs(a)))
    if np.count_nonzero(a) != 1 or abs(abs(a[j]) - 1.0) > 1e-12:
        raise ValueError(f"face predicate {pred.name} has no axis-aligned unit nominal normal")
    sign = float(np.sign(a[j]))
    if not (sign * pred.a_lo[j] > 0 and sign * pred.a_hi[j] > 0):
        raise ValueError(f"face predicate {pred.name}: the normal coefficient interval contains 0")
    return j, sign


def interval_face_offset(not_s, C_out: np.ndarray, x_lo: np.ndarray, x_hi: np.ndarray,
                         r: np.ndarray, delta: float) -> float:
    """Smallest ``c`` such that ``sign * y_j >= c`` implies ``lower([a]^T [y - r, y + r] - [b]) >= delta``
    for every face predicate of ``not_s`` and every tangential ``y`` in the box ``C [x_lo, x_hi]``.

    The lower endpoint is separable, ``sum_k f_k(y_k) - b_hi`` with concave ``f_k``
    (:func:`~rl_stl.stl.interval.interval_term_min`): tangential coordinates take their minimum at a box
    corner; along the normal, ``f_j(sign c) = min(|a_lo_j|, |a_hi_j|) (c - r_j)`` for ``c >= r_j``.
    """
    C = np.asarray(C_out, float)
    y_lo = np.minimum(C * x_lo, C * x_hi).sum(axis=1)
    y_hi = np.maximum(C * x_lo, C * x_hi).sum(axis=1)
    c_out = 0.0
    for pred, _ in iter_predicates(not_s):
        if not isinstance(pred, IntervalPredicate):
            raise TypeError("interval_face_offset needs interval face predicates")
        j, _sign = _face_axis(pred)
        tang = 0.0
        for k in range(C.shape[0]):
            if k != j:
                tang += float(min(interval_term_min(pred.a_lo[k], pred.a_hi[k], y, r[k]) for y in (y_lo[k], y_hi[k])))
        a_min = float(min(abs(pred.a_lo[j]), abs(pred.a_hi[j])))
        c_out = max(c_out, float(r[j]) + max(delta + pred.b_hi - tang, 0.0) / a_min)
    return c_out + 1e-9


def verify_interval_faces(terminal, not_s, C_out: np.ndarray, position_index: dict[str, int],
                          r: np.ndarray, delta: float, tol: float = 1e-9) -> None:
    """Raise unless every state of every face set satisfies its face's interval predicate.

    The lower endpoint is separable and concave per signal coordinate, so its minimum over the face set
    (a product of polygons) is a sum of minima of ``f_k`` at the endpoints of the coordinate's range over
    its block polygon (computed from the polygons' vertices)."""
    C = np.asarray(C_out, float)
    preds = {(_face_axis(q)[0], _face_axis(q)[1]): q for q, _ in iter_predicates(not_s)}
    state_block = {}
    for bname, blk in terminal.blocks.items():
        for col, st in enumerate(blk.states):
            state_block[st] = (bname, col)
    for face in terminal.faces:
        sign = 1.0 if face.name[0] == "+" else -1.0
        st = position_index[face.name[1:]]
        js = np.nonzero(C[:, st])[0]
        if len(js) != 1:
            raise ValueError("face axis is not one signal coordinate")
        j = int(js[0])
        pred = preds[(j, sign * float(np.sign(C[j, st])))]
        total = -pred.b_hi
        for k in range(C.shape[0]):
            ks = np.nonzero(C[k])[0]
            if len(ks) != 1:
                raise ValueError("verify_interval_faces needs a selection output map")
            bname, col = state_block[int(ks[0])]
            vals = C[k, ks[0]] * face.polygons[bname].vertices[:, col]
            total += float(min(interval_term_min(pred.a_lo[k], pred.a_hi[k], v, r[k]) for v in (vals.min(), vals.max())))
        if total < delta - tol:
            raise AssertionError(f"face set {face.name}: interval predicate lower bound {total:.6f} < delta {delta}")


def uncertain_scores(task: PatrolTask, Y: np.ndarray, region: dict) -> dict:
    """Scores of an executed output trace ``Y (T, p)`` of a ``_u`` spec: interval lower robustness (worst
    case over the predicate uncertainty) and the exact robustness against the true ``region``, per window
    ``[s, s + h]`` as :func:`rl_stl.supervisor.receding.realised_robustness`."""
    h = task.horizon
    f_lo, f_true = task.interval_formula(), task.true_formula(region)
    lo = np.array([robustness(f_lo, Y[s:s + h + 1]) for s in range(len(Y) - h)])
    tr = np.array([robustness(f_true, Y[s:s + h + 1]) for s in range(len(Y) - h)])
    return {"satisfied_lower": bool(np.all(lo >= 0)), "min_rho_lower": float(np.min(lo)),
            "satisfied_true": bool(np.all(tr >= 0)), "min_rho_true": float(np.min(tr)),
            "frac_violated_windows_true": float(np.mean(tr < 0)),
            "true_half_width": float(region["half_width"]), "true_angle": float(region["angle"]),
            "true_axis": [float(v) for v in region["axis"]]}


def build_di(cfg: PatrolDIConfig) -> PatrolSetup:
    """Double-integrator patrol; ``d = 2`` axes for (24)/(25), ``d = 3`` for the cube specs."""
    task = PatrolTask(spec=cfg.spec, dt=cfg.dt, pred_offset=cfg.pred_offset, pred_rotation=cfg.pred_rotation,
                      nominal_predicates=cfg.nominal_predicates)
    d = task.dim
    model = zoh_double_integrator(d, cfg.dt)
    K = -dlqr(model.A, model.B, np.diag([cfg.lqr_q[0]] * d + [cfg.lqr_q[1]] * d),
              cfg.lqr_r * np.eye(d))
    w_bar = np.full(d, cfg.w_bar)
    tube = disturbance_tube(model.A, model.B, K, model.B, w_bar, n_max=max(200, 4 * cfg.n_plan))
    ws = task.workspace
    x_lo = np.array([-ws] * d + [-cfg.v_max] * d)
    x_hi = -x_lo
    u_lo, u_hi = -cfg.a_max * np.ones(d), cfg.a_max * np.ones(d)
    C = np.eye(2 * d)[:d]
    names = "xyz"[:d]
    problem = build_patrol_problem(
        task, model, C, x_lo, x_hi, u_lo, u_hi, tube, cfg.delta, cfg.n_plan,
        {a: ((i, d + i), i) for i, a in enumerate(names)}, tuple(names), {a: i for i, a in enumerate(names)},
        np.array([cfg.track_pos] * d + [cfg.track_vel] * d), cfg.gamma0, cfg.effort_weight,
        cfg.n_cost)
    nominal = PatrolPD(tuple(AxisPD(i, d + i, i, cfg.kp, cfg.kd) for i in range(d)),
                       task.waypoints(), u_lo, u_hi, task.waypoint_tol)
    return PatrolSetup(task, model, problem, K, tube, nominal, w_bar)


# ------------------------------------------------------------------- proposers
def nominal_proposer(ctx: ProposalContext) -> np.ndarray:
    """Mission-only proposal: the nominal PD rollout (ignores the specification)."""
    return ctx.u_nom


class ExpertProposer:
    """Solves the synthesis problem itself (rooted at the measured state): an ideal policy.

    Used to generate behaviour-cloning data and as an upper bound on what a learned proposer
    can achieve. ``noise`` (scalar or per input) adds Gaussian noise to its output, clipped to
    the input box (imperfect-learner stand-in).
    """

    def __init__(self, problem: RecedingProblem, settings: ISTLSettings = ISTLSettings(time_limit=2.0),
                 noise: float | np.ndarray = 0.0, seed: int = 0):
        self.problem = problem
        self.settings = settings
        self.noise = noise
        self.rng = np.random.default_rng(seed)
        self._models: dict[int, RecedingISTL] = {}
        self.last = None

    def __call__(self, ctx: ProposalContext) -> np.ndarray:
        mdl = self._models.get(ctx.first_window)
        if mdl is None:
            mdl = self._models[ctx.first_window] = RecedingISTL(self.problem, ctx.first_window, self.settings)
        mdl.update(ctx.x, ctx.history, ctx.x_nom, ctx.u_nom)
        res = mdl.solve()
        self.last = res
        U = res.U if res.has_solution else ctx.u_nom
        if np.any(np.asarray(self.noise) > 0):
            U = np.clip(U + np.asarray(self.noise) * self.rng.standard_normal(U.shape),
                        self.problem.u_lo, self.problem.u_hi)
        return U


# ------------------------------------------------------------------- closed loop
def disturbance_sequence(cfg: PatrolDIConfig, n: int, seed: int) -> np.ndarray:
    d = PatrolTask(spec=cfg.spec).dim
    rng = np.random.default_rng(seed)
    w = np.full(d, cfg.w_bar)
    if cfg.disturbance == "none" or cfg.w_bar == 0:
        return np.zeros((n, d))
    if cfg.disturbance == "uniform":
        return rng.uniform(-w, w, size=(n, d))
    # persistent: piecewise-constant worst-case corners held for 2-8 steps
    out = np.empty((n, d))
    k = 0
    while k < n:
        L = int(rng.integers(2, 9))
        out[k:k + L] = rng.choice([-1.0, 1.0], size=d) * w
        k += L
    return out


def run_di_episode(setup: PatrolSetup, cfg: PatrolDIConfig, sup: RecedingSupervisor,
                   seed: int = 0) -> dict:
    n_steps = round(cfg.episode_seconds / cfg.dt)
    W = disturbance_sequence(cfg, n_steps, seed)
    m = setup.model
    x = np.asarray(cfg.x0, float)
    d = setup.task.dim
    if len(x) == 4 and d == 3:  # planar (px, py, vx, vy) start for the cube specs: z = vz = 0
        x = np.array([x[0], x[1], 0.0, x[2], x[3], 0.0])
    init = sup.initial_plan(x)
    X = [x.copy()]
    U = []
    for k in range(n_steps):
        u, _ = sup.step(k, x)
        u = np.clip(u, setup.problem.u_lo, setup.problem.u_hi)  # actuator saturation
        x = m.A @ x + m.B @ u + m.B @ W[k]
        X.append(x.copy())
        U.append(u)
    X, U = np.array(X), np.array(U)
    Y = X @ setup.problem.C_out.T
    rho = realised_robustness(setup.problem, Y)
    return {"X": X, "U": U, "W": W, "rho_windows": rho, "init": init,
            "steps": sup.step_log, "solves": sup.solve_log}


def waypoints_reached(setup: PatrolSetup, X: np.ndarray, tol: float = 0.25) -> int:
    """Waypoints the trajectory ``X`` reaches in patrol order (within ``tol``), starting at the one
    after the closest (the convention of the nominal controller)."""
    wps = setup.task.waypoints()
    P = X @ setup.problem.C_out.T
    reached, idx = 0, int(np.argmin(np.linalg.norm(wps - P[0], axis=1)) + 1) % len(wps)
    for p in P:
        if np.linalg.norm(wps[idx] - p) < tol:
            reached += 1
            idx = (idx + 1) % len(wps)
    return reached


def waypoint_visits(setup: PatrolSetup, X: np.ndarray, tol: float = 0.25) -> int:
    """Order-free patrol coverage: entries into the ``tol`` ball of a waypoint other than the one
    visited last. :func:`waypoints_reached` stops counting after a missed waypoint; in 3-D the
    detours around the cube edges can pass a waypoint just outside ``tol``, so both are reported."""
    wps = setup.task.waypoints()
    P = X @ setup.problem.C_out.T
    visits, last = 0, -1
    for p in P:
        d = np.linalg.norm(wps - p, axis=1)
        i = int(np.argmin(d))
        if d[i] < tol and i != last:
            visits += 1
            last = i
    return visits


def _patrol_projection(setup: PatrolSetup, X: np.ndarray):
    """Projection of the output samples onto the closed patrol polyline (waypoints in patrol order):
    path parameter ``s`` in ``[0, n)`` and distance to the polyline, per sample."""
    wps = np.asarray(setup.task.waypoints(), float)
    P = X @ setup.problem.C_out.T
    A, B = wps, np.roll(wps, -1, axis=0)
    AB = B - A
    t = np.clip(np.einsum("kij,ij->ki", P[:, None, :] - A[None], AB) / np.sum(AB * AB, axis=1), 0.0, 1.0)
    dist = np.linalg.norm(P[:, None, :] - (A[None] + t[..., None] * AB[None]), axis=2)
    i = np.argmin(dist, axis=1)
    k = np.arange(len(P))
    return i + t[k, i], dist[k, i], len(wps)


def waypoint_miss(setup: PatrolSetup, X: np.ndarray) -> float:
    """Mean distance to a waypoint at the moments the patrol progress (:func:`patrol_progress`) passes it
    [m], linearly interpolated between samples; NaN if no waypoint is passed. Complements the progress
    count: how precisely the patrol goes through its waypoints."""
    wps = np.asarray(setup.task.waypoints(), float)
    P = X @ setup.problem.C_out.T
    s, _, n = _patrol_projection(setup, X)
    su = s[0] + np.concatenate([[0.0], np.cumsum((np.diff(s) + n / 2) % n - n / 2)])
    miss = []
    for k in range(len(su) - 1):
        a, b = su[k], su[k + 1]
        for m in range(int(np.floor(min(a, b))) + 1, int(np.floor(max(a, b))) + 1):
            if b > a:  # forward passes only
                f = (m - a) / (b - a)
                miss.append(np.linalg.norm(P[k] + f * (P[k + 1] - P[k]) - wps[m % n]))
    return float(np.mean(miss)) if miss else float("nan")


def patrol_progress(setup: PatrolSetup, X: np.ndarray) -> float:
    """Tolerance-free patrol progress in waypoint legs: every output sample is projected onto the closed
    polyline through the waypoints in patrol order (parameter ``s`` in ``[0, n)``, leg ``i`` from waypoint
    ``i`` to ``i + 1``), and the signed advance of ``s`` is summed over the trajectory (backtracking
    counts negatively; jumps are wrapped to ``(-n/2, n/2]``). One leg corresponds to one waypoint of
    :func:`waypoints_reached`, without its distance tolerance or its stop after a missed waypoint."""
    s, _, n = _patrol_projection(setup, X)
    ds = (np.diff(s) + n / 2) % n - n / 2
    return float(np.sum(ds))


def episode_summary(setup: PatrolSetup, ep: dict) -> dict:
    steps, solves = ep["steps"], ep["solves"]
    n = len(steps)
    src = [s["source"] for s in steps]
    runtime_solves = [s for s in solves if s["request_step"] >= 0]
    X = ep["X"]
    # Mission: deviation from the nominal patrol is not defined post hoc; report patrol
    # progress (waypoints reached) and speed instead.
    reached = waypoints_reached(setup, X)
    def stat(key):
        v = np.array([s[key] for s in runtime_solves if key in s], float)
        return {"mean": float(np.nanmean(v)) if v.size else float("nan"),
                "p95": float(np.nanpercentile(v, 95)) if v.size else float("nan"),
                "max": float(np.nanmax(v)) if v.size else float("nan")}
    return {
        "n_steps": n,
        "satisfied": bool(np.all(ep["rho_windows"] >= 0)),
        "min_rho": float(np.min(ep["rho_windows"])),
        "frac_proposal": src.count("proposal") / n,
        "frac_cache": src.count("cache") / n,
        "n_runtime_solves": len(runtime_solves),
        "synthesis_call_fraction": len(runtime_solves) / n,
        "n_installed": sum(1 for s in runtime_solves if s.get("installed")),
        "n_deadline_miss": sum(1 for s in runtime_solves if s.get("deadline_miss")),
        "warm_start_infeasible": sum(1 for s in runtime_solves if s.get("warm_start_feasible") is False),
        "solve_wall_s": stat("solve_wall_s"),
        "solve_energy_pkg_j": stat("solve_net_energy_package-0_j"),
        "waypoints_reached": reached,
        "waypoint_visits": waypoint_visits(setup, X),
        "patrol_progress": patrol_progress(setup, X),
        "waypoint_miss": waypoint_miss(setup, X),
        "mean_speed": float(np.mean(np.linalg.norm(X[:, setup.task.dim:2 * setup.task.dim], axis=1))),
        "max_tracking_error": float(max(s["tracking_error"] for s in steps)),
    }
