"""Receding-horizon runtime assurance loop of the L4DC paper (LIRRA, Algorithm 1).

Per planning step ``k`` (period ``dt``), with measured synthesis-model state ``x_k``:

1. ``k = 0`` (``t = 0-``): solve the receding I-STL problem from ``x_0``; the plan is the
   first cached plan.
2. A solve whose result is due at ``k`` is validated (:func:`check_plan`, independent of the
   solver) and installed as the cached plan.
3. Otherwise the policy proposes ``N`` inputs rooted at ``x_k``; the proposal is accepted iff
   it is a feasible point of the synthesis problem (:func:`check_plan`); an accepted
   proposal becomes the cached plan.
4. If it is rejected, the vehicle flies the cached plan and a solve is requested, rooted at
   the cached nominal state ``d`` steps ahead (delay compensation, ``d`` = worst-case
   latency in steps, enforced by the Gurobi time limit). The shifted cached plan is a
   feasible MIP start of that request.

The mission reference ``(x_nom, u_nom)`` of every request and proposal is the nominal patrol
either rerooted at the plan's root state or clock-indexed from the episode start
(``RecedingConfig.reference``, :mod:`rl_stl.policies.nominal`). Clock indexing is by absolute
step, so a delay-compensated request rooted at ``k + d`` tracks ``x_nom(k + d + i)``.

Execution is nominal tube MPC: the command is ``u = u_hat_k + K (x_k - x_hat_k)`` around the
cached nominal ``(x_hat, u_hat)``; past the plan's end the terminal controller extends the
nominal inside its face set. Under the tube assumption (realised error inside ``Z``) the
cached plan is always a valid certificate, so no step is without a certified input.

Variants (``RecedingConfig.variant``) implement the paper baselines:

``shield``           the method above;
``always_on``        solve at every step (NAHS RTA), no policy;
``policy_only``      fly the proposal's first input unchecked;
``raw_trigger``      accept a proposal iff its nominal robustness (no tube, no terminal set,
                     no bounds) is ``>= delta``; otherwise as ``shield``;
``no_delay_comp``    as ``shield`` but the solve is rooted at the measured ``x_k`` and its plan
                     is time-aligned on arrival (``d`` steps late), without re-validation.
"""

from __future__ import annotations

import dataclasses
import enum
import math
import time
from collections.abc import Callable
from typing import Literal

import numpy as np

from rl_stl.evaluation.energy import EnergyMeter
from rl_stl.policies.nominal import ClockReference, PatrolPD
from rl_stl.stl.nahs import first_enforced_window
from rl_stl.stl.robustness import robustness
from rl_stl.synthesis.check import CheckResult, check_plan
from rl_stl.synthesis.istl import History, ISTLCache, ISTLResult, ISTLSettings, RecedingProblem

Variant = Literal["shield", "always_on", "policy_only", "raw_trigger", "no_delay_comp"]


class Source(enum.StrEnum):
    PROPOSAL = "proposal"  # accepted (or, policy_only, unchecked) policy proposal
    CACHE = "cache"  # cached plan / terminal extension
    SOLVER = "solver"  # first step of a just-installed solver plan


@dataclasses.dataclass(frozen=True)
class RecedingConfig:
    variant: Variant = "shield"
    latency_steps: int = 1  # d: worst-case solve latency in planning steps (>= 1)
    latency_mode: Literal["injected", "measured"] = "injected"
    time_fraction: float = 0.8  # Gurobi TimeLimit = time_fraction * d * dt
    query_while_pending: bool = False
    raw_delta: float | None = None  # raw_trigger threshold (default: problem.delta)
    feedback: bool = True  # apply K (x - x_hat)
    reference: Literal["rerooted", "clock"] = "rerooted"  # mission reference (x_nom, u_nom)
    reference_input_scale: float = 1.0  # clock reference: inputs saturate at this x bounds
    initial_time_limit: float = 30.0  # [s] TimeLimit of the t = 0- solve (offline, not latency-bound)


@dataclasses.dataclass
class Plan:
    """Nominal plan with tube bookkeeping; extended past ``N`` by the terminal controller."""

    X: np.ndarray  # (N+1, n), X[0] at root_step
    U: np.ndarray  # (N, m)
    root_step: int
    root_age: int | None  # tube age of X[0] (None: measured root = age 0)
    face: int | None
    source: str
    certified: bool = True
    _ext_X: list = dataclasses.field(default_factory=list)
    _ext_U: list = dataclasses.field(default_factory=list)

    @property
    def n_plan(self) -> int:
        return self.U.shape[0]

    def age(self, k: int) -> int:
        return (0 if self.root_age is None else self.root_age) + (k - self.root_step)

    def _extend(self, j: int, terminal, model) -> None:
        while self.n_plan + len(self._ext_U) <= j:
            x = self._ext_X[-1] if self._ext_X else self.X[-1]
            if terminal is not None and self.face is not None:
                u = terminal.controller(x, self.face)
            else:
                u = np.zeros(self.U.shape[1])  # uncertified plan: coast (baselines only)
            self._ext_U.append(u)
            self._ext_X.append(model.A @ x + model.B @ u)

    def state(self, k: int, terminal=None, model=None) -> np.ndarray:
        j = k - self.root_step
        if j <= self.n_plan:
            return self.X[j]
        self._extend(j - 1, terminal, model)
        return self._ext_X[j - self.n_plan - 1]

    def input(self, k: int, terminal=None, model=None) -> np.ndarray:
        j = k - self.root_step
        if j < self.n_plan:
            return self.U[j]
        self._extend(j, terminal, model)
        return self._ext_U[j - self.n_plan]

    def window(self, k: int, n: int, terminal, model) -> tuple[np.ndarray, np.ndarray]:
        """Nominal ``(X (n+1), U (n))`` starting at step ``k``."""
        X = np.array([self.state(k + i, terminal, model) for i in range(n + 1)])
        U = np.array([self.input(k + i, terminal, model) for i in range(n)])
        return X, U


@dataclasses.dataclass
class ProposalContext:
    k: int
    x: np.ndarray
    history: History  # measured history (local j = 0..h-2) for a plan rooted at x_k
    first_window: int
    x_nom: np.ndarray  # mission reference for the plan rooted at k (rerooted at x_k or clock-indexed)
    u_nom: np.ndarray
    cache: Plan | None
    waypoint_index: int | None = None  # waypoint the reference is heading for (None: nominal.index)


Proposer = Callable[[ProposalContext], np.ndarray]  # -> U (N, m)


@dataclasses.dataclass
class _Pending:
    request_step: int
    release_step: int
    root_step: int
    root_age: int | None
    x0: np.ndarray
    history: History
    first_window: int
    result: ISTLResult
    energy: dict
    start_feasible: bool | None


class RecedingSupervisor:
    def __init__(self, problem: RecedingProblem, K: np.ndarray, nominal: PatrolPD,
                 proposer: Proposer | None, config: RecedingConfig = RecedingConfig(),
                 settings: ISTLSettings = ISTLSettings(), meter: EnergyMeter | None = None):
        if config.latency_steps < 1:
            raise ValueError("latency_steps must be >= 1")
        self.p = problem
        self.K = np.asarray(K, float)
        self.nominal = nominal
        self.reference = (ClockReference(nominal, problem.model, config.reference_input_scale)
                          if config.reference == "clock" else None)
        self.proposer = proposer
        self.cfg = config
        dt = problem.model.dt
        tl = settings.time_limit
        if tl is None:
            tl = config.time_fraction * config.latency_steps * dt
        self.settings = dataclasses.replace(settings, time_limit=tl)
        self.cache_models = ISTLCache(problem, self.settings)
        self.meter = meter or EnergyMeter()
        self.h = problem.h
        self.reset()

    # ----------------------------------------------------------------- state
    def reset(self) -> None:
        self.y_meas: list[np.ndarray] = []
        self.plan: Plan | None = None
        self.pending: _Pending | None = None
        self.step_log: list[dict] = []
        self.solve_log: list[dict] = []

    def _measured_history(self, k_root: int) -> History:
        """History for a plan rooted at ``k_root`` using measured samples only (``k_root <= k``)."""
        p = self.p
        y = np.zeros((self.h - 1, p.C_out.shape[0]))
        for j in range(self.h - 1):
            t = k_root - (self.h - 1) + j
            if t >= 0:
                y[j] = self.y_meas[t]
        return History(y, [None] * (self.h - 1))

    def _delayed_history(self, k: int, k_root: int) -> History:
        """Measured samples up to ``k``, cached nominal (with tube ages) for ``k < t < k_root``."""
        p = self.p
        y = np.zeros((self.h - 1, p.C_out.shape[0]))
        age: list[int | None] = [None] * (self.h - 1)
        for j in range(self.h - 1):
            t = k_root - (self.h - 1) + j
            if t < 0:
                continue
            if t <= k:
                y[j] = self.y_meas[t]
            else:
                y[j] = p.C_out @ self.plan.state(t, p.terminal, p.model)
                age[j] = self.plan.age(t)
        return History(y, age)

    def _mission_reference(self, k_root: int, x0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """``(x_nom (N+1, n), u_nom (N, m))`` for a plan rooted at step ``k_root`` in state ``x0``."""
        if self.reference is not None:
            return self.reference.window(k_root, self.p.n_plan)
        return self.nominal.rollout(self.p.model, x0, self.p.n_plan)

    def _waypoint_index(self, k: int) -> int:
        return self.nominal.index if self.reference is None else self.reference.index_at(k)

    # ----------------------------------------------------------------- solve
    def _solve(self, k: int, x0: np.ndarray, root_step: int, root_age: int | None,
               history: History, start: tuple[np.ndarray, np.ndarray] | None,
               time_limit: float | None = None) -> _Pending:
        p = self.p
        fw = first_enforced_window(root_step, self.h)
        Xn, Un = self._mission_reference(root_step, x0)
        mdl, rebuilt = self.cache_models.get(fw)
        start_ok = None
        if start is not None:
            start_ok = check_plan(p, x0, start[0], start[1], history, root_age, fw).ok
        if time_limit is not None:
            mdl.md.Params.TimeLimit = time_limit
        try:
            with self.meter.measure() as m:
                mdl.update(x0, history, Xn, Un, root_age=root_age, start=start)
                res = mdl.solve()
        finally:
            if time_limit is not None:
                mdl.md.Params.TimeLimit = self.settings.time_limit
        res.rebuilt = rebuilt
        if rebuilt:
            res.update_time += mdl.build_time
        if self.cfg.latency_mode == "injected":
            delay = self.cfg.latency_steps
        else:
            delay = max(1, math.ceil(m.wall_s / p.model.dt - 1e-12))
        if self.cfg.variant == "no_delay_comp":
            release = k + self.cfg.latency_steps
        else:
            release = root_step
        pend = _Pending(k, release if delay <= self.cfg.latency_steps else k + delay, root_step,
                        root_age, x0, history, fw, res, m.as_dict("solve_"), start_ok)
        self.solve_log.append({
            "request_step": k, "release_step": pend.release_step, "root_step": root_step,
            "root_age": -1 if root_age is None else root_age, "status": str(res.status),
            "objective": res.objective, "mip_gap": res.mip_gap, "gurobi_solve_s": res.solve_time,
            "update_s": res.update_time, "rebuilt": rebuilt, "n_binary": res.n_binary,
            "n_vars": res.n_vars, "n_constraints": res.n_constraints,
            "deadline_miss": delay > self.cfg.latency_steps, "warm_start_feasible": start_ok,
            **pend.energy,
        })
        return pend

    def _request(self, k: int, x: np.ndarray) -> None:
        p, d = self.p, self.cfg.latency_steps
        if self.cfg.variant == "no_delay_comp":
            self.pending = self._solve(k, x, k, None, self._measured_history(k), None)
            return
        root = k + d
        x0 = self.plan.state(root, p.terminal, p.model)
        start = self.plan.window(root, p.n_plan, p.terminal, p.model)
        self.pending = self._solve(k, x0, root, self.plan.age(root), self._delayed_history(k, root),
                                   start)

    def _release(self, k: int) -> str | None:
        pend = self.pending
        if pend is None or pend.release_step != k:
            return None
        self.pending = None
        res = pend.result
        log = self.solve_log[-1] if self.solve_log and self.solve_log[-1]["request_step"] == pend.request_step else {}
        if not res.has_solution:
            log["installed"] = False
            return f"solve_{res.status}"
        if log.get("deadline_miss"):
            log["installed"] = False
            return "deadline_miss"
        if self.cfg.variant == "no_delay_comp":
            plan = Plan(res.X, res.U, pend.root_step, None, res.face, "solver", certified=False)
        else:
            chk = check_plan(self.p, pend.x0, res.X, res.U, pend.history, pend.root_age, pend.first_window)
            res.accepted, res.reason, res.rho_lower = chk.ok, chk.reason, chk.rho_lower
            log.update(check_ok=chk.ok, check_reason=chk.reason, rho_lower=chk.rho_lower)
            if not chk.ok:
                log["installed"] = False
                return "solver_plan_rejected"
            plan = Plan(res.X, res.U, pend.root_step, pend.root_age, chk.face, "solver")
        self.plan = plan
        log["installed"] = True
        return "solver_installed"

    # ------------------------------------------------------------------ step
    def initial_plan(self, x0: np.ndarray) -> ISTLResult:
        """``t = 0-`` solve from the measured initial state (time not counted against latency)."""
        self.reset()
        self.nominal.reset(x0)
        if self.reference is not None:
            self.reference.reset(x0)
        pend = self._solve(-1, np.asarray(x0, float), 0, None, self._measured_history(0), None,
                           time_limit=self.cfg.initial_time_limit)
        self.pending = None
        res = pend.result
        self.solve_log[-1]["release_step"] = 0
        if not res.has_solution:
            raise RuntimeError(f"initial synthesis failed: {res.status}")
        chk = check_plan(self.p, x0, res.X, res.U, pend.history, None, pend.first_window)
        self.solve_log[-1].update(check_ok=chk.ok, check_reason=chk.reason, rho_lower=chk.rho_lower,
                                  installed=chk.ok)
        if not chk.ok:
            raise RuntimeError(f"initial plan failed the independent check: {chk.reason}")
        self.plan = Plan(res.X, res.U, 0, None, chk.face, "solver")
        return res

    def step(self, k: int, x: np.ndarray) -> tuple[np.ndarray, dict]:
        """Command ``u`` (synthesis-model input) at step ``k`` for measured state ``x``."""
        p, cfg = self.p, self.cfg
        x = np.asarray(x, float)
        if len(self.y_meas) != k:
            raise ValueError("step() must be called once per consecutive step")
        self.y_meas.append(p.C_out @ x)
        self.nominal(x)  # advance the mission's waypoint index on the realised trajectory
        rec: dict = {"k": k, "variant": cfg.variant}
        if not np.all(np.isfinite(x)):
            rec["event"] = "nonfinite_state"
        rec["release"] = self._release(k)
        source = Source.SOLVER if rec["release"] == "solver_installed" else Source.CACHE
        accepted = None
        want_policy = (cfg.variant != "always_on" and self.proposer is not None
                       and source != Source.SOLVER
                       and (self.pending is None or cfg.query_while_pending))
        if want_policy:
            accepted, chk_info = self._query_policy(k, x)
            rec.update(chk_info)
            if accepted:
                source = Source.PROPOSAL
        need_solve = self.pending is None and (
            cfg.variant == "always_on" or (source == Source.CACHE and accepted is False))
        if need_solve:
            self._request(k, x)
            rec["solve_requested"] = True
        plan = self.plan
        x_hat = plan.state(k, p.terminal, p.model)
        u_hat = plan.input(k, p.terminal, p.model)
        e = x - x_hat
        u = u_hat + (self.K @ e if cfg.feedback else 0.0)
        rec.update(source=str(source), plan_source=plan.source, plan_age=plan.age(k),
                   plan_index=k - plan.root_step, plan_certified=plan.certified,
                   tracking_error=float(np.max(np.abs(e))), pending=self.pending is not None,
                   u=u.tolist(), x=x.tolist(), x_hat=x_hat.tolist())
        self.step_log.append(rec)
        return u, rec

    def _query_policy(self, k: int, x: np.ndarray) -> tuple[bool, dict]:
        p, cfg = self.p, self.cfg
        fw = first_enforced_window(k, self.h)
        hist = self._measured_history(k)
        Xn, Un = self._mission_reference(k, x)
        ctx = ProposalContext(k, x, hist, fw, Xn, Un, self.plan, self._waypoint_index(k))
        t0 = time.perf_counter()
        U = np.asarray(self.proposer(ctx), float).reshape(p.n_plan, p.model.m)
        t_inf = time.perf_counter() - t0
        X = p.model.rollout(x, U)
        t0 = time.perf_counter()
        if cfg.variant == "policy_only":
            ok, info = True, {"check": "unchecked"}
            chk = None
        elif cfg.variant == "raw_trigger":
            sig = np.vstack([hist.y, X @ p.C_out.T])
            from rl_stl.stl.nahs import window_spec
            rho = robustness(window_spec(p.psi, p.n_plan, fw), sig)
            thr = p.delta if cfg.raw_delta is None else cfg.raw_delta
            ok, info, chk = rho >= thr, {"raw_rho": rho}, None
        else:
            chk: CheckResult = check_plan(p, x, X, U, hist, None, fw)
            ok, info = chk.ok, {"check": chk.reason, "rho_lower": chk.rho_lower}
        t_chk = time.perf_counter() - t0
        info.update(proposal_accepted=bool(ok), inference_s=t_inf, check_s=t_chk)
        if ok:
            certified = cfg.variant not in ("policy_only", "raw_trigger")
            self.plan = Plan(X, U, k, None, None if chk is None else chk.face, "proposal",
                             certified=certified)
        return bool(ok), info


def realised_robustness(problem: RecedingProblem, Y: np.ndarray) -> np.ndarray:
    """Nominal robustness of psi at every window start of a measured output trace ``Y``."""
    h = problem.h
    return np.array([robustness(problem.psi, Y[s:s + h + 1]) for s in range(len(Y) - h)])
