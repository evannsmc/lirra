"""Receding fixed-horizon interval-STL synthesis (NAHS (16)) with Gurobi.

At planning step ``t`` the problem is rooted at a state ``x_0`` (measured, or a cached plan's
nominal state when the solve is delay compensated) and plans ``N`` inputs::

    min   sum_{i=1}^{N_c} gamma0^i || W_x (x_i - x_nom_i) ||_1 + w_u sum_i || u_i - u_nom_i ||_1
    s.t.  x_{i+1} = A x_i + B u_i
          x_i in [x_lo + z(o+i), x_hi - z(o+i)],  u_i in [u_lo + kz(o+i), u_hi - kz(o+i)]
          every enforced window of (history ++ C x_1..C x_N) satisfies psi, where a predicate
            a^T y - b at planned step i holds iff a^T C x_i - b >= delta + |a^T C| z(o+i)
          the last b states lie in one face set Q_f of the verified terminal set

``z`` is the interval hull of the error tube (:mod:`rl_stl.synthesis.tube`), ``o`` the
tube age of the root (0 for a measured root). History samples are constants; each has its
own threshold (0 for measured samples: Boolean satisfaction; ``delta + |a| z(age)`` for samples
of a cached plan that have not been measured yet). This is the interval robustness lower
bound of NAHS Sec. 7.1 with the robustness constraint hard (``xi = 0`` in Theorem 2).

Signal layout (see :mod:`rl_stl.stl.nahs`): ``L = h + N`` samples, local index ``j``;
``j = 0..h-2`` history, ``j = h-1`` the root, ``j = h-1+i`` planned step ``i``. The formula is
``psi`` at window starts ``s = first_window .. N-1``.

Encoding: every (node, sample) gets a continuous "holds" variable ``h in [0, 1]``; ``and``
nodes require ``h <= h_child``, ``or`` nodes ``h <= sum h_child``; predicates at planned
samples are binaries with a big-M row; predicates at history samples are binaries fixed by
bounds. A positive ``h`` forces a satisfied predicate below it, so the feasible set is
exactly the set of plans satisfying the thresholded formula; binaries =
``#predicates x #samples`` independent of the number of windows. The model is built once per
:class:`RecedingProblem` and per ``first_window``; every request only changes bounds, RHS
values, fixed binaries and the objective's reference (:meth:`RecedingISTL.update`).

Interval predicates (``_u`` specs). For an :class:`~rl_stl.stl.interval.IntervalPredicate`
``[a]^T y - [b] >= 0`` at planned step ``i`` the sample is the box ``[y_i] = [c - r, c + r]`` with
``c = C x_i`` and ``r = |C| z(o+i)`` (paper eq. (margin)), and the predicate holds iff its lower endpoint

    lower(c, r) = sum_j min(a_lo_j c_j - |a_lo_j| r_j, a_hi_j c_j - |a_hi_j| r_j) - b_hi >= delta.

(:mod:`rl_stl.stl.interval`: the exact minimum over ``[a] x [y] x [b]``.) ``lower`` is concave in ``c``,
so one continuous ``t_j`` per signal coordinate with ``t_j <= a_lo_j c_j - |a_lo_j| r_j`` and
``t_j <= a_hi_j c_j - |a_hi_j| r_j`` (RHS ``-|a| r`` set per request) and the big-M row
``sum_j t_j - b_hi >= delta - M (1 - z)`` encode ``z = 1 => lower >= delta`` exactly (``t_j`` can be
raised to the minimum when ``z = 1``; any ``t_j`` at or below it works when ``z = 0``). ``t_j`` is bounded
below by ``l_j = min_{a, c corners} a c - max|a| r_max`` over the untightened state box and the widest
tube ``r_max = |C| z_inf``, which every admissible ``t_j`` value (in particular the minimum itself)
satisfies, so ``M = delta + b_hi - sum_j l_j`` makes the row inactive for ``z = 0``. History and root
samples are constants: their binary is fixed to ``lower(y, r_age) >= thr`` (``r = 0``, ``thr = 0`` for
measured samples; ``r = |C| z(age)``, ``thr = delta + numerical_margin / 2`` for cached-plan samples: they
were planned with ``delta + numerical_margin`` and the half margin absorbs the rounding of the recomputed
lower endpoint, so the shifted cached plan stays a feasible point). Exact predicates keep the encoding
above unchanged, including their cached-plan threshold ``delta + numerical_margin``: a cached sample whose
recomputed value rounds just below it gets its binary fixed to 0, which occasionally makes a delay-
compensated request infeasible although its warm start passes the check.

Every returned plan is re-checked by :func:`rl_stl.synthesis.check.check_plan`, an
independent numpy implementation of the same constraints, before it may be used.
"""

from __future__ import annotations

import dataclasses
import functools
import time

import gurobipy as gp
import numpy as np
from gurobipy import GRB
from stlpy.STL import LinearPredicate, STLFormula, STLTree

from rl_stl.dynamics.linear import DiscreteLinearModel
from rl_stl.stl.interval import IntervalPredicate
from rl_stl.stl.robustness import formula_horizon
from rl_stl.synthesis.common import SynthesisStatus, gurobi_env
from rl_stl.synthesis.terminal import TerminalSet
from rl_stl.synthesis.tube import Tube


@dataclasses.dataclass(frozen=True)
class ISTLSettings:
    time_limit: float | None = None  # [s] Gurobi TimeLimit (worst-case latency budget)
    mip_gap: float = 1e-4
    threads: int = 1  # fixed for timing/energy comparability
    int_feas_tol: float = 1e-9
    feasibility_tol: float = 1e-9
    numerical_margin: float = 1e-6  # thresholds raised by this; acceptance checks the plain value
    verbose: bool = False


@dataclasses.dataclass(frozen=True, eq=False)
class RecedingProblem:
    """Static data of the receding I-STL problem."""

    model: DiscreteLinearModel
    C_out: np.ndarray  # (p, n) STL signal y = C_out x
    psi: STLFormula  # per-window formula over y
    n_plan: int  # N
    x_lo: np.ndarray
    x_hi: np.ndarray
    u_lo: np.ndarray
    u_hi: np.ndarray
    tube: Tube
    delta: float
    terminal: TerminalSet | None
    terminal_block: int  # b
    track_weight: np.ndarray  # (n,) W_x diagonal
    gamma0: float = 1.0
    n_cost: int | None = None  # N_c (default N)
    effort_weight: float = 0.0
    name: str = "receding_istl"

    @functools.cached_property
    def h(self) -> int:
        return formula_horizon(self.psi)

    @property
    def length(self) -> int:
        return self.h + self.n_plan

    def __post_init__(self) -> None:
        if self.n_plan < self.terminal_block:
            raise ValueError("n_plan must be >= terminal_block")
        if self.tube.n_max < self.n_plan:
            raise ValueError("tube must cover the plan horizon")


@dataclasses.dataclass
class History:
    """Constant samples ``y_j`` for local ``j = 0..h-2`` and their tube ages (None: measured)."""

    y: np.ndarray  # (h-1, p)
    age: list[int | None]


@dataclasses.dataclass
class ISTLResult:
    status: SynthesisStatus
    X: np.ndarray | None  # (N+1, n) nominal plan, X[0] = root
    U: np.ndarray | None  # (N, m)
    face: int | None
    objective: float
    mip_gap: float
    solve_time: float  # Gurobi Runtime
    update_time: float  # model build or in-place update
    wall_time: float
    n_vars: int
    n_binary: int
    n_constraints: int
    gurobi_status: int
    rebuilt: bool
    accepted: bool = False
    reason: str = "not checked"
    rho_lower: float = float("nan")  # independently recomputed (planned samples)

    @property
    def has_solution(self) -> bool:
        return self.X is not None


def _pred_key(pred: LinearPredicate) -> tuple:
    if isinstance(pred, IntervalPredicate):
        return pred.key()
    return (tuple(np.round(pred.a[:, 0], 12)), round(float(pred.b[0]), 12))


class RecedingISTL:
    """Persistent Gurobi model for one :class:`RecedingProblem` and ``first_window``."""

    def __init__(self, problem: RecedingProblem, first_window: int = 0,
                 settings: ISTLSettings = ISTLSettings()):
        t0 = time.perf_counter()
        self.p = problem
        self.first_window = first_window
        self.settings = settings
        self._build()
        self.build_time = time.perf_counter() - t0

    # ------------------------------------------------------------------ build
    def _build(self) -> None:
        p, s = self.p, self.settings
        n, m, N, _h = p.model.n, p.model.m, p.n_plan, p.h
        md = gp.Model(p.name, env=gurobi_env())
        prm = md.Params
        prm.OutputFlag = int(s.verbose)
        prm.MIPGap = s.mip_gap
        prm.IntFeasTol = s.int_feas_tol
        prm.FeasibilityTol = s.feasibility_tol
        prm.Threads = s.threads
        if s.time_limit is not None:
            prm.TimeLimit = s.time_limit
        self.md = md
        self.x = [[md.addVar(lb=-GRB.INFINITY, name=f"x[{i},{k}]") for k in range(n)]
                  for i in range(N + 1)]
        self.u = [[md.addVar(lb=-GRB.INFINITY, name=f"u[{i},{k}]") for k in range(m)]
                  for i in range(N)]
        A, B = p.model.A, p.model.B
        for i in range(N):
            for r in range(n):
                cs = [1.0] + [-A[r, c] for c in range(n)] + [-B[r, c] for c in range(m)]
                vs = [self.x[i + 1][r]] + self.x[i] + self.u[i]
                keep = [(c, v) for c, v in zip(cs, vs) if c != 0.0]
                md.addLConstr(gp.LinExpr([c for c, _ in keep], [v for _, v in keep]), GRB.EQUAL, 0.0)

        # Big-M range of each predicate over the widest (untightened) box.
        self._x_box = (np.asarray(p.x_lo, float), np.asarray(p.x_hi, float))
        self._thr_max = p.delta + s.numerical_margin + float(np.max(np.abs(p.C_out) @ p.tube.z_inf)) + 1e-6
        self._hold: dict[tuple, gp.Var] = {}
        self._pred_rows: list[tuple[gp.Constr, np.ndarray, float, int, float]] = []  # (row, aC, b, i, big_m)
        self._hist_bins: list[tuple[gp.Var, np.ndarray, float, int]] = []  # (z, a, b, j)
        self._root_bins: list[tuple[gp.Var, np.ndarray, float]] = []
        self._pred_bin: dict[tuple, gp.Var] = {}
        # interval predicates: (row, |a|, coordinate, i) epigraph rows, constant-sample binaries
        self._ipred_rows: list[tuple[gp.Constr, float, int, int]] = []
        self._ihist_bins: list[tuple[gp.Var, IntervalPredicate, int]] = []
        self._iroot_bins: list[tuple[gp.Var, IntervalPredicate]] = []
        for sw in range(self.first_window, N):
            md.addLConstr(gp.LinExpr([1.0], [self._holds(p.psi, sw)]), GRB.GREATER_EQUAL, 1.0)

        # Terminal block: one face for the last b states.
        self.face_bins = []
        if p.terminal is not None:
            rows = p.terminal.face_rows()
            self.face_bins = [md.addVar(vtype=GRB.BINARY, name=f"face{f}") for f in range(len(rows))]
            md.addLConstr(gp.LinExpr([1.0] * len(rows), self.face_bins), GRB.EQUAL, 1.0)
            lo, hi = self._x_box
            for f, (H, g) in enumerate(rows):
                big = np.maximum(np.maximum(H * lo, H * hi).sum(axis=1) - g, 0.0) + 1e-6
                for i in range(N - p.terminal_block + 1, N + 1):
                    for r in range(len(g)):
                        idx = np.nonzero(H[r])[0]
                        expr = gp.LinExpr([float(H[r, c]) for c in idx], [self.x[i][c] for c in idx])
                        expr.add(self.face_bins[f], float(big[r]))
                        md.addLConstr(expr, GRB.LESS_EQUAL, float(g[r] + big[r]))

        # Cost epigraphs; RHS carries the nominal reference.
        Nc = N if p.n_cost is None else min(p.n_cost, N)
        self._track_rows = []
        obj = gp.LinExpr()
        for i in range(1, Nc + 1):
            for k in np.nonzero(p.track_weight)[0]:
                e = md.addVar(lb=0.0)
                r1 = md.addLConstr(gp.LinExpr([1.0, -1.0], [e, self.x[i][k]]), GRB.GREATER_EQUAL, 0.0)
                r2 = md.addLConstr(gp.LinExpr([1.0, 1.0], [e, self.x[i][k]]), GRB.GREATER_EQUAL, 0.0)
                self._track_rows.append((r1, r2, i, int(k)))
                obj.add(e, float(p.gamma0**i * p.track_weight[k]))
        self._effort_rows = []
        if p.effort_weight > 0:
            for i in range(N):
                for k in range(m):
                    e = md.addVar(lb=0.0)
                    r1 = md.addLConstr(gp.LinExpr([1.0, -1.0], [e, self.u[i][k]]), GRB.GREATER_EQUAL, 0.0)
                    r2 = md.addLConstr(gp.LinExpr([1.0, 1.0], [e, self.u[i][k]]), GRB.GREATER_EQUAL, 0.0)
                    self._effort_rows.append((r1, r2, i, k))
                    obj.add(e, float(p.effort_weight))
        md.setObjective(obj, GRB.MINIMIZE)
        md.update()

    def _holds(self, node: STLFormula, j: int) -> gp.Var:
        if isinstance(node, IntervalPredicate):
            return self._interval_predicate(node, j)
        if isinstance(node, LinearPredicate):
            return self._predicate(node, j)
        key = (id(node), j)
        v = self._hold.get(key)
        if v is not None:
            return v
        if not isinstance(node, STLTree):
            raise TypeError(f"unsupported node {type(node).__name__}")
        v = self.md.addVar(lb=0.0, ub=1.0)
        self._hold[key] = v
        kids = [self._holds(c, j + dt) for c, dt in zip(node.subformula_list, node.timesteps)]
        if node.combination_type == "and":
            for c in kids:
                self.md.addLConstr(gp.LinExpr([1.0, -1.0], [c, v]), GRB.GREATER_EQUAL, 0.0)
        else:
            self.md.addLConstr(gp.LinExpr([1.0] * len(kids) + [-1.0], kids + [v]), GRB.GREATER_EQUAL, 0.0)
        return v

    def _predicate(self, pred: LinearPredicate, j: int) -> gp.Var:
        key = (_pred_key(pred), j)
        z = self._pred_bin.get(key)
        if z is not None:
            return z
        p = self.p
        z = self.md.addVar(vtype=GRB.BINARY)
        self._pred_bin[key] = z
        a, b = pred.a[:, 0].astype(float), float(pred.b[0])
        i = j - (p.h - 1)
        if i < 0:
            self._hist_bins.append((z, a, b, j))
        elif i == 0:
            self._root_bins.append((z, a, b))
        else:
            aC = a @ p.C_out
            lo, hi = self._x_box
            vmin = float(np.sum(np.minimum(aC * lo, aC * hi)) - b)
            big_m = max(self._thr_max - vmin, 0.0) + 1e-6
            idx = np.nonzero(aC)[0]
            expr = gp.LinExpr([float(aC[c]) for c in idx], [self.x[i][c] for c in idx])
            expr.add(z, -big_m)
            row = self.md.addLConstr(expr, GRB.GREATER_EQUAL, 0.0)  # RHS set per request
            self._pred_rows.append((row, aC, b, i, big_m))
        return z

    def _interval_predicate(self, pred: IntervalPredicate, j: int) -> gp.Var:
        """Binary ``z`` with ``z = 1 => lower([a]^T [y_j] - [b]) >= threshold`` (module docstring)."""
        key = (pred.key(), j)
        z = self._pred_bin.get(key)
        if z is not None:
            return z
        p, s = self.p, self.settings
        z = self.md.addVar(vtype=GRB.BINARY)
        self._pred_bin[key] = z
        i = j - (p.h - 1)
        if i < 0:
            self._ihist_bins.append((z, pred, j))
            return z
        if i == 0:
            self._iroot_bins.append((z, pred))
            return z
        C = p.C_out
        lo, hi = self._x_box
        c_lo = np.minimum(C * lo, C * hi).sum(axis=1)  # box of y = C x over the untightened state box
        c_hi = np.maximum(C * lo, C * hi).sum(axis=1)
        r_max = np.abs(C) @ p.tube.z_inf
        ts, lb_sum = [], 0.0
        for jj in range(C.shape[0]):
            coefs = sorted({float(pred.a_lo[jj]), float(pred.a_hi[jj])})
            if all(a == 0.0 for a in coefs):
                continue
            lb = (min(a * c for a in coefs for c in (c_lo[jj], c_hi[jj]))
                  - max(abs(a) for a in coefs) * r_max[jj] - 1e-6)
            t = self.md.addVar(lb=lb, ub=GRB.INFINITY)
            idx = np.nonzero(C[jj])[0]
            for a in coefs:
                expr = gp.LinExpr([1.0] + [-a * float(C[jj, k]) for k in idx], [t] + [self.x[i][k] for k in idx])
                row = self.md.addLConstr(expr, GRB.LESS_EQUAL, 0.0)  # RHS -|a| r_jj set per request
                self._ipred_rows.append((row, abs(a), jj, i))
            ts.append(t)
            lb_sum += lb
        thr = p.delta + s.numerical_margin
        big_m = max(thr + pred.b_hi - lb_sum, 0.0) + 1e-6
        expr = gp.LinExpr([1.0] * len(ts), ts)
        expr.add(z, -big_m)
        self.md.addLConstr(expr, GRB.GREATER_EQUAL, thr + pred.b_hi - big_m)
        return z

    # ----------------------------------------------------------------- update
    def update(self, x0: np.ndarray, history: History, x_nom: np.ndarray, u_nom: np.ndarray,
               root_age: int | None = None, start: tuple[np.ndarray, np.ndarray] | None = None) -> None:
        """Apply the time-varying data of one request.

        ``x0``: plan root; ``root_age``: tube age of the root (None: measured).
        ``x_nom (N+1, n)`` / ``u_nom (N, m)``: nominal mission reference rollout.
        ``start``: optional MIP start ``(X, U)`` (e.g. the shifted cached plan).
        """
        t0 = time.perf_counter()
        p, s = self.p, self.settings
        N, md, tube = p.n_plan, self.md, p.tube
        off = 0 if root_age is None else int(root_age)
        md.reset(0)
        for k, v in enumerate(self.x[0]):
            v.LB = v.UB = float(x0[k])
        for i in range(1, N + 1):
            z = tube.state(off + i)
            lo, hi = p.x_lo + z, p.x_hi - z
            for k, v in enumerate(self.x[i]):
                v.LB, v.UB = float(lo[k]), float(hi[k])
        for i in range(N):
            kz = tube.input(off + i)
            lo, hi = p.u_lo + kz, p.u_hi - kz
            for k, v in enumerate(self.u[i]):
                v.LB, v.UB = float(lo[k]), float(hi[k])
        margin = p.delta + s.numerical_margin
        for row, aC, b, i, big_m in self._pred_rows:
            thr = margin + float(np.abs(aC) @ tube.state(off + i))
            row.RHS = thr + b - big_m
        y_root = p.C_out @ x0
        for z, a, b in self._root_bins:
            thr = 0.0 if root_age is None else margin + float(np.abs(a @ p.C_out) @ tube.state(off))
            val = 1.0 if a @ y_root - b >= thr else 0.0
            z.LB = z.UB = val
        for z, a, b, j in self._hist_bins:
            age = history.age[j]
            thr = 0.0 if age is None else margin + float(np.abs(a @ p.C_out) @ tube.state(age))
            val = 1.0 if a @ history.y[j] - b >= thr else 0.0
            z.LB = z.UB = val
        if self._ipred_rows or self._iroot_bins or self._ihist_bins:
            absC = np.abs(p.C_out)
            r_plan = {i: absC @ tube.state(off + i) for i in range(1, N + 1)}
            for row, abs_a, jj, i in self._ipred_rows:
                row.RHS = -abs_a * float(r_plan[i][jj])

            # cached-plan samples were planned with ``delta + numerical_margin`` and sit on it up to the
            # rounding of the recomputed lower endpoint (sum of minima); half the margin absorbs that
            # rounding (otherwise the shifted cached plan can turn infeasible) and stays >= delta
            cached_thr = p.delta + 0.5 * s.numerical_margin

            def holds(pred: IntervalPredicate, y: np.ndarray, age: int | None) -> float:
                if age is None:
                    return 1.0 if pred.lower(y) >= 0.0 else 0.0
                return 1.0 if pred.lower(y, absC @ tube.state(age)) >= cached_thr else 0.0

            for z, pred in self._iroot_bins:
                z.LB = z.UB = holds(pred, y_root, None if root_age is None else off)
            for z, pred, j in self._ihist_bins:
                z.LB = z.UB = holds(pred, history.y[j], history.age[j])
        for r1, r2, i, k in self._track_rows:
            r1.RHS = -float(x_nom[i, k])
            r2.RHS = float(x_nom[i, k])
        for r1, r2, i, k in self._effort_rows:
            r1.RHS = -float(u_nom[i, k])
            r2.RHS = float(u_nom[i, k])
        if start is not None:
            Xs, Us = start
            for i in range(N + 1):
                for k, v in enumerate(self.x[i]):
                    v.Start = float(Xs[i, k])
            for i in range(N):
                for k, v in enumerate(self.u[i]):
                    v.Start = float(Us[i, k])
        else:
            for row in self.x + self.u:
                for v in row:
                    v.Start = GRB.UNDEFINED
        self.update_time = time.perf_counter() - t0

    # ------------------------------------------------------------------ solve
    def solve(self) -> ISTLResult:
        t0 = time.perf_counter()
        md, _p = self.md, self.p
        md.update()
        stats = {"n_vars": md.NumVars, "n_binary": md.NumBinVars, "n_constraints": md.NumConstrs}
        try:
            md.optimize()
            code = md.Status
        except gp.GurobiError as exc:
            return ISTLResult(SynthesisStatus.ERROR, None, None, None, np.inf, np.inf, 0.0,
                              self.update_time, time.perf_counter() - t0, gurobi_status=-1,
                              rebuilt=False, reason=f"GurobiError: {exc}", **stats)
        sol = md.SolCount
        if code == GRB.OPTIMAL:
            status = SynthesisStatus.OPTIMAL
        elif code in (GRB.TIME_LIMIT, GRB.SUBOPTIMAL, GRB.INTERRUPTED, GRB.SOLUTION_LIMIT) and sol > 0:
            status = SynthesisStatus.SUBOPTIMAL
        elif code == GRB.TIME_LIMIT:
            status = SynthesisStatus.TIMEOUT
        elif code in (GRB.INFEASIBLE, GRB.INF_OR_UNBD):
            status = SynthesisStatus.INFEASIBLE
        elif code == GRB.NUMERIC:
            status = SynthesisStatus.NUMERIC
        else:
            status = SynthesisStatus.ERROR
        X = U = face = None
        obj = gap = np.inf
        if sol > 0 and status in (SynthesisStatus.OPTIMAL, SynthesisStatus.SUBOPTIMAL):
            X = np.array([[v.X for v in row] for row in self.x])
            U = np.array([[v.X for v in row] for row in self.u])
            if self.face_bins:
                face = int(np.argmax([b.X for b in self.face_bins]))
            obj, gap = float(md.ObjVal), float(md.MIPGap)
        return ISTLResult(status, X, U, face, obj, gap, float(md.Runtime), self.update_time,
                          time.perf_counter() - t0, gurobi_status=int(code), rebuilt=False, **stats)


class ISTLCache:
    """One persistent model per ``first_window`` (only the first ``h-1`` steps differ)."""

    def __init__(self, problem: RecedingProblem, settings: ISTLSettings = ISTLSettings()):
        self.problem = problem
        self.settings = settings
        self._models: dict[int, RecedingISTL] = {}

    def get(self, first_window: int) -> tuple[RecedingISTL, bool]:
        mdl = self._models.get(first_window)
        if mdl is not None:
            return mdl, False
        mdl = RecedingISTL(self.problem, first_window, self.settings)
        self._models[first_window] = mdl
        return mdl, True
