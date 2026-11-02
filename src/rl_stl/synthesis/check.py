"""Independent check of a candidate plan against the receding I-STL constraints.

Used for two purposes with one definition:

* validation of every solver plan before it is installed (recompute robustness
  independently of the solver), and
* the runtime check of a learned proposal: a proposal passes iff it is a feasible point of
  the same synthesis problem, so accepting it keeps the recursive-feasibility argument
  intact without solving.

Written separately from :mod:`rl_stl.synthesis.istl` (plain recursion over the ``stlpy``
tree, no Gurobi). ``rho_lower`` is the interval lower robustness of the enforced windows
with the semantics of the MILP: a planned predicate contributes
``a^T y_i - b - |a^T C| z(o+i)``; a constant history/root sample contributes ``+inf`` if it
meets its threshold and ``-inf`` otherwise. ``rho_lower >= delta`` (and bounds, dynamics and
the terminal block) is the acceptance condition.

An interval predicate ``[a]^T y - [b]`` (:class:`~rl_stl.stl.interval.IntervalPredicate`) contributes
the lower endpoint of its value over ``[a] x [b]`` and the sample box ``[C x_i - |C| z(o+i),
C x_i + |C| z(o+i)]`` (the I-STL lower bound, paper eq. (margin)); a constant sample is Boolean on the
lower endpoint with box radius 0 (measured, threshold 0) or ``|C| z(age)`` (cached plan, threshold
``delta``), as in the MILP.

:class:`CompiledWindows` evaluates the same semantics with numpy gathers compiled once per
``(problem, first_window)`` (a table of predicate values per sample, then one gather and one
min/max per tree level, as in :mod:`rl_stl.stl.jax_robustness`); :func:`interval_robustness`
is the plain recursion kept as the reference implementation (tests check they agree).
"""

from __future__ import annotations

import dataclasses

import numpy as np
from stlpy.STL import LinearPredicate, STLFormula, STLTree

from rl_stl.stl.interval import IntervalPredicate, interval_lower
from rl_stl.synthesis.istl import History, RecedingProblem


@dataclasses.dataclass
class CheckResult:
    ok: bool
    reason: str
    rho_lower: float
    face: int | None
    dynamics_residual: float
    bound_violation: float


def interval_robustness(problem: RecedingProblem, X: np.ndarray, history: History,
                        root_age: int | None, first_window: int) -> float:
    """``rho_lower`` of the enforced windows (see module docstring)."""
    p, tube, h = problem, problem.tube, problem.h
    off = 0 if root_age is None else int(root_age)
    y_hist, y_plan = history.y, X @ p.C_out.T  # y_plan[i] = C x_i, i = 0..N
    memo: dict[tuple[int, int], float] = {}

    absC = np.abs(p.C_out)

    def ipred_value(node: IntervalPredicate, j: int) -> float:
        i = j - (h - 1)
        if i > 0:
            return float(node.lower(y_plan[i], absC @ tube.state(off + i)))
        y, age = (y_plan[0], root_age) if i == 0 else (y_hist[j], history.age[j])
        if age is None:
            return np.inf if node.lower(y) >= 0.0 else -np.inf
        return np.inf if node.lower(y, absC @ tube.state(age)) >= p.delta else -np.inf

    def pred_value(node: LinearPredicate, j: int) -> float:
        if isinstance(node, IntervalPredicate):
            return ipred_value(node, j)
        a, b = node.a[:, 0], float(node.b[0])
        aC = a @ p.C_out
        i = j - (h - 1)
        if i > 0:
            return float(a @ y_plan[i] - b - np.abs(aC) @ tube.state(off + i))
        if i == 0:
            v, age = float(a @ y_plan[0] - b), root_age
        else:
            v, age = float(a @ y_hist[j] - b), history.age[j]
        thr = 0.0 if age is None else p.delta + float(np.abs(aC) @ tube.state(age))
        return np.inf if v >= thr else -np.inf

    def rec(node: STLFormula, j: int) -> float:
        key = (id(node), j)
        if key in memo:
            return memo[key]
        if isinstance(node, LinearPredicate):
            val = pred_value(node, j)
        elif isinstance(node, STLTree):
            vals = [rec(c, j + dt) for c, dt in zip(node.subformula_list, node.timesteps)]
            val = min(vals) if node.combination_type == "and" else max(vals)
        else:
            raise TypeError(type(node).__name__)
        memo[key] = val
        return val

    return min(rec(p.psi, s) for s in range(first_window, p.n_plan))


class CompiledWindows:
    """``rho_lower`` of the enforced windows via precompiled gathers (fast runtime check)."""

    def __init__(self, problem: RecedingProblem, first_window: int):
        self.p = problem
        self._pid: dict[tuple, int] = {}
        self._A: list[np.ndarray] = []
        self._b: list[float] = []
        self._ipreds: dict[int, IntervalPredicate] = {}
        starts = np.arange(first_window, problem.n_plan, dtype=np.int64)
        self._fn = self._build(problem.psi, starts)
        A = np.array(self._A)
        self.A, self.b = A, np.array(self._b)
        self.absAC = np.abs(A @ problem.C_out)
        self.AC = A @ problem.C_out
        # interval predicates: their table rows are overwritten with the I-STL lower endpoint
        self.irows = np.array(sorted(self._ipreds), dtype=np.int64)
        if self.irows.size:
            ip = [self._ipreds[r] for r in self.irows]
            self.ia_lo = np.array([q.a_lo for q in ip])[:, None, :]  # (P_I, 1, p)
            self.ia_hi = np.array([q.a_hi for q in ip])[:, None, :]
            self.ib_hi = np.array([q.b_hi for q in ip])[:, None]
            self.absC = np.abs(problem.C_out)
        self._tube: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self._faces: list | None = None

    def _leaf(self, node: LinearPredicate) -> int:
        if isinstance(node, IntervalPredicate):
            key = node.key()
        else:
            key = (tuple(np.round(node.a[:, 0], 12)), round(float(node.b[0]), 12))
        if key not in self._pid:
            self._pid[key] = len(self._A)
            self._A.append(node.a[:, 0].astype(float))
            self._b.append(float(node.b[0]))
            if isinstance(node, IntervalPredicate):
                self._ipreds[self._pid[key]] = node
        return self._pid[key]

    def _ilower(self, Y: np.ndarray, R: np.ndarray) -> np.ndarray:
        """``(P_I, T)`` lower endpoints of the interval predicates over the boxes ``[Y - R, Y + R]``."""
        return interval_lower(self.ia_lo, self.ia_hi, self.ib_hi, Y[None], R[None])

    def _build(self, node: STLFormula, ts: np.ndarray):
        if isinstance(node, LinearPredicate):
            pid = self._leaf(node)
            return lambda tab: tab[pid, ts]
        if not isinstance(node, STLTree):
            raise TypeError(type(node).__name__)
        groups: dict[int, tuple[STLFormula, list[int]]] = {}
        for child, off in zip(node.subformula_list, node.timesteps):
            groups.setdefault(id(child), (child, []))[1].append(int(off))
        parts = []
        for child, offs in groups.values():
            offs = np.asarray(offs, dtype=np.int64)
            cts = (ts[:, None] + offs[None, :]).ravel()
            uniq, inv = np.unique(cts, return_inverse=True)
            parts.append((self._build(child, uniq), inv, len(offs)))
        n = ts.size
        red = np.min if node.combination_type == "and" else np.max

        def ev(tab):
            return red(np.concatenate([f(tab)[inv].reshape(n, k) for f, inv, k in parts], axis=1), axis=1)
        return ev

    def table(self, X: np.ndarray, history: History, root_age: int | None) -> np.ndarray:
        p, tube, h = self.p, self.p.tube, self.p.h
        off = 0 if root_age is None else int(root_age)
        N = p.n_plan
        Y = X @ p.C_out.T
        tab = np.empty((len(self.b), h - 1 + 1 + N))
        Z = self.tube_arrays(off)[0]
        tab[:, h:] = self.A @ Y[1:].T - self.b[:, None] - self.absAC @ Z.T

        tab[:, h - 1] = self._boolean(self.A @ Y[0] - self.b, root_age)
        if h > 1:
            Vh = self.A @ history.y.T - self.b[:, None]  # (P, h-1)
            thr = np.stack([np.zeros(len(self.b)) if a is None else
                            p.delta + self.absAC @ tube.state(a) for a in history.age], axis=1)
            tab[:, :h - 1] = np.where(Vh >= thr, np.inf, -np.inf)
        if self.irows.size:
            self._interval_table(tab, Y, Z, history, root_age)
        return tab

    def _interval_table(self, tab: np.ndarray, Y: np.ndarray, Z: np.ndarray, history: History,
                        root_age: int | None) -> None:
        """Overwrite the interval predicates' rows of ``tab`` (same semantics as the MILP)."""
        p, h, rows = self.p, self.p.h, self.irows
        tube, delta = p.tube, p.delta
        tab[rows, h:] = self._ilower(Y[1:], Z @ self.absC.T)

        def boolean(y: np.ndarray, age: int | None) -> np.ndarray:
            if age is None:
                return np.where(self._ilower(y[None], np.zeros((1, len(y))))[:, 0] >= 0.0, np.inf, -np.inf)
            v = self._ilower(y[None], (self.absC @ tube.state(age))[None])[:, 0]
            return np.where(v >= delta, np.inf, -np.inf)

        tab[rows, h - 1] = boolean(Y[0], root_age)
        for j in range(h - 1):
            tab[rows, j] = boolean(history.y[j], history.age[j])

    def _boolean(self, v: np.ndarray, age: int | None) -> np.ndarray:
        thr = 0.0 if age is None else self.p.delta + self.absAC @ self.p.tube.state(age)
        return np.where(v >= thr, np.inf, -np.inf)

    def tube_arrays(self, off: int) -> tuple[np.ndarray, np.ndarray]:
        """``(Z (N, n) for i = 1..N, KZ (N, m) for i = 0..N-1)`` (cached per root age)."""
        hit = self._tube.get(off)
        if hit is None:
            N, tube = self.p.n_plan, self.p.tube
            Z = np.stack([tube.state(off + i) for i in range(1, N + 1)])
            KZ = np.stack([tube.input(off + i) for i in range(N)])
            hit = self._tube[off] = (Z, KZ)
        return hit

    def faces(self) -> list[tuple[np.ndarray, np.ndarray]]:
        if self._faces is None:
            self._faces = [] if self.p.terminal is None else self.p.terminal.face_rows()
        return self._faces

    def __call__(self, X: np.ndarray, history: History, root_age: int | None) -> float:
        return float(np.min(self._fn(self.table(X, history, root_age))))


_COMPILED: dict[tuple[int, int], tuple[RecedingProblem, CompiledWindows]] = {}


def compiled_windows(problem: RecedingProblem, first_window: int) -> CompiledWindows:
    key = (id(problem), first_window)
    hit = _COMPILED.get(key)
    if hit is None or hit[0] is not problem:
        hit = (problem, CompiledWindows(problem, first_window))
        _COMPILED[key] = hit
    return hit[1]


def check_plan(problem: RecedingProblem, x0: np.ndarray, X: np.ndarray, U: np.ndarray,
               history: History, root_age: int | None = None, first_window: int = 0,
               tol: float = 1e-6) -> CheckResult:
    """Feasibility of ``(X, U)`` for the request ``(x0, history, root_age, first_window)``."""
    p = problem
    N = p.n_plan
    off = 0 if root_age is None else int(root_age)
    X, U = np.asarray(X, float), np.asarray(U, float)
    if X.shape != (N + 1, p.model.n) or U.shape != (N, p.model.m):
        return CheckResult(False, "shape", -np.inf, None, np.inf, np.inf)
    if not (np.all(np.isfinite(X)) and np.all(np.isfinite(U))):
        return CheckResult(False, "nonfinite", -np.inf, None, np.inf, np.inf)
    dyn = float(np.max(np.abs(X[0] - x0)))
    pred = X[:-1] @ p.model.A.T + U @ p.model.B.T
    dyn = max(dyn, float(np.max(np.abs(X[1:] - pred))))
    cw = compiled_windows(p, first_window)
    Z, KZ = cw.tube_arrays(off)
    viol = max(float(np.max(p.x_lo + Z - X[1:])), float(np.max(X[1:] - p.x_hi + Z)),
               float(np.max(p.u_lo + KZ - U)), float(np.max(U - p.u_hi + KZ)))
    rho = cw(X, history, root_age)
    face = None
    if p.terminal is not None:
        last = X[N - p.terminal_block + 1:]
        for f, (H, g) in enumerate(cw.faces()):
            if float(np.max(last @ H.T - g)) <= tol:
                face = f
                break
    reasons = []
    if dyn > tol * max(1.0, float(np.max(np.abs(X)))):
        reasons.append(f"dynamics residual {dyn:.2e}")
    if viol > tol:
        reasons.append(f"bound violation {viol:.2e}")
    if rho < p.delta - tol:
        reasons.append(f"rho_lower {rho:.4f} < delta {p.delta}")
    if p.terminal is not None and face is None:
        reasons.append("terminal block not in any face set")
    ok = not reasons
    return CheckResult(ok, "ok" if ok else "; ".join(reasons), rho, face, dyn, viol)
