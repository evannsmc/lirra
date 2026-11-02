"""Independent STL robustness evaluation and online (prefix) interval semantics.

Signals in this module are **time-first** arrays of shape ``(T, d)``, the
project convention for trajectories. ``stlpy``'s own ``robustness(y, t)`` uses
``(d, T)``; :func:`stlpy_robustness` performs that conversion.

The evaluator here is written separately from ``stlpy`` (memoised, iterative
over the formula tree) so that robustness of every synthesised trajectory can be
recomputed independently of both the solver and ``stlpy``'s implementation.

Online semantics
----------------
For an observed prefix ``y_0..y_k`` and a formula whose horizon extends past
``k``, ordinary robustness is undefined. :func:`robustness_interval` returns
``[rho_lower, rho_upper]`` enclosing the robustness of **every completion whose
unknown samples lie in the signal box** ``[signal_lo, signal_hi]``:

* ``rho_lower > 0``  every admissible completion satisfies;
* ``rho_upper < 0``  no admissible completion satisfies;
* otherwise           inconclusive.

This is a trace-only (model-free) enclosure: unknown samples are treated
independently at each predicate, so the interval is sound but not tight. It
ignores dynamics, so ``rho_upper >= 0`` does *not* imply that the system can
still satisfy the formula (see the dynamics-constrained oracle in
``rl_stl.evaluation.oracle``).

Future-margin robustness
------------------------
Ordinary robustness of ``y_0:k ++ y_hat`` is capped by the worst *observed*
sample, which can no longer be perturbed. For runtime margins the project uses
the future-margin robustness ``rho_F`` (``observed_until=k``): predicates at
observed times ``<= k`` evaluate to ``+inf`` if satisfied and ``-inf`` if
violated; later predicates keep their quantitative value. Then

* ``rho_F >= 0``  iff  ``rho >= 0`` (same Boolean satisfaction), and
* ``rho_F >= delta`` means every perturbation of the unobserved samples with
  ``L * sup-norm < delta`` preserves satisfaction (``L`` from
  :func:`lipschitz_sup_norm`).

Interval predicates
-------------------
For an :class:`~rl_stl.stl.interval.IntervalPredicate` ``[a]^T y - [b] >= 0`` every evaluator here
uses the I-STL interval of the predicate's value over ``[a]`` and ``[b]``: :func:`robustness` (and
``stlpy``'s own ``robustness``) returns the **lower** robustness (worst case over the predicate
uncertainty, each predicate occurrence independently), :func:`robustness_interval` encloses
``[lower, upper]``, and :func:`interval_lower_robustness` additionally widens each sample to a box
``[y_t - r_t, y_t + r_t]`` (the tube of a predicted sample). Exact predicates keep their values.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
from stlpy.STL import LinearPredicate, NonlinearPredicate, STLFormula, STLTree

from rl_stl.stl.interval import IntervalPredicate, predicate_lower

Interval = tuple[float, float]


def formula_horizon(formula: STLFormula) -> int:
    """Largest time offset referenced by the formula (signal needs horizon+1 samples)."""
    memo: dict[int, int] = {}

    def rec(node: STLFormula) -> int:
        key = id(node)
        if key not in memo:
            if isinstance(node, (LinearPredicate, NonlinearPredicate)):
                memo[key] = 0
            else:
                memo[key] = max(t + rec(c) for c, t in zip(node.subformula_list, node.timesteps))
        return memo[key]

    return rec(formula)


def iter_predicates(formula: STLFormula, t: int = 0) -> Iterator[tuple[LinearPredicate, int]]:
    """Yield ``(predicate, absolute_time)`` for every leaf occurrence."""
    stack = [(formula, t)]
    while stack:
        node, tau = stack.pop()
        if isinstance(node, LinearPredicate):
            yield node, tau
        elif isinstance(node, NonlinearPredicate):
            raise TypeError("only linear predicates are supported")
        else:
            stack.extend((c, tau + dt) for c, dt in zip(node.subformula_list, node.timesteps))


def _pred_value(pred: LinearPredicate, y_t: np.ndarray) -> float:
    if isinstance(pred, IntervalPredicate):
        return float(pred.lower(y_t))
    return float(pred.a[:, 0] @ y_t - pred.b[0])


def _pred_value_range(pred: LinearPredicate, y_t: np.ndarray) -> Interval:
    """``[lower, upper]`` of a predicate at a known sample (degenerate for exact predicates)."""
    if isinstance(pred, IntervalPredicate):
        return float(pred.lower(y_t)), float(pred.upper(y_t))
    v = _pred_value(pred, y_t)
    return v, v


def _pred_box_range(pred: LinearPredicate, lo: np.ndarray, hi: np.ndarray) -> Interval:
    if isinstance(pred, IntervalPredicate):
        c, r = (np.asarray(hi, float) + np.asarray(lo, float)) / 2, (np.asarray(hi, float) - np.asarray(lo, float)) / 2
        return float(pred.lower(c, r)), float(pred.upper(c, r))
    a = pred.a[:, 0]
    vmin = float(np.sum(np.minimum(a * lo, a * hi)) - pred.b[0])
    vmax = float(np.sum(np.maximum(a * lo, a * hi)) - pred.b[0])
    return vmin, vmax


def robustness(
    formula: STLFormula, signal: np.ndarray, t: int = 0, observed_until: int | None = None
) -> float:
    """Quantitative robustness ``rho^phi(signal, t)`` for a complete signal ``(T, d)``.

    With ``observed_until=k`` returns the future-margin robustness ``rho_F``
    (predicates at times ``<= k`` count as ``+/-inf``).
    """
    signal = np.asarray(signal, dtype=float)
    need = t + formula_horizon(formula) + 1
    if signal.ndim != 2 or signal.shape[0] < need:
        raise ValueError(f"signal must be (T>={need}, d); got {signal.shape}")
    if not np.all(np.isfinite(signal[:need])):
        return float("nan")
    memo: dict[tuple[int, int], float] = {}

    def rec(node: STLFormula, tau: int) -> float:
        key = (id(node), tau)
        if key in memo:
            return memo[key]
        if isinstance(node, LinearPredicate):
            val = _pred_value(node, signal[tau])
            if observed_until is not None and tau <= observed_until:
                val = np.inf if val >= 0 else -np.inf
        elif isinstance(node, STLTree):
            vals = [rec(c, tau + dt) for c, dt in zip(node.subformula_list, node.timesteps)]
            val = min(vals) if node.combination_type == "and" else max(vals)
        else:
            raise TypeError(f"unsupported formula node {type(node).__name__}")
        memo[key] = val
        return val

    return rec(formula, t)


def robustness_interval(
    formula: STLFormula,
    prefix: np.ndarray,
    signal_lo: np.ndarray,
    signal_hi: np.ndarray,
    t: int = 0,
    observed_until: int | None = None,
) -> Interval:
    """Enclosure of robustness over all completions of ``prefix`` inside the signal box.

    ``prefix`` has shape ``(k+1, d)`` holding the known samples ``y_0..y_k``.
    Samples at times ``> k`` are unknown and range over ``[signal_lo, signal_hi]``.
    If the prefix already covers the formula horizon the interval is degenerate.
    With ``observed_until=j`` (``j <= k``) predicates at times ``<= j`` are
    Boolean (future-margin semantics); known samples after ``j`` (e.g. a
    prediction) stay quantitative.
    """
    prefix = np.asarray(prefix, dtype=float).reshape(-1, len(signal_lo))
    k = prefix.shape[0] - 1
    lo = np.asarray(signal_lo, dtype=float)
    hi = np.asarray(signal_hi, dtype=float)
    memo: dict[tuple[int, int], Interval] = {}

    def rec(node: STLFormula, tau: int) -> Interval:
        key = (id(node), tau)
        if key in memo:
            return memo[key]
        if isinstance(node, LinearPredicate):
            if tau <= k:
                v_lo, v_hi = _pred_value_range(node, prefix[tau])
                if observed_until is not None and tau <= observed_until:
                    v_lo = np.inf if v_lo >= 0 else -np.inf
                    v_hi = np.inf if v_hi >= 0 else -np.inf
                val = (v_lo, v_hi)
            else:
                val = _pred_box_range(node, lo, hi)
        elif isinstance(node, STLTree):
            ivs = [rec(c, tau + dt) for c, dt in zip(node.subformula_list, node.timesteps)]
            if node.combination_type == "and":
                val = (min(i[0] for i in ivs), min(i[1] for i in ivs))
            else:
                val = (max(i[0] for i in ivs), max(i[1] for i in ivs))
        else:
            raise TypeError(f"unsupported formula node {type(node).__name__}")
        memo[key] = val
        return val

    return rec(formula, t)


def interval_lower_robustness(formula: STLFormula, signal: np.ndarray, radius: np.ndarray | float = 0.0,
                              t: int = 0) -> float:
    """I-STL lower robustness of the interval signal ``[y_t - r_t, y_t + r_t]`` (``signal``, ``radius``:
    ``(T, d)``, ``radius`` may broadcast): each predicate occurrence takes the lower endpoint of its value
    over the sample box and, for interval predicates, over ``[a]`` and ``[b]``; min/max combine the lower
    endpoints (both are monotone). ``radius = 0`` gives :func:`robustness`."""
    signal = np.asarray(signal, dtype=float)
    R = np.broadcast_to(np.asarray(radius, dtype=float), signal.shape)
    need = t + formula_horizon(formula) + 1
    if signal.ndim != 2 or signal.shape[0] < need:
        raise ValueError(f"signal must be (T>={need}, d); got {signal.shape}")
    memo: dict[tuple[int, int], float] = {}

    def rec(node: STLFormula, tau: int) -> float:
        key = (id(node), tau)
        if key in memo:
            return memo[key]
        if isinstance(node, LinearPredicate):
            val = predicate_lower(node, signal[tau], R[tau])
        elif isinstance(node, STLTree):
            vals = [rec(c, tau + dt) for c, dt in zip(node.subformula_list, node.timesteps)]
            val = min(vals) if node.combination_type == "and" else max(vals)
        else:
            raise TypeError(f"unsupported formula node {type(node).__name__}")
        memo[key] = val
        return val

    return rec(formula, t)


def top_level_intervals(
    formula: STLFormula, prefix: np.ndarray, signal_lo: np.ndarray, signal_hi: np.ndarray
) -> dict[str, Interval]:
    """Interval per top-level conjunct: exposes outstanding vs. discharged obligations.

    For ``G safe AND F goal`` a positive lower bound on ``F goal`` means the
    eventuality has been discharged by the prefix; an upper bound below zero on
    either conjunct means it is already violated.
    """
    if isinstance(formula, STLTree) and formula.combination_type == "and":
        children = zip(formula.subformula_list, formula.timesteps)
    else:
        children = [(formula, 0)]
    out: dict[str, Interval] = {}
    for i, (child, dt) in enumerate(children):
        name = child.name or f"conjunct{i}"
        out[name] = robustness_interval(child, prefix, signal_lo, signal_hi, t=dt)
    return out


def stlpy_robustness(formula: STLFormula, signal: np.ndarray, t: int = 0) -> float:
    """``stlpy``'s own robustness for a time-first signal (for cross-checking)."""
    return float(np.asarray(formula.robustness(np.asarray(signal, dtype=float).T, t)).reshape(-1)[0])


def lipschitz_sup_norm(formula: STLFormula) -> float:
    """Constant ``L`` with ``|rho(y) - rho(y')| <= L * max_t ||y_t - y'_t||_inf``.

    Min/max are 1-Lipschitz, so ``L = max_predicates ||a||_1`` (``|| max(|a_lo|, |a_hi|) ||_1`` for an
    interval predicate's lower endpoint).
    """
    return max(float(p.a_abs_max.sum()) if isinstance(p, IntervalPredicate) else float(np.abs(p.a).sum())
               for p, _ in iter_predicates(formula))
