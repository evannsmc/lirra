"""Interval (uncertain) linear predicates ``[a]^T y - [b] >= 0`` (I-STL).

An :class:`IntervalPredicate` is an ``stlpy`` :class:`~stlpy.STL.LinearPredicate` whose normal and
offset are only known to lie in boxes ``a in [a_lo, a_hi]``, ``b in [b_lo, b_hi]``. The parent's
``a`` and ``b`` hold the nominal values (so the formula tree, horizons and names work unchanged);
the bounds define the semantics. Every evaluator in the project dispatches on this class before
:class:`LinearPredicate` (``rl_stl.stl.robustness``, ``rl_stl.synthesis.check``,
``rl_stl.synthesis.istl``, the JAX certificate), so formulas built only from exact predicates take the
old code paths unchanged.

Semantics (paper Sec. 2, eq. (margin)): for a signal sample in the box ``[y] = [c - r, c + r]``
(``r = 0`` for a measured sample) the predicate's value is the interval ``[a]^T [y] - [b]``; its lower
endpoint is

    lower(c, r) = sum_j min_{a_j in {a_lo_j, a_hi_j}} (a_j c_j - |a_j| r_j) - b_hi.

Per coordinate, ``min_{y in [c - r, c + r]} a y = a c - |a| r`` and ``a -> a c - |a| r`` is concave, so
its minimum over ``[a_lo, a_hi]`` is at an endpoint: the formula is the exact minimum of
``a^T y - b`` over the box ``[a] x [y] x [b]`` (the bilinear minimum is attained at a corner). The
upper endpoint is ``sum_j max_a (a_j c_j + |a_j| r_j) - b_lo``. With ``a_lo = a_hi`` and
``b_lo = b_hi`` both reduce to ``a^T c -+ |a|^T r - b``.

``lower`` is concave in ``c`` (a sum of minima of affine functions), so ``lower(C x, r) >= thr`` is a
convex constraint on ``x``; :mod:`rl_stl.synthesis.istl` encodes it exactly with one auxiliary variable
per signal coordinate.
"""

from __future__ import annotations

import numpy as np
from stlpy.STL import LinearPredicate


class IntervalPredicate(LinearPredicate):
    """``[a]^T y - [b] >= 0`` with ``a in [a_lo, a_hi]``, ``b in [b_lo, b_hi]``; ``a``, ``b`` nominal."""

    def __init__(self, a, b, a_lo, a_hi, b_lo, b_hi, name=None):
        super().__init__(a, b, name=name)
        self.a_lo = np.asarray(a_lo, dtype=float).reshape(-1)
        self.a_hi = np.asarray(a_hi, dtype=float).reshape(-1)
        self.b_lo, self.b_hi = float(b_lo), float(b_hi)
        a0, b0 = self.a[:, 0], float(self.b[0])
        if self.a_lo.shape != a0.shape or self.a_hi.shape != a0.shape:
            raise ValueError("a_lo / a_hi must have the shape of a")
        if np.any(self.a_lo > a0 + 1e-12) or np.any(a0 > self.a_hi + 1e-12):
            raise ValueError("nominal a must lie in [a_lo, a_hi]")
        if not self.b_lo - 1e-12 <= b0 <= self.b_hi + 1e-12:
            raise ValueError("nominal b must lie in [b_lo, b_hi]")

    @property
    def is_exact(self) -> bool:
        return bool(np.all(self.a_lo == self.a_hi) and self.b_lo == self.b_hi)

    @property
    def a_abs_max(self) -> np.ndarray:
        """``max(|a_lo|, |a_hi|)``: Lipschitz constant of ``lower`` per coordinate."""
        return np.maximum(np.abs(self.a_lo), np.abs(self.a_hi))

    def key(self) -> tuple:
        """Hashable identity of the predicate's data (shared binaries / table rows)."""
        return ("interval", tuple(np.round(self.a_lo, 12)), tuple(np.round(self.a_hi, 12)),
                round(self.b_lo, 12), round(self.b_hi, 12))

    def lower(self, c, r=0.0) -> np.ndarray:
        """Lower endpoint of ``[a]^T [c - r, c + r] - [b]`` (``c``, ``r``: ``(..., d)``)."""
        return interval_lower(self.a_lo, self.a_hi, self.b_hi, c, r)

    def upper(self, c, r=0.0) -> np.ndarray:
        """Upper endpoint of ``[a]^T [c - r, c + r] - [b]``."""
        return interval_upper(self.a_lo, self.a_hi, self.b_lo, c, r)

    def robustness(self, y, t):
        """``stlpy`` convention (``y`` is ``(d, T)``): the I-STL lower robustness of a point signal."""
        assert isinstance(y, np.ndarray), "y must be a numpy array"
        assert isinstance(t, int), "timestep t must be an integer"
        assert y.shape[0] == self.d, "y must be of shape (d,T)"
        return np.array([float(self.lower(y[:, t]))])

    def negation(self):
        newname = None if self.name is None else "not " + self.name
        return IntervalPredicate(-self.a, -self.b, -self.a_hi, -self.a_lo, -self.b_hi, -self.b_lo,
                                 name=newname)

    def __str__(self):
        if self.name is None:
            return f"{{ IntervalPredicate [{self.a_lo}, {self.a_hi}]*y >= [{self.b_lo}, {self.b_hi}] }}"
        return "{ IntervalPredicate " + self.name + " }"


def interval_term_min(a_lo, a_hi, c, r=0.0):
    """Elementwise ``min_{a in [a_lo, a_hi], y in [c - r, c + r]} a y`` (broadcasting, numpy only)."""
    c = np.asarray(c, dtype=float)
    r = np.asarray(r, dtype=float)
    return np.minimum(a_lo * c - np.abs(a_lo) * r, a_hi * c - np.abs(a_hi) * r)


def interval_term_max(a_lo, a_hi, c, r=0.0):
    c = np.asarray(c, dtype=float)
    r = np.asarray(r, dtype=float)
    return np.maximum(a_lo * c + np.abs(a_lo) * r, a_hi * c + np.abs(a_hi) * r)


def interval_lower(a_lo, a_hi, b_hi, c, r=0.0):
    """``sum_j interval_term_min - b_hi`` over the last axis."""
    return np.sum(interval_term_min(a_lo, a_hi, c, r), axis=-1) - b_hi


def interval_upper(a_lo, a_hi, b_lo, c, r=0.0):
    return np.sum(interval_term_max(a_lo, a_hi, c, r), axis=-1) - b_lo


def predicate_lower(pred: LinearPredicate, c, r=0.0) -> float:
    """Lower endpoint of a predicate (exact or interval) over the box ``[c - r, c + r]``."""
    if isinstance(pred, IntervalPredicate):
        return float(pred.lower(c, r))
    a = pred.a[:, 0]
    return float(a @ np.asarray(c, float) - np.abs(a) @ (np.asarray(r, float) * np.ones_like(a)) - pred.b[0])


def predicate_upper(pred: LinearPredicate, c, r=0.0) -> float:
    if isinstance(pred, IntervalPredicate):
        return float(pred.upper(c, r))
    a = pred.a[:, 0]
    return float(a @ np.asarray(c, float) + np.abs(a) @ (np.asarray(r, float) * np.ones_like(a)) - pred.b[0])
