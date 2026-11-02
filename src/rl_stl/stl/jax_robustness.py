"""Compile ``stlpy`` formulas into jittable JAX robustness functions.

``stlpy`` owns formula construction; this module only evaluates the standard
discrete-time quantitative semantics (Donze and Maler) that ``stlpy``'s
``robustness()`` implements, so the same formula object can be scored inside a
JIT-compiled, vmapped MJX rollout:

- ``LinearPredicate``: ``rho = a^T y_t - b``
- ``STLTree`` "and": ``min_i rho(child_i, t + timesteps_i)``
- ``STLTree`` "or":  ``max_i rho(child_i, t + timesteps_i)``

The tree is walked **once in Python** at compile time. Children that are the same
object (``always``/``eventually`` repeat one subformula over a time interval) are
grouped, their time indices are deduplicated, and each group becomes a single
gather. The resulting function has only static shapes and indices.

Signals are ``(T+1, d)`` arrays (time first), unlike ``stlpy``'s ``(d, T+1)``.
``NonlinearPredicate`` is not supported (the MICP layer does not support it either).

Reward timing is *not* decided here; see :func:`stl_hinge_penalty` and the X2
environment docstring for how windowed robustness is turned into per-step rewards.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import jax.numpy as jnp
import numpy as np
from stlpy.STL import LinearPredicate, NonlinearPredicate, STLFormula, STLTree

Evaluator = Callable[[jnp.ndarray], jnp.ndarray]


class CompiledRobustness:
    """Callable ``y (T+1, d) -> rho`` at the compile-time start times."""

    def __init__(self, formula: STLFormula, start_times: Sequence[int] = (0,)):
        self.start_times = np.asarray(start_times, dtype=np.int64)
        if self.start_times.ndim != 1 or self.start_times.size == 0 or self.start_times.min() < 0:
            raise ValueError("start_times must be a non-empty 1-D sequence of non-negative ints")
        self.dim = formula.d
        self._max_index = [0]
        self._fn = self._build(formula, self.start_times)
        self.required_length = self._max_index[0] + 1

    def __call__(self, y: jnp.ndarray) -> jnp.ndarray:
        if y.shape[-1] != self.dim:
            raise ValueError(f"signal dimension {y.shape[-1]} != formula dimension {self.dim}")
        if y.shape[-2] < self.required_length:
            raise ValueError(f"signal has {y.shape[-2]} samples, formula needs {self.required_length}")
        out = self._fn(y)
        return out[..., 0] if self.start_times.size == 1 else out

    def _build(self, node: STLFormula, ts: np.ndarray) -> Evaluator:
        if isinstance(node, NonlinearPredicate):
            raise TypeError("NonlinearPredicate is not supported")
        if isinstance(node, LinearPredicate):
            a = jnp.asarray(node.a.reshape(-1))
            b = float(node.b[0])
            idx = jnp.asarray(ts)
            self._max_index[0] = max(self._max_index[0], int(ts.max()))
            return lambda y: y[..., idx, :] @ a - b
        if not isinstance(node, STLTree):
            raise TypeError(f"unsupported formula node {type(node)}")

        groups: dict[int, tuple[STLFormula, list[int]]] = {}
        for child, offset in zip(node.subformula_list, node.timesteps, strict=True):
            groups.setdefault(id(child), (child, []))[1].append(int(offset))

        parts = []
        for child, offsets in groups.values():
            offs = np.asarray(offsets, dtype=np.int64)
            child_ts = (ts[:, None] + offs[None, :]).ravel()
            uniq, inverse = np.unique(child_ts, return_inverse=True)
            parts.append((self._build(child, uniq), jnp.asarray(inverse), len(offs)))

        n = ts.size
        reduce = jnp.min if node.combination_type == "and" else jnp.max

        def evaluate(y: jnp.ndarray) -> jnp.ndarray:
            cols = [fn(y)[..., inv].reshape(*y.shape[:-2], n, k) for fn, inv, k in parts]
            return reduce(jnp.concatenate(cols, axis=-1), axis=-1)

        return evaluate


def compile_robustness(formula: STLFormula, start_time: int = 0) -> CompiledRobustness:
    """Robustness of ``formula`` at ``start_time`` for a ``(T+1, d)`` (or batched ``(..., T+1, d)``) signal."""
    return CompiledRobustness(formula, (start_time,))


def compile_robustness_trace(formula: STLFormula, start_times: Sequence[int]) -> CompiledRobustness:
    """Robustness at several start times at once, e.g. every window start of a buffer."""
    return CompiledRobustness(formula, start_times)


def stl_hinge_penalty(rho: jnp.ndarray, delta: float) -> jnp.ndarray:
    """Capped STL term ``max(0, delta - rho)``: zero once the margin ``delta`` is met."""
    return jnp.maximum(0.0, delta - rho)
