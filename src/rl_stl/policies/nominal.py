"""Nominal mission controller of the NAHS patrol: PD toward the current waypoint.

NAHS (`nominal_controller.py:80-93, 164-173`) flies a PD law toward the corners
``(0, r), (r, 0), (0, -r), (-r, 0)`` and switches to the next corner inside ``waypoint_tol``.
Straight legs between corners cut through ``S`` (the midpoint of the first leg is inside),
so the nominal mission violates the specification; the synthesis deviates from it
minimally. The same law, written on the synthesis model ``f_syn``, generates the nominal
reference ``(x_nom, u_nom)`` of every synthesis request and the mission reward of the learned
policy, in one of two forms:

* ``rerooted`` (:meth:`PatrolPD.rollout`): the PD law rolled out from the state the plan is
  rooted at, with the waypoint index advanced on the realised trajectory. Falling behind
  costs nothing, because the reference falls behind too, and the waypoint switch (after which
  the reference cuts through ``S``) happens only when the vehicle reaches the waypoint. A
  long-horizon learner minimises this cost by parking just outside the waypoint tolerance.
* ``clock`` (:class:`ClockReference`): the PD patrol rolled out once from the episode's
  initial state, indexed by absolute step. A plan rooted at step ``r`` tracks
  ``x_nom(r), .., x_nom(r + N)``, so falling behind keeps costing. Its inputs saturate at
  ``input_scale`` times the input bounds: a reference flown at the bounds leaves the planner no
  headroom to catch up after a detour around ``S``, and the phase lag then grows without
  bound (the blimp's PD patrol is saturated 91% of the time).

The controller is generic over axis blocks: for each patrol axis ``a`` (the two planar axes of
(24)/(25), or x, y, z for the cube specs) it commands ``u_a = clip(kp (p_ref_a - p_a) - kd v_a)``;
extra regulated axes (altitude, yaw) use the same law toward fixed set-points. Any number of
waypoints is cycled (4 for the square, 6 for the cube).
"""

from __future__ import annotations

import dataclasses

import numpy as np

from rl_stl.dynamics.linear import DiscreteLinearModel


@dataclasses.dataclass(frozen=True)
class AxisPD:
    pos: int  # state index of the position-like coordinate
    vel: int  # state index of its rate
    input: int
    kp: float
    kd: float


@dataclasses.dataclass
class PatrolPD:
    """PD patrol; ``planar`` (patrol) axes follow the waypoints, ``regulated`` axes their set-points."""

    planar: tuple[AxisPD, ...]  # one per waypoint coordinate
    waypoints: np.ndarray  # (n_wp, len(planar))
    u_lo: np.ndarray
    u_hi: np.ndarray
    waypoint_tol: float = 0.1
    regulated: tuple[tuple[AxisPD, float], ...] = ()
    index: int = 0  # current waypoint

    def __post_init__(self):
        self.waypoints = np.asarray(self.waypoints, float)
        if self.waypoints.ndim != 2 or self.waypoints.shape[1] != len(self.planar):
            raise ValueError(f"waypoints {self.waypoints.shape} do not match {len(self.planar)} patrol axes")

    def reset(self, x: np.ndarray) -> None:
        """Start at the waypoint after the closest one (NAHS starts at the first corner ahead)."""
        p = self.position(x)
        self.index = int(np.argmin(np.linalg.norm(self.waypoints - p, axis=1)) + 1) % len(self.waypoints)

    def position(self, x: np.ndarray) -> np.ndarray:
        return np.array([x[ax.pos] for ax in self.planar])

    def _advance(self, x: np.ndarray, index: int) -> int:
        if np.linalg.norm(self.waypoints[index] - self.position(x)) < self.waypoint_tol:
            return (index + 1) % len(self.waypoints)
        return index

    def _u(self, x: np.ndarray, index: int) -> np.ndarray:
        u = np.zeros(len(self.u_lo))
        ref = self.waypoints[index]
        for ax, r in zip(self.planar, ref):
            u[ax.input] = ax.kp * (r - x[ax.pos]) - ax.kd * x[ax.vel]
        for ax, sp in self.regulated:
            u[ax.input] = ax.kp * (sp - x[ax.pos]) - ax.kd * x[ax.vel]
        return np.clip(u, self.u_lo, self.u_hi)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Input at ``x``; advances the internal waypoint index."""
        self.index = self._advance(x, self.index)
        return self._u(x, self.index)

    def rollout(self, model: DiscreteLinearModel, x0: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
        """Nominal reference ``(X (n+1, nx), U (n, nu))`` from ``x0`` (internal index untouched)."""
        X = np.empty((n + 1, model.n))
        U = np.empty((n, model.m))
        X[0] = x0
        idx = self.index
        for i in range(n):
            idx = self._advance(X[i], idx)
            U[i] = self._u(X[i], idx)
            X[i + 1] = model.step(X[i], U[i])
        return X, U


class ClockReference:
    """Clock-indexed nominal patrol: the PD law rolled out on ``f_syn`` from the episode start.

    ``X[t]`` is the nominal state at absolute step ``t`` and ``U[t]`` its input (the same
    advance-then-command order as :meth:`PatrolPD.rollout`, which this reproduces for
    ``t = 0..n`` from the same initial state and index). The rollout is deterministic
    (no disturbance) and extended on demand. ``input_scale`` in ``(0, 1]`` shrinks the input
    box of the reference (not of the plans).
    """

    def __init__(self, nominal: PatrolPD, model: DiscreteLinearModel, input_scale: float = 1.0):
        if not 0.0 < input_scale <= 1.0:
            raise ValueError("input_scale must be in (0, 1]")
        self.nominal, self.model, self.input_scale = nominal, model, float(input_scale)
        self.X: list[np.ndarray] = []
        self.U: list[np.ndarray] = []
        self.index: list[int] = []  # waypoint index used to command U[t]

    def reset(self, x0: np.ndarray) -> None:
        self.nominal.reset(x0)
        self.X, self.U, self.index = [np.asarray(x0, float).copy()], [], []
        self._idx = self.nominal.index

    def _extend(self, t: int) -> None:
        while len(self.U) < t:
            x = self.X[-1]
            self._idx = self.nominal._advance(x, self._idx)
            s = self.input_scale
            u = np.clip(self.nominal._u(x, self._idx), s * self.nominal.u_lo, s * self.nominal.u_hi)
            self.index.append(self._idx)
            self.U.append(u)
            self.X.append(self.model.step(x, u))

    def window(self, k: int, n: int) -> tuple[np.ndarray, np.ndarray]:
        """``(X[k..k+n] (n+1, nx), U[k..k+n-1] (n, nu))`` for a plan rooted at step ``k >= 0``."""
        if k < 0:
            raise ValueError("the clock reference starts at step 0")
        self._extend(k + n)
        return np.stack(self.X[k:k + n + 1]), np.stack(self.U[k:k + n])

    def index_at(self, k: int) -> int:
        """Waypoint the reference is heading for at step ``k``."""
        self._extend(k + 1)
        return self.index[k]
