"""STL specifications built with ``stlpy``.

Predicates are written with unit-norm axis-aligned normals so that robustness is
directly in metres (or m/s) and is 1-Lipschitz with respect to the sup-norm of
the corresponding signal component. This is used when converting a signal
deviation bound into a robustness margin.
"""

from __future__ import annotations

import dataclasses

import numpy as np
from stlpy.STL import LinearPredicate, STLFormula

Box = tuple[float, float, float, float]  # (x_min, x_max, y_min, y_max)


def _axis(d: int, index: int, sign: float) -> np.ndarray:
    a = np.zeros(d)
    a[index] = sign
    return a


def inside_box(box: Box, ix: int, iy: int, d: int, name: str) -> STLFormula:
    """Conjunction of four half-planes; robustness = signed distance-like margin."""
    x_min, x_max, y_min, y_max = box
    preds = [
        LinearPredicate(_axis(d, ix, +1.0), x_min, name=f"{name}:x>=xmin"),
        LinearPredicate(_axis(d, ix, -1.0), -x_max, name=f"{name}:x<=xmax"),
        LinearPredicate(_axis(d, iy, +1.0), y_min, name=f"{name}:y>=ymin"),
        LinearPredicate(_axis(d, iy, -1.0), -y_max, name=f"{name}:y<=ymax"),
    ]
    formula = preds[0] & preds[1] & preds[2] & preds[3]
    formula.simplify()
    formula.name = name
    return formula


def outside_box(box: Box, ix: int, iy: int, d: int, name: str) -> STLFormula:
    """Disjunction of four half-planes (outside an axis-aligned rectangle)."""
    x_min, x_max, y_min, y_max = box
    preds = [
        LinearPredicate(_axis(d, ix, -1.0), -x_min, name=f"{name}:x<=xmin"),
        LinearPredicate(_axis(d, ix, +1.0), x_max, name=f"{name}:x>=xmax"),
        LinearPredicate(_axis(d, iy, -1.0), -y_min, name=f"{name}:y<=ymin"),
        LinearPredicate(_axis(d, iy, +1.0), y_max, name=f"{name}:y>=ymax"),
    ]
    formula = preds[0] | preds[1] | preds[2] | preds[3]
    formula.simplify()
    formula.name = name
    return formula


@dataclasses.dataclass(frozen=True)
class ReachAvoidTrackScenario:
    """2-D double-integrator reach-avoid-track benchmark (theorem/debug stage).

    Specification over the state signal ``y = [px, py, vx, vy]``::

        phi = G_[0,N] (in_workspace AND outside every obstacle)
              AND F_[0,N_goal] (in_goal)

    Tracking a straight-line reference that passes through the obstacle is the
    separate task objective ``J_task``; it is deliberately in conflict with the
    specification.
    """

    dt: float = 0.1
    horizon: int = 40  # N: formula horizon in steps
    goal_deadline: int = 35  # N_goal <= N
    workspace: Box = (0.0, 10.0, 0.0, 10.0)
    obstacles: tuple[Box, ...] = ((4.0, 6.0, 3.5, 6.5),)
    goal: Box = (8.0, 9.5, 8.0, 9.5)
    start: tuple[float, float] = (1.0, 1.0)
    v_max: float = 4.0  # |v_i| <= v_max  (hard state constraint, not STL)
    a_max: float = 4.0  # |a_i| <= a_max  (hard input constraint)
    reference_arrival_step: int = 30  # step at which the reference reaches the goal centre

    def __post_init__(self) -> None:
        if not 0 < self.goal_deadline <= self.horizon:
            raise ValueError("goal_deadline must be in (0, horizon]")

    @property
    def signal_dim(self) -> int:
        return 4

    def specification(self) -> STLFormula:
        d = self.signal_dim
        safe = inside_box(self.workspace, 0, 1, d, "workspace")
        for i, obs in enumerate(self.obstacles):
            safe = safe & outside_box(obs, 0, 1, d, f"obstacle{i}")
        safe.name = "safe"
        always_safe = safe.always(0, self.horizon)
        always_safe.name = "always_safe"
        reach = inside_box(self.goal, 0, 1, d, "goal").eventually(0, self.goal_deadline)
        reach.name = "reach_goal"
        spec = always_safe & reach
        spec.name = "reach_avoid_track"
        return spec

    def state_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Hard state box; also the signal domain for online interval semantics."""
        x_min, x_max, y_min, y_max = self.workspace
        lo = np.array([x_min, y_min, -self.v_max, -self.v_max])
        hi = np.array([x_max, y_max, self.v_max, self.v_max])
        return lo, hi

    def input_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return -self.a_max * np.ones(2), self.a_max * np.ones(2)

    def x0(self) -> np.ndarray:
        return np.array([self.start[0], self.start[1], 0.0, 0.0])

    def goal_center(self) -> np.ndarray:
        x_min, x_max, y_min, y_max = self.goal
        return np.array([(x_min + x_max) / 2, (y_min + y_max) / 2])

    def reference(self) -> np.ndarray:
        """Position reference ``(N+1, 2)``: straight line start -> goal centre, then hold."""
        start = np.asarray(self.start, dtype=float)
        goal = self.goal_center()
        s = np.clip(np.arange(self.horizon + 1) / self.reference_arrival_step, 0.0, 1.0)
        return start[None, :] + s[:, None] * (goal - start)[None, :]

    def reference_states(self) -> np.ndarray:
        """State reference ``(N+1, 4)`` with finite-difference velocities."""
        p = self.reference()
        v = np.zeros_like(p)
        v[:-1] = (p[1:] - p[:-1]) / self.dt
        return np.hstack([p, v])
