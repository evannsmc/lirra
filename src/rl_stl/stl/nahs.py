"""NAHS patrol task: keep-out square, formulas (24)/(25), receding window specification.

Baird, Morales-Cuadrado and Coogan (NAHS 2026 submission, Sec. 11):

    theta = (x in S) => F_[0,4] (x notin S)                 (24)
    phi   = (x in S) => F_[0,3] G_[0,1] (x notin S)         (25)

with time bounds in seconds and ``S = {|px| < 1.41, |py| < 1.41}`` (2 m taped square
inflated by the 0.38 m blimp radius and the 0.03 m tape). The reference code
(`rta_fixed.py:395-397`) converts seconds to steps as ``F_[0, round(4/dt) - 1]`` and
``F_[0, round(3/dt) - 1] G_[0, round(1/dt)]``; :func:`seconds_to_steps` follows that
convention so that ``dt = 0.5`` reproduces their 7-step formulas exactly.

``(x in S) => F_[0,a] notS`` equals ``notS OR F_[0,a] notS = F_[0,a] notS`` (the
eventually includes offset 0), so theta is built as the eventually alone. phi keeps
the disjunction (``notS`` now does not imply ``F G notS``).

The signal is the planar position ``y = (px, py)``; every predicate is
``+-px - 1.41 >= 0`` or ``+-py - 1.41 >= 0`` (unit normals, robustness in metres).

3-D cube variants (theta^+, phi^+)
---------------------------------
``theta_plus`` and ``phi_plus`` keep the temporal structure of (24)/(25) and replace the square by
the cube ``S+ = {|px|, |py|, |pz| < 1.41}`` (signal ``y = (px, py, pz)``, z relative to the cube
centre). The patrol waypoints are the six face centres moved outward by the 2-D offset
(``waypoint_radius - half_width``), visited in the order ``+y, +x, +z, -y, -x, -z``: consecutive
faces are adjacent, so each straight leg cuts an edge region of ``S+`` as the 2-D legs cut the
corners of ``S``. Restricted to the plane the cycle is the 2-D order ``+y, +x, -y, -x``.

Uncertain predicates (theta_u, phi_u, theta_plus_u, phi_plus_u)
---------------------------------------------------------------
The ``_u`` specs keep the temporal structure and the signal of their exact counterpart and replace
every face predicate by an interval predicate ``[a]^T y - [b] >= 0``
(:class:`~rl_stl.stl.interval.IntervalPredicate`). The true keep-out region is the nominal square
(cube) rotated by an unknown angle of at most ``pred_rotation`` (0.05 rad, any axis in 3-D) about its
centre, with an unknown half-width ``b in [half_width - pred_offset, half_width + pred_offset]``
(0.05 m). ``[b]`` is that interval; ``[a]`` is the interval hull of the face's rotated unit normal:
for the ``+x`` face ``a_x in [cos(pred_rotation), 1]`` and every tangential component in
``[-sin(pred_rotation), sin(pred_rotation)]`` (signs flipped for the ``-`` faces). Every true region
is therefore inside the predicates' uncertainty set, and the I-STL lower bound
(:mod:`rl_stl.stl.interval`) is a worst case over all of them.

``PatrolTask.nominal_predicates`` is an ablation: the loop plans and checks with the exact nominal
predicates while the episode is scored against a true region drawn inside the uncertainty set
(:meth:`PatrolTask.sample_true_region`, :meth:`PatrolTask.true_formula`).

Receding window
---------------
At planning step ``t`` the synthesis signal is ``y_{t-h+1} .. y_t`` (measured) followed
by ``y_{t+1} .. y_{t+N}`` (planned), ``L = h + N`` samples, where ``h`` is the formula
horizon. The formula is enforced on every window ``[s, s+h]`` that contains a planned
sample: local starts ``s = 0 .. N-1`` (NAHS (16): ``k = t-h+1 .. t+N-h``). Windows that
would start before the episode start are not enforced (:func:`window_spec`).
"""

from __future__ import annotations

import dataclasses
from typing import Literal

import numpy as np
from stlpy.STL import LinearPredicate, STLFormula, STLTree

from rl_stl.stl.interval import IntervalPredicate
from rl_stl.stl.robustness import formula_horizon

SpecName = Literal["theta", "phi", "theta_plus", "phi_plus", "theta_u", "phi_u", "theta_plus_u", "phi_plus_u"]
EXACT_SPECS: tuple[str, ...] = ("theta", "phi", "theta_plus", "phi_plus")
UNCERTAIN_SPECS: tuple[str, ...] = ("theta_u", "phi_u", "theta_plus_u", "phi_plus_u")
SPECS: tuple[str, ...] = EXACT_SPECS + UNCERTAIN_SPECS


def spec_dim(spec: str) -> int:
    """Signal dimension: 3 for the cube specs (``*_plus``, ``*_plus_u``), else 2."""
    return 3 if "_plus" in spec else 2


def spec_base(spec: str) -> str:
    """Temporal structure ``theta`` (24) or ``phi`` (25)."""
    return spec.removesuffix("_u").removesuffix("_plus")


def spec_uncertain(spec: str) -> bool:
    """Interval (uncertain) predicates: the ``_u`` specs."""
    return spec.endswith("_u")


def seconds_to_steps(seconds: float, dt: float) -> int:
    """NAHS convention for the upper bound of ``F_[0, seconds]``: ``round(s/dt) - 1``."""
    return round(seconds / dt) - 1


def outside_box(half_width: float, d: int = 2, axes: tuple[int, ...] | None = None) -> STLFormula:
    """``x notin S``: disjunction of the ``2 len(axes)`` half-spaces outside ``|y_i| < half_width``.

    ``axes`` (default ``0..d-1``) are the signal components that bound ``S``; two axes give the NAHS
    square, three the cube ``S+``.
    """
    axes = tuple(range(d)) if axes is None else tuple(axes)
    names = "xyz"
    preds = []
    for idx in axes:
        for sign in (-1.0, 1.0):
            a = np.zeros(d)
            a[idx] = sign
            tag = f"p{names[idx] if idx < 3 else idx}{'<=-c' if sign < 0 else '>=c'}"
            preds.append(LinearPredicate(a, half_width, name=tag))
    f = preds[0]
    for q in preds[1:]:
        f = f | q
    f.simplify()
    f.name = "notS"
    return f


def interval_outside_box(half_width: float, d: int = 2, axes: tuple[int, ...] | None = None,
                         offset_bound: float = 0.05, rotation_bound: float = 0.05) -> STLFormula:
    """``x notin S`` for an uncertain box: interval predicates ``[a]^T y - [b] >= 0`` per face.

    ``[b] = [half_width - offset_bound, half_width + offset_bound]``; ``[a]`` is the box hull of the
    face's outward unit normal rotated (within the subspace of ``axes``) by at most ``rotation_bound``:
    normal component ``sign * [cos(rotation_bound), 1]``, other box axes ``[-sin, sin]``, signal
    components outside ``axes`` 0. The nominal ``a``, ``b`` are those of :func:`outside_box`.
    """
    axes = tuple(range(d)) if axes is None else tuple(axes)
    if not 0.0 <= rotation_bound < np.pi / 2 or offset_bound < 0.0:
        raise ValueError("need 0 <= rotation_bound < pi/2 and offset_bound >= 0")
    cr, sr = float(np.cos(rotation_bound)), float(np.sin(rotation_bound))
    names = "xyz"
    preds = []
    for idx in axes:
        for sign in (-1.0, 1.0):
            a = np.zeros(d)
            a[idx] = sign
            a_lo, a_hi = np.zeros(d), np.zeros(d)
            for k in axes:
                if k == idx:
                    a_lo[k], a_hi[k] = (cr, 1.0) if sign > 0 else (-1.0, -cr)
                else:
                    a_lo[k], a_hi[k] = -sr, sr
            tag = f"p{names[idx] if idx < 3 else idx}{'<=-[c]' if sign < 0 else '>=[c]'}"
            preds.append(IntervalPredicate(a, half_width, a_lo, a_hi, half_width - offset_bound,
                                           half_width + offset_bound, name=tag))
    f = preds[0]
    for q in preds[1:]:
        f = f | q
    f.simplify()
    f.name = "notS_u"
    return f


def rotated_outside_box(half_width: float, R: np.ndarray, d: int | None = None,
                        axes: tuple[int, ...] | None = None) -> STLFormula:
    """``x notin R S_b``: exact predicates of the box ``|y_k| < half_width`` rotated by ``R``.

    ``R`` is a rotation of the subspace of ``axes`` (``len(axes) x len(axes)``); face ``(k, sign)`` has
    the outward normal ``sign * R e_k`` (embedded in the signal space) and offset ``half_width``.
    """
    R = np.asarray(R, dtype=float)
    d = R.shape[0] if d is None else d
    axes = tuple(range(R.shape[0])) if axes is None else tuple(axes)
    names = "xyz"
    preds = []
    for col, idx in enumerate(axes):
        for sign in (-1.0, 1.0):
            a = np.zeros(d)
            a[list(axes)] = sign * R[:, col]
            tag = f"p{names[idx] if idx < 3 else idx}{'<=-c' if sign < 0 else '>=c'}_true"
            preds.append(LinearPredicate(a, half_width, name=tag))
    f = preds[0]
    for q in preds[1:]:
        f = f | q
    f.simplify()
    f.name = "notS_true"
    return f


def rotation_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues rotation about the unit ``axis`` (3-D), or the planar rotation (``axis`` ignored, 2-D)."""
    axis = np.asarray(axis, dtype=float)
    if axis.size == 1 or axis.size == 0:
        c, s = np.cos(angle), np.sin(angle)
        return np.array([[c, -s], [s, c]])
    k = axis / np.linalg.norm(axis)
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)


def outside_square(half_width: float, d: int = 2, ix: int = 0, iy: int = 1) -> STLFormula:
    """``x notin S``: disjunction of the four half-planes outside ``|px|, |py| < half_width``."""
    return outside_box(half_width, d, (ix, iy))


@dataclasses.dataclass(frozen=True)
class PatrolTask:
    """Geometry and timing of the NAHS patrol experiment (all lengths in metres)."""

    spec: SpecName = "theta"
    dt: float = 0.5
    half_width: float = 1.41  # S = {|px|, |py| < half_width}
    waypoint_radius: float = 1.52  # nominal patrol corners (0, +-r), (+-r, 0)
    waypoint_tol: float = 0.1  # switch to the next waypoint inside this distance
    altitude: float = 1.5  # [m] above the floor of the f_syn origin (NAHS: z = -1.5 m NED)
    workspace: float = 2.5  # |px|, |py| (and |pz| for the cube specs) <= workspace (arena replica, NAHS logs span +-2.13)
    theta_seconds: float = 4.0
    phi_eventually_seconds: float = 3.0
    phi_always_seconds: float = 1.0
    pred_offset: float = 0.05  # _u specs: |true half-width - half_width| <= pred_offset [m]
    pred_rotation: float = 0.05  # _u specs: rotation angle of the true region <= pred_rotation [rad]
    nominal_predicates: bool = False  # _u specs, ablation: plan/check with the exact nominal predicates

    def __post_init__(self):
        if self.spec not in SPECS:
            raise ValueError(f"unknown spec {self.spec!r}; expected one of {SPECS}")

    @property
    def dim(self) -> int:
        """Dimension of the STL signal (planar position, or 3-D position for the cube specs)."""
        return spec_dim(self.spec)

    @property
    def base_spec(self) -> str:
        """Temporal structure: ``theta`` (24) or ``phi`` (25)."""
        return spec_base(self.spec)

    @property
    def uncertain(self) -> bool:
        """``_u`` spec: the true keep-out region is uncertain (scored against a drawn true region)."""
        return spec_uncertain(self.spec)

    @property
    def interval_predicates(self) -> bool:
        """The planning formula uses interval predicates (``_u`` spec without the nominal ablation)."""
        return self.uncertain and not self.nominal_predicates

    @property
    def exact_spec(self) -> str:
        """The exact counterpart (``theta_plus_u -> theta_plus``)."""
        return self.spec.removesuffix("_u")

    def outside_formula(self) -> STLFormula:
        """``x notin S`` of the planning formula (interval predicates for the ``_u`` specs)."""
        if self.interval_predicates:
            return interval_outside_box(self.half_width, self.dim, offset_bound=self.pred_offset,
                                        rotation_bound=self.pred_rotation)
        return outside_box(self.half_width, self.dim)

    def _temporal(self, not_s: STLFormula) -> STLFormula:
        if self.base_spec == "theta":
            f = not_s.eventually(0, seconds_to_steps(self.theta_seconds, self.dt))
            f.name = self.spec
            return f
        held = not_s.always(0, round(self.phi_always_seconds / self.dt))
        f = not_s | held.eventually(0, seconds_to_steps(self.phi_eventually_seconds, self.dt))
        f.name = self.spec
        return f

    def formula(self) -> STLFormula:
        """The per-window formula psi (24) or (25) (or its cube version) in steps."""
        return self._temporal(self.outside_formula())

    def interval_formula(self) -> STLFormula:
        """psi with the interval predicates (``_u`` specs, also under the nominal ablation)."""
        if not self.uncertain:
            raise ValueError(f"{self.spec} has exact predicates")
        return self._temporal(interval_outside_box(self.half_width, self.dim, offset_bound=self.pred_offset,
                                                   rotation_bound=self.pred_rotation))

    def true_formula(self, region: dict) -> STLFormula:
        """psi with the exact predicates of a true region (:meth:`sample_true_region`)."""
        return self._temporal(rotated_outside_box(region["half_width"], region["R"], self.dim))

    def sample_true_region(self, rng: np.random.Generator) -> dict:
        """A true keep-out region inside the predicate uncertainty set (uniform half-width; 2-D: uniform
        angle in ``[-pred_rotation, pred_rotation]``; 3-D: uniform axis on the sphere, uniform angle in
        ``[0, pred_rotation]``). Exact specs: the nominal region."""
        if not self.uncertain:
            return {"half_width": self.half_width, "angle": 0.0,
                    "axis": [0.0, 0.0, 1.0] if self.dim == 3 else [1.0], "R": np.eye(self.dim)}
        b = float(rng.uniform(self.half_width - self.pred_offset, self.half_width + self.pred_offset))
        if self.dim == 2:
            angle = float(rng.uniform(-self.pred_rotation, self.pred_rotation))
            axis = np.array([1.0])
        else:
            g = rng.standard_normal(3)
            axis = g / np.linalg.norm(g)
            angle = float(rng.uniform(0.0, self.pred_rotation))
        return {"half_width": b, "angle": angle, "axis": axis.tolist(), "R": rotation_matrix(axis, angle)}

    @property
    def horizon(self) -> int:
        """Formula horizon ``h`` in steps."""
        return formula_horizon(self.formula())

    @property
    def terminal_block(self) -> int:
        """Number ``b`` of consecutive terminal states that must lie in one face set.

        theta needs one outside sample per window: a single terminal state in an invariant
        subset of ``S^c`` keeps every later window satisfied (b = 1). phi needs
        ``G_[0,hG]``: ``b = hG + 1`` consecutive outside samples in the same face set.
        """
        if self.base_spec == "theta":
            return 1
        return round(self.phi_always_seconds / self.dt) + 1

    def waypoints(self) -> np.ndarray:
        """Patrol waypoints ``(n_wp, dim)``: face centres of ``S`` moved outward by
        ``waypoint_radius - half_width``. 2-D: ``(0, r), (r, 0), (0, -r), (-r, 0)``; cube:
        ``+y, +x, +z, -y, -x, -z`` (consecutive faces adjacent)."""
        r = self.waypoint_radius
        if self.dim == 2:
            return np.array([[0.0, r], [r, 0.0], [0.0, -r], [-r, 0.0]])
        return np.array([[0.0, r, 0.0], [r, 0.0, 0.0], [0.0, 0.0, r],
                         [0.0, -r, 0.0], [-r, 0.0, 0.0], [0.0, 0.0, -r]])


def window_spec(psi: STLFormula, n_plan: int, first_window: int = 0) -> STLFormula:
    """Conjunction of ``psi`` at local window starts ``first_window .. n_plan - 1``.

    The signal has ``h + n_plan`` samples (``h = horizon(psi)``). ``first_window > 0``
    drops windows that would start before the episode start (fewer than ``h - 1``
    measured samples exist).
    """
    starts = list(range(first_window, n_plan))
    if not starts:
        raise ValueError("no window to enforce")
    f = STLTree([psi] * len(starts), "and", starts, name=f"G_[{first_window},{n_plan - 1}] {psi.name}")
    return f


def first_enforced_window(t: int, h: int) -> int:
    """Local start of the first window that begins at absolute time ``>= 0``."""
    return max(0, h - 1 - t)
