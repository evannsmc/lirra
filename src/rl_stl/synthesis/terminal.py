"""Verified terminal sets (persistently feasible sets) for the NAHS patrol formulas.

Construction (NAHS Prop. 6 with Prop. 8's "commit to one disjunct"): ``S^c`` is the union of
four half-planes (faces) ``px >= c``, ``px <= -c``, ``py >= c``, ``py <= -c``. For each face
``f`` a set ``Q_f`` is built that is **control invariant for the nominal dynamics** under
the tightened state/input constraints and lies inside the face. A plan whose last ``b``
nominal states lie in one ``Q_f`` can be extended forever by the terminal controller
without leaving ``Q_f``; for theta (``b = 1``) and phi (``b = hG + 1``) every later window
then contains the outside samples it needs (see ``rl_stl.stl.nahs.PatrolTask.terminal_block``).

Decomposition. All synthesis models used here are, at the linearisation point, a product of
decoupled 2-state axis blocks driven by one input each (double integrator ``(p, v)`` per
axis; the blimp's ``(v, p)`` lateral, vertical and yaw blocks, whose inputs also excite
roll/pitch states that feed back into nothing). With a box input set the product of
per-axis control invariant polygons is control invariant for the full model. Every block's
decoupling is checked numerically (:func:`axis_blocks`).

Per axis the maximal control invariant subset of a polygon is computed by the standard
iteration ``Omega <- Omega cap Pre(Omega)`` and **verified** afterwards: every vertex of the
result has an admissible input mapping it back into the set (sufficient for convex sets).
"""

from __future__ import annotations

import dataclasses

import numpy as np
from scipy.optimize import linprog
from scipy.spatial import ConvexHull, HalfspaceIntersection

from rl_stl.dynamics.linear import DiscreteLinearModel

_TOL = 1e-9


# ----------------------------------------------------------------------------- polygons
@dataclasses.dataclass(frozen=True, eq=False)
class Polygon:
    """Bounded convex polygon ``{x in R^2 : H x <= g}`` with its vertices (CCW)."""

    H: np.ndarray  # (k, 2)
    g: np.ndarray  # (k,)
    vertices: np.ndarray  # (v, 2)

    def contains(self, x: np.ndarray, tol: float = 1e-7) -> bool:
        return bool(np.all(self.H @ np.asarray(x, float) <= self.g + tol))

    @property
    def area(self) -> float:
        return float(ConvexHull(self.vertices).volume) if len(self.vertices) >= 3 else 0.0

    @staticmethod
    def from_box(lo: np.ndarray, hi: np.ndarray) -> Polygon:
        H = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
        g = np.array([hi[0], -lo[0], hi[1], -lo[1]], dtype=float)
        return polygon_from_halfspaces(H, g)


def _normalise(H: np.ndarray, g: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    nrm = np.linalg.norm(H, axis=1)
    keep = nrm > 1e-12
    return H[keep] / nrm[keep, None], g[keep] / nrm[keep]


def polygon_from_halfspaces(H: np.ndarray, g: np.ndarray) -> Polygon | None:
    """Polygon of ``H x <= g`` (``None`` if empty or degenerate)."""
    H, g = _normalise(np.asarray(H, float), np.asarray(g, float))
    # Chebyshev centre: max r s.t. H x + r ||h|| <= g (rows normalised).
    res = linprog(c=[0.0, 0.0, -1.0], A_ub=np.hstack([H, np.ones((len(H), 1))]), b_ub=g,
                  bounds=[(None, None), (None, None), (0.0, None)], method="highs")
    if res.status != 0 or res.x[2] <= 1e-9:
        return None
    centre = res.x[:2]
    hs = HalfspaceIntersection(np.hstack([H, -g[:, None]]), centre)
    pts = hs.intersections
    hull = ConvexHull(pts)
    verts = pts[hull.vertices]
    eq = hull.equations  # n.x + off <= 0
    return Polygon(H=eq[:, :2], g=-eq[:, 2], vertices=verts)


def polygon_from_points(points: np.ndarray) -> Polygon:
    hull = ConvexHull(points)
    eq = hull.equations
    return Polygon(H=eq[:, :2], g=-eq[:, 2], vertices=points[hull.vertices])


def intersect(p: Polygon, q: Polygon) -> Polygon | None:
    return polygon_from_halfspaces(np.vstack([p.H, q.H]), np.concatenate([p.g, q.g]))


def input_interval(A: np.ndarray, B: np.ndarray, x: np.ndarray, P: Polygon,
                   u_lo: float, u_hi: float) -> tuple[float, float]:
    """``{u in [u_lo, u_hi] : A x + B u in P}`` as an interval (``lo > hi`` if empty)."""
    hb = P.H @ B  # coefficient of u in each row
    r = P.g - P.H @ (A @ x)
    lo, hi = u_lo, u_hi
    for c, ri in zip(hb, r):
        if abs(c) < 1e-14:
            if ri < -1e-9:
                return 1.0, 0.0
        elif c > 0:
            hi = min(hi, ri / c)
        else:
            lo = max(lo, ri / c)
    return lo, hi


def pre_set(A: np.ndarray, B: np.ndarray, P: Polygon, u_lo: float, u_hi: float) -> Polygon:
    """``{x : exists u in [u_lo, u_hi], A x + B u in P}`` for invertible ``A``."""
    Ainv = np.linalg.inv(A)
    pts = np.vstack([(Ainv @ (v - B * u)) for v in P.vertices for u in (u_lo, u_hi)])
    return polygon_from_points(pts)


def max_control_invariant(A: np.ndarray, B: np.ndarray, region: Polygon, u_lo: float,
                          u_hi: float, max_iter: int = 500) -> Polygon | None:
    """Maximal control invariant subset of ``region`` (``None`` if empty), verified."""
    omega = region
    for _ in range(max_iter):
        pre = pre_set(A, B, omega, u_lo, u_hi)
        if all(pre.contains(v, tol=1e-9) for v in omega.vertices):
            break
        nxt = intersect(omega, pre)
        if nxt is None:
            return None
        omega = nxt
    else:
        raise RuntimeError("invariant-set iteration did not converge")
    verify_invariant(A, B, omega, u_lo, u_hi)
    return omega


def verify_invariant(A: np.ndarray, B: np.ndarray, P: Polygon, u_lo: float, u_hi: float,
                     tol: float = 1e-7) -> None:
    """Raise unless every vertex of ``P`` can be mapped into ``P`` (convexity => whole set)."""
    for v in P.vertices:
        lo, hi = input_interval(A, B, v, P, u_lo, u_hi)
        if lo > hi + tol:
            raise AssertionError(f"vertex {v} has no admissible input (interval [{lo}, {hi}])")


# ------------------------------------------------------------------------- axis blocks
@dataclasses.dataclass(frozen=True, eq=False)
class AxisBlock:
    """Decoupled 2-state block ``z+ = A z + B u_j`` of the synthesis model."""

    name: str
    states: tuple[int, int]
    input: int
    A: np.ndarray  # (2, 2)
    B: np.ndarray  # (2,)


def axis_blocks(model: DiscreteLinearModel, spec: dict[str, tuple[tuple[int, int], int]],
                tol: float = 1e-10) -> dict[str, AxisBlock]:
    """Extract and check decoupled blocks ``{name: ((i0, i1), input)}``."""
    out = {}
    n, m = model.n, model.m
    for name, (states, j) in spec.items():
        idx = list(states)
        others = [i for i in range(n) if i not in idx]
        if np.max(np.abs(model.A[np.ix_(idx, others)]), initial=0.0) > tol:
            raise ValueError(f"block {name}: depends on other states")
        other_u = [k for k in range(m) if k != j]
        if np.max(np.abs(model.B[np.ix_(idx, other_u)]), initial=0.0) > tol:
            raise ValueError(f"block {name}: depends on other inputs")
        out[name] = AxisBlock(name, (int(states[0]), int(states[1])), int(j),
                              model.A[np.ix_(idx, idx)].copy(), model.B[idx, j].copy())
    return out


# ------------------------------------------------------------------------ terminal set
@dataclasses.dataclass(frozen=True, eq=False)
class FaceSet:
    """``Q_f``: product of per-block invariant polygons (blocks not listed are free)."""

    name: str
    polygons: dict[str, Polygon]

    def rows(self, blocks: dict[str, AxisBlock], n: int) -> tuple[np.ndarray, np.ndarray]:
        """H-representation ``H x <= g`` in the full state."""
        Hs, gs = [], []
        for bname, poly in self.polygons.items():
            blk = blocks[bname]
            H = np.zeros((len(poly.g), n))
            H[:, blk.states[0]] = poly.H[:, 0]
            H[:, blk.states[1]] = poly.H[:, 1]
            Hs.append(H)
            gs.append(poly.g)
        return np.vstack(Hs), np.concatenate(gs)

    def contains(self, blocks: dict[str, AxisBlock], x: np.ndarray, tol: float = 1e-7) -> bool:
        return all(p.contains(x[list(blocks[b].states)], tol) for b, p in self.polygons.items())


@dataclasses.dataclass(frozen=True, eq=False)
class TerminalSet:
    """Union of face sets; a plan commits to one face for its last ``b`` states."""

    faces: tuple[FaceSet, ...]
    blocks: dict[str, AxisBlock]
    u_lo: np.ndarray  # tightened input box used for invariance
    u_hi: np.ndarray
    n: int
    description: str = ""

    def face_rows(self) -> list[tuple[np.ndarray, np.ndarray]]:
        return [f.rows(self.blocks, self.n) for f in self.faces]

    def containing_faces(self, x: np.ndarray, tol: float = 1e-7) -> list[int]:
        return [i for i, f in enumerate(self.faces) if f.contains(self.blocks, x, tol)]

    def controller(self, x: np.ndarray, face: int, u_pref: np.ndarray | None = None) -> np.ndarray:
        """Nominal terminal input keeping ``x`` in face ``face`` (closest to ``u_pref``).

        Inputs not driving any listed block are set to ``u_pref`` (default 0) clipped.
        """
        m = len(self.u_lo)
        u = np.zeros(m) if u_pref is None else np.asarray(u_pref, float).copy()
        u = np.clip(u, self.u_lo, self.u_hi)
        for bname, poly in self.faces[face].polygons.items():
            blk = self.blocks[bname]
            z = x[list(blk.states)]
            lo, hi = input_interval(blk.A, blk.B, z, poly, self.u_lo[blk.input], self.u_hi[blk.input])
            if lo > hi + 1e-7:
                raise ValueError(f"state {z} of block {bname} is outside its invariant set")
            u[blk.input] = float(np.clip(u[blk.input], lo, max(lo, hi)))
        return u


def patrol_terminal_set(model: DiscreteLinearModel, blocks: dict[str, AxisBlock],
                        plane: tuple[str, str], c_out: float, x_lo: np.ndarray,
                        x_hi: np.ndarray, u_lo: np.ndarray, u_hi: np.ndarray,
                        position_index: dict[str, int]) -> TerminalSet:
    """Four face sets for ``S^c = {|p_a| >= c_out for some planar axis a}``.

    ``x_lo/x_hi/u_lo/u_hi`` are the (tightened) constraint boxes; ``c_out`` already includes
    the robustness margin and the output tube. For face ``(+, ax)`` the planar axis ``ax`` is
    confined to ``p_ax in [c_out, x_hi]``; every other block keeps its full box.
    """
    def region(bname: str, p_lo: float | None = None, p_hi: float | None = None) -> Polygon:
        blk = blocks[bname]
        lo = np.array([x_lo[blk.states[0]], x_lo[blk.states[1]]], dtype=float)
        hi = np.array([x_hi[blk.states[0]], x_hi[blk.states[1]]], dtype=float)
        k = blk.states.index(position_index[bname]) if bname in position_index else None
        if k is not None:
            if p_lo is not None:
                lo[k] = max(lo[k], p_lo)
            if p_hi is not None:
                hi[k] = min(hi[k], p_hi)
        if np.any(~np.isfinite(lo)) or np.any(~np.isfinite(hi)) or np.any(lo >= hi):
            raise ValueError(f"block {bname}: unbounded or empty region {lo}..{hi}")
        return Polygon.from_box(lo, hi)

    def inv(bname: str, reg: Polygon) -> Polygon:
        blk = blocks[bname]
        P = max_control_invariant(blk.A, blk.B, reg, float(u_lo[blk.input]), float(u_hi[blk.input]))
        if P is None:
            raise ValueError(f"empty invariant set for block {bname}")
        return P

    free = {b: inv(b, region(b)) for b in blocks}
    faces = []
    for ax in plane:
        for sign in (+1, -1):
            reg = region(ax, p_lo=c_out) if sign > 0 else region(ax, p_hi=-c_out)
            polys = dict(free)
            polys[ax] = inv(ax, reg)
            faces.append(FaceSet(f"{'+' if sign > 0 else '-'}{ax}", polys))
    return TerminalSet(tuple(faces), blocks, np.asarray(u_lo, float), np.asarray(u_hi, float),
                       model.n, description=f"patrol faces, c_out={c_out:.4f}")
