"""Learned segment proposer: features, differentiable certificate, network, BC training.

The policy maps an observation to ``N`` synthesis-model inputs (a trajectory segment through
the ``f_syn`` rollout). One feature definition is used at runtime (numpy, from a
:class:`~rl_stl.supervisor.receding.ProposalContext`) and in training (JAX).

Observation (STL context and the disturbance *bound*, never the realised
disturbance):

* state ``x`` scaled by the state box half-widths;
* measured output history ``y_{t-h+1}..y_{t-1}`` (scaled), a validity mask (samples before
  the episode start are not enforced), and each sample's outside-``S`` margin;
* one-hot of the waypoint the mission reference is heading for (4 waypoints for the square, 6 for
  the cube specs);
* the disturbance-bound level ``w / w_max`` (for the ``_u`` specs followed by the predicate
  uncertainty bounds ``(pred_offset, pred_rotation)``, scaled the same way; they are equal at every
  level, so the two features are constant 1: they document the bound and keep the interface ready
  for randomised predicate bounds);
* clock-indexed reference only (``LevelTables.ref_offsets``): the reference minus the state,
  ``(x_nom(k + j) - x_k)`` scaled, at the offsets ``j`` (the rerooted reference starts at
  ``x_k`` and is a function of the other features, the clock reference is not).

Action: ``a in R^{N x m}``; inputs ``u_i = c_i + r_i tanh(a_i)`` inside the tightened input box
``[u_lo + kz(i), u_hi - kz(i)]`` of a measured root, so input feasibility holds by
construction.

:func:`patrol_certificate` is a JAX implementation of the runtime check
(:func:`rl_stl.synthesis.check.check_plan`) for the patrol formulas: it returns
``min(rho_lower - delta, terminal margin, state-bound margin)``, non-negative iff the
proposal is a feasible point of the synthesis problem (up to tolerances). It is
differentiable almost everywhere and is used as a hinge in the BC loss and as the PPO
reward; the numpy check remains the authority at runtime.

Uncertain-predicate specs (``*_u``, :attr:`LevelTables.interval`): the face predicates are interval
predicates ``[a]^T y - [b] >= 0`` and every place that used the unit-normal margin ``|y_i| - c`` uses
the I-STL lower endpoint instead (:func:`signal_margin`: history margin feature, Boolean history/root
samples, and the planned margin with the tube box in :func:`patrol_certificate`).
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from rl_stl.stl.nahs import PatrolTask, seconds_to_steps, spec_base
from rl_stl.stl.robustness import iter_predicates

BIG = 1e3


# ----------------------------------------------------------------- level tables
@dataclasses.dataclass(frozen=True, eq=False)
class LevelTables:
    """Per-disturbance-level constants of one patrol benchmark (numpy, stacked over levels)."""

    w_levels: np.ndarray  # (Lw, q) observed uncertainty descriptor per level (w_bar or eps)
    z: np.ndarray  # (Lw, N+1, n) state tube hull, age 0 root
    kz: np.ndarray  # (Lw, N, m) input tube hull
    H: np.ndarray  # (Lw, F, R, n) terminal face rows (padded with zero rows, g = BIG)
    g: np.ndarray  # (Lw, F, R)
    A: np.ndarray
    B: np.ndarray
    C_out: np.ndarray
    x_lo: np.ndarray
    x_hi: np.ndarray
    u_lo: np.ndarray
    u_hi: np.ndarray
    delta: float
    half_width: float
    spec: str
    h: int
    hF: int
    hG: int
    b: int
    n_plan: int
    waypoints: np.ndarray
    w_max: np.ndarray
    ref_offsets: tuple[int, ...] = ()  # clock reference: x_nom(k + j) - x_k features at these j
    ref_input_scale: float = 1.0  # clock reference: its inputs saturate at this x bounds
    pred_a_lo: np.ndarray | None = None  # (F, p) interval face predicates (``_u`` specs), None: exact
    pred_a_hi: np.ndarray | None = None
    pred_b_hi: np.ndarray | None = None  # (F,)

    @property
    def interval(self) -> bool:
        """Interval face predicates (``_u`` specs planned with them)."""
        return self.pred_a_lo is not None

    @property
    def n(self) -> int:
        return self.A.shape[0]

    @property
    def m(self) -> int:
        return self.B.shape[1]

    @property
    def obs_size(self) -> int:
        p = self.C_out.shape[0]
        return (self.n + (self.h - 1) * (p + 2) + self.n_waypoints + self.w_levels.shape[1]
                + len(self.ref_offsets) * self.n)

    @property
    def p(self) -> int:
        """Dimension of the STL signal ``y = C_out x`` (2: square, 3: cube)."""
        return self.C_out.shape[0]

    @property
    def n_waypoints(self) -> int:
        return self.waypoints.shape[0]

    @property
    def base_spec(self) -> str:
        """``theta`` or ``phi``: the temporal structure (the cube and ``_u`` specs share it)."""
        return spec_base(self.spec)

    @property
    def reference(self) -> str:
        return "clock" if self.ref_offsets else "rerooted"

    @property
    def action_size(self) -> int:
        return self.n_plan * self.m


def make_level_tables(problems: list, task: PatrolTask, waypoints: np.ndarray,
                      w_levels: np.ndarray, reference: str = "rerooted",
                      ref_input_scale: float = 1.0) -> LevelTables:
    """Stack the problems built for each disturbance level (same ``N``, model, bounds).

    ``reference="clock"`` adds the reference-offset features at ``j = 0, N//2, N``;
    ``ref_input_scale`` is the clock reference's input saturation (ignored when rerooted)."""
    p0 = problems[0]
    N = p0.n_plan
    zs, kzs, Hs, gs = [], [], [], []
    R = max(max(len(g) for _, g in p.terminal.face_rows()) for p in problems)
    F = len(p0.terminal.faces)
    for p in problems:
        zs.append(np.stack([p.tube.state(i) for i in range(N + 1)]))
        kzs.append(np.stack([p.tube.input(i) for i in range(N)]))
        H = np.zeros((F, R, p.model.n))
        g = np.full((F, R), BIG)
        for f, (Hf, gf) in enumerate(p.terminal.face_rows()):
            H[f, :len(gf)] = Hf
            g[f, :len(gf)] = gf
        Hs.append(H)
        gs.append(g)
    theta = task.base_spec == "theta"
    hF = seconds_to_steps(task.theta_seconds if theta else task.phi_eventually_seconds, task.dt)
    hG = 0 if theta else int(round(task.phi_always_seconds / task.dt))
    if p0.C_out.shape[0] != task.dim or np.asarray(waypoints).shape[1] != task.dim:
        raise ValueError("C_out / waypoints do not match the task dimension")
    w_levels = np.asarray(w_levels, float)
    ipred = {}
    if task.interval_predicates:
        preds = [q for q, _ in iter_predicates(task.outside_formula())]
        ipred = {"pred_a_lo": np.array([q.a_lo for q in preds]), "pred_a_hi": np.array([q.a_hi for q in preds]),
                 "pred_b_hi": np.array([q.b_hi for q in preds])}
    return LevelTables(
        w_levels=w_levels, z=np.stack(zs), kz=np.stack(kzs), H=np.stack(Hs), g=np.stack(gs),
        A=p0.model.A, B=p0.model.B, C_out=p0.C_out, x_lo=p0.x_lo, x_hi=p0.x_hi, u_lo=p0.u_lo,
        u_hi=p0.u_hi, delta=p0.delta, half_width=task.half_width, spec=task.spec, h=p0.h, hF=hF,
        hG=hG, b=p0.terminal_block, n_plan=N, waypoints=np.asarray(waypoints, float),
        w_max=np.maximum(w_levels.max(axis=0), 1e-9),
        ref_offsets={"rerooted": (), "clock": (0, N // 2, N)}[reference],
        ref_input_scale=float(ref_input_scale) if reference == "clock" else 1.0, **ipred)


# --------------------------------------------------------------------- features
def outside_margin(y, half_width: float, xp=np):
    """``max_i |y_i| - c``: signed outside-``S`` margin (>= 0 outside) over every signal axis
    (square for 2-D ``y``, cube for 3-D ``y``)."""
    return xp.max(xp.abs(y), -1) - half_width


def interval_margin(T: LevelTables, y, r=0.0, xp=np):
    """``max_f lower([a_f]^T [y - r, y + r] - [b_f])``: I-STL lower endpoint of ``x notin S`` (interval
    face predicates; ``y``, ``r``: ``(..., p)``)."""
    a_lo, a_hi = xp.asarray(T.pred_a_lo), xp.asarray(T.pred_a_hi)  # (F, p)
    y = y[..., None, :]
    r = xp.asarray(r)[..., None, :] if xp.ndim(xp.asarray(r)) else r
    t = xp.minimum(a_lo * y - xp.abs(a_lo) * r, a_hi * y - xp.abs(a_hi) * r)
    return xp.max(xp.sum(t, -1) - xp.asarray(T.pred_b_hi), -1)


def signal_margin(T: LevelTables, y, xp=np):
    """Outside-``S`` margin of measured samples: :func:`outside_margin` for exact predicates, the
    interval lower endpoint for the ``_u`` specs (``>= 0`` iff the sample satisfies ``x notin S``)."""
    if T.interval:
        return interval_margin(T, y, 0.0, xp)
    return outside_margin(y, T.half_width, xp)


def features(T: LevelTables, x, hist_y, hist_mask, waypoint_index, level, xp=np, x_ref=None):
    """Observation vector (numpy or ``jax.numpy`` via ``xp``); ``x_ref (N+1, n)``: the mission
    reference of a plan rooted here (needed iff ``T.ref_offsets``)."""
    scale_x = (T.x_hi - T.x_lo) / 2.0
    ws = np.abs(T.C_out) @ scale_x  # half-range of each signal axis
    marg = xp.clip(signal_margin(T, hist_y, xp), -1.0, 1.0) * hist_mask
    wp = (xp.arange(T.n_waypoints) == waypoint_index).astype(x.dtype if xp is np else jnp.float32)
    w = T.w_levels[level] / T.w_max if xp is np else jnp.asarray(T.w_levels)[level] / T.w_max
    parts = [x / scale_x, (hist_y / ws).reshape(-1) * xp.repeat(hist_mask, T.p), hist_mask, marg, wp, w]
    if T.ref_offsets:
        if x_ref is None:
            raise ValueError("clock-reference features need x_ref")
        parts.append(((x_ref[np.asarray(T.ref_offsets)] - x) / scale_x).reshape(-1))
    return xp.concatenate(parts)


def squash(T: LevelTables, a, level, xp=np):
    """Raw network output ``(N*m,)`` -> inputs ``(N, m)`` inside the tightened box."""
    kz = T.kz[level] if xp is np else jnp.asarray(T.kz)[level]
    lo, hi = T.u_lo + kz, T.u_hi - kz
    a = a.reshape(T.n_plan, T.m)
    return (lo + hi) / 2 + (hi - lo) / 2 * xp.tanh(a)


def unsquash(T: LevelTables, U: np.ndarray, level: int, eps: float = 1e-6) -> np.ndarray:
    lo, hi = T.u_lo + T.kz[level], T.u_hi - T.kz[level]
    s = np.clip((U - (lo + hi) / 2) / np.maximum((hi - lo) / 2, 1e-12), -1 + eps, 1 - eps)
    return np.arctanh(s).reshape(-1)


# ------------------------------------------------------------------ certificate
def rollout_jax(A, B, x0, U):
    def f(x, u):
        xn = A @ x + B @ u
        return xn, xn
    _, X = jax.lax.scan(f, x0, U)
    return jnp.concatenate([x0[None], X], 0)


def patrol_certificate(T: LevelTables, x0, U, hist_y, hist_mask, first_window, level):
    """``(cert, parts)``; ``cert >= 0`` iff the plan passes the runtime check (JAX)."""
    A, B, C = jnp.asarray(T.A), jnp.asarray(T.B), jnp.asarray(T.C_out)
    z = jnp.asarray(T.z)[level]
    X = rollout_jax(A, B, x0, U)  # (N+1, n)
    Y = X @ C.T
    zy = z @ jnp.abs(C).T  # (N+1, p)
    c = T.half_width
    if T.interval:  # interval face predicates: I-STL lower endpoint over [a], [b] and the tube box
        planned = interval_margin(T, Y[1:], zy[1:], jnp)
    else:
        # every predicate is +-y_i - c >= 0 with a unit normal: tightened margin |y_i| - c - zy_i
        planned = jnp.max(jnp.abs(Y[1:]) - c - zy[1:], -1)
    planned = planned - T.delta
    hist = jnp.where(signal_margin(T, hist_y, jnp) >= 0, BIG, -BIG)
    root = jnp.where(signal_margin(T, Y[0], jnp) >= 0, BIG, -BIG)
    M = jnp.concatenate([hist, root[None], planned])  # local j = 0..h+N-1
    N, hF, hG = T.n_plan, T.hF, T.hG
    starts = jnp.arange(N)
    if T.base_spec == "theta":
        idx = starts[:, None] + jnp.arange(hF + 1)[None, :]
        win = jnp.max(M[idx], axis=1)
    else:
        L = M.shape[0]
        jj = jnp.arange(L - hG)
        G = jnp.min(M[jj[:, None] + jnp.arange(hG + 1)[None, :]], axis=1)
        idx = starts[:, None] + jnp.arange(hF + 1)[None, :]
        win = jnp.maximum(M[starts], jnp.max(G[idx], axis=1))
    win = jnp.where(starts >= first_window, win, BIG)
    rho = jnp.min(win)
    # terminal block: last b states in one face
    Hf, gf = jnp.asarray(T.H)[level], jnp.asarray(T.g)[level]  # (F, R, n), (F, R)
    last = X[N - T.b + 1:]  # (b, n)
    face_m = jnp.min(gf[:, None, :] - jnp.einsum("frn,bn->fbr", Hf, last), axis=(1, 2))
    term = jnp.max(face_m)
    lo = jnp.asarray(T.x_lo) + z[1:]
    hi = jnp.asarray(T.x_hi) - z[1:]
    bnd = jnp.min(jnp.minimum(X[1:] - lo, hi - X[1:]))
    cert = jnp.minimum(jnp.minimum(rho, term), bnd)
    return cert, {"rho_margin": rho, "terminal": term, "bounds": bnd}


# ---------------------------------------------------------------------- network
def init_mlp(key, sizes: list[int]) -> list[dict]:
    params = []
    for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
        key, k = jax.random.split(key)
        scale = jnp.sqrt(2.0 / a) if i < len(sizes) - 2 else 1e-2 / jnp.sqrt(a)
        params.append({"W": scale * jax.random.normal(k, (a, b)), "b": jnp.zeros(b)})
    return params


def mlp(params: list[dict], x):
    for layer in params[:-1]:
        x = jax.nn.swish(x @ layer["W"] + layer["b"])
    return x @ params[-1]["W"] + params[-1]["b"]


@dataclasses.dataclass
class SegmentPolicy:
    """Normalised MLP ``obs -> raw action``; ``obs_mean/std`` from the training data."""

    params: list
    obs_mean: np.ndarray
    obs_std: np.ndarray
    hidden: tuple[int, ...]
    reference: str = "rerooted"  # mission reference the policy was trained with
    ref_input_scale: float = 1.0  # and the clock reference's input saturation

    def raw(self, obs):
        return mlp(self.params, (obs - self.obs_mean) / self.obs_std)

    def save(self, path: str | Path) -> None:
        """Plain ``.npz`` of arrays (no pickle)."""
        arrays = {"obs_mean": self.obs_mean, "obs_std": self.obs_std,
                  "hidden": np.asarray(self.hidden, dtype=np.int64), "reference": np.asarray(self.reference),
                  "ref_input_scale": np.asarray(self.ref_input_scale)}
        for i, layer in enumerate(self.params):
            arrays[f"W{i}"] = np.asarray(layer["W"])
            arrays[f"b{i}"] = np.asarray(layer["b"])
        np.savez(path, **arrays)

    @staticmethod
    def load(path: str | Path) -> SegmentPolicy:
        d = np.load(path, allow_pickle=False)
        n_layers = sum(1 for k in d.files if k.startswith("W"))
        params = [{"W": jnp.asarray(d[f"W{i}"]), "b": jnp.asarray(d[f"b{i}"])} for i in range(n_layers)]
        ref = str(d["reference"]) if "reference" in d.files else "rerooted"
        scale = float(d["ref_input_scale"]) if "ref_input_scale" in d.files else 1.0
        return SegmentPolicy(params, d["obs_mean"], d["obs_std"], tuple(int(v) for v in d["hidden"]), ref, scale)


class LearnedProposer:
    """Runtime proposer (numpy in, numpy out; jitted network on CPU)."""

    def __init__(self, policy: SegmentPolicy, T: LevelTables, level: int, nominal):
        if policy.reference != T.reference:
            raise ValueError(f"policy trained with the {policy.reference} reference, tables use {T.reference}")
        if policy.ref_input_scale != T.ref_input_scale:
            raise ValueError(f"policy trained with reference input scale {policy.ref_input_scale}, "
                             f"tables use {T.ref_input_scale}")
        self.policy, self.T, self.level, self.nominal = policy, T, level, nominal
        self._fn = jax.jit(lambda o: squash(T, policy.raw(o), level, jnp))

    def observation(self, ctx) -> np.ndarray:
        h = self.T.h
        mask = (np.arange(h - 1) >= ctx.first_window).astype(float)
        idx = self.nominal.index if ctx.waypoint_index is None else ctx.waypoint_index
        return features(self.T, ctx.x, ctx.history.y, mask, idx, self.level, x_ref=ctx.x_nom)

    def __call__(self, ctx) -> np.ndarray:
        obs = self.observation(ctx).astype(np.float32)
        return np.asarray(self._fn(obs), dtype=float)
