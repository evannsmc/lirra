"""Render one checked-loop patrol episode as an MP4 and optionally a GIF (any system, any segment policy).

Runs the shield (checked loop with delay-compensated synthesis) with a learned proposer from
one evaluation start state, records the vehicle, the knot point of the mission reference
x_nom(t) and the plan in force at every planning step, and draws them over the keep-out square
S (green) and the waypoints (red). The plan is colored by its source: accepted proposal, solver
or cached plan. Views (``--view``, one or more):

* ``topdown``: a 2-D plot (waypoints as red stars), one frame per planning step;
* ``mujoco``: the MuJoCo vehicle rendered offscreen from the simulator state recorded at every
  planning step, with S as a translucent green slab at flight altitude, the waypoints as red
  spheres, the knot point as a purple sphere, the plan and the trail as segments and the counters
  as text. The double integrator is simulated in NumPy; its view draws a point mass in a MuJoCo
  scene at ``DI_ALTITUDE``. Needs ``MUJOCO_GL=egl`` on a GPU node;
* ``composite``: one 1920x1080 video with the top-down view on the left, the MuJoCo view top
  right, the two robustness histories below it on a shared time axis and, above all panels, a
  status bar: "RL in control" (green) or "RL rejected" (yellow), and the conditions of the check
  that the latest rejected proposal failed (``REASONS``). After a rejection the vehicle flies the
  last verified plan and then the solver's plan until a proposal passes again; that whole stretch
  shows as rejected;
* ``both``: ``topdown`` and ``mujoco``.

Cube specs (``theta_plus``, ``phi_plus``): the ``topdown`` panel becomes a 3-D perspective view of
the keep-out cube S+ and the six waypoints, and the MuJoCo view draws S+ as a translucent cube
centred at the cube altitude (``altitude_plus``; ``DI_CUBE_ALTITUDE`` for the double integrator).

Uncertain-predicate specs (``theta_u``, ``phi_u``, ``theta_plus_u``, ``phi_plus_u``): the true S has an
unknown half-width within ``pred_offset`` of the nominal one and an unknown rotation of at most
``pred_rotation`` about its centre. The views draw the band that S can occupy (between the smallest and the
largest region over that set: the intersection and the union of all such squares in 2-D, the extreme cubes
as wireframes in 3-D) and the true S of the episode, drawn as ``evaluate.py`` draws it for the same start
index (``TRUE_REGION_SEED + seed``). The robustness panel shows the interval lower robustness rho^I (worst
case over the uncertainty set, the quantity the loop plans and checks with) and the robustness against
the true S, and the summary adds ``satisfied_true``, ``min_rho_true`` and the true parameters.

The realized robustness is the nominal robustness rho(y, t - h) of the formula on the realized
output window [t - h, t] (``realised_robustness``), plotted at the window's end t, when its last
sample has been measured; the plot and the video therefore never use positions not yet flown, and
the first h steps, before a whole window exists, have no value. The RL
proposal robustness rho^I is the interval lower robustness of the measured history followed by the
proposed segment that the check compares with delta (``check_plan``); it exists at the steps the
policy is queried, not while a solve is pending. A proposal can also be rejected with
rho^I >= delta, for a bound or the terminal set. Yellow vertical lines mark the steps at which an
STL solve was requested.

Writes ``<system>_<spec>[_mujoco|_composite].mp4``, poster frames (the last frame; PDF for the
top-down view), ``<system>_<spec>_rho.pdf`` and ``<system>_<spec>_rho_rl.pdf``, with ``--gif`` a GIF of each video, and the episode
summary to ``runs/patrol_videos/<system>/<run>/``.

    MUJOCO_GL=egl uv run --no-sync --with imageio-ffmpeg python experiments/patrol/render_episode.py \
        --system blimp --spec theta --level 2 --view topdown mujoco composite \
        --policy runs/patrol_ppo/blimp/<run>/policy.npz

``imageio-ffmpeg`` supplies an ffmpeg with libx264 (some system ffmpeg builds have no
browser-playable encoder); without it the ffmpeg on PATH is used (topdown view only).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from common import feedback, latency_default, load_system
from matplotlib.animation import FFMpegWriter
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgba
from matplotlib.patches import Polygon, Rectangle

from rl_stl.evaluation.patrol import uncertain_scores
from rl_stl.policies.segment import LearnedProposer, SegmentPolicy
from rl_stl.stl.nahs import rotation_matrix
from rl_stl.stl.robustness import robustness
from rl_stl.supervisor.receding import RecedingConfig, RecedingSupervisor, realised_robustness
from rl_stl.synthesis.istl import ISTLSettings
from rl_stl.utils.provenance import create_run_dir, write_summary

try:  # an ffmpeg build with libx264, if installed
    import imageio_ffmpeg

    FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
    matplotlib.rcParams["animation.ffmpeg_path"] = FFMPEG
except ImportError:
    imageio_ffmpeg, FFMPEG = None, "ffmpeg"

SECONDS = {"di": 60.0, "x2": 60.0, "blimp": 120.0}
TITLE = {"di": "Double integrator", "x2": "Skydio X2", "blimp": "Blimp"}
SRC_COLOR = {"proposal": "#1f6fd1", "solver": "#f2b400", "cache": "#7a7a7a"}
SOLVE = SRC_COLOR["solver"]
SRC_LABEL = {"proposal": "accepted proposal", "solver": "solver plan", "cache": "cached plan"}
KEEPOUT, WAYPOINT, KNOT, VEHICLE, TRAIL = "#1f7a3a", "#d62728", "#8a3fbf", "#0b3d91", "#333333"
RHO = "#0b3d91"
SPEC_TEX = {"theta": r"\vartheta", "phi": r"\varphi", "theta_plus": r"\vartheta^{+}", "phi_plus": r"\varphi^{+}"}
SPEC_UNI = {"theta": "θ", "phi": "φ", "theta_plus": "θ⁺", "phi_plus": "φ⁺"}
# uncertain-predicate specs: same symbol with a subscript u
SPEC_TEX.update({k + "_u": v + r"_{u}" for k, v in list(SPEC_TEX.items())})
SPEC_UNI.update({k + "_u": v + "ᵤ" for k, v in list(SPEC_UNI.items())})
RL_OK = "#2a8a3a"
TRUE_S = "#0e4d22"  # _u specs: outline of the episode's true S
TRUE_REGION_SEED = 1_000_003  # as evaluate.py: true region of start index s from default_rng(TRUE_REGION_SEED + s)
REASONS = (  # conditions of check_plan: markers in its reason string, box name, description, color
    (("dynamics residual", "shape", "nonfinite"), "Dynamics", "plan breaks the\nsynthesis model", "#4a2c74"),
    (("bound violation",), "Bounds", "state or input limit\n(tube-tightened)", "#8b1e2d"),
    (("rho_lower",), "Robustness", r"$\rho^{I} < \delta$", "#1b3a6b"),
    (("terminal block",), "Terminal set", "plan does not\nend in $Q$", "#145a52"),
)
STATES = (("RL in control", "its plan passed the STL check", RL_OK),
          ("RL rejected", "the STL solver re-plans; a\nverified plan flies meanwhile", SOLVE))
DI_ALTITUDE = 1.0  # height of the double integrator's point mass in the 3-D view [m]
DI_CUBE_ALTITUDE = 2.75  # cube specs: centre of S+ in the double integrator's 3-D view (as the vehicles)
DI_MJCF = """<mujoco>
  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1=".86 .87 .89" rgb2=".72 .74 .77" width="512" height="512"/>
    <material name="grid" texture="grid" texrepeat="40 40" reflectance="0"/>
  </asset>
  <worldbody>
    <light pos="0 0 6" dir="0 0 -1" directional="true"/>
    <geom name="floor" type="plane" size="10 10 0.05" material="grid"/>
    <body name="point_mass" pos="0 0 1"><freejoint/><geom type="sphere" size="0.12" rgba=".04 .24 .57 1"/></body>
  </worldbody>
</mujoco>"""


def mujoco_handles(sim):
    """``(model, data)`` of a MuJoCo patrol simulator, or None (double integrator)."""
    if hasattr(sim, "model") and hasattr(sim, "data"):
        return sim.model, sim.data
    inner = getattr(sim, "sim", None)
    if inner is not None and hasattr(inner, "model"):
        return inner.model, inner.data
    return None


def run(args):
    """One shield episode; per step the realized state, the plan in force, its source, x_nom and,
    for the MuJoCo vehicles, the simulator state."""
    policy = SegmentPolicy.load(args.policy)
    system, setups, T = load_system(args.system, args.spec, policy.reference)
    st = system.build(args.level)
    p = st.problem
    rng = np.random.default_rng(12345)  # the evaluation starts of evaluate.py
    x0 = [system.eval_start(rng, setups[0]) for _ in range(args.seed + 1)][args.seed]
    sup = RecedingSupervisor(p, st.K, st.nominal, LearnedProposer(policy, T, args.level, st.nominal),
                             RecedingConfig(variant="shield", latency_steps=latency_default(system),
                                            reference=policy.reference,
                                            reference_input_scale=T.ref_input_scale,
                                            feedback=feedback(system)), ISTLSettings())
    n = round(SECONDS[args.system] / system.dt)
    sim = system.make_sim(st, args.seed, n)
    mj = mujoco_handles(sim)

    def snap():
        return None if mj is None else (mj[1].qpos.copy(), mj[1].qvel.copy())

    x = sim.reset(np.asarray(x0, float))
    sup.initial_plan(x)
    X, plans, sources, queries, snaps = [x.copy()], [], [], [], [snap()]
    for k in range(n):
        u, rec = sup.step(k, x)
        plans.append(np.array([p.C_out @ sup.plan.state(j, p.terminal, p.model)
                               for j in range(k, k + p.n_plan + 1)]))
        sources.append(rec.get("source", "cache"))
        if "proposal_accepted" in rec:
            ok = rec["proposal_accepted"]
            failed = ([np.nan] * len(REASONS) if ok
                      else [any(m in rec["check"] for m in marks) for marks, *_ in REASONS])
            queries.append((rec["rho_lower"], ok, *failed))
        else:
            queries.append((np.nan,) * (2 + len(REASONS)))
        x = sim.step(sup.plan.state(k, p.terminal, p.model), sup.plan.input(k, p.terminal, p.model), u)
        X.append(x.copy())
        snaps.append(snap())
    if sup.reference is not None:
        knots = np.array([p.C_out @ sup.reference.window(k, 1)[0][0] for k in range(n + 1)])
    else:  # rerooted reference: no clock-indexed knot points
        knots = None
    P = np.array(X) @ p.C_out.T
    if mj is None:  # double integrator
        mj, snaps = point_mass_scene(P, altitude(system, st))
    # (n, 2 + len(REASONS)): rho^I of the RL proposal, 1/0 accepted and, for a rejection, 1/0 per
    # condition failed; NaN where the policy was not queried
    queries = np.array(queries, float)
    queries[~np.isfinite(queries[:, 0]), 0] = np.nan  # malformed proposal: no robustness
    return system, st, policy, P, knots, plans, sources, queries, sup.solve_log, mj, snaps


def point_mass_scene(P, alt: float = DI_ALTITUDE):
    """``((model, data), snaps)``: a sphere following the positions ``P`` (planar at height ``alt``, or
    3-D around the cube centre at ``alt``). Only draws the double integrator, simulated in NumPy."""
    import mujoco

    model = mujoco.MjModel.from_xml_string(DI_MJCF)
    snaps = [(np.array([*lift(q, alt), 1.0, 0.0, 0.0, 0.0]), np.zeros(6)) for q in P]
    return (model, mujoco.MjData(model)), snaps


def altitude(system, st) -> float:
    """Height of the f_syn origin (the centre of S) in the 3-D views."""
    if st.task.dim == 3:
        return getattr(system.cfg, "altitude_plus", DI_CUBE_ALTITUDE) if system.name != "di" else DI_CUBE_ALTITUDE
    return getattr(system.cfg, "altitude", DI_ALTITUDE)


def lift(q, alt: float) -> np.ndarray:
    """World position of a signal sample: planar ``(px, py)`` at height ``alt``, or ``(px, py, pz)``
    with ``pz`` relative to ``alt``."""
    q = np.asarray(q, float)
    return np.array([q[0], q[1], alt + (q[2] if len(q) > 2 else 0.0)])


def true_region(task, seed: int) -> dict | None:
    """``_u`` specs: the true keep-out region of start index ``seed`` (the one ``evaluate.py`` scores
    against); None for the exact specs."""
    if not task.uncertain:
        return None
    return task.sample_true_region(np.random.default_rng(TRUE_REGION_SEED + seed))


def band_outlines(task, n: int = 1440, m: int = 201) -> tuple[np.ndarray, np.ndarray]:
    """2-D ``_u`` specs: closed outlines ``(n, 2)`` of the largest and the smallest keep-out region over the
    uncertainty set, i.e. the union and the intersection of the squares of half-width
    ``b in [c - pred_offset, c + pred_offset]`` rotated by ``|alpha| <= pred_rotation``. Both are star
    shaped about the centre: radius ``max`` / ``min`` over ``alpha`` of ``b / max(|cos|, |sin|)``."""
    c, o, A = task.half_width, task.pred_offset, task.pred_rotation
    th = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    ph = th[:, None] - np.linspace(-A, A, m)[None, :]
    r = 1.0 / np.maximum(np.abs(np.cos(ph)), np.abs(np.sin(ph)))
    u = np.c_[np.cos(th), np.sin(th)]
    return (c + o) * r.max(axis=1)[:, None] * u, (c - o) * r.min(axis=1)[:, None] * u


def square_corners(region: dict) -> np.ndarray:
    """2-D: the four corners of a (rotated) square region, counter-clockwise."""
    b = region["half_width"]
    return np.array([[-b, -b], [b, -b], [b, b], [-b, b]]) @ np.asarray(region["R"]).T


CUBE_EDGES = [(i, j) for i in range(8) for j in range(i + 1, 8) if (i ^ j).bit_count() == 1]
CUBE_FACES = ((0, 1, 3, 2), (4, 5, 7, 6), (0, 1, 5, 4), (2, 3, 7, 6), (0, 2, 6, 4), (1, 3, 7, 5))


def cube_vertices(b: float, R: np.ndarray | None = None) -> np.ndarray:
    """The eight vertices of the cube ``|y_k| <= b`` rotated by ``R`` (index bits: x, y, z sign)."""
    v = np.array([[sx, sy, sz] for sx in (-b, b) for sy in (-b, b) for sz in (-b, b)])
    return v if R is None else v @ np.asarray(R).T


def extreme_cubes(task) -> list[np.ndarray]:
    """3-D ``_u`` specs: vertices of the extreme cubes of the uncertainty set (smallest and largest
    half-width, unrotated and rotated by ``+-pred_rotation`` about each axis)."""
    c, o, A = task.half_width, task.pred_offset, task.pred_rotation
    rots = [np.eye(3)] + [rotation_matrix(np.eye(3)[k], s * A) for k in range(3) for s in (-1.0, 1.0)]
    return [cube_vertices(b, R) for b in (c - o, c + o) for R in rots]


def status(sources, queries):
    """``status(k)``: ``(rejected, failed)`` at step ``k``, ``failed`` the conditions (``REASONS``) that the
    latest rejected proposal failed. A rejection lasts until a proposal is accepted again."""
    n = len(sources)
    failed = np.zeros((n, len(REASONS)), bool)
    last = failed[0]
    for k in range(n):
        if queries[k, 1] == 0.0:
            last = queries[k, 2:] == 1.0
        if sources[k] != "proposal":
            failed[k] = last

    def at(k):
        k = min(k, n - 1)
        return sources[k] != "proposal", failed[k]
    return at


def status_bar(fig, y=0.85, h=0.085):
    """Boxes of the two control states and the four rejection conditions across the top of ``fig``;
    returns ``frame(rejected, failed)``, which fills the boxes that apply with their color."""
    import matplotlib.patheffects as pe
    from matplotlib.patches import FancyBboxPatch

    specs = [(0.04 + 0.172 * i, 0.16, name, desc, color) for i, (name, desc, color) in enumerate(STATES)]
    specs += [(0.435 + 0.1355 * i, 0.128, name, desc, color) for i, (_, name, desc, color) in enumerate(REASONS)]
    fig.text(0.04, y + h + 0.008, "Control state", fontsize=10, color="0.35")
    fig.text(0.435, y + h + 0.008, "Why the STL check rejected the RL plan", fontsize=10, color="0.35")
    fig.text(0.4035, y + h / 2, "\u2192", ha="center", va="center", fontsize=26, color="0.45")
    boxes = []
    for x, w, name, desc, color in specs:
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.006", lw=1.6,
                             transform=fig.transFigure)
        fig.add_artist(box)
        texts = (fig.text(x + w / 2, y + 0.67 * h, name, ha="center", va="center", fontsize=15, fontweight="bold"),
                 fig.text(x + w / 2, y + 0.27 * h, desc, ha="center", va="center", fontsize=9.5, linespacing=1.1))
        boxes.append((box, texts, color))
    halo = [pe.withStroke(linewidth=1.0, foreground="#7a5c00")]  # the white name on yellow needs an edge

    def frame(rejected, failed):
        for i, (box, (name, desc), color) in enumerate(boxes):
            on = i == int(rejected) if i < len(STATES) else bool(rejected and failed[i - len(STATES)])
            ink = "#a57c00" if color == SOLVE else color  # yellow text is unreadable on white
            box.set_facecolor(color if on else "white")
            box.set_edgecolor(to_rgba(color, 1.0 if on else 0.45))
            for t in (name, desc):
                t.set_color("white" if on else to_rgba(ink, 0.6))
            name.set_path_effects(halo if on and color == SOLVE else [])
    return frame


def summarize(args, system, st, P, knots, sources, solve_log):
    p, n = st.problem, len(sources)
    rho = realised_robustness(p, P)
    solve_steps = sorted({s["request_step"] for s in solve_log if s["request_step"] >= 0})
    wps = st.task.waypoints()
    reach_t, idx = [], int(np.argmin(np.linalg.norm(wps - P[0], axis=1)) + 1) % len(wps)
    for k, q in enumerate(P):  # waypoints reached in patrol order within 0.25 m, as in evaluate.py
        if np.linalg.norm(wps[idx] - q) < 0.25:
            reach_t.append(k)
            idx = (idx + 1) % len(wps)
    out = {"system": args.system, "spec": args.spec, "level": args.level,
           "level_value": float(system.levels[args.level]), "seed": args.seed, "dt": system.dt, "steps": n,
           "satisfied": bool(np.all(rho >= 0)), "min_rho": float(rho.min()),
           "accepted": sources.count("proposal") / n, "solves": len(solve_steps),
           "solves_per_step": len(solve_steps) / n, "waypoints": len(reach_t), "policy": args.policy}
    if knots is not None:
        gap = np.linalg.norm(P - knots, axis=1)
        out.update(mean_gap_to_reference=float(gap.mean()), max_gap_to_reference=float(gap.max()))
    region, rho_true = true_region(st.task, args.seed), None
    if region is not None:  # rho is then the interval lower robustness
        out.update(uncertain_scores(st.task, P, region))
        h = st.task.horizon
        f_true = st.task.true_formula(region)
        rho_true = np.array([robustness(f_true, P[s:s + h + 1]) for s in range(len(P) - h)])
    return out, rho, rho_true, region, solve_steps, reach_t


def counters(args, system, st, P, sources, solve_steps, reach_t, region=None):
    """``lines(k)``: the text panel at step ``k`` (shared by all views); ``_u`` specs count the time in
    the episode's true S."""
    n, c = len(sources), st.task.half_width
    if region is None:
        inside = np.all(np.abs(P) < c, axis=1)
    else:
        inside = np.all(np.abs(P @ np.asarray(region["R"])) < region["half_width"], axis=1)
    stay = np.zeros(len(P))
    for k in range(1, len(P)):
        stay[k] = stay[k - 1] + system.dt if inside[k] else 0.0
    rule = "limit 4 s" if args.spec.startswith("theta") else "out 1 s within 3 s"

    def lines(k):
        src = sources[min(k, n - 1)]
        acc = sources[:min(k + 1, n)].count("proposal") / min(k + 1, n)
        return [f"t = {k * system.dt:6.2f} s",
                f"plan    {SRC_LABEL[src]}",
                f"accepted {100 * acc:5.1f} %   solves {sum(1 for s in solve_steps if s <= k):3d}",
                f"waypoints reached {sum(1 for t in reach_t if t <= k):2d}",
                f"time in {'true S' if region is not None else 'S'} {stay[k]:4.1f} s ({rule})"]
    return lines


def title(args, system, policy):
    unc = (f"w̄ = {system.levels[args.level]:g} m/s²" if args.system == "di"
           else f"ε = {system.levels[args.level]:g} m")
    head = f"{TITLE[args.system]}, formula {SPEC_UNI[args.spec]}, {unc}"
    t = getattr(args, "task", None)  # set by main() once the system is built
    if t is not None and t.uncertain:
        head += f", S: half-width ±{t.pred_offset:g} m, rotation ±{t.pred_rotation:g} rad"
    return (head, f"checked loop, {args.label} proposer ({policy.reference} reference)")


def uncertainty_handles(st, region):
    """Legend entries of the uncertain S (``_u`` specs), else none."""
    if region is None:
        return []
    t = st.task
    return [Rectangle((0, 0), 1, 1, facecolor="#2e9e4f", alpha=0.3, edgecolor=KEEPOUT, ls="--", lw=0.9,
                      label=f"possible S (±{t.pred_offset:g} m, ±{t.pred_rotation:g} rad)"),
            plt.Line2D([], [], color=TRUE_S, lw=1.8, label="true S (this episode)")]


def keepout_2d(ax, st, region):
    """S on the top-down axes: the nominal square, or (``_u`` specs) the band between the smallest and the
    largest possible S and the episode's true S."""
    c = st.task.half_width
    if region is None:
        ax.add_patch(Rectangle((-c, -c), 2 * c, 2 * c, facecolor="#2e9e4f", alpha=0.18, edgecolor=KEEPOUT,
                               lw=1.6, zorder=0))
        return
    from matplotlib.patches import PathPatch
    from matplotlib.path import Path as MPath

    outer, inner = band_outlines(st.task)
    ring = MPath.make_compound_path(MPath(np.r_[outer, outer[:1]], closed=True),
                                    MPath(np.r_[inner, inner[:1]][::-1], closed=True))
    ax.add_patch(PathPatch(ring, facecolor="#2e9e4f", alpha=0.30, edgecolor="none", zorder=0))
    ax.add_patch(Polygon(inner, closed=True, facecolor="#2e9e4f", alpha=0.12, edgecolor="none", zorder=0))
    for o in (outer, inner):
        ax.add_patch(Polygon(o, closed=True, fill=False, edgecolor=KEEPOUT, lw=0.9, ls="--", zorder=1))
    ax.add_patch(Polygon(square_corners(region), closed=True, fill=False, edgecolor=TRUE_S, lw=1.8, zorder=2))


def keepout_3d(ax, st, region):
    """S+ in the perspective view: the nominal cube, or (``_u`` specs) the extreme cubes of the
    uncertainty set as wireframes and the episode's true cube."""
    from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

    if region is None:
        v = cube_vertices(st.task.half_width)
        ax.add_collection3d(Poly3DCollection([[v[i] for i in f] for f in CUBE_FACES], facecolor="#2e9e4f",
                                             alpha=0.10, edgecolor=KEEPOUT, lw=1.0))
        return
    segs = [[v[i], v[j]] for v in extreme_cubes(st.task) for i, j in CUBE_EDGES]
    ax.add_collection3d(Line3DCollection(segs, colors=KEEPOUT, linewidths=0.6, alpha=0.45, linestyles="--"))
    v = cube_vertices(region["half_width"], region["R"])
    ax.add_collection3d(Poly3DCollection([[v[i] for i in f] for f in CUBE_FACES], facecolor="#2e9e4f",
                                         alpha=0.10, edgecolor=TRUE_S, lw=1.4))


def topdown_axes(ax, st, P, knots, plans, sources, lines, info_fontsize=9, region=None):
    """Static top-down scene on ``ax``; returns ``(frame, handles)``, ``frame(k)`` drawing step ``k``."""
    n, wps = len(sources), st.task.waypoints()
    keepout_2d(ax, st, region)
    ax.text(0, 0, "S (keep out)", ha="center", va="center", color=KEEPOUT, fontsize=12, fontweight="bold")
    if knots is not None:
        ax.plot(knots[:, 0], knots[:, 1], ls="--", color="0.7", lw=0.9, zorder=1)
    ax.scatter(wps[:, 0], wps[:, 1], marker="*", s=260, color=WAYPOINT, edgecolors="#7a0e10", linewidths=0.8,
               zorder=6)
    ax.set_aspect("equal")
    ax.set_xlim(-2.35, 2.35)
    ax.set_ylim(-2.35, 2.35)
    ax.set_xlabel("$p_x$ [m]")
    ax.set_ylabel("$p_y$ [m]")
    ax.grid(lw=0.4, alpha=0.5)

    trail = LineCollection([], colors=TRAIL, linewidths=1.3, alpha=0.8, zorder=3)
    ax.add_collection(trail)
    plan_line, = ax.plot([], [], lw=2.4, zorder=4, solid_capstyle="round")
    plan_pts = ax.scatter([], [], s=10, zorder=4)
    link, = ax.plot([], [], color=KNOT, lw=1.0, ls=":", zorder=5)
    knot = ax.scatter([], [], s=110, facecolors="none", edgecolors=KNOT, linewidths=2.0, zorder=7)
    veh = ax.scatter([], [], s=90, color=VEHICLE, edgecolors="white", linewidths=1.2, zorder=8)
    info = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", ha="left", fontsize=info_fontsize,
                   family="monospace", zorder=10,
                   bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.85, "edgecolor": "0.8"})
    handles = [
        plt.Line2D([], [], marker="o", ls="", color=VEHICLE, mec="white", ms=8, label="vehicle"),
        plt.Line2D([], [], color=TRAIL, lw=1.3, label="vehicle trail"),
        plt.Line2D([], [], marker="o", ls="", mfc="none", mec=KNOT, mew=2, ms=9, label="knot point $x^{nom}(t)$"),
        plt.Line2D([], [], color="0.7", ls="--", label="reference path"),
        plt.Line2D([], [], marker="*", ls="", color=WAYPOINT, ms=13, label="waypoint"),
        plt.Line2D([], [], color=SRC_COLOR["proposal"], lw=2.4, label="plan: accepted proposal"),
        plt.Line2D([], [], color=SRC_COLOR["solver"], lw=2.4, label="plan: solver"),
        plt.Line2D([], [], color=SRC_COLOR["cache"], lw=2.4, label="plan: cached"),
    ] + uncertainty_handles(st, region)

    def frame(k):
        pts = P[:k + 1]
        trail.set_segments([pts[i:i + 2] for i in range(len(pts) - 1)])
        src, pl = sources[min(k, n - 1)], plans[min(k, n - 1)]
        plan_line.set_data(pl[:, 0], pl[:, 1])
        plan_line.set_color(SRC_COLOR[src])
        plan_pts.set_offsets(pl)
        plan_pts.set_color(SRC_COLOR[src])
        veh.set_offsets([P[k]])
        if knots is not None:
            knot.set_offsets([knots[k]])
            link.set_data([P[k, 0], knots[k, 0]], [P[k, 1], knots[k, 1]])
        info.set_text("\n".join(lines(k)))
    return frame, handles


def cube_axes(ax, st, P, knots, plans, sources, lines, info_fontsize=9, region=None):
    """3-D counterpart of :func:`topdown_axes` for the cube specs (``ax`` has ``projection="3d"``)."""
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    n, wps = len(sources), st.task.waypoints()
    keepout_3d(ax, st, region)
    if knots is not None:
        ax.plot(knots[:, 0], knots[:, 1], knots[:, 2], ls="--", color="0.7", lw=0.8)
    ax.scatter(wps[:, 0], wps[:, 1], wps[:, 2], marker="*", s=200, color=WAYPOINT, edgecolors="#7a0e10",
               linewidths=0.8, depthshade=False)
    lim = 2.35
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel("$p_x$ [m]")
    ax.set_ylabel("$p_y$ [m]")
    ax.set_zlabel("$p_z$ [m]")
    ax.view_init(elev=22, azim=-60)
    trail = Line3DCollection([np.repeat(P[:1], 2, axis=0)], colors=TRAIL, linewidths=1.2, alpha=0.8)
    ax.add_collection3d(trail, autolim=False)
    plan_line, = ax.plot([], [], [], lw=2.4, solid_capstyle="round")
    link, = ax.plot([], [], [], color=KNOT, lw=1.0, ls=":")
    knot, = ax.plot([], [], [], marker="o", ls="", mfc="none", mec=KNOT, mew=2, ms=9)
    veh, = ax.plot([], [], [], marker="o", ls="", color=VEHICLE, mec="white", ms=8)
    info = ax.text2D(0.0, 1.0, "", transform=ax.transAxes, va="top", ha="left", fontsize=info_fontsize,
                     family="monospace", zorder=10,
                     bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.85, "edgecolor": "0.8"})
    handles = [
        plt.Line2D([], [], marker="o", ls="", color=VEHICLE, mec="white", ms=8, label="vehicle"),
        plt.Line2D([], [], color=TRAIL, lw=1.3, label="vehicle trail"),
        plt.Line2D([], [], marker="o", ls="", mfc="none", mec=KNOT, mew=2, ms=9, label="knot point $x^{nom}(t)$"),
        plt.Line2D([], [], color="0.7", ls="--", label="reference path"),
        plt.Line2D([], [], marker="*", ls="", color=WAYPOINT, ms=13, label="waypoint"),
        plt.Line2D([], [], color=SRC_COLOR["proposal"], lw=2.4, label="plan: accepted proposal"),
        plt.Line2D([], [], color=SRC_COLOR["solver"], lw=2.4, label="plan: solver"),
        plt.Line2D([], [], color=SRC_COLOR["cache"], lw=2.4, label="plan: cached"),
    ] + uncertainty_handles(st, region)

    def frame(k):
        pts = P[:k + 1]
        trail.set_segments([pts[i:i + 2] for i in range(len(pts) - 1)] or [np.repeat(pts[:1], 2, axis=0)])
        src, pl = sources[min(k, n - 1)], plans[min(k, n - 1)]
        plan_line.set_data_3d(pl[:, 0], pl[:, 1], pl[:, 2])
        plan_line.set_color(SRC_COLOR[src])
        veh.set_data_3d([P[k, 0]], [P[k, 1]], [P[k, 2]])
        if knots is not None:
            knot.set_data_3d([knots[k, 0]], [knots[k, 1]], [knots[k, 2]])
            link.set_data_3d([P[k, 0], knots[k, 0]], [P[k, 1], knots[k, 1]], [P[k, 2], knots[k, 2]])
        info.set_text("\n".join(lines(k)))
    return frame, handles


def scene_axes(fig, rect, st, *args, **kw):
    """Top-down axes (2-D specs) or 3-D perspective axes (cube specs) at ``rect`` of ``fig``."""
    if st.task.dim == 3:
        ax = fig.add_axes(rect, projection="3d")
        return ax, cube_axes(ax, st, *args, **kw)
    ax = fig.add_axes(rect)
    return ax, topdown_axes(ax, st, *args, **kw)


def rho_axes(ax, args, system, st, rho, solve_steps, n, rho_true=None):
    """Robustness history on ``ax``, each window at its end; returns ``frame(k)`` revealing the windows
    that end at or before step ``k``. ``rho_true`` (``_u`` specs): the robustness against the true S,
    drawn with ``rho`` (then the interval lower robustness)."""
    dt, h = system.dt, st.problem.h
    t = (np.arange(len(rho)) + h) * dt  # rho[s] is the window [s, s + h], known at step s + h
    both = rho if rho_true is None else np.r_[rho, rho_true]
    lo, hi = min(0.0, both.min()), max(both.max(), 0.1)
    pad = 0.08 * (hi - lo)
    ax.axhline(0.0, color="0.25", lw=0.9, zorder=1)
    ax.axvspan(0.0, h * dt, color="0.9", lw=0, zorder=0)
    ax.text(h / 2 * dt, (lo + hi) / 2, "window incomplete", ha="center", va="center", rotation=90,
            fontsize=7, color="0.45")
    ts = np.array(solve_steps, float) * dt
    solves = solve_lines(ax, ts)
    true_line = None
    if rho_true is not None:
        true_line, = ax.plot([], [], color=TRUE_S, lw=1.2, ls="--", zorder=4, label=r"$\rho$ against the true S")
    line, = ax.plot([], [], color=RHO, lw=1.4, zorder=4,
                    label=None if rho_true is None else r"$\rho^{I}$ (worst case over the possible S)")
    now = ax.axvline(0.0, color="0.55", lw=0.8, ls=":", zorder=2)
    dot = ax.scatter([], [], s=55, color=RHO, edgecolors="white", linewidths=1.0, zorder=5)
    ax.set_xlim(0.0, n * dt)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlabel("$t$ [s] (window end)")
    ax.set_ylabel(rf"$\rho^{{{SPEC_TEX[args.spec]}}}(y, t-h)$ [m]")
    ax.grid(lw=0.4, alpha=0.5)

    def frame(k):
        j = k - h  # the latest window that has been fully measured
        line.set_data(t[:max(j + 1, 0)], rho[:max(j + 1, 0)])
        if true_line is not None:
            true_line.set_data(t[:max(j + 1, 0)], rho_true[:max(j + 1, 0)])
        dot.set_offsets([[t[j], rho[j]]] if j >= 0 else np.empty((0, 2)))
        now.set_xdata([k * dt, k * dt])
        for s, ln in zip(ts, solves):
            ln.set_visible(s <= k * dt)
    return frame


def solve_lines(ax, ts):
    """Yellow vertical lines at the solve-request times ``ts``, hidden until a frame reveals them."""
    return [ax.axvline(s, color=SOLVE, lw=1.8, alpha=0.9, zorder=2, visible=False,
                       label="STL solve requested" if i == 0 else "_nolegend_") for i, s in enumerate(ts)]


def rl_axes(ax, system, st, queries, solve_steps, n):
    """Robustness of the measured history followed by the RL proposal at every query, against the
    acceptance threshold delta (red); returns ``frame(k)`` revealing it up to step ``k``."""
    dt, delta = system.dt, st.problem.delta
    t = np.arange(n) * dt
    r, acc = queries[:, 0], queries[:, 1]
    fin = r[np.isfinite(r)]
    lo = min(0.0, delta, fin.min() if fin.size else 0.0)
    hi = max(delta + 0.1, fin.max() if fin.size else 0.0)
    pad = 0.08 * (hi - lo)
    ax.axhline(0.0, color="0.25", lw=0.9, zorder=1)
    ax.axhline(delta, color=WAYPOINT, lw=1.6, zorder=3, label=rf"$\delta$ = {delta:g} m (acceptance threshold)")
    ts = np.array(solve_steps, float) * dt
    solves = solve_lines(ax, ts)
    line, = ax.plot([], [], color=SRC_COLOR["proposal"], lw=0.9, alpha=0.6, zorder=4)
    ok = ax.scatter(np.empty(0), np.empty(0), s=9, color=SRC_COLOR["proposal"], zorder=5,
                    label="RL proposal accepted")
    bad = ax.scatter(np.empty(0), np.empty(0), s=24, marker="x", color="0.1", linewidths=1.1, zorder=5,
                     label="RL proposal rejected")
    now = ax.axvline(0.0, color="0.55", lw=0.8, ls=":", zorder=2)
    dot = ax.scatter(np.empty(0), np.empty(0), s=55, edgecolors="white", linewidths=1.0, zorder=6)
    ax.set_xlim(0.0, n * dt)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlabel("$t$ [s] (check time)")
    ax.set_ylabel(r"$\rho^{I}$(history + RL plan) [m]")
    ax.grid(lw=0.4, alpha=0.5)

    def frame(k):
        j = min(k, n - 1) + 1
        line.set_data(t[:j], r[:j])
        for sc, flag in ((ok, 1.0), (bad, 0.0)):
            m = (acc[:j] == flag) & np.isfinite(r[:j])
            sc.set_offsets(np.c_[t[:j][m], r[:j][m]])
        if k < n and np.isfinite(r[k]):
            dot.set_offsets([[t[k], r[k]]])
            dot.set_facecolor(SRC_COLOR["proposal"] if acc[k] == 1.0 else "0.1")
        else:
            dot.set_offsets(np.empty((0, 2)))
        now.set_xdata([k * dt, k * dt])
        for s, ln in zip(ts, solves):
            ln.set_visible(s <= k * dt)
    return frame


def render_rl_pdf(args, system, st, policy, queries, solve_steps, n, run_dir):
    """Robustness of history + RL proposal at every query over the whole episode."""
    plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans"})
    fig, ax = plt.subplots(figsize=(6.4, 2.6))
    frame = rl_axes(ax, system, st, queries, solve_steps, n)
    frame(n)
    ax.lines[-1].set_visible(False)  # no current-time line
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=7, frameon=False, borderaxespad=0.0)
    r, acc = queries[:, 0], queries[:, 1]
    rej = acc == 0.0
    head = f"{title(args, system, policy)[0]}: {int(np.sum(acc == 1.0))} of {int(np.sum(np.isfinite(acc)))} " \
           "RL proposals accepted"
    if rej.any():
        head += (f";\n{int(rej.sum())} rejected, {int(np.sum(rej & (r >= st.problem.delta)))} of them with "
                 r"$\rho^{I} \geq \delta$ (a bound or the terminal set failed)")
    ax.set_title(head, fontsize=9, loc="left")
    fig.tight_layout()
    out = run_dir / f"{args.system}_{args.spec}_rho_rl.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def render_rho_pdf(args, system, st, policy, rho, solve_steps, n, run_dir, rho_true=None):
    """Robustness of the formula against window end over the whole episode."""
    plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans"})
    fig, ax = plt.subplots(figsize=(6.4, 2.6))
    frame = rho_axes(ax, args, system, st, rho, solve_steps, n, rho_true)
    frame(n)  # the whole history
    ax.lines[-1].set_visible(False)  # without the current-time line
    ax.collections[-1].set_visible(False)  # and dot
    i = int(np.argmin(rho))
    t_min = (i + st.problem.h) * system.dt  # end of the worst window
    ax.scatter([t_min], [rho[i]], s=28, marker="v", color=WAYPOINT, zorder=6, label="minimum")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=7, frameon=False, borderaxespad=0.0)
    head = title(args, system, policy)[0]
    if rho_true is None:
        head += f", min $\\rho$ = {rho[i]:.3f} m, window ending at $t$ = {t_min:g} s"
    else:
        ax.set_ylabel(r"$\rho(y, t-h)$ [m]")
        head = head.replace(", S:", ",\nS:") + (f"; min $\\rho^{{I}}$ = {rho[i]:.3f} m at $t$ = {t_min:g} s, "
                                                 f"min $\\rho$ (true S) = {rho_true.min():.3f} m")
    ax.set_title(head, fontsize=9, loc="left")
    fig.tight_layout()
    out = run_dir / f"{args.system}_{args.spec}_rho.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def render_topdown(args, system, st, policy, P, knots, plans, sources, lines, run_dir, region=None):
    n = len(sources)
    plt.rcParams.update({"font.size": 10, "font.family": "DejaVu Sans"})
    fig = plt.figure(figsize=(6.4, 6.4), dpi=150)
    ax, (frame, handles) = scene_axes(fig, [0.1, 0.17, 0.87, 0.73], st, P, knots, plans, sources, lines,
                                      region=region)
    ax.set_title("\n".join(title(args, system, policy)), fontsize=11 if region is None else 9.5)
    fig.legend(handles=handles, loc="lower center", ncol=4 if region is None else 5, frameon=False,
               fontsize=7.5 if region is None else 6.5, columnspacing=1.0)

    stem = f"{args.system}_{args.spec}"
    writer = FFMpegWriter(fps=args.fps, bitrate=1800, codec="libx264", extra_args=["-pix_fmt", "yuv420p"])
    with writer.saving(fig, str(run_dir / f"{stem}.mp4"), dpi=150):
        for k in range(n + 1):
            frame(k)
            writer.grab_frame()
    fig.savefig(run_dir / f"{stem}_poster.pdf")
    plt.close(fig)
    return run_dir / f"{stem}.mp4"


def mujoco_frames(args, system, st, policy, knots, plans, sources, lines, mj, snaps, width, height,
                  overlay=True, camera=None, region=None):
    """Offscreen MuJoCo render of the recorded simulator states with the task drawn into the scene;
    yields one RGB frame per step, with the title and counters drawn on it when ``overlay``.
    ``camera``: ``(distance, elevation, lookat_y)`` of the free camera, default ``(--cam-distance, -58, 0.1)``."""
    import mujoco
    from PIL import Image, ImageDraw, ImageFont

    from rl_stl.utils.render import add_box, add_segment, add_sphere

    model, data = mj
    n, c, wps = len(sources), st.task.half_width, st.task.waypoints()
    alt = altitude(system, st)
    cube = st.task.dim == 3
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, width)  # offscreen framebuffer
    model.vis.global_.offheight = max(model.vis.global_.offheight, height)
    model.vis.headlight.ambient[:] = 0.45
    model.vis.headlight.diffuse[:] = 0.65
    model.vis.headlight.specular[:] = 0.15
    has_floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor") >= 0
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    distance, elevation, lookat_y = camera or (args.cam_distance, -58.0, 0.1)
    if cube:  # the cube is 2.8 m tall: step back and look from lower down
        distance, elevation = 1.7 * distance, -22.0
    cam.lookat[:] = [0.0, lookat_y, alt if cube else alt * 0.6]
    cam.distance = distance
    cam.elevation = elevation
    cam.azimuth = 90.0
    world = np.array([s[0][:3] for s in snaps])  # free-joint position of the vehicle
    if region is None:
        keepout = [(np.array([c, c, c if cube else 0.12]), None)]
        band = []
    else:  # the true S as the box, the possible S as thin outlines
        R = np.asarray(region["R"])
        b = region["half_width"]
        if cube:
            keepout = [(np.full(3, b), R)]
            band = [(v[i] + [0, 0, alt], v[j] + [0, 0, alt]) for v in extreme_cubes(st.task) for i, j in CUBE_EDGES]
        else:
            R3 = np.eye(3)
            R3[:2, :2] = R
            keepout = [(np.array([b, b, 0.12]), R3)]
            band = [(lift(o[i], alt + 0.125), lift(o[(i + 1) % len(o)], alt + 0.125))
                    for o in (q[::8] for q in band_outlines(st.task)) for i in range(len(o))]
    rgba = {k: np.array(to_rgba(v)) for k, v in SRC_COLOR.items()}
    from matplotlib import font_manager

    font = ImageFont.truetype(font_manager.findfont("DejaVu Sans"), max(14, height // 40))  # has Greek
    head = "  |  ".join(title(args, system, policy))
    with mujoco.Renderer(model, height=height, width=width, max_geom=4000 + len(band)) as renderer:
        for k in range(n + 1):
            data.qpos[:], data.qvel[:] = snaps[k]
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=cam)
            sc = renderer.scene
            sc.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False  # the slab's shadow reads as a second S
            sc.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = False
            # backdrop floor, under the model's own floor (the blimp arena's ends inside the view)
            add_box(sc, np.array([0.0, 0.0, -0.02 if has_floor else 0.0]), np.array([10.0, 10.0, 0.005]),
                    rgba=(0.82, 0.83, 0.85, 1.0))
            for half, rot in keepout:
                add_box(sc, np.array([0.0, 0.0, alt]), half, rgba=(0.12, 0.6, 0.27, 0.18 if cube else 0.28), rot=rot)
            for a, b_ in band:
                add_segment(sc, a, b_, 0.008, (0.06, 0.35, 0.15, 0.75))
            for w in wps:
                add_sphere(sc, lift(w, alt), 0.08, (0.84, 0.15, 0.16, 1.0))
            for i in range(k):
                add_segment(sc, world[i], world[i + 1], 0.012, (0.2, 0.2, 0.2, 0.85))
            src, pl = sources[min(k, n - 1)], plans[min(k, n - 1)]
            for i in range(len(pl) - 1):
                add_segment(sc, lift(pl[i], alt), lift(pl[i + 1], alt), 0.022, rgba[src])
            if knots is not None:
                kp = lift(knots[k], alt)
                add_sphere(sc, kp, 0.06, (0.54, 0.25, 0.75, 0.9))
                add_segment(sc, world[k], kp, 0.006, (0.54, 0.25, 0.75, 0.7))
            img = renderer.render()
            if overlay:
                frame = Image.fromarray(img)
                draw = ImageDraw.Draw(frame, "RGBA")
                draw.rectangle([0, 0, width, font.size + 14], fill=(255, 255, 255, 215))
                draw.text((10, 6), head, fill=(20, 20, 20, 255), font=font)
                txt = "\n".join(lines(k))
                box = draw.multiline_textbbox((12, font.size + 24), txt, font=font, spacing=4)
                draw.rectangle([box[0] - 6, box[1] - 6, box[2] + 6, box[3] + 6], fill=(255, 255, 255, 215))
                draw.multiline_text((12, font.size + 24), txt, fill=(20, 20, 20, 255), font=font, spacing=4)
                img = np.asarray(frame)
            yield np.ascontiguousarray(img)


def render_mujoco(args, system, st, policy, knots, plans, sources, lines, mj, snaps, run_dir, region=None):
    from PIL import Image

    stem = f"{args.system}_{args.spec}_mujoco"
    mp4 = run_dir / f"{stem}.mp4"
    writer = imageio_ffmpeg.write_frames(str(mp4), (args.width, args.height), fps=args.fps, codec="libx264",
                                         pix_fmt_out="yuv420p", quality=7, macro_block_size=8)
    writer.send(None)
    img = None
    for img in mujoco_frames(args, system, st, policy, knots, plans, sources, lines, mj, snaps,
                             args.width, args.height, region=region):
        writer.send(img)
    writer.close()
    Image.fromarray(img).save(run_dir / f"{stem}_poster.png")
    return mp4


# Composite layout in figure fractions of a 16 x 9 in figure at 120 dpi (1920 x 1080 pixels): the
# title and the status bar across the top, the top-down view as a square on the left
# with its legend below, the MuJoCo view top right (rendered at 1.25 times its on-screen size,
# 1212 x 452, with a closer camera than the stand-alone view) and below it the realized robustness
# and the RL proposal robustness on a shared time axis.
COMPOSITE_DPI = 120
TOPDOWN_BOX = [0.04, 0.15, 0.39, 0.655]
SIM_BOX = [0.47, 0.475, 0.505, 0.335]
RHO_BOX = [0.525, 0.29, 0.45, 0.13]
RL_BOX = [0.525, 0.075, 0.45, 0.13]
SIM_PX = (1212, 452)
SIM_CAMERA = (4.9, -58.0, 0.1)


def render_composite(args, system, st, policy, P, knots, plans, sources, queries, lines, rho, solve_steps, mj,
                     snaps, run_dir, rho_true=None, region=None):
    n = len(sources)
    plt.rcParams.update({"font.size": 11, "font.family": "DejaVu Sans"})
    fig = plt.figure(figsize=(16, 9), dpi=COMPOSITE_DPI)
    fig.suptitle("   |   ".join(title(args, system, policy)), fontsize=13, y=0.982)
    authority = status(sources, queries)
    bar_frame = status_bar(fig)
    ax_sim, ax_rho, ax_rl = (fig.add_axes(b) for b in (SIM_BOX, RHO_BOX, RL_BOX))
    ax_td, (td_frame, handles) = scene_axes(fig, TOPDOWN_BOX, st, P, knots, plans, sources, lines, info_fontsize=10,
                                            region=region)
    ax_td.set_title("Perspective view of S+" if st.task.dim == 3 else "Top-down view", fontsize=12)
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(TOPDOWN_BOX[0] + TOPDOWN_BOX[2] / 2, 0.0),
               ncol=4 if region is None else 5, frameon=False, fontsize=9 if region is None else 8,
               columnspacing=1.0)
    rho_frame = rho_axes(ax_rho, args, system, st, rho, solve_steps, n, rho_true)
    if rho_true is None:
        ax_rho.set_title(rf"Robustness of ${SPEC_TEX[args.spec]}$ on the realized trace, window ending at $t$"
                         + ("   (yellow: STL solve requested)" if solve_steps else ""), fontsize=11)
    else:
        ax_rho.set_title(rf"Robustness of ${SPEC_TEX[args.spec]}$ on the realized trace: $\rho^{{I}}$ (solid, worst "
                         r"case over S) and against the true S (dashed)", fontsize=11)
    ax_rho.tick_params(labelbottom=False)  # shares the time axis of the panel below
    ax_rho.set_xlabel("")
    ax_rho.set_ylabel(rf"$\rho^{{{SPEC_TEX[args.spec]}}}(y, t-h)$ [m]" if rho_true is None
                      else r"$\rho(y, t-h)$ [m]", fontsize=10)
    rl_frame = rl_axes(ax_rl, system, st, queries, solve_steps, n)
    ax_rl.set_title(r"Robustness of history + RL proposal at each check (red: threshold $\delta$; "
                    r"$\times$: rejected)", fontsize=11)
    ax_rl.set_ylabel(r"$\rho^{I}$ [m]", fontsize=10)
    frames = mujoco_frames(args, system, st, policy, knots, plans, sources, lines, mj, snaps, *SIM_PX,
                           overlay=False, camera=SIM_CAMERA, region=region)  # one frame per step, consumed with the other panels
    sim = ax_sim.imshow(next(frames), aspect="auto", interpolation="antialiased")
    ax_sim.set_axis_off()
    ax_sim.set_title("3-D view (point mass; the double integrator is simulated in NumPy)" if args.system == "di"
                     else "MuJoCo simulation", fontsize=12)

    stem = f"{args.system}_{args.spec}_composite"
    writer = FFMpegWriter(fps=args.fps, bitrate=6000, codec="libx264", extra_args=["-pix_fmt", "yuv420p"])
    with writer.saving(fig, str(run_dir / f"{stem}.mp4"), dpi=COMPOSITE_DPI):
        for k in range(n + 1):
            if k > 0:
                sim.set_data(next(frames))
            td_frame(k)
            rho_frame(k)
            rl_frame(k)
            bar_frame(*authority(k))
            writer.grab_frame()
    frames.close()  # release the GL context now, not at interpreter exit
    fig.savefig(run_dir / f"{stem}_poster.png", dpi=COMPOSITE_DPI)
    plt.close(fig)
    return run_dir / f"{stem}.mp4"


def to_gif(mp4: Path, width: int, fps: int) -> Path:
    """Palette-optimized GIF of an MP4 (ffmpeg palettegen / paletteuse)."""
    gif = mp4.with_suffix(".gif")
    vf = (f"fps={fps},scale={width}:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=128[p];"
          f"[b][p]paletteuse=dither=bayer:bayer_scale=4")
    subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-i", str(mp4), "-vf", vf, str(gif)], check=True)
    return gif


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--system", default="di", choices=["di", "x2", "blimp"])
    ap.add_argument("--spec", default="theta", choices=["theta", "phi", "theta_plus", "phi_plus", "theta_u", "phi_u", "theta_plus_u", "phi_plus_u"])
    ap.add_argument("--policy", required=True, help="SegmentPolicy .npz")
    ap.add_argument("--label", default="PPO", help="proposer name shown in the title")
    ap.add_argument("--level", type=int, default=0, help="uncertainty level index of the system")
    ap.add_argument("--seed", type=int, default=0, help="index of the evaluation start state")
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--view", nargs="+", default=["topdown"], choices=["topdown", "mujoco", "composite", "both"])
    ap.add_argument("--width", type=int, default=960, help="MuJoCo view")
    ap.add_argument("--height", type=int, default=720, help="MuJoCo view")
    ap.add_argument("--cam-distance", type=float, default=6.0, help="MuJoCo view")
    ap.add_argument("--gif", action="store_true", help="also write a GIF of each video")
    ap.add_argument("--gif-width", type=int, default=480)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    views = set(args.view) | ({"topdown", "mujoco"} if "both" in args.view else set())
    if views & {"mujoco", "composite"} and imageio_ffmpeg is None:
        raise SystemExit("--view mujoco/composite needs imageio-ffmpeg (uv run --with imageio-ffmpeg)")
    run_dir = create_run_dir(f"patrol_videos/{args.system}", vars(args), seed=args.seed,
                             tag=args.tag or f"{args.spec}_level{args.level}")
    system, st, policy, P, knots, plans, sources, queries, solve_log, mj, snaps = run(args)
    args.task = st.task  # for the titles (after the config was saved)
    summary, rho, rho_true, region, solve_steps, reach_t = summarize(args, system, st, P, knots, sources, solve_log)
    print(json.dumps(summary), flush=True)
    lines = counters(args, system, st, P, sources, solve_steps, reach_t, region)
    outputs = [render_rho_pdf(args, system, st, policy, rho, solve_steps, len(sources), run_dir, rho_true),
               render_rl_pdf(args, system, st, policy, queries, solve_steps, len(sources), run_dir)]
    videos = []
    if "topdown" in views:
        videos.append(render_topdown(args, system, st, policy, P, knots, plans, sources, lines, run_dir, region))
    if "mujoco" in views:
        videos.append(render_mujoco(args, system, st, policy, knots, plans, sources, lines, mj, snaps, run_dir,
                                    region))
    if "composite" in views:
        videos.append(render_composite(args, system, st, policy, P, knots, plans, sources, queries, lines, rho,
                                       solve_steps, mj, snaps, run_dir, rho_true, region))
    if args.gif:
        videos += [to_gif(v, args.gif_width, min(args.fps, 10)) for v in list(videos)]
    write_summary(run_dir, summary)
    for v in outputs + videos:
        print("wrote", v)


if __name__ == "__main__":
    main()
