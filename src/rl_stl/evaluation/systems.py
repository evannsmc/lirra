"""The three patrol benchmarks behind one interface: double integrator, Skydio X2, blimp.

A :class:`PatrolSystem` provides

* ``build(level)``: the :class:`~rl_stl.evaluation.patrol.PatrolSetup` (``f_syn``, receding
  I-STL problem with verified terminal set, tube, nominal patrol) for one uncertainty level;
* ``descriptor(level)``: what the policy observes about the uncertainty (the disturbance bound
  ``w_bar`` for the double integrator, the tracking-tube radius ``eps`` for the MuJoCo
  vehicles), never the realised disturbance;
* ``make_sim(setup, seed)``: the closed loop below the planner. Each planning step it receives
  the cached plan's nominal ``(x_hat_k, u_hat_k)`` and the supervisor command and returns the
  next measured ``f_syn`` state.

Double integrator: ``f_sim = f_syn + B w`` with ``w`` inside ``W`` and feedback
``u = u_hat + K (x - x_hat)`` (tube from ``W`` and ``K``, the guarantee's assumption holds).
X2 and blimp: MuJoCo ``f_sim``; the low-level tracker follows the ZOH-interpolated nominal
between planning points, and the tube is a constant empirical box ``eps`` on the planar
position (NAHS's choice), so ``K = 0`` at the planning level. Whether the realised tracking
error stays inside ``eps`` is measured and reported, not assumed.

Cube specs (``theta_plus``, ``phi_plus``): the signal is the 3-D position, the patrol runs on x,
y and z through six waypoints, the tube box covers x, y and z, and the workspace is
``|p_i| <= 2.5`` m on every axis. The MuJoCo vehicles fly the cube centred at
``altitude_plus`` (2.75 m, so the lowest workspace point is 0.25 m above the floor). The X2's
z bounds are raised to its planar ones (z carries patrol legs); the blimp keeps its physical z
thrust bounds and z speed limit and patrols z with its altitude PD gains.

Uncertain-predicate specs (``*_u``): the same builds with interval face predicates (bounds
``pred_offset``, ``pred_rotation`` in each config, equal at every uncertainty level) and wider terminal
face sets. ``nominal_predicates`` (config field, or ``get_system(..., nominal_predicates=True)``) is the
ablation that plans and checks with the exact predicates. The descriptor of a ``_u`` spec appends
``(pred_offset, pred_rotation)`` to the tube/disturbance bound, so the policy observes every
uncertainty bound (constant across levels: the feature is 1 after the ``w / w_max`` scaling).
"""

from __future__ import annotations

import dataclasses
import math
from typing import Literal

import mujoco
import numpy as np

from rl_stl.dynamics.linear import zoh_double_integrator
from rl_stl.evaluation.patrol import (
    PatrolDIConfig,
    PatrolSetup,
    build_di,
    build_patrol_problem,
    disturbance_sequence,
)
from rl_stl.policies.nominal import AxisPD, PatrolPD
from rl_stl.stl.nahs import PatrolTask, spec_uncertain
from rl_stl.synthesis.tube import constant_tube

SystemName = Literal["di", "x2", "blimp"]


# ============================================================== double integrator
class DISim:
    def __init__(self, setup: PatrolSetup, cfg: PatrolDIConfig, seed: int, n_steps: int,
                 disturbance: str):
        self.s, self.m = setup, setup.model
        self.W = disturbance_sequence(dataclasses.replace(cfg, disturbance=disturbance), n_steps, seed)
        self.k = 0
        self.max_err = 0.0

    def reset(self, x0: np.ndarray) -> np.ndarray:
        self.x = np.asarray(x0, float).copy()
        self.k = 0
        return self.x.copy()

    def step(self, x_hat: np.ndarray, u_hat: np.ndarray, u_cmd: np.ndarray) -> np.ndarray:
        u = np.clip(u_cmd, self.s.problem.u_lo, self.s.problem.u_hi)
        self.x = self.m.A @ self.x + self.m.B @ u + self.m.B @ self.W[self.k]
        self.k += 1
        return self.x.copy()

    def info(self) -> dict:
        return {}


@dataclasses.dataclass(frozen=True)
class DISystem:
    spec: str = "theta"
    levels: tuple[float, ...] = (0.0, 0.01, 0.02, 0.03, 0.04)
    cfg: PatrolDIConfig = PatrolDIConfig(kp=0.25, kd=1.0)
    name: str = "di"

    @property
    def dt(self) -> float:
        return self.cfg.dt

    def build(self, level: int) -> PatrolSetup:
        return build_di(dataclasses.replace(self.cfg, spec=self.spec, w_bar=self.levels[level]))

    def descriptor(self, level: int) -> np.ndarray:
        return _descriptor(self.spec, self.levels[level], self.cfg)

    def make_sim(self, setup: PatrolSetup, seed: int, n_steps: int, disturbance: str = "persistent"):
        return DISim(setup, dataclasses.replace(self.cfg, spec=self.spec, w_bar=float(setup.w_bar[0])), seed,
                     n_steps, disturbance)

    def random_start(self, rng, setup: PatrolSetup) -> np.ndarray:
        p = _ring_position(rng, setup.task)
        v = _start_velocity(rng, p, 0.6 * self.cfg.v_max)
        return np.concatenate([p, v])

    def eval_start(self, rng, setup: PatrolSetup) -> np.ndarray:
        return np.concatenate([_ring_position(rng, setup.task, 1.55, 2.2), np.zeros(setup.task.dim)])


def _descriptor(spec: str, level_value: float, cfg) -> np.ndarray:
    """Observed uncertainty bound: the level per signal axis, plus ``(pred_offset, pred_rotation)`` for the
    ``_u`` specs."""
    d = np.full(PatrolTask(spec=spec).dim, level_value)
    if spec_uncertain(spec):
        d = np.concatenate([d, [cfg.pred_offset, cfg.pred_rotation]])
    return d


def _task(spec: str, cfg, altitude: float) -> PatrolTask:
    return PatrolTask(spec=spec, dt=cfg.dt, altitude=altitude, pred_offset=cfg.pred_offset,
                      pred_rotation=cfg.pred_rotation, nominal_predicates=cfg.nominal_predicates)


def _ring_position(rng, task: PatrolTask, r_lo: float | None = None, r_hi: float | None = None):
    """Position at radius ``[r_lo, r_hi]`` from the centre of ``S``: on a ring (2-D) or, for the cube
    specs, a spherical shell (direction uniform on the sphere)."""
    half, ws = task.half_width, task.workspace
    r_lo = half + 0.1 if r_lo is None else r_lo
    r_hi = min(ws - 0.2, half + 0.9) if r_hi is None else r_hi
    if task.dim == 2:
        ang = rng.uniform(0, 2 * np.pi)
        direction = np.array([np.cos(ang), np.sin(ang)])
    else:
        g = rng.standard_normal(task.dim)
        direction = g / np.linalg.norm(g)
    r = rng.uniform(r_lo, r_hi)
    return np.clip(r * direction, -ws + 0.2, ws - 0.2)


def _start_velocity(rng, p, v_max):
    if rng.random() < 0.5:  # boundary-rich: moving inward
        return -p / np.linalg.norm(p) * rng.uniform(0.0, v_max)
    return rng.uniform(-0.5, 0.5, len(p)) * v_max


# ================================================================== Skydio X2
@dataclasses.dataclass(frozen=True)
class X2PatrolConfig:
    dt: float = 0.2
    n_plan: int = 30
    delta: float = 0.02
    altitude: float = 1.5
    v_max_xy: float = 1.5
    v_max_z: float = 0.5
    z_band: float = 0.5  # |p_z - altitude| <= z_band
    a_max_xy: float = 1.0  # planning accel bound (well inside the tilt-limited box)
    a_max_z: float = 1.0
    kp: float = 0.25
    kd: float = 1.0
    kp_z: float = 1.0
    kd_z: float = 1.5
    track_pos: float = 1.0
    track_vel: float = 0.2
    gamma0: float = 0.97
    effort_weight: float = 0.01
    altitude_plus: float = 2.75  # cube specs: centre of S+ above the floor
    pred_offset: float = 0.05  # _u specs: half-width uncertainty [m]
    pred_rotation: float = 0.05  # _u specs: rotation uncertainty [rad]
    nominal_predicates: bool = False  # _u specs ablation: plan/check with the exact nominal predicates


class X2Sim:
    """Host-side MuJoCo X2; the outer-loop tracker follows the ZOH nominal at the physics rate."""

    def __init__(self, setup: PatrolSetup, cfg: X2PatrolConfig, seed: int):
        from rl_stl.envs.skydio_x2.host_sim import X2HostSim

        self.s, self.cfg = setup, cfg
        self.sim = X2HostSim()
        self.x2 = self.sim.x2
        self.n_sub = int(round(cfg.dt / self.sim.dt))
        if abs(self.n_sub * self.sim.dt - cfg.dt) > 1e-9:
            raise ValueError("planning dt must be a multiple of the physics step")
        self.max_err = 0.0
        self.max_tilt = 0.0

    def _offset(self):
        return np.array([0.0, 0.0, self.s.task.altitude, 0.0, 0.0, 0.0])

    def reset(self, x0: np.ndarray) -> np.ndarray:
        from rl_stl.dynamics.quadrotor_fsyn import mujoco_state_from_fsyn

        qpos, qvel = mujoco_state_from_fsyn(self.x2, np.asarray(x0, float) + self._offset())
        self.sim.reset(qpos, qvel)
        return self.measure()

    def measure(self) -> np.ndarray:
        from rl_stl.dynamics.quadrotor_fsyn import fsyn_state_from_mujoco

        return fsyn_state_from_mujoco(self.x2, self.sim.data.qpos, self.sim.data.qvel) - self._offset()

    def step(self, x_hat: np.ndarray, u_hat: np.ndarray, u_cmd: np.ndarray) -> np.ndarray:
        off = self._offset()
        h = self.sim.dt
        for j in range(self.n_sub):
            tau = j * h
            p_ref = x_hat[:3] + x_hat[3:] * tau + 0.5 * u_hat * tau**2 + off[:3]
            v_ref = x_hat[3:] + u_hat * tau
            self.sim.step_tracking(p_ref, v_ref, u_hat)
            x = self.measure()
            p = self.s.task.dim  # tracking error on the STL signal axes
            self.max_err = max(self.max_err, float(np.max(np.abs(x[:p] + off[:p] - p_ref[:p]))))
            q = self.sim.data.qpos[3:7]
            tilt = math.degrees(math.acos(max(-1.0, min(1.0, 1 - 2 * (q[1] ** 2 + q[2] ** 2)))))
            self.max_tilt = max(self.max_tilt, tilt)
        return self.measure()

    def info(self) -> dict:
        return {"max_tracking_error_sim": self.max_err, "max_tilt_deg": self.max_tilt}


@dataclasses.dataclass(frozen=True)
class X2System:
    spec: str = "theta"
    levels: tuple[float, ...] = (0.05, 0.10, 0.15)  # constant planar tube eps [m]
    cfg: X2PatrolConfig = X2PatrolConfig()
    name: str = "x2"

    @property
    def dt(self) -> float:
        return self.cfg.dt

    def build(self, level: int) -> PatrolSetup:
        c = self.cfg
        cube = PatrolTask(spec=self.spec).dim == 3
        task = _task(self.spec, c, c.altitude_plus if cube else c.altitude)
        model = zoh_double_integrator(3, c.dt)
        eps = self.levels[level]
        n, m = 6, 3
        ws = task.workspace
        if cube:  # z is a patrol axis: planar bounds and PD on z, tube on x, y, z
            tube = constant_tube(n, m, np.array([eps, eps, eps, 0, 0, 0]), n_max=max(200, 4 * c.n_plan))
            x_lo = np.array([-ws, -ws, -ws, -c.v_max_xy, -c.v_max_xy, -c.v_max_xy])
            u_hi = np.array([c.a_max_xy] * 3)
        else:
            tube = constant_tube(n, m, np.array([eps, eps, 0, 0, 0, 0]), n_max=max(200, 4 * c.n_plan))
            x_lo = np.array([-ws, -ws, -c.z_band, -c.v_max_xy, -c.v_max_xy, -c.v_max_z])
            u_hi = np.array([c.a_max_xy, c.a_max_xy, c.a_max_z])
        x_hi = -x_lo
        u_lo = -u_hi
        plane = ("x", "y", "z") if cube else ("x", "y")
        C = np.eye(6)[:len(plane)]
        problem = build_patrol_problem(
            task, model, C, x_lo, x_hi, u_lo, u_hi, tube, c.delta, c.n_plan,
            {"x": ((0, 3), 0), "y": ((1, 4), 1), "z": ((2, 5), 2)}, plane,
            {"x": 0, "y": 1, "z": 2},
            np.array([c.track_pos] * 3 + [c.track_vel] * 3), c.gamma0, c.effort_weight)
        if cube:
            nominal = PatrolPD(tuple(AxisPD(i, 3 + i, i, c.kp, c.kd) for i in range(3)),
                               task.waypoints(), u_lo, u_hi, task.waypoint_tol)
        else:
            nominal = PatrolPD((AxisPD(0, 3, 0, c.kp, c.kd), AxisPD(1, 4, 1, c.kp, c.kd)),
                               task.waypoints(), u_lo, u_hi, task.waypoint_tol,
                               regulated=((AxisPD(2, 5, 2, c.kp_z, c.kd_z), 0.0),))
        return PatrolSetup(task, model, problem, np.zeros((3, 6)), tube, nominal, np.zeros(3))

    def descriptor(self, level: int) -> np.ndarray:
        return _descriptor(self.spec, self.levels[level], self.cfg)

    def make_sim(self, setup: PatrolSetup, seed: int, n_steps: int, disturbance: str = "none"):
        return X2Sim(setup, self.cfg, seed)

    def random_start(self, rng, setup: PatrolSetup) -> np.ndarray:
        p = _ring_position(rng, setup.task)
        v = _start_velocity(rng, p, 0.4 * self.cfg.v_max_xy)
        if setup.task.dim == 3:
            return np.concatenate([p, v])
        return np.concatenate([p, [0.0], v, [0.0]])

    def eval_start(self, rng, setup: PatrolSetup) -> np.ndarray:
        p = _ring_position(rng, setup.task, 1.55, 2.2)
        return np.concatenate([p, np.zeros(6 - len(p))])


# ===================================================================== blimp
@dataclasses.dataclass(frozen=True)
class BlimpPatrolConfig:
    dt: float = 0.5  # NAHS RTA rate
    n_plan: int = 12  # NAHS N
    delta: float = 0.02  # NAHS zeta (sim)
    altitude: float = 1.5
    v_max_xy: float = 0.5
    v_max_z: float = 0.3
    z_band: float = 0.5
    psi_band: float = 0.5
    w_max: float = 0.3
    input_scale: float = 0.4  # NAHS alpha
    kp: float = 0.09  # nominal PD (N/m), NAHS Kp on the planar axes
    kd: float = 0.15
    kp_z: float = 0.1
    kd_z: float = 0.4
    kp_psi: float = 0.003
    kd_psi: float = 0.009
    track_pos: float = 1.0
    track_vel: float = 0.5
    track_psi: float = 0.1
    gamma0: float = 0.5  # NAHS gamma0
    n_cost: int | None = 4  # NAHS ell = 4
    effort_weight: float = 0.0
    tracker_hz: float = 50.0
    # tracker gains (world PD on the f_syn reference + feedforward)
    tk_p: float = 0.08
    tk_d: float = 0.2
    tk_pz: float = 0.15
    tk_dz: float = 0.3
    tk_ppsi: float = 0.004
    tk_dpsi: float = 0.006
    altitude_plus: float = 2.75  # cube specs: centre of S+ above the floor
    pred_offset: float = 0.05  # _u specs: half-width uncertainty [m]
    pred_rotation: float = 0.05  # _u specs: rotation uncertainty [rad]
    nominal_predicates: bool = False  # _u specs ablation: plan/check with the exact nominal predicates


class BlimpSim:
    """MuJoCo blimp; tracker at ``tracker_hz`` on the ZOH-propagated ``f_syn`` reference."""

    def __init__(self, setup: PatrolSetup, cfg: BlimpPatrolConfig, seed: int):
        from rl_stl.envs.blimp.model import BlimpParams, load_blimp_model

        self.s, self.cfg = setup, cfg
        self.P = BlimpParams()
        self.model = load_blimp_model(self.P)
        self.data = mujoco.MjData(self.model)
        self.bid = self.model.body("blimp").id
        self._sid = self.model.site("thrust_point").id
        self.n_sub = int(round(cfg.dt / self.P.timestep))
        self.n_ctrl = int(round(1.0 / cfg.tracker_hz / self.P.timestep))
        self.max_err = 0.0
        self.max_tilt = 0.0
        # f_syn at the tracker period propagates the reference between planning points (exact ZOH)
        from rl_stl.envs.blimp.model import blimp_fsyn

        sub = blimp_fsyn(self.P.timestep * self.n_ctrl, self.P)
        self._Ah, self._Bh = sub.A, sub.B

    def reset(self, x0: np.ndarray) -> np.ndarray:
        mujoco.mj_resetData(self.model, self.data)
        x0 = np.asarray(x0, float)
        self.data.qpos[:3] = [x0[0], x0[1], x0[2] + self.s.task.altitude]
        self.data.qpos[3:7] = [math.cos(x0[3] / 2), 0, 0, math.sin(x0[3] / 2)]
        self.data.qvel[:3] = x0[4:7]
        self.data.qvel[5] = x0[7]
        mujoco.mj_forward(self.model, self.data)
        return self.measure()

    def measure(self) -> np.ndarray:
        from rl_stl.envs.blimp.model import state_from_mujoco

        return state_from_mujoco(self.data.qpos, self.data.qvel, self.s.task.altitude)

    def _control(self, x_ref: np.ndarray, u_ff: np.ndarray) -> np.ndarray:
        from rl_stl.envs.blimp.model import motor_command

        c, x = self.cfg, self.measure()
        e = x_ref - x
        e[3] = math.atan2(math.sin(e[3]), math.cos(e[3]))
        # Velocity damping is measured at the thrust point (gondola), collocated with the lateral
        # force: damping the CB velocity instead pumps energy into the swing mode (non-collocated).
        vel = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_SITE, self._sid, vel, 0)
        e_v_g = x_ref[4:6] - vel[3:5]
        f_w = u_ff[:2] + c.tk_p * e[:2] + c.tk_d * e_v_g
        psi = x[3]
        cs, sn = math.cos(psi), math.sin(psi)
        f_b = np.array([cs * f_w[0] + sn * f_w[1], -sn * f_w[0] + cs * f_w[1]])
        f_z = u_ff[2] + c.tk_pz * e[2] + c.tk_dz * e[6]
        tau = u_ff[3] + c.tk_ppsi * e[3] + c.tk_dpsi * e[7]
        return motor_command(np.array([f_b[0], f_b[1], f_z, tau]), self.P)

    def step(self, x_hat: np.ndarray, u_hat: np.ndarray, u_cmd: np.ndarray) -> np.ndarray:
        from rl_stl.envs.blimp.model import buoyancy_wrench

        x_ref = np.asarray(x_hat, float).copy()
        ctrl = None
        for j in range(self.n_sub):
            if j % self.n_ctrl == 0:
                if j > 0:
                    x_ref = self._Ah @ x_ref + self._Bh @ u_hat
                ctrl = self._control(x_ref, u_hat)
            self.data.ctrl[:] = ctrl
            self.data.xfrc_applied[self.bid] = buoyancy_wrench(self.data, self.model, self.P)
            mujoco.mj_step(self.model, self.data)
            if j % self.n_ctrl == self.n_ctrl - 1:
                x = self.measure()
                p = self.s.task.dim  # tracking error on the STL signal axes
                self.max_err = max(self.max_err, float(np.max(np.abs(x[:p] - x_ref[:p]))))
                q = self.data.qpos[3:7]
                tilt = math.degrees(math.acos(max(-1.0, min(1.0, 1 - 2 * (q[1] ** 2 + q[2] ** 2)))))
                self.max_tilt = max(self.max_tilt, tilt)
        return self.measure()

    def info(self) -> dict:
        return {"max_tracking_error_sim": self.max_err, "max_tilt_deg": self.max_tilt}


@dataclasses.dataclass(frozen=True)
class BlimpSystem:
    spec: str = "theta"
    levels: tuple[float, ...] = (0.05, 0.10, 0.15)  # constant planar tube eps [m] (NAHS: 0.07-0.18)
    cfg: BlimpPatrolConfig = BlimpPatrolConfig()
    name: str = "blimp"

    @property
    def dt(self) -> float:
        return self.cfg.dt

    def build(self, level: int) -> PatrolSetup:
        from rl_stl.envs.blimp.model import BlimpParams, blimp_fsyn

        c = self.cfg
        cube = PatrolTask(spec=self.spec).dim == 3
        task = _task(self.spec, c, c.altitude_plus if cube else c.altitude)
        model = blimp_fsyn(c.dt, BlimpParams())
        eps = self.levels[level]
        tube = constant_tube(8, 4, np.array([eps, eps, eps if cube else 0, 0, 0, 0, 0, 0]),
                             n_max=max(200, 4 * c.n_plan))
        ws = task.workspace
        z_lo = ws if cube else c.z_band
        x_lo = np.array([-ws, -ws, -z_lo, -c.psi_band, -c.v_max_xy, -c.v_max_xy, -c.v_max_z, -c.w_max])
        x_hi = -x_lo
        # NAHS U (rta_fixed.py:343-346) scaled by alpha; ENU: f_z up positive
        u_hi = c.input_scale * np.array([0.04752, 0.04752, 0.0908, 0.0005])
        u_lo = c.input_scale * np.array([-0.04752, -0.04752, -0.0366, -0.0005])
        plane = ("x", "y", "z") if cube else ("x", "y")
        C = np.eye(8)[:len(plane)]
        problem = build_patrol_problem(
            task, model, C, x_lo, x_hi, u_lo, u_hi, tube, c.delta, c.n_plan,
            {"x": ((0, 4), 0), "y": ((1, 5), 1), "z": ((2, 6), 2), "yaw": ((3, 7), 3)}, plane,
            {"x": 0, "y": 1, "z": 2, "yaw": 3},
            np.array([c.track_pos, c.track_pos, c.track_pos, c.track_psi,
                      c.track_vel, c.track_vel, c.track_vel, c.track_psi]),
            c.gamma0, c.effort_weight, c.n_cost)
        yaw = (AxisPD(3, 7, 3, c.kp_psi, c.kd_psi), 0.0)
        z_pd = AxisPD(2, 6, 2, c.kp_z, c.kd_z)
        if cube:  # z patrols with the altitude gains
            nominal = PatrolPD((AxisPD(0, 4, 0, c.kp, c.kd), AxisPD(1, 5, 1, c.kp, c.kd), z_pd),
                               task.waypoints(), u_lo, u_hi, task.waypoint_tol, regulated=(yaw,))
        else:
            nominal = PatrolPD((AxisPD(0, 4, 0, c.kp, c.kd), AxisPD(1, 5, 1, c.kp, c.kd)),
                               task.waypoints(), u_lo, u_hi, task.waypoint_tol,
                               regulated=((z_pd, 0.0), yaw))
        return PatrolSetup(task, model, problem, np.zeros((4, 8)), tube, nominal, np.zeros(4))

    def descriptor(self, level: int) -> np.ndarray:
        return _descriptor(self.spec, self.levels[level], self.cfg)

    def make_sim(self, setup: PatrolSetup, seed: int, n_steps: int, disturbance: str = "none"):
        return BlimpSim(setup, self.cfg, seed)

    def random_start(self, rng, setup: PatrolSetup) -> np.ndarray:
        p = _ring_position(rng, setup.task)
        v = _start_velocity(rng, p, 0.3 * self.cfg.v_max_xy)
        if setup.task.dim == 3:
            v[2] = np.clip(v[2], -self.cfg.v_max_z, self.cfg.v_max_z)
            return np.concatenate([p, [0.0], v, [0.0]])
        return np.concatenate([p, [0.0, 0.0], v, [0.0, 0.0]])

    def eval_start(self, rng, setup: PatrolSetup) -> np.ndarray:
        # NAHS simulation starts at (1.5, 1.5, -1.5) NED
        p = _ring_position(rng, setup.task, 1.55, 2.2)
        return np.concatenate([p, np.zeros(8 - len(p))])


def get_system(name: SystemName, spec: str = "theta", nominal_predicates: bool = False, **kw):
    """``nominal_predicates``: ``_u`` spec ablation (plan and check with the exact nominal predicates)."""
    cls = {"di": DISystem, "x2": X2System, "blimp": BlimpSystem}.get(name)
    if cls is None:
        raise ValueError(name)
    if nominal_predicates:
        if not spec_uncertain(spec):
            raise ValueError("nominal_predicates is an ablation of the _u specs")
        cfg = kw.pop("cfg", cls.__dataclass_fields__["cfg"].default)
        kw["cfg"] = dataclasses.replace(cfg, nominal_predicates=True)
    return cls(spec=spec, **kw)


# ================================================================= closed loop
def run_episode(system, setup: PatrolSetup, sup, x0: np.ndarray, seconds: float, seed: int,
                disturbance: str | None = None) -> dict:
    """Closed loop: supervisor at the planning rate, ``make_sim`` below it."""
    from rl_stl.supervisor.receding import realised_robustness

    n_steps = int(round(seconds / system.dt))
    sim = system.make_sim(setup, seed, n_steps, **({} if disturbance is None else {"disturbance": disturbance}))
    x = sim.reset(np.asarray(x0, float))
    init = sup.initial_plan(x)
    X, U = [x.copy()], []
    for k in range(n_steps):
        u, rec = sup.step(k, x)
        x_hat = sup.plan.state(k, setup.problem.terminal, setup.problem.model)
        u_hat = sup.plan.input(k, setup.problem.terminal, setup.problem.model)
        x = sim.step(x_hat, u_hat, u)
        X.append(x.copy())
        U.append(u)
    X, U = np.array(X), np.array(U)
    rho = realised_robustness(setup.problem, X @ setup.problem.C_out.T)
    return {"X": X, "U": U, "rho_windows": rho, "init": init, "steps": sup.step_log,
            "solves": sup.solve_log, "sim_info": sim.info()}
