"""Reduced-order synthesis model ``f_syn`` v1 for the Skydio X2: 3-D double integrator.

State ``z = [p_com (3), v_com (3)]`` in the world frame (COM position and velocity),
input ``u = a_com`` (3, world-frame COM acceleration excluding gravity), exact
zero-order-hold discretization at the synthesis step ``dt``::

    z+ = A z + B u,   A = [[I, dt I], [0, I]],   B = [[dt^2/2 I], [dt I]]

The stlpy output is ``y = [p, v, u]`` (9-D), mirroring ``stlpy.systems.DoubleIntegrator``.

Input bounds are a box ``a_min <= u <= a_max`` chosen *inside* the set of COM
accelerations the X2 can produce at quasi-steady attitude:

- thrust vector per unit mass ``f = u + g e_z`` must have tilt ``<= tilt_max``;
  the worst case is the smallest vertical component, so
  ``a_xy <= (g - a_down) tan(tilt_max) / sqrt(2)`` (both horizontal axes at the
  box corner simultaneously);
- ``|f| <= headroom * T_max / m`` so that ``1 - headroom`` of the collective range
  stays available for attitude control;
- ``g - a_down >= T_min / m``.

These are necessary conditions for tracking, not a proof: attitude/rate transients,
motor saturation and fluid drag create mismatch that is measured by
``experiments/reduced_quadrotor/fsyn_mismatch.py`` and must be represented as a
bounded disturbance before any guarantee is claimed.
"""

from __future__ import annotations

import dataclasses
import math

import mujoco
import numpy as np
from stlpy.systems import LinearSystem

from rl_stl.envs.skydio_x2.model import X2Params

FSYN_VERSION = "fsyn_di3d_v1"


@dataclasses.dataclass(frozen=True)
class FsynConfig:
    dt: float = 0.1
    tilt_max_deg: float = 25.0
    accel_up_max: float = 3.0
    accel_down_max: float = 3.0
    thrust_headroom_frac: float = 0.8
    vel_max: tuple[float, float, float] = (2.0, 2.0, 1.5)


@dataclasses.dataclass(frozen=True)
class DoubleIntegrator3D:
    dt: float
    a_min: np.ndarray  # (3,)
    a_max: np.ndarray  # (3,)
    v_max: np.ndarray  # (3,)
    version: str = FSYN_VERSION

    @property
    def A(self) -> np.ndarray:
        eye, zero = np.eye(3), np.zeros((3, 3))
        return np.block([[eye, self.dt * eye], [zero, eye]])

    @property
    def B(self) -> np.ndarray:
        eye = np.eye(3)
        return np.vstack([0.5 * self.dt**2 * eye, self.dt * eye])

    def step(self, z: np.ndarray, u: np.ndarray) -> np.ndarray:
        return self.A @ z + self.B @ u

    def rollout(self, z0: np.ndarray, us: np.ndarray) -> np.ndarray:
        """``(H+1, 6)`` states for inputs ``(H, 3)``."""
        zs = [np.asarray(z0, dtype=float)]
        for u in us:
            zs.append(self.step(zs[-1], u))
        return np.array(zs)

    def intra_step_reference(self, z: np.ndarray, u: np.ndarray, tau: float) -> tuple[np.ndarray, np.ndarray]:
        """Exact continuous position/velocity ``tau`` seconds into a ZOH interval."""
        p, v = z[:3], z[3:]
        return p + v * tau + 0.5 * u * tau**2, v + u * tau

    def stlpy_system(self) -> LinearSystem:
        eye, zero = np.eye(3), np.zeros((3, 3))
        C = np.block([[eye, zero], [zero, eye], [zero, zero]])
        D = np.vstack([zero, zero, eye])
        return LinearSystem(self.A, self.B, C, D)

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "dt": self.dt,
            "state": "p_com_world(3), v_com_world(3)",
            "input": "a_com_world(3), gravity excluded",
            "output": "y = [p, v, u] (9)",
            "a_min": self.a_min.tolist(),
            "a_max": self.a_max.tolist(),
            "v_max": self.v_max.tolist(),
        }


def accel_box_from_limits(x2: X2Params, cfg: FsynConfig) -> tuple[np.ndarray, np.ndarray]:
    g, m = x2.gravity, x2.mass
    t_min, t_max = 4 * x2.motor_min, 4 * x2.motor_max
    f_z_low = g - cfg.accel_down_max
    if f_z_low * m < t_min or f_z_low <= 0:
        raise ValueError("accel_down_max exceeds what the minimum thrust allows")
    a_xy_tilt = f_z_low * math.tan(math.radians(cfg.tilt_max_deg)) / math.sqrt(2)
    f_max = cfg.thrust_headroom_frac * t_max / m
    rem = f_max**2 - (g + cfg.accel_up_max) ** 2
    if rem <= 0:
        raise ValueError("accel_up_max exceeds the thrust headroom")
    a_xy = min(a_xy_tilt, math.sqrt(rem / 2))
    return (
        np.array([-a_xy, -a_xy, -cfg.accel_down_max]),
        np.array([a_xy, a_xy, cfg.accel_up_max]),
    )


def make_fsyn(x2: X2Params, cfg: FsynConfig = FsynConfig()) -> DoubleIntegrator3D:
    a_min, a_max = accel_box_from_limits(x2, cfg)
    return DoubleIntegrator3D(dt=cfg.dt, a_min=a_min, a_max=a_max, v_max=np.asarray(cfg.vel_max, dtype=float))


def fsyn_state_from_mujoco(x2: X2Params, qpos: np.ndarray, qvel: np.ndarray) -> np.ndarray:
    """MuJoCo free-joint state -> ``z = [p_com, v_com]`` (world)."""
    rot = np.zeros(9)
    mujoco.mju_quat2Mat(rot, np.asarray(qpos[3:7], dtype=float))
    rot = rot.reshape(3, 3)
    p_com = qpos[:3] + rot @ x2.com_body
    v_com = qvel[:3] + rot @ np.cross(qvel[3:6], x2.com_body)
    return np.concatenate([p_com, v_com])


def mujoco_state_from_fsyn(
    x2: X2Params,
    z: np.ndarray,
    quat: np.ndarray = (1.0, 0.0, 0.0, 0.0),
    w_body: np.ndarray = (0.0, 0.0, 0.0),
) -> tuple[np.ndarray, np.ndarray]:
    """``z`` plus attitude/body rates -> free-joint ``qpos (7)``, ``qvel (6)`` (exact inverse)."""
    quat = np.asarray(quat, dtype=float)
    w_body = np.asarray(w_body, dtype=float)
    rot = np.zeros(9)
    mujoco.mju_quat2Mat(rot, quat)
    rot = rot.reshape(3, 3)
    origin = z[:3] - rot @ x2.com_body
    v_origin = z[3:] - rot @ np.cross(w_body, x2.com_body)
    return np.concatenate([origin, quat]), np.concatenate([v_origin, w_body])
