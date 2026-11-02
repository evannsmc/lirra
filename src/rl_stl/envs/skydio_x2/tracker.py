"""Outer-loop tracker: ``f_syn`` reference (p, v, a) -> normalized CTBR action.

This lets synthesized plans and learned policies drive MuJoCo through the *same*
action interface (:mod:`rl_stl.envs.skydio_x2.ctbr`)::

    a_cmd  = a_ref + Kp (p_ref - p) + Kd (v_ref - v)               (world, COM)
    f_des  = m (a_cmd + g e_z), horizontal part limited to tilt_max
    T      = f_des . z_body                                         (projected thrust)
    z_des  = f_des / |f_des|
    w_tilt = K_att_xy * R^T (z_body x z_des)                        (reduced attitude, body frame)
    w_yaw  = K_att_z * wrap(yaw_ref - yaw) * R^T e_z                (heading, lower priority)
    action = command_to_normalized(T, w_tilt + w_yaw)

The tilt and heading errors are decoupled on purpose: the full SO(3) error
``1/2 vee(R_des^T R - R^T R_des)`` loses its yaw component and flips the sign of the
tilt correction near a 180 deg heading error, which diverged from random-yaw resets.

Pure ``jax.numpy``; single sample (vmap for batches).
"""

from __future__ import annotations

import dataclasses
import math
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from rl_stl.envs.skydio_x2 import frames
from rl_stl.envs.skydio_x2.ctbr import CTBRParams, command_to_normalized


@dataclasses.dataclass(frozen=True)
class TrackerConfig:
    kp_pos: tuple[float, float, float] = (2.0, 2.0, 3.0)
    kd_vel: tuple[float, float, float] = (2.5, 2.5, 3.5)
    k_att: tuple[float, float, float] = (6.0, 6.0, 3.0)  # (tilt, tilt, yaw) [1/s]; tilt gains must match
    tilt_max_deg: float = 35.0
    yaw_ref: float = 0.0


class TrackerParams(NamedTuple):
    mass: jnp.ndarray
    gravity: jnp.ndarray
    kp: jnp.ndarray
    kd: jnp.ndarray
    k_att: jnp.ndarray
    tan_tilt_max: jnp.ndarray
    yaw_ref: jnp.ndarray


def make_tracker_params(mass: float, gravity: float, cfg: TrackerConfig = TrackerConfig()) -> TrackerParams:
    if cfg.k_att[0] != cfg.k_att[1]:
        raise ValueError("reduced-attitude tilt law uses one gain: k_att[0] must equal k_att[1]")
    f = jnp.asarray
    return TrackerParams(
        mass=f(mass),
        gravity=f(gravity),
        kp=f(np.asarray(cfg.kp_pos, dtype=float)),
        kd=f(np.asarray(cfg.kd_vel, dtype=float)),
        k_att=f(np.asarray(cfg.k_att, dtype=float)),
        tan_tilt_max=f(math.tan(math.radians(cfg.tilt_max_deg))),
        yaw_ref=f(cfg.yaw_ref),
    )


def tracker_action(
    quat: jnp.ndarray,
    p_com: jnp.ndarray,
    v_com: jnp.ndarray,
    p_ref: jnp.ndarray,
    v_ref: jnp.ndarray,
    a_ref: jnp.ndarray,
    tp: TrackerParams,
    cp: CTBRParams,
) -> jnp.ndarray:
    a_cmd = a_ref + tp.kp * (p_ref - p_com) + tp.kd * (v_ref - v_com)
    f_des = tp.mass * (a_cmd + jnp.array([0.0, 0.0, 1.0]) * tp.gravity)
    f_z = jnp.maximum(f_des[2], 0.05 * tp.mass * tp.gravity)
    f_h = f_des[:2]
    h_norm = jnp.linalg.norm(f_h)
    h_max = f_z * tp.tan_tilt_max
    f_h = f_h * jnp.minimum(1.0, h_max / jnp.maximum(h_norm, 1e-9))
    f_des = jnp.concatenate([f_h, f_z[None]])

    R = frames.quat_to_rotmat(quat)
    thrust = jnp.maximum(jnp.dot(f_des, R[:, 2]), 0.0)

    z_des = f_des / jnp.linalg.norm(f_des)
    z_body = R[:, 2]
    w_tilt = tp.k_att[:2].mean() * (R.T @ jnp.cross(z_body, z_des))
    yaw = jnp.arctan2(R[1, 0], R[0, 0])
    yaw_err = jnp.arctan2(jnp.sin(tp.yaw_ref - yaw), jnp.cos(tp.yaw_ref - yaw))
    w_yaw = tp.k_att[2] * yaw_err * R[2, :]  # R^T e_z
    w_cmd = w_tilt + w_yaw
    return command_to_normalized(thrust, w_cmd, cp)
