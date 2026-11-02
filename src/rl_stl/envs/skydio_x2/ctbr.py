"""Normalized collective-thrust-and-body-rate (CTBR) action interface for the X2.

Pipeline (pure ``jax.numpy``; jit/vmap-able), applied at every physics substep:

1. **Normalized command** ``a = [a_T, a_p, a_q, a_r] in [-1, 1]^4`` (clipped).
   Collective thrust moves above/below hover using the available margins::

       T_cmd = T_hover + a_T * (T_max - T_hover)   if a_T >= 0
       T_cmd = T_hover + a_T * (T_hover - T_min)   if a_T <  0

   so ``a_T = -1, 0, +1`` map to ``T_min, T_hover, T_max``. Body-rate command
   ``w_des = [a_p, a_q, a_r] * rate_max`` (body frame, rad/s).

2. **Rate loop** (P + gyroscopic feedforward, always on)::

       tau = J @ (K * (w_des - w_meas)) + w_meas x (J @ w_meas)

   With exact inertia this gives decoupled first-order rate dynamics
   ``dw/dt = K (w_des - w)`` (time constant ``1/K``), i.e. ``K`` is the rate-loop
   bandwidth in rad/s. ``J`` is the full inertia about the COM in body axes from
   the compiled MJCF (it has a non-zero ``I_xz``).

3. **Allocation** ``f = A^{-1} [T, tau_x, tau_y, tau_z]`` with the X-configuration
   matrix ``A`` built from the MJCF sites and gears (:mod:`rl_stl.envs.skydio_x2.model`).

4. **Prioritized desaturation** into ``[motor_min, motor_max]``, priority
   roll/pitch > collective thrust > yaw:

   a. roll/pitch torques are scaled down only if their motor spread alone exceeds
      the motor range (physically unavoidable);
   b. collective thrust is shifted (per-motor base clipped) so that the roll/pitch
      differential fits;
   c. yaw torque is scaled into whatever headroom remains, without moving thrust;
   d. a final clip guards against round-off.

   The realized wrench ``A @ f`` and the scale/shift factors are returned for logging.
"""

from __future__ import annotations

import dataclasses
import math
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from rl_stl.envs.skydio_x2.model import X2Params


@dataclasses.dataclass(frozen=True)
class CTBRConfig:
    rate_max: tuple[float, float, float] = (2 * math.pi, 2 * math.pi, math.pi)
    rate_gain: tuple[float, float, float] = (20.0, 20.0, 10.0)  # rad/s bandwidth per axis
    # Collective thrust range as fractions of the physical range [4*motor_min, 4*motor_max].
    # 0.0 / 1.0 = full physical range (interview default). Shrink to reserve attitude headroom.
    collective_min_frac: float = 0.0
    collective_max_frac: float = 1.0


class CTBRParams(NamedTuple):
    inertia: jnp.ndarray  # (3, 3)
    allocation: jnp.ndarray  # (4, 4)
    allocation_inv: jnp.ndarray  # (4, 4)
    motor_min: jnp.ndarray
    motor_max: jnp.ndarray
    thrust_hover: jnp.ndarray
    thrust_min: jnp.ndarray
    thrust_max: jnp.ndarray
    rate_max: jnp.ndarray  # (3,)
    rate_gain: jnp.ndarray  # (3,)


class MixInfo(NamedTuple):
    wrench_des: jnp.ndarray  # (4,) [T, tau_x, tau_y, tau_z] requested
    wrench_realized: jnp.ndarray  # (4,) after desaturation
    rp_scale: jnp.ndarray  # roll/pitch torque scale in [0, 1]
    thrust_shift: jnp.ndarray  # realized - requested collective thrust (N)
    yaw_scale: jnp.ndarray  # yaw torque scale in [0, 1]
    saturated: jnp.ndarray  # bool


def make_ctbr_params(x2: X2Params, cfg: CTBRConfig = CTBRConfig()) -> CTBRParams:
    a_inv = np.asarray(x2.allocation_inv)
    if not np.allclose(a_inv[:, 0], a_inv[0, 0], atol=1e-9):
        raise ValueError("desaturation assumes equal per-motor thrust share (symmetric X configuration)")
    full_lo, full_hi = 4 * x2.motor_min, 4 * x2.motor_max
    t_min = full_lo + cfg.collective_min_frac * (full_hi - full_lo)
    t_max = full_lo + cfg.collective_max_frac * (full_hi - full_lo)
    if not (t_min < x2.hover_thrust_total < t_max):
        raise ValueError(f"hover thrust {x2.hover_thrust_total} outside collective range [{t_min}, {t_max}]")
    f = jnp.asarray
    return CTBRParams(
        inertia=f(x2.inertia_com_body),
        allocation=f(x2.allocation),
        allocation_inv=f(a_inv),
        motor_min=f(x2.motor_min),
        motor_max=f(x2.motor_max),
        thrust_hover=f(x2.hover_thrust_total),
        thrust_min=f(t_min),
        thrust_max=f(t_max),
        rate_max=f(np.asarray(cfg.rate_max, dtype=float)),
        rate_gain=f(np.asarray(cfg.rate_gain, dtype=float)),
    )


def normalized_to_command(action: jnp.ndarray, p: CTBRParams) -> tuple[jnp.ndarray, jnp.ndarray]:
    a = jnp.clip(action, -1.0, 1.0)
    a_t = a[..., 0]
    thrust = jnp.where(
        a_t >= 0,
        p.thrust_hover + a_t * (p.thrust_max - p.thrust_hover),
        p.thrust_hover + a_t * (p.thrust_hover - p.thrust_min),
    )
    rates = a[..., 1:4] * p.rate_max
    return thrust, rates


def command_to_normalized(thrust: jnp.ndarray, rates: jnp.ndarray, p: CTBRParams) -> jnp.ndarray:
    """Inverse of :func:`normalized_to_command` (saturating)."""
    a_t = jnp.where(
        thrust >= p.thrust_hover,
        (thrust - p.thrust_hover) / (p.thrust_max - p.thrust_hover),
        (thrust - p.thrust_hover) / (p.thrust_hover - p.thrust_min),
    )
    a = jnp.concatenate([jnp.asarray(a_t)[..., None], rates / p.rate_max], -1)
    return jnp.clip(a, -1.0, 1.0)


def rate_control_torque(rate_des: jnp.ndarray, rate_meas: jnp.ndarray, p: CTBRParams) -> jnp.ndarray:
    jw = jnp.einsum("ij,...j->...i", p.inertia, rate_meas)
    accel_des = p.rate_gain * (rate_des - rate_meas)
    return jnp.einsum("ij,...j->...i", p.inertia, accel_des) + jnp.cross(rate_meas, jw)


def mix(thrust: jnp.ndarray, torque: jnp.ndarray, p: CTBRParams) -> tuple[jnp.ndarray, MixInfo]:
    """Wrench -> motor thrusts with prioritized desaturation (single sample; vmap for batches)."""
    fmin, fmax = p.motor_min, p.motor_max
    cols = p.allocation_inv
    f_rp = cols[:, 1] * torque[0] + cols[:, 2] * torque[1]
    f_yaw = cols[:, 3] * torque[2]
    base = cols[0, 0] * thrust  # per-motor share of collective thrust

    # (a) roll/pitch scaling only if the differential alone cannot fit.
    spread = jnp.max(f_rp) - jnp.min(f_rp)
    rp_scale = jnp.minimum(1.0, (fmax - fmin) / jnp.maximum(spread, 1e-12))
    f_rp = rp_scale * f_rp

    # (b) shift collective thrust so that base + f_rp fits in [fmin, fmax].
    lo = fmin - jnp.min(f_rp)
    hi = fmax - jnp.max(f_rp)
    base_new = jnp.clip(base, lo, jnp.maximum(lo, hi))

    # (c) scale yaw into the remaining headroom without moving thrust.
    g = base_new + f_rp
    pos = f_yaw > 1e-12
    neg = f_yaw < -1e-12
    safe = jnp.where(pos | neg, f_yaw, 1.0)
    limit = jnp.where(pos, (fmax - g) / safe, jnp.where(neg, (fmin - g) / safe, jnp.inf))
    yaw_scale = jnp.clip(jnp.min(limit), 0.0, 1.0)

    # (d) final guard clip.
    motors = jnp.clip(g + yaw_scale * f_yaw, fmin, fmax)

    wrench_des = jnp.concatenate([jnp.asarray(thrust)[None], torque])
    wrench_realized = p.allocation @ motors
    info = MixInfo(
        wrench_des=wrench_des,
        wrench_realized=wrench_realized,
        rp_scale=rp_scale,
        thrust_shift=(base_new - base) / cols[0, 0],
        yaw_scale=yaw_scale,
        saturated=(rp_scale < 1.0) | (jnp.abs(base_new - base) > 1e-9) | (yaw_scale < 1.0),
    )
    return motors, info


def ctbr_motor_thrusts(
    action: jnp.ndarray, rate_meas: jnp.ndarray, p: CTBRParams
) -> tuple[jnp.ndarray, MixInfo]:
    """Normalized CTBR action + measured body rates -> motor thrusts (N) for ``data.ctrl``."""
    thrust, rate_des = normalized_to_command(action, p)
    torque = rate_control_torque(rate_des, rate_meas, p)
    return mix(thrust, torque, p)
