"""Rotation and frame conventions shared by the X2 environment, controllers and tests.

Conventions (identical to MuJoCo, verified in ``tests/test_frames.py``):

- Quaternions are ``q = [w, x, y, z]`` (scalar first), unit norm, and rotate
  **body -> world**: ``v_world = R(q) @ v_body``.
- World frame is ENU-like with ``+z`` up and gravity ``[0, 0, -9.81]``.
- Body frame of the X2: ``+x`` forward (towards rotors 3/4), ``+y`` left, ``+z``
  up along the thrust direction.
- Free-joint ``qvel[0:3]`` is the **world-frame** linear velocity of the body
  frame origin; ``qvel[3:6]`` is the **body-frame** angular velocity (this is what
  the gyro measures).
- Euler angles, when used for logging only, are intrinsic Z-Y-X (yaw, pitch, roll).

All functions are pure ``jax.numpy`` and work under ``jit``/``vmap``; they also
accept NumPy arrays.
"""

from __future__ import annotations

import jax.numpy as jnp


def quat_to_rotmat(q: jnp.ndarray) -> jnp.ndarray:
    """Rotation matrix (body -> world) from a unit quaternion ``[w, x, y, z]``."""
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    r00 = 1 - 2 * (y * y + z * z)
    r01 = 2 * (x * y - w * z)
    r02 = 2 * (x * z + w * y)
    r10 = 2 * (x * y + w * z)
    r11 = 1 - 2 * (x * x + z * z)
    r12 = 2 * (y * z - w * x)
    r20 = 2 * (x * z - w * y)
    r21 = 2 * (y * z + w * x)
    r22 = 1 - 2 * (x * x + y * y)
    return jnp.stack(
        [jnp.stack([r00, r01, r02], -1), jnp.stack([r10, r11, r12], -1), jnp.stack([r20, r21, r22], -1)],
        -2,
    )


def rotmat_to_quat(R: jnp.ndarray) -> jnp.ndarray:
    """Unit quaternion ``[w, x, y, z]`` with ``w >= 0`` from a rotation matrix (branch-free)."""
    m00, m11, m22 = R[..., 0, 0], R[..., 1, 1], R[..., 2, 2]
    w = 0.5 * jnp.sqrt(jnp.maximum(0.0, 1 + m00 + m11 + m22))
    x = 0.5 * jnp.sqrt(jnp.maximum(0.0, 1 + m00 - m11 - m22))
    y = 0.5 * jnp.sqrt(jnp.maximum(0.0, 1 - m00 + m11 - m22))
    z = 0.5 * jnp.sqrt(jnp.maximum(0.0, 1 - m00 - m11 + m22))
    x = jnp.copysign(x, R[..., 2, 1] - R[..., 1, 2])
    y = jnp.copysign(y, R[..., 0, 2] - R[..., 2, 0])
    z = jnp.copysign(z, R[..., 1, 0] - R[..., 0, 1])
    q = jnp.stack([w, x, y, z], -1)
    return q / jnp.linalg.norm(q, axis=-1, keepdims=True)


def quat_mul(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """Hamilton product ``a * b`` (apply ``b`` first, then ``a``)."""
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return jnp.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        -1,
    )


def quat_conj(q: jnp.ndarray) -> jnp.ndarray:
    return q * jnp.array([1.0, -1.0, -1.0, -1.0])


def quat_from_axis_angle(axis: jnp.ndarray, angle: jnp.ndarray) -> jnp.ndarray:
    axis = axis / jnp.linalg.norm(axis, axis=-1, keepdims=True)
    half = 0.5 * jnp.asarray(angle)[..., None]
    return jnp.concatenate([jnp.cos(half), jnp.sin(half) * axis], -1)


def quat_from_euler_zyx(roll: jnp.ndarray, pitch: jnp.ndarray, yaw: jnp.ndarray) -> jnp.ndarray:
    """Quaternion for intrinsic yaw (z), then pitch (y), then roll (x)."""
    ez = jnp.array([0.0, 0.0, 1.0])
    ey = jnp.array([0.0, 1.0, 0.0])
    ex = jnp.array([1.0, 0.0, 0.0])
    return quat_mul(
        quat_mul(quat_from_axis_angle(ez, yaw), quat_from_axis_angle(ey, pitch)),
        quat_from_axis_angle(ex, roll),
    )


def rotmat_to_euler_zyx(R: jnp.ndarray) -> jnp.ndarray:
    """``[roll, pitch, yaw]`` for ``R = Rz(yaw) Ry(pitch) Rx(roll)`` (logging only; singular at |pitch|=pi/2)."""
    roll = jnp.arctan2(R[..., 2, 1], R[..., 2, 2])
    pitch = -jnp.arcsin(jnp.clip(R[..., 2, 0], -1.0, 1.0))
    yaw = jnp.arctan2(R[..., 1, 0], R[..., 0, 0])
    return jnp.stack([roll, pitch, yaw], -1)


def rotate(q: jnp.ndarray, v_body: jnp.ndarray) -> jnp.ndarray:
    """Body-frame vector -> world frame."""
    return jnp.einsum("...ij,...j->...i", quat_to_rotmat(q), v_body)


def rotate_inv(q: jnp.ndarray, v_world: jnp.ndarray) -> jnp.ndarray:
    """World-frame vector -> body frame."""
    return jnp.einsum("...ji,...j->...i", quat_to_rotmat(q), v_world)


def tilt_cosine(q: jnp.ndarray) -> jnp.ndarray:
    """cos(angle between body z and world z) = R[2, 2]."""
    return 1 - 2 * (q[..., 1] ** 2 + q[..., 2] ** 2)


def skew(v: jnp.ndarray) -> jnp.ndarray:
    x, y, z = v[..., 0], v[..., 1], v[..., 2]
    o = jnp.zeros_like(x)
    return jnp.stack(
        [jnp.stack([o, -z, y], -1), jnp.stack([z, o, -x], -1), jnp.stack([-y, x, o], -1)], -2
    )


def vee(S: jnp.ndarray) -> jnp.ndarray:
    return jnp.stack([S[..., 2, 1], S[..., 0, 2], S[..., 1, 0]], -1)


def point_velocity_world(
    q: jnp.ndarray, v_origin_world: jnp.ndarray, w_body: jnp.ndarray, r_body: jnp.ndarray
) -> jnp.ndarray:
    """World velocity of a body-fixed point at ``r_body`` from free-joint velocities."""
    return v_origin_world + rotate(q, jnp.cross(w_body, r_body))
