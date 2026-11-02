"""MuJoCo blimp ``f_sim`` matched to the NAHS ``BlimpSim`` model, and its ``f_syn``.

Parameters are those that drive the NAHS experiments (`gtmab_model/utilities.py`, see
``docs/blimp_model.md``): Fossen 6-DOF model about the centre of buoyancy (CB) with the
centre of gravity (CG) ``r_gb = 0.05 m`` below it, neutral buoyancy, added mass at the CB,
linear damping at the CB, thrust applied at the gondola ``d_vt = 0.295 m`` below the CB.

MuJoCo realisation (world ENU, z up; body frame x forward, z up):

* free joint whose origin is the CB; body inertial at ``(0, 0, -r_gb)`` with ``m_RB`` and the
  CG inertia ``I``;
* added mass ``(m_Ax, m_Ay, m_Az)`` as **armature** of the three translational dofs (they act
  at the joint origin = CB and carry no gravity, as in Fossen's ``M_A``). The free joint's
  translational dofs are world-frame, so this equals the body-frame added mass up to the
  small roll/pitch angles (``m_Ax = m_Ay`` makes it exact under yaw);
* linear damping ``D_v`` / ``D_omega`` as dof damping (translational at the CB, rotational in
  the body frame); MuJoCo's own fluid model is disabled;
* buoyancy ``m_RB g`` upward at the CB, applied every step through ``xfrc_applied``
  (:func:`buoyancy_wrench`), which MuJoCo applies at the CoM, hence the added torque;
* six thrusters on the gondola (four lateral, two vertical) as site actuators; the
  generalised input ``u = (f_x, f_y, f_z, tau_z)`` (body frame, at the gondola) is mapped to
  them by :func:`allocation`.

``f_syn`` (:func:`blimp_fsyn`) is the NAHS linearisation about hover with zero yaw, restricted
to its four decoupled blocks ``(v_x, p_x)``, ``(v_y, p_y)``, ``(v_z, p_z)``, ``(omega_z, psi)``
(roll/pitch are driven by the lateral forces but feed back into nothing in that model and are
unconstrained, as in NAHS); ZOH-discretised at the planning step.
"""

from __future__ import annotations

import dataclasses

import mujoco
import numpy as np
import scipy.linalg

from rl_stl.dynamics.linear import DiscreteLinearModel

F_SIM_VERSION = "blimp-mjcf-v1"
F_SYN_VERSION = "blimp-fsyn-nahs-lin-v1"


@dataclasses.dataclass(frozen=True)
class BlimpParams:
    """NAHS ``BlimpSim`` constants (``utilities.py`` line numbers in ``docs/blimp_model.md``)."""

    m_rb: float = 0.1249  # rigid-body mass [kg]
    inertia: tuple[float, float, float] = (0.005821, 0.005821, 0.005821)  # about the CG [kg m^2]
    r_gb: float = 0.05  # CG below CB [m]
    d_vt: float = 0.295  # thrust point below CB [m] (H_env/2 + H_gondola)
    added_mass: tuple[float, float, float] = (0.0466, 0.0466, 0.0545)  # at CB [kg]
    damping_v: tuple[float, float, float] = (0.0125, 0.0125, 0.048)  # [N s/m]
    damping_w: tuple[float, float, float] = (0.000862, 0.000862, 0.000862)  # [N m s]
    gravity: float = 9.8
    envelope_radii: tuple[float, float, float] = (0.385, 0.385, 0.255)  # D_env/2, D_env/2, H_env/2
    gondola_half: tuple[float, float, float] = (0.06, 0.06, 0.02)
    thruster_arm: float = 0.05  # lateral thruster offset from the gondola centre [m]
    motor_lateral_max: float = 0.0594  # per motor, both directions [N]
    motor_up_max: float = 0.0454
    motor_down_max: float = 0.0183
    timestep: float = 0.0025  # 400 Hz (NAHS simulator node)


def mjcf(p: BlimpParams = BlimpParams(), arena: float = 2.5, ceiling: float = 3.0) -> str:
    rx, ry, rz = p.envelope_radii
    gx, gy, gz = p.gondola_half
    a, zt = p.thruster_arm, -p.d_vt
    lat, up, dn = p.motor_lateral_max, p.motor_up_max, p.motor_down_max
    ix, iy, iz = p.inertia
    return f"""
<mujoco model="nahs_blimp">
  <compiler angle="radian"/>
  <option timestep="{p.timestep}" gravity="0 0 {-p.gravity}" density="0" viscosity="0" integrator="implicitfast"/>
  <visual><global offwidth="960" offheight="720"/></visual>
  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1=".85 .85 .85" rgb2=".75 .75 .75" width="512" height="512"/>
    <material name="grid" texture="grid" texrepeat="10 10"/>
  </asset>
  <worldbody>
    <light pos="0 0 {ceiling + 2}" dir="0 0 -1" diffuse=".8 .8 .8"/>
    <geom name="floor" type="plane" size="{arena + 1} {arena + 1} .1" material="grid" contype="0" conaffinity="0"/>
    <camera name="overview" pos="0 {-2.2 * arena} {2.2 * arena}" xyaxes="1 0 0 0 .7 .7"/>
    <body name="blimp" pos="0 0 1.5">
      <freejoint name="root"/>
      <inertial pos="0 0 {-p.r_gb}" mass="{p.m_rb}" diaginertia="{ix} {iy} {iz}"/>
      <geom name="envelope" type="ellipsoid" size="{rx} {ry} {rz}" rgba=".9 .9 .95 .5" contype="0" conaffinity="0" mass="0"/>
      <geom name="gondola" type="box" pos="0 0 {zt + gz}" size="{gx} {gy} {gz}" rgba=".2 .2 .25 1" contype="0" conaffinity="0" mass="0"/>
      <site name="cb" pos="0 0 0" size=".01"/>
      <site name="thrust_point" pos="0 0 {zt}" size=".01"/>
      <site name="t1" pos="0 {a} {zt}" zaxis="1 0 0" size=".008"/>
      <site name="t2" pos="0 {-a} {zt}" zaxis="1 0 0" size=".008"/>
      <site name="t3" pos="{a} 0 {zt}" zaxis="0 1 0" size=".008"/>
      <site name="t4" pos="{-a} 0 {zt}" zaxis="0 1 0" size=".008"/>
      <site name="t5" pos="0 {a} {zt}" zaxis="0 0 1" size=".008"/>
      <site name="t6" pos="0 {-a} {zt}" zaxis="0 0 1" size=".008"/>
    </body>
  </worldbody>
  <actuator>
    <motor name="m1" site="t1" gear="0 0 1 0 0 0" ctrllimited="true" ctrlrange="{-lat} {lat}"/>
    <motor name="m2" site="t2" gear="0 0 1 0 0 0" ctrllimited="true" ctrlrange="{-lat} {lat}"/>
    <motor name="m3" site="t3" gear="0 0 1 0 0 0" ctrllimited="true" ctrlrange="{-lat} {lat}"/>
    <motor name="m4" site="t4" gear="0 0 1 0 0 0" ctrllimited="true" ctrlrange="{-lat} {lat}"/>
    <motor name="m5" site="t5" gear="0 0 1 0 0 0" ctrllimited="true" ctrlrange="{-dn} {up}"/>
    <motor name="m6" site="t6" gear="0 0 1 0 0 0" ctrllimited="true" ctrlrange="{-dn} {up}"/>
  </actuator>
</mujoco>
"""


def load_blimp_model(p: BlimpParams = BlimpParams()) -> mujoco.MjModel:
    model = mujoco.MjModel.from_xml_string(mjcf(p))
    model.dof_armature[0:3] = p.added_mass
    model.dof_damping[0:3] = p.damping_v
    model.dof_damping[3:6] = p.damping_w
    return model


def allocation(p: BlimpParams = BlimpParams()) -> np.ndarray:
    """``u = M f`` with ``u = (f_x, f_y, f_z, tau_z)`` (body, at the gondola) and motor thrusts ``f``.

    Lateral pairs: t1/t2 push along +x at y = +-a (tau_z = -a f1 + a f2), t3/t4 along +y at
    x = +-a (tau_z = a f3 - a f4); t5/t6 push along +z.
    """
    a = p.thruster_arm
    return np.array([
        [1, 1, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 0],
        [0, 0, 0, 0, 1, 1],
        [-a, a, a, -a, 0, 0],
    ], dtype=float)


def motor_command(u: np.ndarray, p: BlimpParams = BlimpParams()) -> np.ndarray:
    """Minimum-norm motor thrusts for the generalised input (clipped to the motor ranges)."""
    M = allocation(p)
    f = np.linalg.pinv(M) @ np.asarray(u, float)
    lo = np.array([-p.motor_lateral_max] * 4 + [-p.motor_down_max] * 2)
    hi = np.array([p.motor_lateral_max] * 4 + [p.motor_up_max] * 2)
    return np.clip(f, lo, hi)


def buoyancy_wrench(data: mujoco.MjData, model: mujoco.MjModel, p: BlimpParams = BlimpParams()) -> np.ndarray:
    """``xfrc_applied`` row (force, torque about the CoM) of buoyancy ``m_RB g`` at the CB."""
    bid = model.body("blimp").id
    F = np.array([0.0, 0.0, p.m_rb * p.gravity])
    r = data.xpos[bid] - data.xipos[bid]  # CoM -> CB (joint origin)
    return np.concatenate([F, np.cross(r, F)])


# ------------------------------------------------------------------------ f_syn
@dataclasses.dataclass(frozen=True)
class BlimpLinear:
    """Continuous NAHS linearisation coefficients (``BlimpSim.py:54-74, 128-141``)."""

    a_v: float  # v_x, v_y: v' = -a_v v + b_v f
    b_v: float
    a_vz: float
    b_vz: float
    a_w: float  # roll/pitch rate damping
    k_restore: float  # m g r_gb / I_xy
    b_tilt: float  # (d_vt - r_gb) / I_xy
    a_wz: float
    b_wz: float


def blimp_linear(p: BlimpParams = BlimpParams()) -> BlimpLinear:
    mx, mz = p.m_rb + p.added_mass[0], p.m_rb + p.added_mass[2]
    i_xy = p.inertia[0] + p.m_rb * p.r_gb**2
    return BlimpLinear(
        a_v=p.damping_v[0] / mx, b_v=1.0 / mx, a_vz=p.damping_v[2] / mz, b_vz=1.0 / mz,
        a_w=p.damping_w[0] / i_xy, k_restore=p.m_rb * p.gravity * p.r_gb / i_xy,
        b_tilt=(p.d_vt - p.r_gb) / i_xy, a_wz=p.damping_w[2] / p.inertia[2], b_wz=1.0 / p.inertia[2])


def blimp_fsyn(dt: float, p: BlimpParams = BlimpParams()) -> DiscreteLinearModel:
    """8-state ``f_syn``: ``x = (p_x, p_y, p_z, psi, v_x, v_y, v_z, omega_z)``, ``u = (f_x, f_y, f_z, tau_z)``.

    Positions in the world frame relative to the arena centre (``p_z`` = height); exact ZOH.
    """
    L = blimp_linear(p)
    Ac = np.zeros((8, 8))
    Bc = np.zeros((8, 4))
    Ac[0:4, 4:8] = np.eye(4)
    Ac[4, 4] = Ac[5, 5] = -L.a_v
    Ac[6, 6] = -L.a_vz
    Ac[7, 7] = -L.a_wz
    Bc[4, 0] = Bc[5, 1] = L.b_v
    Bc[6, 2] = L.b_vz
    Bc[7, 3] = L.b_wz
    M = scipy.linalg.expm(np.block([[Ac, Bc], [np.zeros((4, 12))]]) * dt)
    A, B = M[:8, :8], M[:8, 8:]
    return DiscreteLinearModel(
        A=A, B=B, C=np.eye(8), D=np.zeros((8, 4)), dt=float(dt), name="blimp_fsyn_nahs",
        version=F_SYN_VERSION,
        state_names=("px", "py", "pz", "psi", "vx", "vy", "vz", "wz"),
        input_names=("fx", "fy", "fz", "tauz"))


def state_from_mujoco(qpos: np.ndarray, qvel: np.ndarray, z_ref: float = 0.0) -> np.ndarray:
    """``f_syn`` state of the CB from MuJoCo ``qpos/qvel`` (yaw from the quaternion).

    Velocities: world-frame CB velocity (``f_syn`` inputs ``f_x, f_y`` are world-frame horizontal
    forces; the tracker rotates them into the body frame), yaw rate = body z rate (MuJoCo free
    joint angular velocity is body-frame).
    """
    w, x, y, z = qpos[3:7]
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return np.array([qpos[0], qpos[1], qpos[2] - z_ref, yaw, qvel[0], qvel[1], qvel[2], qvel[5]])
