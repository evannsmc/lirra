"""Skydio X2 simulation model ``f_sim`` from MuJoCo Menagerie.

The asset is loaded from the Menagerie checkout that MuJoCo Playground pins
(``mujoco_playground._src.mjx_env.MENAGERIE_COMMIT_SHA``). That commit
(1b86ece, 2025-11-07) predates the upstream Menagerie fix 9bac002 "fix x2 thrust
gear" (2025-12-12), in which rotors 3 and 4 had swapped yaw-drag signs. With the
unfixed gears the yaw-torque row of the allocation matrix is proportional to the
roll-torque row, so yaw cannot be controlled independently. :func:`load_x2_spec`
applies exactly the upstream fix via ``MjSpec`` and asserts the original values,
so a future Menagerie bump that already contains the fix fails loudly instead of
being patched twice. No other part of the model is modified.

Everything physical used by controllers (mass, inertia, COM, rotor geometry,
allocation matrix, motor limits) is derived from the compiled ``MjModel`` by
:func:`x2_params_from_model`, never hard-coded.
"""

from __future__ import annotations

import dataclasses
import subprocess
from pathlib import Path

import mujoco
import numpy as np

F_SIM_VERSION = "x2_menagerie-1b86ece_gearfix-9bac002_v1"

# Upstream Menagerie commit 9bac002 changed only these two yaw-drag gear entries.
_YAW_GEAR_PINNED = {"thrust1": -0.0201, "thrust2": 0.0201, "thrust3": 0.0201, "thrust4": -0.0201}
_YAW_GEAR_FIXED = {"thrust1": -0.0201, "thrust2": 0.0201, "thrust3": -0.0201, "thrust4": 0.0201}
MOTOR_NAMES = ("thrust1", "thrust2", "thrust3", "thrust4")


def menagerie_x2_dir() -> Path:
    from mujoco_playground._src import mjx_env

    mjx_env.ensure_menagerie_exists()
    return Path(str(mjx_env.MENAGERIE_PATH)) / "skydio_x2"


def menagerie_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(menagerie_x2_dir().parent), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True, timeout=10,
        ).stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        return None


def load_x2_spec(scene: bool = True, apply_gear_fix: bool = True) -> mujoco.MjSpec:
    """Menagerie X2 (optionally with ground plane scene) as an ``MjSpec``."""
    path = menagerie_x2_dir() / ("scene.xml" if scene else "x2.xml")
    spec = mujoco.MjSpec.from_file(str(path))
    if apply_gear_fix:
        for name in MOTOR_NAMES:
            act = spec.actuator(name)
            gear = np.array(act.gear, dtype=float)
            if not np.isclose(gear[5], _YAW_GEAR_PINNED[name]):
                raise RuntimeError(
                    f"X2 actuator {name} yaw gear is {gear[5]}, expected pinned value "
                    f"{_YAW_GEAR_PINNED[name]}; Menagerie changed, revisit the gear fix."
                )
            gear[5] = _YAW_GEAR_FIXED[name]
            act.gear = gear
    return spec


def load_x2_model(
    scene: bool = True,
    fluid: bool = True,
    timestep: float | None = None,
    apply_gear_fix: bool = True,
) -> mujoco.MjModel:
    """Compiled X2 ``MjModel``. ``fluid=False`` zeroes air density/viscosity (ablation only)."""
    spec = load_x2_spec(scene=scene, apply_gear_fix=apply_gear_fix)
    model = spec.compile()
    if not fluid:
        model.opt.density = 0.0
        model.opt.viscosity = 0.0
    if timestep is not None:
        model.opt.timestep = timestep
    return model


@dataclasses.dataclass(frozen=True)
class X2Params:
    """Physical parameters of ``f_sim`` extracted from the compiled model (SI units)."""

    mass: float
    gravity: float
    inertia_com_body: np.ndarray  # (3, 3) about COM, body axes
    com_body: np.ndarray  # (3,) COM in body frame
    rotor_pos_body: np.ndarray  # (4, 3) thrust site positions in body frame
    yaw_coeff: np.ndarray  # (4,) yaw torque per Newton of thrust (gear[5])
    allocation: np.ndarray  # (4, 4) [T, tau_x, tau_y, tau_z]^T = allocation @ f
    allocation_inv: np.ndarray  # (4, 4)
    motor_min: float
    motor_max: float
    hover_thrust_total: float
    timestep: float
    density: float
    viscosity: float

    @property
    def hover_thrust_per_motor(self) -> float:
        return self.hover_thrust_total / 4.0


def x2_params_from_model(model: mujoco.MjModel) -> X2Params:
    body_id = model.body("x2").id
    mass = float(model.body_subtreemass[body_id])
    rot = np.zeros(9)
    mujoco.mju_quat2Mat(rot, model.body_iquat[body_id])
    rot = rot.reshape(3, 3)
    inertia = rot @ np.diag(model.body_inertia[body_id]) @ rot.T
    com = np.array(model.body_ipos[body_id])

    rotor_pos = np.zeros((4, 3))
    yaw = np.zeros(4)
    allocation = np.zeros((4, 4))
    ctrl_lo, ctrl_hi = [], []
    for i, name in enumerate(MOTOR_NAMES):
        act = model.actuator(name)
        if model.actuator_trntype[act.id] != mujoco.mjtTrn.mjTRN_SITE:
            raise ValueError(f"{name} is not a site actuator")
        site_id = model.actuator_trnid[act.id, 0]
        if model.site_bodyid[site_id] != body_id:
            raise ValueError(f"{name} site is not on the x2 body")
        site_rot = np.zeros(9)
        mujoco.mju_quat2Mat(site_rot, model.site_quat[site_id])
        site_rot = site_rot.reshape(3, 3)
        gear = model.actuator_gear[act.id]
        force_body = site_rot @ gear[:3]
        torque_body = np.cross(model.site_pos[site_id] - com, force_body) + site_rot @ gear[3:]
        # Body wrench per unit ctrl. Collective thrust is the body-z force component.
        allocation[0, i] = force_body[2]
        allocation[1:, i] = torque_body
        rotor_pos[i] = model.site_pos[site_id]
        yaw[i] = gear[5]
        ctrl_lo.append(model.actuator_ctrlrange[act.id, 0])
        ctrl_hi.append(model.actuator_ctrlrange[act.id, 1])
    if not (np.allclose(ctrl_lo, ctrl_lo[0]) and np.allclose(ctrl_hi, ctrl_hi[0])):
        raise ValueError("motors have different control ranges")
    gravity = float(-model.opt.gravity[2])
    if np.linalg.matrix_rank(allocation) < 4:
        raise ValueError(
            f"X2 allocation matrix is rank-deficient (rank {np.linalg.matrix_rank(allocation)}); "
            "yaw gears are probably the unfixed pinned values"
        )
    return X2Params(
        mass=mass,
        gravity=gravity,
        inertia_com_body=inertia,
        com_body=com,
        rotor_pos_body=rotor_pos,
        yaw_coeff=yaw,
        allocation=allocation,
        allocation_inv=np.linalg.inv(allocation),
        motor_min=float(ctrl_lo[0]),
        motor_max=float(ctrl_hi[0]),
        hover_thrust_total=mass * gravity,
        timestep=float(model.opt.timestep),
        density=float(model.opt.density),
        viscosity=float(model.opt.viscosity),
    )


def com_state_from_qpos_qvel(
    params: X2Params, qpos: np.ndarray, qvel: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """COM position and world velocity from free-joint ``qpos``/``qvel`` (NumPy)."""
    rot = np.zeros(9)
    mujoco.mju_quat2Mat(rot, np.asarray(qpos[3:7], dtype=float))
    rot = rot.reshape(3, 3)
    p_com = qpos[:3] + rot @ params.com_body
    v_com = qvel[:3] + rot @ np.cross(qvel[3:6], params.com_body)
    return p_com, v_com
