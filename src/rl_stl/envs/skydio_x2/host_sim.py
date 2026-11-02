"""Host-side (C MuJoCo) X2 simulation driven through the CTBR interface.

Used for single evaluation episodes, synthesis/supervisor experiments and mismatch
measurement, where a host-side ``stlpy`` solver must stay outside the JIT-compiled
MJX batch. The per-physics-step controller (tracker + rate loop + mixer) is a
jitted JAX function, so the control law is bit-for-bit the one used in the MJX env
(up to float32/float64 differences).
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import mujoco
import numpy as np

from rl_stl.envs.skydio_x2 import frames
from rl_stl.envs.skydio_x2.ctbr import CTBRConfig, ctbr_motor_thrusts, make_ctbr_params
from rl_stl.envs.skydio_x2.model import load_x2_model, x2_params_from_model
from rl_stl.envs.skydio_x2.tracker import TrackerConfig, make_tracker_params, tracker_action


@dataclasses.dataclass
class StepLog:
    action: np.ndarray
    motors: np.ndarray
    wrench_realized: np.ndarray
    saturated: bool


class X2HostSim:
    def __init__(
        self,
        ctbr_cfg: CTBRConfig = CTBRConfig(),
        tracker_cfg: TrackerConfig = TrackerConfig(),
        fluid: bool = True,
    ):
        self.model = load_x2_model(fluid=fluid)
        self.data = mujoco.MjData(self.model)
        self.x2 = x2_params_from_model(self.model)
        self.ctbr = make_ctbr_params(self.x2, ctbr_cfg)
        self.tracker = make_tracker_params(self.x2.mass, self.x2.gravity, tracker_cfg)
        com = jnp.asarray(self.x2.com_body)
        ctbr, tracker = self.ctbr, self.tracker

        def track(qpos, qvel, p_ref, v_ref, a_ref):
            quat, w = qpos[3:7], qvel[3:6]
            p_com = qpos[:3] + frames.rotate(quat, com)
            v_com = frames.point_velocity_world(quat, qvel[:3], w, com)
            action = tracker_action(quat, p_com, v_com, p_ref, v_ref, a_ref, tracker, ctbr)
            motors, info = ctbr_motor_thrusts(action, w, ctbr)
            return action, motors, info

        def act(qvel, action):
            return ctbr_motor_thrusts(action, qvel[3:6], ctbr)

        self._track = jax.jit(track)
        self._act = jax.jit(act)

    @property
    def dt(self) -> float:
        return float(self.model.opt.timestep)

    def reset(self, qpos: np.ndarray, qvel: np.ndarray) -> None:
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel
        mujoco.mj_forward(self.model, self.data)

    def step_tracking(self, p_ref, v_ref, a_ref) -> StepLog:
        action, motors, info = self._track(
            jnp.asarray(self.data.qpos), jnp.asarray(self.data.qvel),
            jnp.asarray(p_ref), jnp.asarray(v_ref), jnp.asarray(a_ref),
        )
        return self._apply(action, motors, info)

    def step_action(self, action) -> StepLog:
        motors, info = self._act(jnp.asarray(self.data.qvel), jnp.asarray(action))
        return self._apply(jnp.asarray(action), motors, info)

    def _apply(self, action, motors, info) -> StepLog:
        self.data.ctrl[:] = np.asarray(motors)
        mujoco.mj_step(self.model, self.data)
        return StepLog(
            action=np.asarray(action),
            motors=np.asarray(motors),
            wrench_realized=np.asarray(info.wrench_realized),
            saturated=bool(info.saturated),
        )
