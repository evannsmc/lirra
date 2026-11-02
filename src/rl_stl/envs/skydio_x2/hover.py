"""MuJoCo Playground (MJX/JAX) hover task for the Menagerie Skydio X2.

Action
    Normalized CTBR ``a = [T, p, q, r] in [-1, 1]^4`` (:mod:`.ctbr`). The action is
    held for ``n_substeps = ctrl_dt / sim_dt`` physics steps; the body-rate loop and
    the prioritized mixer run at every physics step on the measured body rates.

Observation (flat, float32; frames stated per block)
    - ``pos_err_world`` (3): ``p_target - p_com`` in world frame [m]; ``p_target`` is
      fixed (hover) or sampled per episode within ``target_range`` (waypoint variant)
    - ``vel_com_world`` (3): COM linear velocity, world frame [m/s]
    - ``rot_body_to_world`` (9): rotation matrix, row-major (encodes gravity direction and yaw)
    - ``ang_vel_body`` (3): body-frame angular velocity (gyro) [rad/s]
    - ``last_action`` (4)
    - ``context`` (3): ``[time_to_deadline / deadline, in_box_now, episode_time_fraction]``
      -- STL/task context slots (clock and instantaneous obligation predicate)
    - ``stl_window`` (``window * 3``, only if ``stl.enabled``): the last ``window``
      COM positions relative to the target, oldest first, so the windowed STL
      penalty below is a function of the observation (Markov).

Reward (task terms; each term is logged under ``reward/<name>``)
    ``pos`` = exp(-|e_p|^2 / sigma_pos^2), ``vel`` = exp(-|v|^2 / sigma_vel^2),
    ``upright`` = (1 + R_zz) / 2, ``ang_vel`` = |w|^2, ``action_rate`` = |a - a_prev|^2,
    ``alive`` = 1; ``reward = sum(scale_i * term_i)`` (penalty scales are negative).

Optional STL term (``stl.enabled``; disabled by default = task reward only)
    Specification (built with stlpy, selected by ``stl.spec``) over the signal
    ``y = p_com - p_target``:

    - ``hover_box`` (default): ``phi = G_[0, window-1] (y in box(halfwidth))`` --
      stay inside the box for the whole window;
    - ``periodic_exit``: ``phi = F_[0, window-1] (y not in box(halfwidth))`` -- leave
      the box at least once per window, i.e. never dwell inside it for ``window``
      consecutive samples. Scored at every step, this is the sliding-window form of
      ``G F_[0, window-1] (not in box)``.

    At control step ``k`` the realized window ``y_{k-window+1..k}`` is scored with the
    compiled JAX robustness and the step receives ``-stl_scale * max(0, delta - rho_k)``.
    ``rho_k`` is computed (and logged) whether or not the term is enabled, so a
    task-only policy can be checked against the same spec. Timing and credit
    assignment are explicit:

    - the term is **zero before** ``k_deadline + window - 1`` (``k_deadline =
      deadline_s / ctrl_dt``): the obligation holds from the deadline on, and only
      fully post-deadline windows are scored;
    - ``rho_k`` is a *past-time sliding-window* value: one sample that decides the
      window (an out-of-box sample for ``hover_box``, the only exit for
      ``periodic_exit``) enters the robustness of each of the ``window`` steps whose
      window contains it (credit is spread over the window, not assigned to a single
      transition);
    - no trajectory-level robustness is copied to all transitions.

Termination
    NaN/Inf in state or observation, tilt beyond ``max_tilt_deg``, COM below
    ``min_z``, or position error above ``max_pos_err``.

Floor contacts are disabled by default (``contacts=False``): the task terminates
above the floor, and disabling contacts removes collision cost from the batch.

Brax's ``BraxAutoResetWrapper`` with ``full_reset=False`` restores ``data``/``obs``
but not ``info``; the env therefore re-initializes its per-episode info whenever
``data.time == 0`` at the start of a step.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from ml_collections import config_dict
from mujoco import mjx
from mujoco_playground._src import mjx_env
from stlpy.STL import LinearPredicate

from rl_stl.envs.skydio_x2 import frames
from rl_stl.envs.skydio_x2.ctbr import CTBRConfig, ctbr_motor_thrusts, make_ctbr_params
from rl_stl.envs.skydio_x2.model import (
    F_SIM_VERSION,
    load_x2_model,
    menagerie_x2_dir,
    x2_params_from_model,
)
from rl_stl.stl.jax_robustness import compile_robustness, stl_hinge_penalty
from rl_stl.utils.config import to_dict

ENV_VERSION = "x2_hover_ctbr_v1"


@dataclasses.dataclass(frozen=True)
class ResetConfig:
    pos_range: tuple[float, float, float] = (0.5, 0.5, 0.3)
    roll_pitch_deg: float = 15.0
    yaw_range: float = math.pi
    lin_vel: float = 0.3
    ang_vel: float = 0.3


@dataclasses.dataclass(frozen=True)
class RewardScales:
    pos: float = 1.0
    vel: float = 0.2
    upright: float = 0.5
    ang_vel: float = -0.01
    action_rate: float = -0.05
    alive: float = 0.2
    stl: float = 1.0  # lambda_stl; only active when stl.enabled


@dataclasses.dataclass(frozen=True)
class RewardConfig:
    scales: RewardScales = RewardScales()
    sigma_pos: float = 0.5
    sigma_vel: float = 1.0


@dataclasses.dataclass(frozen=True)
class TerminationConfig:
    max_tilt_deg: float = 80.0
    min_z: float = 0.2
    max_pos_err: float = 3.0


@dataclasses.dataclass(frozen=True)
class STLRewardConfig:
    enabled: bool = False
    spec: str = "hover_box"  # key of STL_SPECS
    window: int = 25
    box_halfwidth: tuple[float, float, float] = (0.2, 0.2, 0.2)
    delta: float = 0.05
    deadline_s: float = 3.0


@dataclasses.dataclass(frozen=True)
class HoverEnvConfig:
    """Single source of truth for the hover env; converted to ``ml_collections`` for Playground."""

    ctrl_dt: float = 0.02
    sim_dt: float = 0.01
    episode_length: int = 500
    impl: str = "jax"
    naconmax: int = 0
    njmax: int = 0
    fluid: bool = True
    contacts: bool = False
    ctbr: CTBRConfig = CTBRConfig()
    target: tuple[float, float, float] = (0.0, 0.0, 1.5)
    # Waypoint variant: per-episode target = target + U(-target_range, target_range). Zeros = hover.
    target_range: tuple[float, float, float] = (0.0, 0.0, 0.0)
    reset: ResetConfig = ResetConfig()
    reward: RewardConfig = RewardConfig()
    termination: TerminationConfig = TerminationConfig()
    stl: STLRewardConfig = STLRewardConfig()


def to_config_dict(cfg: HoverEnvConfig) -> config_dict.ConfigDict:
    return config_dict.ConfigDict(to_dict(cfg))


def default_config() -> config_dict.ConfigDict:
    return to_config_dict(HoverEnvConfig())


def hover_box_formula(halfwidth, window: int):
    """``G_[0, window-1] (|y_i| <= h_i for i = 0,1,2)`` over ``y = p_com - p_target``."""
    preds = None
    for i, h in enumerate(halfwidth):
        e = np.zeros(3)
        e[i] = 1.0
        for pred in (LinearPredicate(e, -h), LinearPredicate(-e, -h)):  # y_i >= -h, -y_i >= -h
            preds = pred if preds is None else preds & pred
    return preds.always(0, window - 1)


def periodic_exit_formula(halfwidth, window: int):
    """``F_[0, window-1] (|y_i| >= h_i for some i)`` over ``y = p_com - p_target``.

    Robustness is ``max_{j in window} max_i (|y_{j,i}| - h_i)``: negative iff the COM was
    strictly inside the box at every sample of the window.
    """
    preds = None
    for i, h in enumerate(halfwidth):
        e = np.zeros(3)
        e[i] = 1.0
        for pred in (LinearPredicate(e, h), LinearPredicate(-e, h)):  # y_i >= h, -y_i >= h
            preds = pred if preds is None else preds | pred
    return preds.eventually(0, window - 1)


STL_SPECS = {"hover_box": hover_box_formula, "periodic_exit": periodic_exit_formula}


class X2Hover(mjx_env.MjxEnv):
    def __init__(
        self,
        config: config_dict.ConfigDict | None = None,
        config_overrides: dict[str, Any] | None = None,
    ):
        super().__init__(config or default_config(), config_overrides=config_overrides)
        cfg = self._config
        self._mj_model = load_x2_model(fluid=cfg.fluid, timestep=cfg.sim_dt)
        if not cfg.contacts:
            self._mj_model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
        self._mjx_model = mjx.put_model(self._mj_model, impl=cfg.impl)
        self._xml_path = str(menagerie_x2_dir() / "scene.xml")

        self.x2 = x2_params_from_model(self._mj_model)
        self.ctbr_params = make_ctbr_params(
            self.x2,
            CTBRConfig(
                rate_max=tuple(cfg.ctbr.rate_max),
                rate_gain=tuple(cfg.ctbr.rate_gain),
                collective_min_frac=cfg.ctbr.collective_min_frac,
                collective_max_frac=cfg.ctbr.collective_max_frac,
            ),
        )
        self._com = jnp.asarray(self.x2.com_body)
        self._target = jnp.asarray(cfg.target, dtype=jnp.float32)
        self._halfwidth = jnp.asarray(cfg.stl.box_halfwidth, dtype=jnp.float32)
        self._cos_max_tilt = math.cos(math.radians(cfg.termination.max_tilt_deg))
        self._deadline_steps = round(cfg.stl.deadline_s / cfg.ctrl_dt)
        if cfg.stl.spec not in STL_SPECS:
            raise ValueError(f"unknown stl.spec {cfg.stl.spec!r}; choose from {sorted(STL_SPECS)}")
        self.stl_formula = STL_SPECS[cfg.stl.spec](cfg.stl.box_halfwidth, cfg.stl.window)
        self._stl_rho = compile_robustness(self.stl_formula)
        self.version = {"env": ENV_VERSION, "f_sim": F_SIM_VERSION}

    # ------------------------------------------------------------------ helpers
    def _com_state(self, data: mjx.Data):
        quat, w = data.qpos[3:7], data.qvel[3:6]
        p_com = data.qpos[:3] + frames.rotate(quat, self._com)
        v_com = frames.point_velocity_world(quat, data.qvel[:3], w, self._com)
        return p_com, v_com, quat, w

    def _init_info(self, rng: jax.Array, rel_pos: jax.Array, target: jax.Array) -> dict[str, Any]:
        return {
            "rng": rng,
            "x2_target": target,
            "x2_last_action": jnp.zeros(4),
            "x2_step": jnp.zeros((), dtype=jnp.int32),
            "x2_stl_buffer": jnp.tile(rel_pos, (self._config.stl.window, 1)),
        }

    # --------------------------------------------------------------------- API
    def reset(self, rng: jax.Array) -> mjx_env.State:
        cfg = self._config
        rng, k_tgt, k_pos, k_rp, k_yaw, k_v, k_w = jax.random.split(rng, 7)
        target_range = jnp.asarray(cfg.target_range)
        target = self._target + jax.random.uniform(k_tgt, (3,), minval=-target_range, maxval=target_range)
        pos_range = jnp.asarray(cfg.reset.pos_range)
        p_com = target + jax.random.uniform(k_pos, (3,), minval=-pos_range, maxval=pos_range)
        rp = jax.random.uniform(k_rp, (2,), minval=-1.0, maxval=1.0) * math.radians(cfg.reset.roll_pitch_deg)
        yaw = jax.random.uniform(k_yaw, (), minval=-cfg.reset.yaw_range, maxval=cfg.reset.yaw_range)
        quat = frames.quat_from_euler_zyx(rp[0], rp[1], yaw)
        v_com = jax.random.uniform(k_v, (3,), minval=-cfg.reset.lin_vel, maxval=cfg.reset.lin_vel)
        w = jax.random.uniform(k_w, (3,), minval=-cfg.reset.ang_vel, maxval=cfg.reset.ang_vel)
        origin = p_com - frames.rotate(quat, self._com)
        v_origin = v_com - frames.rotate(quat, jnp.cross(w, self._com))
        qpos = jnp.concatenate([origin, quat])
        qvel = jnp.concatenate([v_origin, w])
        data = mjx_env.make_data(
            self._mj_model, qpos=qpos, qvel=qvel, impl=self._mjx_model.impl.value,
            naconmax=cfg.naconmax, njmax=cfg.njmax,
        )
        data = mjx.forward(self._mjx_model, data)
        info = self._init_info(rng, p_com - target, target)
        metrics = {
            **{f"reward/{k}": jnp.zeros(()) for k in cfg.reward.scales},
            "stl/rho_window": jnp.zeros(()),
            "stl/penalty": jnp.zeros(()),
            "ctbr/saturated_frac": jnp.zeros(()),
            "task/pos_err": jnp.linalg.norm(p_com - target),
        }
        obs = self._get_obs(data, info)
        return mjx_env.State(data, obs, jnp.zeros(()), jnp.zeros(()), metrics, info)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        cfg = self._config
        data0 = state.data
        # Re-initialize per-episode info after an auto-reset that restored only data/obs.
        first = data0.time < 0.5 * cfg.sim_dt
        p0, _, _, _ = self._com_state(data0)
        target = state.info["x2_target"]
        fresh = self._init_info(state.info["rng"], p0 - target, target)
        info = {**state.info}
        for key in ("x2_last_action", "x2_step", "x2_stl_buffer"):
            info[key] = jnp.where(first, fresh[key], state.info[key])

        action = jnp.clip(action, -1.0, 1.0)
        ctbr = self.ctbr_params

        def substep(data, _):
            motors, mix_info = ctbr_motor_thrusts(action, data.qvel[3:6], ctbr)
            data = data.replace(ctrl=motors)
            data = mjx.step(self._mjx_model, data)
            return data, mix_info.saturated

        data, saturated = jax.lax.scan(substep, data0, (), self.n_substeps)

        p_com, v_com, quat, w = self._com_state(data)
        rel = p_com - target
        step_idx = info["x2_step"] + 1
        buffer = jnp.concatenate([info["x2_stl_buffer"][1:], rel[None]], axis=0)

        R = frames.quat_to_rotmat(quat)
        terms = {
            "pos": jnp.exp(-jnp.sum(rel**2) / cfg.reward.sigma_pos**2),
            "vel": jnp.exp(-jnp.sum(v_com**2) / cfg.reward.sigma_vel**2),
            "upright": 0.5 * (1.0 + R[2, 2]),
            "ang_vel": jnp.sum(w**2),
            "action_rate": jnp.sum((action - info["x2_last_action"]) ** 2),
            "alive": jnp.ones(()),
        }
        rho = self._stl_rho(buffer)
        scored = step_idx >= self._deadline_steps + cfg.stl.window - 1
        penalty = jnp.where(scored & cfg.stl.enabled, stl_hinge_penalty(rho, cfg.stl.delta), 0.0)
        terms["stl"] = -penalty
        scales = cfg.reward.scales
        reward = sum(scales[k] * v for k, v in terms.items())

        info.update(x2_last_action=action, x2_step=step_idx, x2_stl_buffer=buffer)
        obs = self._get_obs(data, info)
        done = (
            ~jnp.isfinite(data.qpos).all()
            | ~jnp.isfinite(data.qvel).all()
            | ~jnp.isfinite(obs).all()
            | (R[2, 2] < self._cos_max_tilt)
            | (p_com[2] < cfg.termination.min_z)
            | (jnp.linalg.norm(rel) > cfg.termination.max_pos_err)
        )
        reward = jnp.where(jnp.isfinite(reward), reward, 0.0)
        obs = jnp.where(jnp.isfinite(obs), obs, 0.0)

        metrics = dict(state.metrics)
        for k, v in terms.items():
            metrics[f"reward/{k}"] = v
        metrics["stl/rho_window"] = jnp.where(scored, rho, 0.0)
        metrics["stl/penalty"] = penalty
        metrics["ctbr/saturated_frac"] = jnp.mean(saturated.astype(jnp.float32))
        metrics["task/pos_err"] = jnp.linalg.norm(rel)
        return state.replace(data=data, obs=obs, reward=reward, done=done.astype(jnp.float32),
                             metrics=metrics, info=info)

    def _get_obs(self, data: mjx.Data, info: dict[str, Any]) -> jax.Array:
        cfg = self._config
        p_com, v_com, quat, w = self._com_state(data)
        rel = p_com - info["x2_target"]
        t = info["x2_step"] * cfg.ctrl_dt
        deadline = max(cfg.stl.deadline_s, 1e-6)
        in_box = jnp.all(jnp.abs(rel) <= self._halfwidth).astype(jnp.float32)
        context = jnp.stack([
            jnp.maximum(0.0, cfg.stl.deadline_s - t) / deadline,
            in_box,
            info["x2_step"] / cfg.episode_length,
        ])
        parts = [-rel, v_com, frames.quat_to_rotmat(quat).reshape(9), w, info["x2_last_action"], context]
        if cfg.stl.enabled:
            parts.append(info["x2_stl_buffer"].reshape(-1))
        return jnp.concatenate(parts).astype(jnp.float32)

    # -------------------------------------------------------------- properties
    @property
    def xml_path(self) -> str:
        return self._xml_path

    @property
    def action_size(self) -> int:
        return 4

    @property
    def mj_model(self) -> mujoco.MjModel:
        return self._mj_model

    @property
    def mjx_model(self) -> mjx.Model:
        return self._mjx_model
