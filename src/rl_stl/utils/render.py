"""Small MuJoCo video helpers shared by the X2 rollout/evaluation scripts.

Rendering needs an OpenGL context: on a headless machine use ``MUJOCO_GL=egl``.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import mujoco
import numpy as np


def pick_h264_encoder() -> str:
    """mediapy's default codec='h264' resolves to whatever ffmpeg registers under
    that name; on some cluster ffmpeg builds that is the v4l2m2m hardware encoder,
    which fails without a /dev/video* device. Prefer a software encoder."""
    try:
        out = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True,
                             timeout=10, check=False).stdout
    except (OSError, subprocess.SubprocessError):
        return "h264"
    for name in ("libx264", "libopenh264", "mpeg4"):
        if name in out:
            return name
    return "h264"


def add_box(scene, center: np.ndarray, halfwidth: np.ndarray, rgba=(0.2, 0.85, 0.2, 0.25),
            rot: np.ndarray | None = None) -> None:
    """Append a translucent box geom to an already updated ``mujoco.MjvScene`` (``rot``: 3x3 world
    orientation of the box, default axis aligned)."""
    if scene.ngeom >= scene.maxgeom:
        return
    mat = np.eye(3) if rot is None else np.asarray(rot, np.float64)
    mujoco.mjv_initGeom(
        scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_BOX, np.asarray(halfwidth, np.float64),
        np.asarray(center, np.float64), mat.flatten(), np.asarray(rgba, np.float32),
    )
    scene.ngeom += 1


def add_sphere(scene, center: np.ndarray, radius: float, rgba) -> None:
    """Append a sphere geom to an already updated ``mujoco.MjvScene``."""
    if scene.ngeom >= scene.maxgeom:
        return
    mujoco.mjv_initGeom(
        scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE, np.full(3, radius, np.float64),
        np.asarray(center, np.float64), np.eye(3).flatten(), np.asarray(rgba, np.float32),
    )
    scene.ngeom += 1


def add_segment(scene, start: np.ndarray, end: np.ndarray, width: float, rgba) -> None:
    """Append a capsule from ``start`` to ``end`` to an already updated ``mujoco.MjvScene``."""
    if scene.ngeom >= scene.maxgeom:
        return
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.eye(3).flatten(),
                        np.asarray(rgba, np.float32))
    mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, width, np.asarray(start, np.float64),
                         np.asarray(end, np.float64))
    scene.ngeom += 1


def render_qpos_video(mj_model: mujoco.MjModel, qpos: np.ndarray, path: str | Path, fps: float,
                      box_center: np.ndarray | None = None, box_halfwidth: np.ndarray | None = None,
                      width: int = 640, height: int = 480, camera: str | int = "track") -> int:
    """Render one frame per row of ``qpos`` (kinematic replay) and write an mp4. Returns frame count."""
    import mediapy

    data = mujoco.MjData(mj_model)
    names = {mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_CAMERA, i) for i in range(mj_model.ncam)}
    cam = camera if (not isinstance(camera, str) or camera in names) else -1
    frames = []
    with mujoco.Renderer(mj_model, height=height, width=width) as renderer:
        for q in qpos:
            data.qpos[:] = q
            mujoco.mj_forward(mj_model, data)
            renderer.update_scene(data, camera=cam)
            if box_center is not None and box_halfwidth is not None:
                add_box(renderer.scene, box_center, box_halfwidth)
            frames.append(renderer.render().copy())
    mediapy.write_video(str(path), frames, fps=fps, codec=pick_h264_encoder())
    return len(frames)
