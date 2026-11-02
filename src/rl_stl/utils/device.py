"""CPU/GPU switch for JAX, MJX and Brax.

The same code runs on the laptop (CPU smoke tests) and on the cluster (GPU
training). Select the device with, in order of precedence:

1. the ``device`` argument (``"cpu"``, ``"gpu"`` or ``"auto"``),
2. the ``RL_STL_DEVICE`` environment variable,
3. ``"auto"``: GPU if ``nvidia-smi`` exists and the JAX CUDA plugin is installed
   (``uv sync --extra cuda12``), otherwise CPU.

JAX reads its platform configuration at import time, so
:func:`configure_jax_device` must run **before** the first ``import jax``
anywhere in the process (including indirect imports via brax/playground).
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import sys
import warnings

_VALID = ("cpu", "gpu", "auto")


def _cuda_available() -> bool:
    has_driver = shutil.which("nvidia-smi") is not None
    has_plugin = importlib.util.find_spec("jax_cuda12_plugin") is not None
    return has_driver and has_plugin


def resolve_device(device: str | None = None) -> str:
    choice = (device or os.environ.get("RL_STL_DEVICE") or "auto").lower()
    if choice not in _VALID:
        raise ValueError(f"device must be one of {_VALID}, got {choice!r}")
    if choice == "auto":
        return "gpu" if _cuda_available() else "cpu"
    if choice == "gpu" and not _cuda_available():
        raise RuntimeError(
            "device='gpu' requested but no CUDA driver/JAX CUDA plugin found. "
            "On the cluster run `uv sync --extra cuda12` inside a GPU job."
        )
    return choice


def configure_jax_device(device: str | None = None) -> str:
    """Set JAX/XLA environment variables for the resolved device and return it."""
    resolved = resolve_device(device)
    if "jax" in sys.modules:
        current = os.environ.get("JAX_PLATFORMS")
        if (resolved == "cpu") != (current == "cpu"):
            warnings.warn(
                "jax was imported before configure_jax_device(); the device choice may not apply.",
                stacklevel=2,
            )
    if resolved == "cpu":
        os.environ["JAX_PLATFORMS"] = "cpu"
    else:
        if os.environ.get("JAX_PLATFORMS") == "cpu":
            del os.environ["JAX_PLATFORMS"]
        # Recommended by MuJoCo Playground for MJX on NVIDIA GPUs.
        flags = os.environ.get("XLA_FLAGS", "")
        if "--xla_gpu_triton_gemm_any" not in flags:
            os.environ["XLA_FLAGS"] = f"{flags} --xla_gpu_triton_gemm_any=True".strip()
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
        os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ["RL_STL_RESOLVED_DEVICE"] = resolved
    return resolved


def jax_device_summary() -> dict[str, object]:
    """Describe the devices JAX actually sees (imports jax)."""
    import jax

    return {
        "resolved": os.environ.get("RL_STL_RESOLVED_DEVICE"),
        "jax_version": jax.__version__,
        "devices": [str(d) for d in jax.devices()],
        "device_count": jax.device_count(),
    }
