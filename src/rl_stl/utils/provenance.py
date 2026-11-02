"""Run directories with provenance: resolved config, seeds, git revision, versions, hardware.

Every experiment script should call :func:`create_run_dir` before doing work and
:func:`write_summary` at the end. Nothing here imports JAX, so it is safe to use
before :func:`rl_stl.utils.device.configure_jax_device`.
"""

from __future__ import annotations

import datetime as _dt
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from rl_stl.utils.config import save_config, to_dict

REPO_ROOT = Path(__file__).resolve().parents[3]

TRACKED_PACKAGES = (
    "mujoco",
    "mujoco-mjx",
    "playground",
    "brax",
    "jax",
    "jax-cuda12-plugin",
    "flax",
    "optax",
    "stlpy",
    "gurobipy",
    "numpy",
    "scipy",
)


def git_revision(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    def run(*args: str) -> str | None:
        try:
            return subprocess.run(
                ["git", "-C", str(repo_root), *args],
                capture_output=True, text=True, check=True, timeout=10,
            ).stdout.strip()
        except (subprocess.SubprocessError, FileNotFoundError):
            return None

    status = run("status", "--porcelain")
    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(status) if status is not None else None,
    }


def package_versions(packages: tuple[str, ...] = TRACKED_PACKAGES) -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for name in packages:
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = None
    return out


def hardware_info() -> dict[str, Any]:
    cpu_model = platform.processor() or None
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    cpu_model = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    gpus = None
    if shutil.which("nvidia-smi"):
        try:
            gpus = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                capture_output=True, text=True, check=True, timeout=10,
            ).stdout.strip().splitlines()
        except subprocess.SubprocessError:
            gpus = None
    return {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_model": cpu_model,
        "cpu_count": os.cpu_count(),
        "gpus": gpus,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "jax_platforms_env": os.environ.get("JAX_PLATFORMS"),
    }


def run_metadata(seed: int | Mapping[str, int] | None = None, **extra: Any) -> dict[str, Any]:
    return {
        "created_utc": _dt.datetime.now(_dt.UTC).isoformat(),
        "seed": to_dict(seed),
        "git": git_revision(),
        "versions": package_versions(),
        "hardware": hardware_info(),
        **to_dict(extra),
    }


def create_run_dir(
    experiment: str,
    config: Any,
    seed: int | Mapping[str, int] | None = None,
    root: str | Path | None = None,
    tag: str | None = None,
    **extra: Any,
) -> Path:
    """Create ``runs/<experiment>/<timestamp>[_tag]/`` with config.yaml and metadata.json."""
    root = Path(root) if root is not None else REPO_ROOT / "runs"
    stamp = _dt.datetime.now().astimezone().strftime("%Y%m%d-%H%M%S-%f")
    name = f"{stamp}_{tag}" if tag else stamp
    run_dir = root / experiment / name
    run_dir.mkdir(parents=True, exist_ok=False)
    save_config(config, run_dir / "config.yaml")
    with open(run_dir / "metadata.json", "w") as f:
        json.dump(run_metadata(seed=seed, **extra), f, indent=2)
    return run_dir


def write_summary(run_dir: str | Path, summary: Mapping[str, Any]) -> None:
    with open(Path(run_dir) / "summary.json", "w") as f:
        json.dump(to_dict(summary), f, indent=2, default=str)
