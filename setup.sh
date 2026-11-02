#!/usr/bin/env bash
# =============================================================================
# rl-stl: create/update the uv environment (.venv) and install rl_stl from src/
# in editable mode, so code changes apply without reinstalling.
#
#   ./setup.sh            # cuda12 extra if nvidia-smi sees a GPU, otherwise CPU
#   ./setup.sh --gpu      # force cuda12 (e.g. on a login node where no GPU is visible)
#   ./setup.sh --cpu      # force CPU
#   ./setup.sh --wandb    # also install the optional wandb extra
#
# A script cannot activate a venv in the shell that ran it, so afterwards run:
#
#   source .venv/bin/activate
#
# Safe to re-run: `uv sync` makes .venv match uv.lock exactly, so switching
# between --cpu and --gpu adds or removes the CUDA wheels.
# =============================================================================
set -euo pipefail

usage() {
    cat <<'EOF'
usage: ./setup.sh [--cpu | --gpu] [--wandb]

  --cpu     install CPU-only JAX
  --gpu     install the cuda12 extra (jax[cuda12])
  --wandb   install the optional wandb extra
  -h        show this help

Without --cpu/--gpu the cuda12 extra is used iff `nvidia-smi -L` finds a GPU.
EOF
}

log() { printf '==> %s\n' "$*"; }
warn() { printf 'warning: %s\n' "$*" >&2; }

device=auto
extras=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --cpu) device=cpu ;;
        --gpu) device=gpu ;;
        --wandb) extras+=(--extra wandb) ;;
        -h | --help) usage; exit 0 ;;
        *) warn "unknown option: $1"; usage >&2; exit 2 ;;
    esac
    shift
done

cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"

# ---- uv (manages Python 3.12 and all packages) --------------------------------
if ! command -v uv >/dev/null 2>&1; then
    log "uv not found; installing it to ~/.local/bin"
    curl -LsSf https://astral.sh/uv/install.sh | env UV_NO_MODIFY_PATH=1 sh
    export PATH="${HOME}/.local/bin:${PATH}"
fi
log "$(uv --version)"

# ---- sync ---------------------------------------------------------------------
if [[ "$device" == auto ]]; then
    if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
        device=gpu
    else
        device=cpu
    fi
fi
if [[ "$device" == gpu ]]; then
    extras+=(--extra cuda12)
fi

# --locked fails if pyproject.toml has changed since uv.lock was written; run
# `uv lock` deliberately in that case. The project itself is installed editable
# (uv's default for a project with a [build-system]), along with the dev group.
log "uv sync --locked${extras[*]:+ ${extras[*]}} (${device})"
uv sync --locked "${extras[@]}"

py=.venv/bin/python

# ---- smoke check --------------------------------------------------------------
log "checking the editable install and the JAX device"
"$py" - <<'EOF'
from pathlib import Path

from rl_stl.utils.device import configure_jax_device, jax_device_summary

# Before jax is imported. "auto" so a --gpu sync on a node without a GPU (login
# node) is still checked, on the CPU.
configure_jax_device("auto")

import rl_stl

pkg = Path(rl_stl.__file__).resolve().parent
src = Path("src/rl_stl").resolve()
assert pkg == src, f"rl_stl is imported from {pkg}, not the editable source {src}"
print(f"rl_stl -> {pkg} (editable)")
print(jax_device_summary())
EOF

# ---- Jupyter kernel -----------------------------------------------------------
log "registering the Jupyter kernel 'rl-stl'"
"$py" -m ipykernel install --user --name rl-stl --display-name "Python 3.12 (rl-stl)"

# ---- Gurobi licence (warning only) --------------------------------------------
# Search order used by Gurobi: $GRB_LICENSE_FILE, /opt/gurobi/gurobi.lic,
# ~/gurobi.lic, or WLS credentials in the environment.
lic=""
if [[ -n "${GRB_LICENSE_FILE:-}" ]]; then
    if [[ -f "$GRB_LICENSE_FILE" ]]; then
        lic="$GRB_LICENSE_FILE"
    else
        warn "GRB_LICENSE_FILE=${GRB_LICENSE_FILE} is not a file"
    fi
elif [[ -f /opt/gurobi/gurobi.lic ]]; then
    lic=/opt/gurobi/gurobi.lic
elif [[ -f "${HOME}/gurobi.lic" ]]; then
    lic="${HOME}/gurobi.lic"
elif [[ -n "${WLSACCESSID:-}" && -n "${WLSSECRET:-}" && -n "${LICENSEID:-}" ]]; then
    lic="WLS environment variables"
fi
if [[ -n "$lic" ]]; then
    log "Gurobi licence: ${lic}"
else
    warn "no Gurobi licence found; gurobipy will use its size-limited licence" \
        "(double-integrator L1 problems only). See docs/guides/gurobi_setup.pdf"
fi

cat <<EOF

Environment ready (${device}). Activate it in your shell with:

    source .venv/bin/activate
EOF
