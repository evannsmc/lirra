#!/usr/bin/env bash
# Run an experiment with readable Intel RAPL energy counters (pyJoules / rl_stl.evaluation.energy).
#
#   experiments/run_with_energy.sh experiments/patrol/run_patrol.py --config ... [args...]
#
# Recent kernels make /sys/class/powercap/intel-rapl:*/energy_uj readable by root only. This
# script makes them world-readable (once per boot; needs sudo) and then runs the Python script
# as the current user via `uv run`, so results and .venv stay user-owned. Set
# RL_STL_SKIP_RAPL=1 to skip the sudo step (energy is then recorded as NaN).
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <script.py> [args...]" >&2
  exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
rapl=/sys/class/powercap/intel-rapl

if [[ "${RL_STL_SKIP_RAPL:-0}" != 1 ]]; then
  if [[ ! -d "$rapl" ]]; then
    echo "[energy] no RAPL powercap interface on this machine; energy will be NaN" >&2
  elif ! cat "$rapl"/intel-rapl:0/energy_uj >/dev/null 2>&1; then
    echo "[energy] RAPL counters are root-only; running: sudo chmod -R a+r $rapl" >&2
    sudo chmod -R a+r "$rapl" || echo "[energy] sudo failed (e.g. on a cluster); energy will be NaN" >&2
  fi
fi

# Fewer background threads -> less package-energy noise from this process.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export XLA_FLAGS="${XLA_FLAGS:---xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1}"

cd "$repo_root"
exec uv run --no-sync python "$@"
