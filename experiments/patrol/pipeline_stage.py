"""Run one stage of the patrol pipeline, resolving its input from the newest matching run.

Stages and the runs they read/write (``<sys>`` = di | x2 | blimp, ``<spec>`` = theta | phi):

    dataset   writes runs/patrol_dataset/<sys>/*_<spec>
    bc        reads the newest dataset          -> runs/patrol_bc/<sys>/*_<spec>_bc
    ppo       reads the newest BC policy        -> runs/patrol_ppo/<sys>/*_<spec>_ppo
    eval      reads the newest PPO policy (falls back to BC) -> runs/patrol_eval/<sys>/*_<spec>_<pairs>

Stages can therefore be chained (e.g. as batch jobs) without hand-copied paths:

    uv run python experiments/patrol/pipeline_stage.py bc --system blimp --spec theta
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent

DATASET_EPISODES = {"di": 400, "x2": 300, "blimp": 400}
PPO_STEPS = {"di": 30_000_000, "x2": 30_000_000, "blimp": 30_000_000}


def newest(pattern: str, name: str) -> Path:
    hits = sorted(REPO.glob(pattern))
    hits = [h for h in hits if (h / name).exists()]
    if not hits:
        raise SystemExit(f"no run matching {pattern} with {name}; run the previous stage first")
    return hits[-1] / name


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("stage", choices=["dataset", "bc", "ppo", "eval"])
    ap.add_argument("--system", required=True, choices=["di", "x2", "blimp"])
    ap.add_argument("--spec", required=True, choices=["theta", "phi", "theta_plus", "phi_plus", "theta_u", "phi_u", "theta_plus_u", "phi_plus_u"])
    ap.add_argument("--pairs", nargs="*", default=None, help="eval: subset of loop/proposer pairs")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--extra", nargs=argparse.REMAINDER, default=[], help="passed to the stage script")
    args = ap.parse_args()
    s, sp = args.system, args.spec
    py = sys.executable
    if args.stage == "dataset":
        workers = max(1, len(os.sched_getaffinity(0)) - 2)
        cmd = [py, str(HERE / "dataset.py"), "--system", s, "--spec", sp, "--episodes",
               str(DATASET_EPISODES[s]), "--workers", str(workers), "--tag", sp]
    elif args.stage == "bc":
        data = newest(f"runs/patrol_dataset/{s}/*_{sp}", "dataset.npz")
        cmd = [py, str(HERE / "train_bc.py"), "--system", s, "--spec", sp, "--data", str(data),
               "--tag", f"{sp}_bc"]
    elif args.stage == "ppo":
        init = newest(f"runs/patrol_bc/{s}/*_{sp}_bc", "policy.npz")
        cmd = [py, str(HERE / "train_ppo.py"), "--system", s, "--spec", sp, "--init", str(init),
               "--num-timesteps", str(PPO_STEPS[s]), "--num-envs", "2048", "--batch-size", "1024",
               "--num-minibatches", "16", "--device", "auto", "--tag", f"{sp}_ppo"]
    else:
        policy = None
        for pattern in (f"runs/patrol_ppo/{s}/*_{sp}_ppo", f"runs/patrol_bc/{s}/*_{sp}_bc"):
            try:
                policy = newest(pattern, "policy.npz")
                break
            except SystemExit:
                continue
        learned = {"policy_only", "shield", "raw_trigger", "no_delay_comp"}
        if policy is None and (not args.pairs or learned & set(args.pairs)):
            raise SystemExit("no PPO or BC policy found; run the bc/ppo stages first")
        tag = f"{sp}_" + ("-".join(args.pairs) if args.pairs else "all")
        cmd = [py, str(HERE / "evaluate.py"), "--system", s, "--spec", sp,
               "--seeds", str(args.seeds), "--tag", tag]
        if policy is not None:
            cmd += ["--policy", str(policy)]
        if args.pairs:
            cmd += ["--pairs", *args.pairs]
    cmd += args.extra
    print("+", " ".join(cmd), flush=True)
    raise SystemExit(subprocess.call(cmd, cwd=REPO))


if __name__ == "__main__":
    main()
