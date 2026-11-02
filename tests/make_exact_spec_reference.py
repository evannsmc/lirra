"""Write ``tests/data/exact_spec_reference.npz``: synthesized plans of the exact patrol specs.

The reference is the regression baseline of ``tests/test_interval_predicates.py``: for every system
(double integrator, X2, blimp) and exact spec (theta, phi, theta_plus, phi_plus) it stores the plan of
three fixed requests (the t = 0- request with only the in-episode windows enforced; a mid-episode
request with every window enforced and a history that dwells inside ``S`` and then at the first
waypoint; the same request delay compensated, with the last two history samples and the root taken
from a cached plan, tube ages 1, 2, 3), the terminal face offsets, the independent check's
``rho_lower`` and the realised robustness of the history followed by the plan. It was generated from the code before the interval
predicates were added (git revision recorded in the file); rerun only to move the baseline on purpose.

    .venv/bin/python tests/make_exact_spec_reference.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "patrol"))

SYSTEMS = ("di", "x2", "blimp")
EXACT_SPECS = ("theta", "phi", "theta_plus", "phi_plus")
LEVEL = 1
OUT = ROOT / "tests" / "data" / "exact_spec_reference.npz"


def requests(system, setup):
    """Three fixed requests ``(name, x0, history, root_age, first_window)`` per setup."""
    from rl_stl.stl.nahs import first_enforced_window
    from rl_stl.synthesis.istl import History

    p = setup.problem
    x0 = system.eval_start(np.random.default_rng(0), setup)
    hz = History(np.zeros((p.h - 1, p.C_out.shape[0])), [None] * (p.h - 1))
    hy = np.zeros((p.h - 1, p.C_out.shape[0]))  # inside S, then outside at the first waypoint
    hy[(p.h - 1) // 2:] = setup.task.waypoints()[0]
    hx = History(hy, [None] * (p.h - 1))
    ha = History(hy.copy(), [None] * (p.h - 3) + [1, 2])
    fw = first_enforced_window(100, p.h)
    return [("t0", x0, hz, None, first_enforced_window(0, p.h)), ("mid", x0, hx, None, fw),
            ("aged", x0, ha, 3, fw)]


def compute() -> dict[str, np.ndarray]:
    from common import load_system

    from rl_stl.supervisor.receding import realised_robustness
    from rl_stl.synthesis.check import check_plan
    from rl_stl.synthesis.istl import ISTLSettings, RecedingISTL

    out: dict[str, np.ndarray] = {}
    for sysname in SYSTEMS:
        for spec in EXACT_SPECS:
            system, setups, _ = load_system(sysname, spec, "rerooted")
            st = setups[LEVEL]
            p = st.problem
            key = f"{sysname}/{spec}"
            out[f"{key}/face_g"] = np.concatenate([g for _, g in p.terminal.face_rows()])
            for name, x0, hist, age, fw in requests(system, st):
                st.nominal.reset(x0)
                Xn, Un = st.nominal.rollout(p.model, x0, p.n_plan)
                mdl = RecedingISTL(p, fw, ISTLSettings(time_limit=120.0))
                mdl.update(x0, hist, Xn, Un, root_age=age)
                res = mdl.solve()
                k = f"{key}/{name}"
                out[f"{k}/status"] = np.asarray(str(res.status))
                if res.has_solution:
                    chk = check_plan(p, x0, res.X, res.U, hist, age, fw)
                    out[f"{k}/X"], out[f"{k}/U"] = res.X, res.U
                    out[f"{k}/objective"] = np.asarray(res.objective)
                    out[f"{k}/face"] = np.asarray(-1 if res.face is None else res.face)
                    out[f"{k}/rho_lower"] = np.asarray(chk.rho_lower)
                    out[f"{k}/check_ok"] = np.asarray(chk.ok)
                    out[f"{k}/n_binary"] = np.asarray(res.n_binary)
                    Y = np.vstack([hist.y, res.X @ p.C_out.T])
                    out[f"{k}/realised"] = realised_robustness(p, Y)
                print(k, res.status, res.objective, res.solve_time, flush=True)
    return out


def main() -> None:
    rev = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                         check=True).stdout.strip()
    out = compute()
    out["git_revision"] = np.asarray(rev)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUT, **out)
    print("wrote", OUT, len(out), "arrays")


if __name__ == "__main__":
    main()
