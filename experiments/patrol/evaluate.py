"""Evaluate supervisor variants and proposers on a patrol benchmark (DI, X2 or blimp).

Every (variant, proposer) pair runs on the same initial states, uncertainty levels and
disturbance seeds. Writes ``episodes.jsonl`` (one summary per episode), ``table.csv``
(aggregates) and ``summary.json`` (table + per-call time/energy benchmarks of a solve, a
policy inference and a check). Run through ``experiments/run_with_energy.sh`` to record CPU
energy (RAPL); without readable counters energy is NaN.

    experiments/run_with_energy.sh experiments/patrol/evaluate.py --system blimp --spec theta \
        --policy runs/patrol_bc/blimp/<run>/policy.npz --seeds 10

Uncertain-predicate specs (``*_u``): every episode draws a true keep-out region inside the predicate
uncertainty set (seeded by the episode seed only, so every pair and level of a seed sees the same
region) and the summary adds the interval lower robustness (``satisfied_lower``, ``min_rho_lower``),
the exact robustness against the true region (``satisfied_true``, ``min_rho_true``,
``frac_violated_windows_true``) and the true parameters (``true_half_width``, ``true_angle``,
``true_axis``). ``--nominal-predicates`` is the ablation that plans and checks with the exact nominal
predicates (``satisfied``/``min_rho`` then refer to the nominal formula) while the scores above still
use the interval predicates and the true region.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from common import feedback, latency_default, level_tables_summary, load_system

from rl_stl.evaluation.energy import EnergyMeter, benchmark
from rl_stl.evaluation.patrol import (
    ExpertProposer,
    nominal_proposer,
    uncertain_scores,
    waypoint_visits,
    waypoints_reached,
)
from rl_stl.evaluation.systems import run_episode
from rl_stl.policies.segment import LearnedProposer, SegmentPolicy
from rl_stl.stl.nahs import SPECS, first_enforced_window
from rl_stl.supervisor.receding import ProposalContext, RecedingConfig, RecedingSupervisor
from rl_stl.synthesis.check import check_plan
from rl_stl.synthesis.istl import History, ISTLSettings
from rl_stl.utils.provenance import create_run_dir, write_summary

PAIRS = {
    "always_on": ("always_on", None),
    "policy_only": ("policy_only", "learned"),
    "shield": ("shield", "learned"),
    "raw_trigger": ("raw_trigger", "learned"),
    "no_delay_comp": ("no_delay_comp", "learned"),
    "shield_nominal": ("shield", "nominal"),
    "shield_expert": ("shield", "expert"),
    "nominal_only": ("policy_only", "nominal"),
}


TRUE_REGION_SEED = 1_000_003  # true region of episode seed s: default_rng(TRUE_REGION_SEED + s)


def summarize(setup, ep: dict, eps_planar: float | None, region: dict | None = None) -> dict:
    steps, solves = ep["steps"], ep["solves"]
    n = len(steps)
    src = [s["source"] for s in steps]
    rt = [s for s in solves if s["request_step"] >= 0]
    X = ep["X"]
    reached = waypoints_reached(setup, X)
    # tracking error on the STL signal axes (planar for (24)/(25), x, y, z for the cube specs); the key
    # keeps its 2-D name so that the tables read both
    C = setup.problem.C_out
    planar_err = np.array([np.max(np.abs(C @ (np.asarray(r["x"]) - np.asarray(r["x_hat"])))) for r in steps])

    def stat(key, vals=None):
        v = np.array([s[key] for s in rt if key in s] if vals is None else vals, float)
        if v.size == 0 or np.all(np.isnan(v)):
            return {"mean": float("nan"), "p95": float("nan"), "max": float("nan"), "sum": 0.0}
        return {"mean": float(np.nanmean(v)), "p95": float(np.nanpercentile(v, 95)),
                "max": float(np.nanmax(v)), "sum": float(np.nansum(v))}

    out = {
        "n_steps": n, "satisfied": bool(np.all(ep["rho_windows"] >= 0)),
        "min_rho": float(np.min(ep["rho_windows"])),
        "frac_violated_windows": float(np.mean(ep["rho_windows"] < 0)),
        "frac_proposal": src.count("proposal") / n, "frac_cache": src.count("cache") / n,
        "n_runtime_solves": len(rt), "synthesis_call_fraction": len(rt) / n,
        "n_installed": sum(1 for s in rt if s.get("installed")),
        "n_deadline_miss": sum(1 for s in rt if s.get("deadline_miss")),
        "n_solve_no_plan": sum(1 for s in rt if s["status"] not in ("optimal", "suboptimal")),
        "warm_start_infeasible": sum(1 for s in rt if s.get("warm_start_feasible") is False),
        "solve_wall_s": stat("solve_wall_s"), "solve_cpu_s": stat("solve_cpu_s"),
        "solve_energy_pkg_j": stat("solve_net_energy_package-0_j"),
        "inference_s": stat(None, [r["inference_s"] for r in steps if "inference_s" in r]),
        "check_s": stat(None, [r["check_s"] for r in steps if "check_s" in r]),
        "waypoints_reached": reached,
        "waypoint_visits": waypoint_visits(setup, X),
        "max_planar_tracking_error": float(planar_err.max()),
        **ep["sim_info"],
    }
    if eps_planar is not None:
        out["frac_steps_outside_tube"] = float(np.mean(planar_err > eps_planar + 1e-9))
    if setup.task.uncertain:
        out.update(uncertain_scores(setup.task, X @ C.T, region))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--system", default="di", choices=["di", "x2", "blimp"])
    ap.add_argument("--spec", default="theta", choices=list(SPECS))
    ap.add_argument("--policy", default=None)
    ap.add_argument("--pairs", nargs="+", default=list(PAIRS))
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--levels", type=int, nargs="+", default=None)
    ap.add_argument("--latency", type=int, default=None)
    ap.add_argument("--seconds", type=float, default=None)
    ap.add_argument("--idle-seconds", type=float, default=3.0)
    ap.add_argument("--latency-mode", default="injected", choices=["injected", "measured"],
                    help="injected: every result is used d steps after its request; measured: the solve's wall "
                         "time sets the delay, and a result later than d steps is discarded (deadline miss)")
    ap.add_argument("--reference", default=None, choices=["rerooted", "clock"],
                    help="mission reference of the synthesis objective (default: the policy's, else rerooted)")
    ap.add_argument("--nominal-predicates", action="store_true",
                    help="_u specs ablation: plan and check with the exact nominal predicates, scored against "
                         "the interval predicates and the true region")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    policy = SegmentPolicy.load(args.policy) if args.policy else None
    if args.reference is None:
        args.reference = policy.reference if policy else "rerooted"
    system, setups, T = load_system(args.system, args.spec, args.reference, args.nominal_predicates)
    levels = args.levels if args.levels is not None else list(range(len(setups)))
    d = args.latency if args.latency is not None else latency_default(system)
    seconds = args.seconds if args.seconds is not None else {"di": 60.0, "x2": 60.0, "blimp": 120.0}[system.name]
    meter = EnergyMeter(idle_seconds=args.idle_seconds)
    run_dir = create_run_dir(f"patrol_eval/{args.system}", vars(args), tag=args.tag or args.spec,
                             energy=meter.describe(), tables=level_tables_summary(T), latency_steps=d)
    print(run_dir, meter.describe(), flush=True)
    rng = np.random.default_rng(12345)
    starts = [system.eval_start(rng, setups[0]) for _ in range(args.seeds)]
    regions = [setups[0].task.sample_true_region(np.random.default_rng(TRUE_REGION_SEED + s))
               for s in range(args.seeds)]

    rows = []
    with open(run_dir / "episodes.jsonl", "w") as fout:
        for name in args.pairs:
            variant, prop = PAIRS[name]
            if prop == "learned" and policy is None:
                continue
            for level in levels:
                for seed in range(args.seeds):
                    st = system.build(level)
                    if prop == "learned":
                        P = LearnedProposer(policy, T, level, st.nominal)
                    elif prop == "expert":
                        P = ExpertProposer(st.problem)
                    elif prop == "nominal":
                        P = nominal_proposer
                    else:
                        P = None
                    sup = RecedingSupervisor(st.problem, st.K, st.nominal, P,
                                             RecedingConfig(variant=variant, latency_steps=d,
                                                            reference=args.reference,
                                                            reference_input_scale=T.ref_input_scale,
                                                            latency_mode=args.latency_mode,
                                                            feedback=feedback(system)),
                                             ISTLSettings(), meter=meter)
                    t0 = time.perf_counter()
                    try:
                        ep = run_episode(system, st, sup, starts[seed], seconds, seed)
                    except Exception as exc:  # logged, not hidden: counts as a failed episode
                        rec = {"pair": name, "level": level, "seed": seed, "error": repr(exc)}
                        fout.write(json.dumps(rec) + "\n")
                        rows.append(rec)
                        print(name, level, seed, "ERROR", repr(exc), flush=True)
                        continue
                    eps = None if system.name == "di" else float(system.levels[level])
                    s = summarize(st, ep, eps, regions[seed])
                    if sup.reference is not None:  # tracking error to the clock reference x_nom(t)
                        C = st.problem.C_out
                        Xn = sup.reference.window(0, len(ep["X"]) - 1)[0]
                        gap = np.linalg.norm(ep["X"] @ C.T - Xn @ C.T, axis=1)
                        s.update(ref_gap_mean=float(gap.mean()), ref_gap_max=float(gap.max()))
                    s.update(pair=name, variant=variant, proposer=prop, level=level,
                             level_value=float(system.levels[level]), seed=seed,
                             wall_s=time.perf_counter() - t0)
                    fout.write(json.dumps(s, default=float) + "\n")
                    fout.flush()
                    rows.append(s)
                    print(name, level, seed, {k: s[k] for k in ("satisfied", "min_rho", "frac_proposal",
                          "synthesis_call_fraction", "waypoints_reached", "satisfied_true", "min_rho_true")
                          if k in s}, flush=True)

    import pandas as pd
    df = pd.json_normalize([r for r in rows if "error" not in r])
    agg = df.groupby(["pair", "level"]).agg(
        episodes=("seed", "count"), satisfied=("satisfied", "mean"), min_rho=("min_rho", "min"),
        frac_violated_windows=("frac_violated_windows", "mean"),
        frac_proposal=("frac_proposal", "mean"), synthesis_call_fraction=("synthesis_call_fraction", "mean"),
        waypoints=("waypoints_reached", "mean"), waypoint_visits=("waypoint_visits", "mean"),
        solve_wall_mean_s=("solve_wall_s.mean", "mean"),
        **({"ref_gap_mean": ("ref_gap_mean", "mean")} if "ref_gap_mean" in df else {}),
        **({"satisfied_true": ("satisfied_true", "mean"), "min_rho_true": ("min_rho_true", "min"),
            "satisfied_lower": ("satisfied_lower", "mean"), "min_rho_lower": ("min_rho_lower", "min")}
           if "satisfied_true" in df else {}),
        solve_wall_max_s=("solve_wall_s.max", "max"), solve_wall_total_s=("solve_wall_s.sum", "mean"),
        solve_energy_total_j=("solve_energy_pkg_j.sum", "mean"),
        deadline_miss=("n_deadline_miss", "sum"), no_plan=("n_solve_no_plan", "sum"),
        inference_mean_s=("inference_s.mean", "mean"), check_mean_s=("check_s.mean", "mean"),
    ).reset_index()
    errors = [r for r in rows if "error" in r]
    agg.to_csv(run_dir / "table.csv", index=False)
    print(agg.to_string(), flush=True)

    # per-call benchmarks at a representative state
    st = setups[levels[0]]
    p = st.problem
    x0 = starts[0]
    st.nominal.reset(x0)
    Xn, Un = st.nominal.rollout(p.model, x0, p.n_plan)
    fw = first_enforced_window(100, p.h)
    hist = History(np.tile(p.C_out @ x0, (p.h - 1, 1)), [None] * (p.h - 1))
    expert = ExpertProposer(p, ISTLSettings())
    ctx = ProposalContext(100, x0, hist, fw, Xn, Un, None)
    bench = {"solve": benchmark(lambda: expert(ctx), meter, repeats=30, warmup=3)}
    U = expert(ctx)
    X = p.model.rollout(x0, U)
    bench["check"] = benchmark(lambda: check_plan(p, x0, X, U, hist, None, fw), meter, repeats=300)
    if policy is not None:
        lp = LearnedProposer(policy, T, levels[0], st.nominal)
        bench["inference"] = benchmark(lambda: lp(ctx), meter, repeats=2000)
    write_summary(run_dir, {"table": agg.to_dict(orient="records"), "benchmarks": bench,
                            "errors": errors, "energy": meter.describe(), "latency_steps": d,
                            "seconds": seconds})
    print(json.dumps(bench, indent=1), flush=True)


if __name__ == "__main__":
    main()
