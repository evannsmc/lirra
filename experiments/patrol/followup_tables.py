"""Aggregate the follow-up evaluation runs into the paper's result tables.

Reads runs/patrol_eval/<system>/*_<spec>_fu_<tag>/episodes.jsonl and writes
docs/results/patrol_followup.{md,json}: the main table with the full pipeline and the tracking
error to the clock reference, the full-pipeline table with both PPO seeds, the measured-latency
table, the identification of the X2 tracking tube from every checked-loop X2 episode, and the
performance table per platform (computation per RL call, per check and per solve, solver calls,
waypoints, constraint violation) for the solver alone, the RL proposer alone, the first stage and
the full pipeline.

    uv run --no-sync python experiments/patrol/followup_tables.py
    uv run --no-sync python experiments/patrol/followup_tables.py --specs theta_plus phi_plus --out patrol_followup_plus

The cube specs (theta^+, phi^+) are tabulated separately with ``--specs``; the waypoint columns then
also report order-free visits (``waypoint_visits``, see ``rl_stl.evaluation.patrol``).
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
EVAL = REPO / "runs" / "patrol_eval"
OUT = REPO / "docs" / "results"
SYSTEMS, SPECS = ["di", "x2", "blimp"], ["theta", "phi"]
NAMES = {"always_on": "Always-on synthesis", "bc1": "First stage: cloned", "ppo1": "First stage: PPO",
         "mbase": "Margin, original data", "mall": "Margin, original + DAgger",
         "ppoB": "PPO B (seed 0)", "ppoB_s1": "PPO B (seed 1)", "ppoC": "PPO C (seed 0)", "ppoC_s1": "PPO C (seed 1)"}


def episodes(system: str, spec: str, tag: str) -> list[dict] | None:
    ds = sorted(d for d in (EVAL / system).glob(f"*_{spec}_fu_{tag}") if (d / "episodes.jsonl").exists())
    if not ds:
        return None
    with open(ds[-1] / "episodes.jsonl") as fh:
        E = [e for e in map(json.loads, fh) if "n_steps" in e]
    return E or None


def stats(E: list[dict] | None) -> dict | None:
    if not E:
        return None

    def g(k):
        return np.array([e[k] for e in E], float)
    out = {"episodes": len(E), "satisfied": int(g("satisfied").sum()), "min_rho": float(g("min_rho").min()),
           "accepted": float(g("frac_proposal").mean()), "calls": float(g("synthesis_call_fraction").mean()),
           "waypoints": float(g("waypoints_reached").mean()),
           "visits": float(g("waypoint_visits").mean()) if "waypoint_visits" in E[0] else None,
           "warm_start_infeasible": int(g("warm_start_infeasible").sum()),
           "requests": int(g("n_runtime_solves").sum()), "deadline_miss": int(g("n_deadline_miss").sum()),
           "no_plan": int(g("n_solve_no_plan").sum())}
    if "ref_gap_mean" in E[0]:
        out.update(ref_gap_mean=float(g("ref_gap_mean").mean()), ref_gap_max=float(g("ref_gap_max").max()))
    walls = [e["solve_wall_s"] for e in E if np.isfinite(e["solve_wall_s"]["mean"])]
    if walls:
        out.update(solve_mean=float(np.mean([w["mean"] for w in walls])),
                   solve_p95=float(np.mean([w["p95"] for w in walls])),
                   solve_max=float(np.max([w["max"] for w in walls])))
    return out


def tube_identification(specs: list[str]) -> dict:
    """Per tube level: max planning-rate tracking error, steps outside the tube, warm-start failures and
    violations over every checked-loop X2 episode with the clock reference (shield and always-on)."""
    rows = {}
    for d in sorted((EVAL / "x2").glob("2026*")):
        # "_theta_(clock|dg|fu)_" does not match "_theta_plus_fu_": the 2-D and cube runs stay apart
        if not re.search(rf"_({'|'.join(specs)})_(clock|dg|fu)_", d.name) or not (d / "episodes.jsonl").exists():
            continue
        if not re.search(r"(shield|always_on)", d.name) or re.search(r"(no_delay|raw|measured)", d.name):
            continue
        with open(d / "episodes.jsonl") as fh:
            eps = [e for e in map(json.loads, fh) if "n_steps" in e]
        for e in eps:
            r = rows.setdefault(e["level_value"], {"episodes": 0, "violated": 0, "warm_start_infeasible": 0,
                                                   "requests": 0, "max_error": 0.0, "outside": []})
            r["episodes"] += 1
            r["violated"] += int(not e["satisfied"])
            r["warm_start_infeasible"] += e["warm_start_infeasible"]
            r["requests"] += e["n_runtime_solves"]
            r["max_error"] = max(r["max_error"], e["max_planar_tracking_error"])
            r["outside"].append(e["frac_steps_outside_tube"])
    return {str(k): {**{a: b for a, b in v.items() if a != "outside"}, "frac_outside_mean": float(np.mean(v["outside"])),
                     "frac_outside_max": float(np.max(v["outside"]))} for k, v in sorted(rows.items())}


PERF_LOOPS = [("always_on", "STL solver only (always-on)"), ("ppoC_policy_only", "RL only, no check (full pipeline)"),
              ("ppo1_shield", "First stage, checked"), ("ppoC_shield", "Full pipeline, checked")]


def performance(specs: list[str]) -> dict:
    """Per platform and loop, pooling the specs (60 episodes): computation and mission metrics."""
    out = {}
    for s in SYSTEMS:
        for tag, _ in PERF_LOOPS:
            E = [e for sp in specs for e in (episodes(s, sp, tag) or [])]
            if not E:
                continue
            out[f"{s}/{tag}"] = loop_perf(E)
    return out


def loop_perf(E: list[dict]) -> dict:
    """Computation and mission metrics of one loop, pooled over the episodes ``E``."""
    def tot(key, field="sum"):
        return float(sum(e[key][field] for e in E if key in e and np.isfinite(e[key][field])))

    def per_call(key):  # mean over all calls: total time / number of calls
        n = sum(e[key]["sum"] / e[key]["mean"] for e in E
                if key in e and np.isfinite(e[key]["mean"]) and e[key]["mean"] > 0)
        return 1e3 * tot(key) / n if n else None
    steps = sum(e["n_steps"] for e in E)
    solves = sum(e["n_runtime_solves"] for e in E)
    rl, chk = per_call("inference_s"), per_call("check_s")
    return {
        "episodes": len(E), "solves_per_episode": solves / len(E),
        "solve_ms": 1e3 * tot("solve_wall_s") / solves if solves else None,
        "rl_ms": rl, "check_ms": chk,
        "rl_check_ms": (rl or 0.0) + (chk or 0.0) if rl is not None else None,
        "compute_ms_per_step": 1e3 * (tot("solve_wall_s") + tot("inference_s") + tot("check_s")) / steps,
        "waypoints": float(np.mean([e["waypoints_reached"] for e in E])),
        "visits": (float(np.mean([e["waypoint_visits"] for e in E])) if "waypoint_visits" in E[0] else None),
        "violated_episodes": int(sum(not e["satisfied"] for e in E)),
        "violated_windows_pct": 100.0 * float(np.mean([e["frac_violated_windows"] for e in E])),
        "min_rho": float(min(e["min_rho"] for e in E)),
        "ref_gap_mean": (float(np.mean([e["ref_gap_mean"] for e in E])) if "ref_gap_mean" in E[0] else None),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--specs", nargs="+", default=SPECS)
    ap.add_argument("--out", default="patrol_followup", help="file stem under docs/results")
    args = ap.parse_args()
    specs = args.specs
    cube = any("_plus" in sp for sp in specs)
    res = {"main": {}, "pipeline": {}, "measured": {}, "unchecked": {}}
    for s in SYSTEMS:
        for p in specs:
            key = f"{s}/{p}"
            res["main"][key] = {n: stats(episodes(s, p, t)) for n, t in
                                [("always_on", "always_on"), ("bc1", "bc1_shield"), ("ppo1", "ppo1_shield"),
                                 ("ppoC", "ppoC_shield")]}
            res["pipeline"][key] = {n: stats(episodes(s, p, f"{n}_shield")) for n in
                                    ["mbase", "mall", "ppoB", "ppoC", "ppoB_s1", "ppoC_s1"]}
            res["unchecked"][key] = {n: stats(episodes(s, p, f"{n}_policy_only")) for n in
                                     ["bc1", "ppo1", "ppoB_s1", "ppoC_s1"]}
            res["measured"][key] = {n: stats(episodes(s, p, t)) for n, t in
                                    [("always_on", "always_on_measured"), ("bc1", "bc1_shield_measured"),
                                     ("ppo1", "ppo1_shield_measured"), ("ppoC", "ppoC_shield_measured")]}
    res["x2_tube"] = tube_identification(specs)
    res["performance"] = performance(specs)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{args.out}.json").write_text(json.dumps(res, indent=1))

    def cell(st, *keys, fmt="{:.3f}"):
        if st is None:
            return "--"
        return " / ".join(fmt.format(st[k]) if isinstance(st.get(k), float) else str(st.get(k, "--")) for k in keys)

    wp_keys = ("waypoints", "visits") if cube else ("waypoints",)
    wp_title = "waypoints in order / visits" if cube else "waypoints"
    lines = ["# Patrol follow-up results" + (" (3-D cube specs " + ", ".join(specs) + ")" if cube else ""), "",
             "Generated by `experiments/patrol/followup_tables.py` from the follow-up evaluation runs.",
             "Checked loop unless noted; 30 episodes per cell (10 starts x 3 uncertainty levels).", ""]
    for title, part, keys in [(f"Main table: satisfied / calls per step / accepted / {wp_title} / gap to x_nom (m)",
                               "main", ("satisfied", "calls", "accepted", *wp_keys, "ref_gap_mean")),
                              (f"Full pipeline: satisfied / calls per step / accepted / {wp_title} / gap to x_nom (m)",
                               "pipeline", ("satisfied", "calls", "accepted", *wp_keys, "ref_gap_mean")),
                              ("Proposer alone: satisfied / min robustness", "unchecked", ("satisfied", "min_rho")),
                              (("Measured latency: satisfied / calls / deadline misses / requests / solve p95 (s) / "
                                "solve max (s)"), "measured",
                               ("satisfied", "calls", "deadline_miss", "requests", "solve_p95", "solve_max"))]:
        lines += [f"## {title}", ""]
        names = list(next(iter(res[part].values())).keys())
        lines += ["| system / spec | " + " | ".join(NAMES[n] for n in names) + " |",
                  "|---|" + "---|" * len(names)]
        for key, row in res[part].items():
            lines.append(f"| {key} | " + " | ".join(cell(row[n], *keys) for n in names) + " |")
        lines.append("")
    lines += ["## X2 tracking tube (checked loops, clock reference)", "",
              (f"| tube eps (m) | episodes | violated | warm-start failures / requests | max {'3-D' if cube else 'planar'} error (m) | "
               "steps outside tube (mean / max) |"), "|---|---|---|---|---|---|"]
    for k, v in res["x2_tube"].items():
        lines.append(f"| {k} | {v['episodes']} | {v['violated']} | {v['warm_start_infeasible']} / {v['requests']} | "
                     f"{v['max_error']:.3f} | {v['frac_outside_mean']:.3f} / {v['frac_outside_max']:.3f} |")
    lines += ["", (f"## Performance per platform ({' and '.join(specs)} pooled, 60 episodes; times on one CPU core; "
                   "CPU energy not measured on the cluster)"), "",
              ("| platform | loop | solves / episode | per solve (ms) | per RL call (ms) | RL + check (ms) | "
               f"compute per step (ms) | {wp_title} | violated episodes | violated windows (%) | min rho | gap (m) |"),
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]

    def f(v, fmt):
        return "--" if v is None else fmt.format(v)
    for key, v in res["performance"].items():
        s, tag = key.split("/")
        lines.append(f"| {s} | {dict(PERF_LOOPS)[tag]} | {v['solves_per_episode']:.1f} | {f(v['solve_ms'], '{:.1f}')} | "
                     f"{f(v['rl_ms'], '{:.2f}')} | {f(v['rl_check_ms'], '{:.2f}')} | {v['compute_ms_per_step']:.2f} | "
                     f"{v['waypoints']:.1f}{' / ' + f(v.get('visits'), '{:.1f}') if cube else ''} | {v['violated_episodes']} | {v['violated_windows_pct']:.2f} | "
                     f"{v['min_rho']:.3f} | {f(v['ref_gap_mean'], '{:.2f}')} |")
    (OUT / f"{args.out}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
