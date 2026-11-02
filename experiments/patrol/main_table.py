"""Numbers of the paper's main table (``tab:main``): I-STL always-on vs LIRRA (PPO C, seed 0, checked loop).

Reads the evaluation runs (``runs/patrol_eval/<system>/*_<spec>_fu_always_on`` and ``*_<spec>_fu_ppoC_shield``, injected latency d = 1) for the
X2 and the blimp, specs theta, phi, theta_plus, phi_plus and their uncertain-predicate versions theta_u, phi_u,
theta_plus_u, phi_plus_u, and writes ``docs/results/main_table.md``.

Definitions (the same as the performance table of ``followup_tables.py``):

* Fallbacks/step: mean over episodes of ``synthesis_call_fraction`` (runtime solve requests / planning steps;
  the t = 0- initial solve is excluded).
* Time [ms/step]: (sum over episodes of ``solve_wall_s.sum`` + ``inference_s.sum`` + ``check_s.sum``) / sum of
  ``n_steps``, i.e. total online computation divided by total planning steps (pooled over the 30 episodes).
* Waypoints: mean ``waypoints_reached`` (in patrol order); the cube specs also report ``waypoint_visits``.
* Violations: episodes with ``satisfied == False`` / episodes that ran; errored episodes are listed separately. For
  the ``_u`` specs the column counts violations of the true keep-out region drawn per episode inside the predicate
  uncertainty set (``satisfied_true``); ``satisfied`` is then the interval lower robustness, which implies it.
* Nominal-predicate ablation (``_u`` specs): always-on synthesis that plans and checks with the nominal predicates
  (``*_fu_always_on_nominal``), scored against the same true regions.
* Per call: solve = total solve wall / number of runtime solves; inference and check = total time / number of calls.

CPU energy is not tabulated: the in-loop RAPL numbers are not valid measurements (see the energy section of the
output and ``energy_diagnostic``).

    uv run --no-sync python experiments/patrol/main_table.py
"""

from __future__ import annotations

import argparse
import json

import numpy as np
from followup_tables import EVAL, OUT, loop_perf, stats

SYSTEMS = ["x2", "blimp"]
SPECS = ["theta", "phi", "theta_plus", "phi_plus", "theta_u", "phi_u", "theta_plus_u", "phi_plus_u"]
LOOPS = {"istl": "always_on", "lirra": "ppoC_shield"}
# run family of the LIRRA rows per system: "fu" = the follow-up runs of the original proposers, "fulm" = the
# blimp proposers retrained with as many expert labels as the X2. Always-on
# I-STL and the nominal-predicate ablation do not depend on the proposer and always come from "fu".
RUNS = {"x2": "fu", "blimp": "fulm"}


def episodes(system: str, spec: str, tag: str, run: str = "fu") -> list[dict] | None:
    """Completed episodes of the latest ``*_<spec>_<run>_<tag>`` evaluation."""
    ds = sorted(d for d in (EVAL / system).glob(f"*_{spec}_{run}_{tag}") if (d / "episodes.jsonl").exists())
    if not ds:
        return None
    with open(ds[-1] / "episodes.jsonl") as fh:
        return [e for e in map(json.loads, fh) if "n_steps" in e] or None


def errors(system: str, spec: str, tag: str, run: str = "fu") -> list[dict]:
    """Episodes that raised (logged by evaluate.py as records with an ``error`` field)."""
    ds = sorted(d for d in (EVAL / system).glob(f"*_{spec}_{run}_{tag}") if (d / "episodes.jsonl").exists())
    with open(ds[-1] / "episodes.jsonl") as fh:
        return [e for e in map(json.loads, fh) if "error" in e]


def energy_diagnostic(E: list[dict]) -> dict:
    """In-loop net package energy of the solves (idle baseline subtracted), per planning step and as the
    implied net power of the solver (net energy / solve wall time), with its spread over episodes."""
    en = np.array([e["solve_energy_pkg_j"]["sum"] for e in E])
    wall = np.array([e["solve_wall_s"]["sum"] for e in E])
    p = en[wall > 0] / wall[wall > 0]
    return {"j_per_step": en.sum() / sum(e["n_steps"] for e in E), "net_w": en.sum() / wall.sum(),
            "net_w_min": float(p.min()), "net_w_max": float(p.max())}


def true_region(E: list[dict]) -> dict:
    """``_u`` specs: violations of the true keep-out region (episodes, mean % of windows) and its min robustness."""
    if "satisfied_true" not in E[0]:
        return {}
    return {"true_violated": int(sum(not e["satisfied_true"] for e in E)),
            "true_windows_pct": 100.0 * float(np.mean([e["frac_violated_windows_true"] for e in E])),
            "min_rho_true": float(min(e["min_rho_true"] for e in E)),
            "lower_satisfied": int(sum(e["satisfied_lower"] for e in E))}


def collect() -> dict:
    res = {}
    for s in SYSTEMS:
        for sp in SPECS:
            r = {}
            for name, tag in LOOPS.items():
                run = RUNS[s] if name == "lirra" else "fu"
                E = episodes(s, sp, tag, run)
                r[name] = {**stats(E), **loop_perf(E), "errors": errors(s, sp, tag, run),
                           "energy": energy_diagnostic(E), **true_region(E), "run": run}
            if sp.endswith("_u"):
                E = episodes(s, sp, "always_on_nominal")
                r["nominal"] = {**stats(E), **true_region(E), "errors": errors(s, sp, "always_on_nominal")}
            res[f"{s}/{sp}"] = r
    return res


def fr(v: float) -> str:  # .003
    return f"{v:.3f}".lstrip("0")


def ms(v: float | None, n: int = 1) -> str:
    return "--" if v is None else f"{v:.{n}f}"


def render(res: dict) -> str:
    sym = {"theta": "theta", "phi": "phi", "theta_plus": "theta+", "phi_plus": "phi+", "theta_u": "theta^u",
           "phi_u": "phi^u", "theta_plus_u": "theta+^u", "phi_plus_u": "phi+^u"}
    L = ["# Main table (tab:main): I-STL always-on vs LIRRA (PPO C, seed 0)", "",
         "Generated by `experiments/patrol/main_table.py` from the follow-up runs (`*_<spec>_fu_always_on`, "
         "`*_<spec>_<run>_ppoC_shield`, LIRRA run family per system: "
         + ", ".join(f"{k} `{v}`" for k, v in RUNS.items()) + "; `fulm` = blimp proposers retrained with the X2 label "
         "count; injected latency d = 1, 10 starts x 3 uncertainty levels). Time per step = "
         "(solve + inference + check wall time) / planning steps, pooled over episodes; one Gurobi thread, "
         "several evaluations sharing the node. Waypoints for the cube specs: in patrol order (order-free visits). "
         "Uncertain-predicate specs (^u): Viol. counts violations of the true keep-out region.",
         "", "| System | Spec | Fallbacks/step | Time I-STL [ms/step] | Time LIRRA [ms/step] | Waypoints I-STL | "
         "Waypoints LIRRA | Viol. (LIRRA) |", "|---|---|---|---|---|---|---|---|"]
    for key, r in res.items():
        s, sp = key.split("/")
        a, b = r["istl"], r["lirra"]
        wp = (lambda x: f"{x['waypoints']:.1f}" + (f" ({x['visits']:.1f})" if x.get("visits") is not None
                                                     and "_plus" in sp else ""))
        viol = b.get("true_violated", b["violated_episodes"])
        L.append(f"| {s} | {sym[sp]} | {fr(b['calls'])} | {ms(a['compute_ms_per_step'])} | "
                 f"{ms(b['compute_ms_per_step'])} | {wp(a)} | {wp(b)} | "
                 f"{viol}/{b['episodes']} |")
    L += ["", "## Uncertain predicates: true-region violations", "",
          "True keep-out region drawn per episode inside the predicate uncertainty set (half-width +-0.05 m, "
          "rotation +-0.05 rad; same region for every loop of a start). Nominal = always-on synthesis planning and "
          "checking with the nominal predicates. Episodes violated / episodes (mean % of violated windows); "
          "min rho against the true region.", "",
          "| System | Spec | I-STL (interval) | LIRRA (interval) | I-STL (nominal predicates) | min rho true "
          "I-STL / LIRRA / nominal | nominal: interval lower bound satisfied |", "|---|---|---|---|---|---|---|"]
    for key, r in res.items():
        s, sp = key.split("/")
        if "nominal" not in r:
            continue
        a, b, n = r["istl"], r["lirra"], r["nominal"]

        def tv(x):
            return f"{x['true_violated']}/{x['episodes']} ({x['true_windows_pct']:.2f}%)"
        L.append(f"| {s} | {sym[sp]} | {tv(a)} | {tv(b)} | {tv(n)} | {a['min_rho_true']:+.3f} / "
                 f"{b['min_rho_true']:+.3f} / {n['min_rho_true']:+.3f} | {n['lower_satisfied']}/{n['episodes']} |")
    L += ["", "## Details", "",
          "| System | Spec | I-STL sat. / episodes | I-STL errored | I-STL deadline misses | LIRRA sat. / episodes | "
          "LIRRA errored | LIRRA deadline misses | LIRRA fallbacks / episode | solve per fallback (ms) | "
          "I-STL solve per call (ms) | inference per call (ms) | check per call (ms) | solves without plan (I-STL / LIRRA) |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for key, r in res.items():
        s, sp = key.split("/")
        a, b = r["istl"], r["lirra"]

        def err(x):
            return "; ".join(f"level {e['level']} start {e['seed']}: {e['error']}" for e in x["errors"]) or "0"
        L.append(f"| {s} | {sym[sp]} | {a['satisfied']}/{a['episodes']} | {err(a)} | {a['deadline_miss']} | "
                 f"{b['satisfied']}/{b['episodes']} | {err(b)} | {b['deadline_miss']} | "
                 f"{b['solves_per_episode']:.2f} | {ms(b['solve_ms'])} | {ms(a['solve_ms'])} | "
                 f"{ms(b['rl_ms'], 2)} | {ms(b['check_ms'], 2)} | {a['no_plan']} / {b['no_plan']} |")
    L += ["", "Deadline misses are 0 by construction in these runs (injected latency: every result is used d steps "
          "after its request); measured-latency runs exist only for theta/phi (`*_fu_*_measured`). Solves without plan: "
          "runtime solves whose Gurobi status was neither optimal nor suboptimal (e.g. the 0.8 d dt time limit hit "
          "without an incumbent); the vehicle keeps the cached plan.", "",
          "## CPU energy: not valid", "",
          "In-loop net package energy of the solves (RAPL `package-0`, idle power measured once over 3 s at process "
          "start and subtracted), per planning step and as implied net solver power. Not usable for the paper: "
          "theta/phi values are negative, the implied power of the same solver differs between loops, the node has "
          "two sockets but only `package-0` is read, and 8-14 evaluations shared the package. Inference and check "
          "energy are not measured in the loop at all (only micro-benchmarks at one state in summary.json).", "",
          "| System | Spec | I-STL J/step | I-STL net W (episode min..max) | LIRRA solve J/step | "
          "LIRRA net W (episode min..max) |", "|---|---|---|---|---|---|"]
    for key, r in res.items():
        s, sp = key.split("/")
        a, b = r["istl"]["energy"], r["lirra"]["energy"]
        L.append(f"| {s} | {sym[sp]} | {a['j_per_step']:.3f} | {a['net_w']:.1f} ({a['net_w_min']:.1f}..{a['net_w_max']:.1f}) | "
                 f"{b['j_per_step']:.4f} | {b['net_w']:.1f} ({b['net_w_min']:.1f}..{b['net_w_max']:.1f}) |")
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--blimp-runs", default=RUNS["blimp"], choices=["fu", "fulm"],
                    help="LIRRA rows of the blimp: original proposers (fu) or label-matched retraining (fulm)")
    ap.add_argument("--out", default="main_table", help="file stem under docs/results")
    args = ap.parse_args()
    RUNS["blimp"] = args.blimp_runs
    text = render(collect())
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{args.out}.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
