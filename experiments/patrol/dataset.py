"""Expert dataset for the patrol benchmarks (behaviour cloning / DAgger), any system.

Episodes run the full shield (delay-compensated synthesis fallback) in the system's closed
loop (``f_syn + w`` for the double integrator, MuJoCo + tracker for the X2 and the blimp),
driven by a mixture of proposers: expert, noisy expert, nominal mission controller and, for
DAgger rounds, a learned policy. At every policy query the expert problem rooted at the
measured state is solved and stored with the observation, so labels cover the states the
closed loop visits, including boundary states (random starts near ``S`` moving inward),
deadline pressure (dwell inside ``S`` in the history) and every uncertainty level.

Each sample stores the observation, the expert input sequence, the level, the driver, and the
solver status, objective, recomputed robustness and terminal face.

DAgger round: drive with a learned policy (``--policy``, ``--mix 0 0 0 1``) and aggregate the
new labels with the earlier datasets in ``train_bc.py --data``. The policy's proposals can be
perturbed (``--policy-noise``, fraction of the input half-range): Gaussian noise on the first
``--noise-steps`` inputs of the proposed sequence, the ones flown before the next re-plan.
Noise on the whole sequence mostly breaks the terminal block, so the check rejects the proposal
and the solver flies, which visits the expert's states instead of the policy's.

    uv run python experiments/patrol/dataset.py --system blimp --spec theta --episodes 300
    uv run python experiments/patrol/dataset.py --system blimp --spec theta --reference clock \
        --policy runs/patrol_bc/blimp/<run>/policy.npz --mix 0 0 0 1 --policy-noise 0 0.05 0.1
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from common import feedback, latency_default, level_tables_summary, load_system

from rl_stl.evaluation.patrol import ExpertProposer, nominal_proposer
from rl_stl.policies.segment import LearnedProposer, SegmentPolicy, features, unsquash
from rl_stl.supervisor.receding import RecedingConfig, RecedingSupervisor
from rl_stl.synthesis.check import check_plan
from rl_stl.synthesis.istl import ISTLSettings
from rl_stl.utils.provenance import create_run_dir, write_summary


class NoisyProposer:
    """Learned proposer with Gaussian noise on its first ``steps`` inputs (DAgger exploration)."""

    def __init__(self, inner: LearnedProposer, sigma: np.ndarray, steps: int, lo: np.ndarray, hi: np.ndarray,
                 seed: int):
        self.inner, self.sigma, self.steps, self.lo, self.hi = inner, sigma, steps, lo, hi
        self.rng = np.random.default_rng(seed)

    def __call__(self, ctx):
        U = np.array(self.inner(ctx))
        k = min(self.steps, len(U))
        U[:k] = np.clip(U[:k] + self.sigma * self.rng.standard_normal(U[:k].shape), self.lo, self.hi)
        return U


class Labeller:
    """Wraps a driving proposer; labels every query with the expert solution."""

    def __init__(self, drive, expert: ExpertProposer, T, level: int, nominal, store: list, kind: str):
        self.drive, self.expert, self.T, self.level, self.nominal = drive, expert, T, level, nominal
        self.store, self.kind = store, kind

    def __call__(self, ctx):
        U = self.expert(ctx)
        res = self.expert.last
        if res.has_solution:
            chk = check_plan(self.expert.problem, ctx.x, res.X, res.U, ctx.history, None, ctx.first_window)
            mask = (np.arange(self.T.h - 1) >= ctx.first_window).astype(float)
            idx = self.nominal.index if ctx.waypoint_index is None else ctx.waypoint_index
            obs = features(self.T, ctx.x, ctx.history.y, mask, idx, self.level, x_ref=ctx.x_nom)
            self.store.append({
                "obs": obs.astype(np.float32), "U": U.astype(np.float32),
                "a": unsquash(self.T, U, self.level).astype(np.float32), "level": self.level,
                "x": ctx.x, "hist_y": ctx.history.y, "hist_mask": mask,
                "first_window": ctx.first_window, "k": ctx.k, "objective": res.objective,
                "status": str(res.status), "rho_lower": chk.rho_lower, "check_ok": chk.ok,
                "face": -1 if chk.face is None else chk.face, "solve_s": res.solve_time,
                "driver": self.kind})
        return U if self.drive is None else self.drive(ctx)


def worker(args) -> list[dict]:
    system_name, spec, episodes, seed, seconds, policy_path, mix, reference, policy_noise, noise_steps = args
    system, setups, T = load_system(system_name, spec, reference)
    rng = np.random.default_rng(seed)
    policy = SegmentPolicy.load(policy_path) if policy_path else None
    out: list[dict] = []
    for _ in range(episodes):
        level = int(rng.integers(len(setups)))
        st = system.build(level)  # fresh nominal controller state per episode
        kind = rng.choice(["expert", "noisy", "nominal", "policy"], p=mix)
        if kind == "policy" and policy is None:
            kind = "noisy"
        expert = ExpertProposer(st.problem, ISTLSettings(time_limit=2.0))
        if kind == "expert":
            drive = None
        elif kind == "noisy":
            scale = (st.problem.u_hi - st.problem.u_lo) / 2
            drive = ExpertProposer(st.problem, ISTLSettings(time_limit=2.0),
                                   noise=float(rng.choice([0.04, 0.1, 0.2])) * scale,
                                   seed=int(rng.integers(1 << 30)))
        elif kind == "nominal":
            drive = nominal_proposer
        else:
            drive = LearnedProposer(policy, T, level, st.nominal)
            sigma = float(rng.choice(policy_noise))
            if sigma > 0:
                lo, hi = T.u_lo + T.kz[level][0], T.u_hi - T.kz[level][0]
                drive = NoisyProposer(drive, sigma * (hi - lo) / 2, noise_steps, lo, hi,
                                      int(rng.integers(1 << 30)))
            kind = f"policy_sigma{sigma:g}"
        lab = Labeller(drive, expert, T, level, st.nominal, out, str(kind))
        sup = RecedingSupervisor(st.problem, st.K, st.nominal, lab,
                                 RecedingConfig(variant="shield", latency_steps=latency_default(system),
                                                query_while_pending=True, feedback=feedback(system),
                                                reference=reference, reference_input_scale=T.ref_input_scale),
                                 ISTLSettings())
        x = system.random_start(rng, st)
        n_steps = int(round(seconds / system.dt))
        sim = system.make_sim(st, int(rng.integers(1 << 30)), n_steps,
                              **({"disturbance": str(rng.choice(["uniform", "persistent"]))} if system.name == "di" else {}))
        x = sim.reset(x)
        try:
            sup.initial_plan(x)
        except RuntimeError:
            continue
        for k in range(n_steps):
            u, _ = sup.step(k, x)
            x_hat = sup.plan.state(k, st.problem.terminal, st.problem.model)
            u_hat = sup.plan.input(k, st.problem.terminal, st.problem.model)
            x = sim.step(x_hat, u_hat, u)
            if not np.all(np.isfinite(x)):
                break
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--system", default="di", choices=["di", "x2", "blimp"])
    ap.add_argument("--spec", default="theta", choices=["theta", "phi", "theta_plus", "phi_plus", "theta_u", "phi_u", "theta_plus_u", "phi_plus_u"])
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--workers", type=int, default=max(1, len(os.sched_getaffinity(0)) - 2),
                    help="default: allocated cores - 2 (respects the SLURM cgroup)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--policy", default=None, help="SegmentPolicy .npz for a DAgger round")
    ap.add_argument("--mix", type=float, nargs=4, default=[0.3, 0.4, 0.3, 0.0],
                    help="probabilities of expert/noisy/nominal/policy drivers")
    ap.add_argument("--policy-noise", type=float, nargs="+", default=[0.0],
                    help="per-episode choice of the noise std on policy proposals, fraction of the input half-range")
    ap.add_argument("--noise-steps", type=int, default=1, help="number of leading proposal inputs perturbed")
    ap.add_argument("--reference", default="rerooted", choices=["rerooted", "clock"],
                    help="mission reference of the expert objective and the observation")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    mix = np.asarray(args.mix) / np.sum(args.mix)
    _, _, T = load_system(args.system, args.spec, args.reference)
    run_dir = create_run_dir(f"patrol_dataset/{args.system}", vars(args), seed=args.seed,
                             tag=args.tag or args.spec, tables=level_tables_summary(T))
    per = [args.episodes // args.workers + (i < args.episodes % args.workers) for i in range(args.workers)]
    jobs = [(args.system, args.spec, n, args.seed * 1000 + i, args.seconds, args.policy, mix, args.reference,
             args.policy_noise, args.noise_steps)
            for i, n in enumerate(per) if n > 0]
    t0 = time.perf_counter()
    with mp.get_context("spawn").Pool(len(jobs)) as pool:
        parts = pool.map(worker, jobs)
    samples = [s for p in parts for s in p]
    arrays = {k: np.stack([np.asarray(s[k]) for s in samples]) for k in samples[0]}
    np.savez_compressed(run_dir / "dataset.npz", **arrays)
    summary = {"n_samples": len(samples), "wall_s": time.perf_counter() - t0,
               "frac_check_ok": float(np.mean(arrays["check_ok"])),
               "status_counts": {str(s): int(np.sum(arrays["status"] == s)) for s in np.unique(arrays["status"])},
               "solve_s_mean": float(np.mean(arrays["solve_s"])),
               "driver_counts": {str(d): int(np.sum(arrays["driver"] == d)) for d in np.unique(arrays["driver"])}}
    write_summary(run_dir, summary)
    print(run_dir, summary)


if __name__ == "__main__":
    main()
