"""Behaviour cloning of the expert synthesis segments (any patrol system).

Loss per sample (inputs normalised by the input half-range ``r``)::

    || (squash(pi(o)) - U*) / r ||^2 / (N m)  +  lambda_cert * max(0, eps_cert - cert(pi(o)))

``cert`` is the differentiable runtime check (:func:`patrol_certificate`): the second term
pushes proposals to be feasible points of the synthesis problem with a margin, which a
plain MSE does not (small errors already break the terminal-set or margin constraints).

    uv run python experiments/patrol/train_bc.py --system di --spec theta \
        --data runs/patrol_dataset/di/<run>/dataset.npz
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import jax
import jax.numpy as jnp
import numpy as np
import optax
from common import level_tables_summary, load_system

from rl_stl.policies.segment import SegmentPolicy, init_mlp, mlp, patrol_certificate, squash
from rl_stl.utils.provenance import create_run_dir, write_summary


def load(paths: list[str]) -> dict[str, np.ndarray]:
    parts = [dict(np.load(p, allow_pickle=False)) for p in paths]
    keys = [k for k in parts[0] if all(k in p for p in parts)]  # e.g. "driver" exists only in DAgger rounds
    return {k: np.concatenate([p[k] for p in parts]) for k in keys}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--system", default="di", choices=["di", "x2", "blimp"])
    ap.add_argument("--spec", default="theta", choices=["theta", "phi", "theta_plus", "phi_plus", "theta_u", "phi_u", "theta_plus_u", "phi_plus_u"])
    ap.add_argument("--hidden", type=int, nargs="+", default=[512, 512, 512])
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lambda-cert", type=float, default=1.0)
    ap.add_argument("--eps-cert", type=float, default=0.01)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init", default=None, help="warm-start from a SegmentPolicy .npz")
    ap.add_argument("--reference", default="rerooted", choices=["rerooted", "clock"],
                    help="mission reference the dataset was generated with")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    _, _, T = load_system(args.system, args.spec, args.reference)
    D = load(args.data)
    if D["obs"].shape[1] != T.obs_size:
        raise SystemExit(f"dataset obs width {D['obs'].shape[1]} != {T.obs_size} ({args.reference} reference)")
    D = {k: v for k, v in D.items() if v.dtype.kind in "fiub"}
    keep = D["check_ok"].astype(bool)
    D = {k: v[keep] for k, v in D.items()}
    n = len(D["obs"])
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n)
    n_val = int(args.val_frac * n)
    val, tr = perm[:n_val], perm[n_val:]
    obs_mean = D["obs"][tr].mean(0)
    obs_std = D["obs"][tr].std(0) + 1e-3
    run_dir = create_run_dir(f"patrol_bc/{args.system}", vars(args), seed=args.seed,
                             tag=args.tag or args.spec, tables=level_tables_summary(T))

    key = jax.random.PRNGKey(args.seed)
    if args.init:
        pol0 = SegmentPolicy.load(args.init)
        params, obs_mean, obs_std = pol0.params, pol0.obs_mean, pol0.obs_std
    else:
        params = init_mlp(key, [T.obs_size, *args.hidden, T.action_size])
    r_half = (T.u_hi - T.u_lo) / 2

    J = {k: jnp.asarray(D[k]) for k in ("obs", "U", "level", "x", "hist_y", "hist_mask", "first_window")}

    def sample_loss(prm, i):
        o = (J["obs"][i] - obs_mean) / obs_std
        U = squash(T, mlp(prm, o), J["level"][i], jnp)
        mse = jnp.mean(((U - J["U"][i]) / r_half) ** 2)
        cert, _ = patrol_certificate(T, J["x"][i], U, J["hist_y"][i], J["hist_mask"][i],
                                     J["first_window"][i], J["level"][i])
        return mse + args.lambda_cert * jnp.maximum(0.0, args.eps_cert - cert), (mse, cert)

    def batch_loss(prm, idx):
        loss, (mse, cert) = jax.vmap(sample_loss, in_axes=(None, 0))(prm, idx)
        return jnp.mean(loss), (jnp.mean(mse), jnp.mean(cert >= 0))

    steps_per_epoch = max(1, len(tr) // args.batch)
    sched = optax.cosine_decay_schedule(args.lr, args.epochs * steps_per_epoch, alpha=0.02)
    opt = optax.adamw(sched, weight_decay=1e-5)
    opt_state = opt.init(params)

    @jax.jit
    def train_step(prm, st, idx):
        (loss, aux), g = jax.value_and_grad(batch_loss, has_aux=True)(prm, idx)
        upd, st = opt.update(g, st, prm)
        return optax.apply_updates(prm, upd), st, loss, aux

    eval_fn = jax.jit(batch_loss)
    t0 = time.perf_counter()
    log = []
    for ep in range(args.epochs):
        order = rng.permutation(tr)
        for s in range(steps_per_epoch):
            idx = jnp.asarray(order[s * args.batch:(s + 1) * args.batch])
            params, opt_state, loss, (mse, okf) = train_step(params, opt_state, idx)
        if ep % 10 == 0 or ep == args.epochs - 1:
            vl, (vmse, vok) = eval_fn(params, jnp.asarray(val))
            rec = {"epoch": ep, "train_loss": float(loss), "val_loss": float(vl), "val_mse": float(vmse),
                   "val_cert_ok": float(vok), "wall_s": time.perf_counter() - t0}
            log.append(rec)
            print(rec, flush=True)
    pol = SegmentPolicy(params, obs_mean, obs_std, tuple(args.hidden), T.reference, T.ref_input_scale)
    pol.save(run_dir / "policy.npz")
    write_summary(run_dir, {"n_train": len(tr), "n_val": len(val), "final": log[-1], "log": log,
                            "obs_size": T.obs_size, "action_size": T.action_size})
    print(run_dir)


if __name__ == "__main__":
    main()
