"""PPO fine-tuning of the segment proposer on the planning-level patrol env (BC -> PPO).

The policy network is initialised from a behaviour-cloned :class:`SegmentPolicy` (same MLP,
same fixed observation normalisation; Brax's own normalisation is off), so PPO starts from
the distilled synthesis expert. Without ``--init`` it trains from scratch (PPO-only baseline).
The result is saved back as a ``SegmentPolicy`` (``policy.npz``) for the runtime evaluation.

    uv run python experiments/patrol/train_ppo.py --system di --spec theta \
        --init runs/patrol_bc/di/<run>/policy.npz --num-timesteps 20000000 --device gpu
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "skydio_x2"))

from rl_stl.utils.device import configure_jax_device

AP = argparse.ArgumentParser(description=__doc__.split("\n")[0])
AP.add_argument("--system", default="di", choices=["di", "x2", "blimp"])
AP.add_argument("--spec", default="theta", choices=["theta", "phi", "theta_plus", "phi_plus", "theta_u", "phi_u", "theta_plus_u", "phi_plus_u"])
AP.add_argument("--init", default=None, help="BC SegmentPolicy .npz")
AP.add_argument("--hidden", type=int, nargs="+", default=[512, 512, 512], help="only without --init")
AP.add_argument("--num-timesteps", type=int, default=2_000_000)
AP.add_argument("--num-envs", type=int, default=256)
AP.add_argument("--batch-size", type=int, default=256)
AP.add_argument("--num-minibatches", type=int, default=8)
AP.add_argument("--unroll-length", type=int, default=10)
# 1e-4 diverges from a BC start (tanh_normal std head runs away: mean std 0.05 -> 48, KL ~1e11 on
# the blimp); 1e-5 and 3e-6 are stable, 1e-5 improves faster
AP.add_argument("--learning-rate", type=float, default=1e-5)
AP.add_argument("--entropy-cost", type=float, default=1e-4)
AP.add_argument("--init-std", type=float, default=0.05)
AP.add_argument("--episode-steps", type=int, default=120)
AP.add_argument("--c-cost", type=float, default=None, help="weight of the plan cost J (default: PlanEnvConfig)")
AP.add_argument("--c-cert", type=float, default=None, help="weight of the certificate hinge (default: PlanEnvConfig)")
AP.add_argument("--c-ok", type=float, default=None, help="bonus for a certified proposal (default: PlanEnvConfig)")
AP.add_argument("--eps-cert", type=float, default=None, help="certification margin of the reward (default: PlanEnvConfig)")
AP.add_argument("--fallback", action="store_true", help="fly the cached plan when a proposal fails the check")
AP.add_argument("--c-syn", type=float, default=None, help="price of a rejected proposal's synthesis call (default: PlanEnvConfig)")
AP.add_argument("--c-wp", type=float, default=None,
                help="path-accuracy bonus per waypoint pass, scaled by max(0, 1 - miss/wp_radius) (default 0: off)")
AP.add_argument("--wp-radius", type=float, default=None, help="pass distance [m] at which the bonus is 0 (default 0.5)")
AP.add_argument("--wp-target", default=None, choices=["executed", "plan"],
                help="passes scored by --c-wp: the executed step or the proposed segment (default executed)")
AP.add_argument("--num-evals", type=int, default=10)
AP.add_argument("--seed", type=int, default=0)
AP.add_argument("--device", default="auto", choices=["cpu", "gpu", "auto"])
AP.add_argument("--reference", default=None, choices=["rerooted", "clock"],
                help="mission reference of the reward (default: the BC policy's, else rerooted)")
AP.add_argument("--tag", default=None)
ARGS = AP.parse_args()
DEVICE = configure_jax_device(ARGS.device)

import functools  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from brax.training.agents.ppo import networks as ppo_networks  # noqa: E402
from brax.training.agents.ppo import train as ppo  # noqa: E402
from common import level_tables_summary, load_system  # noqa: E402
from mujoco_playground._src import wrapper  # noqa: E402
from train_common import ProgressLogger  # noqa: E402

from rl_stl.envs.patrol_plan import PatrolPlanEnv, PlanEnvConfig  # noqa: E402
from rl_stl.policies.segment import SegmentPolicy  # noqa: E402
from rl_stl.utils.device import jax_device_summary  # noqa: E402
from rl_stl.utils.provenance import create_run_dir, write_summary  # noqa: E402


def main() -> None:
    bc = SegmentPolicy.load(ARGS.init) if ARGS.init else None
    if ARGS.reference is None:
        ARGS.reference = bc.reference if bc else "rerooted"
    elif bc is not None and bc.reference != ARGS.reference:
        raise SystemExit(f"--init policy uses the {bc.reference} reference, not {ARGS.reference}")
    system, setups, T = load_system(ARGS.system, ARGS.spec, ARGS.reference)
    if bc is not None and bc.ref_input_scale != T.ref_input_scale:
        raise SystemExit(f"--init policy uses reference input scale {bc.ref_input_scale}, not {T.ref_input_scale}")
    p = setups[0].problem
    hidden = tuple(bc.hidden) if bc else tuple(ARGS.hidden)
    obs_mean = bc.obs_mean if bc else np.zeros(T.obs_size, np.float32)
    obs_std = bc.obs_std if bc else np.ones(T.obs_size, np.float32)
    weights = {k: v for k, v in dict(c_cost=ARGS.c_cost, c_cert=ARGS.c_cert, c_ok=ARGS.c_ok, eps_cert=ARGS.eps_cert,
                                     c_syn=ARGS.c_syn, c_wp=ARGS.c_wp, wp_radius=ARGS.wp_radius,
                                     wp_target=ARGS.wp_target).items() if v is not None}
    cfg = PlanEnvConfig(episode_steps=ARGS.episode_steps, **weights, fallback=ARGS.fallback,
                        uncertainty="disturbance" if system.name == "di" else "tube")
    env = PatrolPlanEnv(T, setups[0].nominal, p.track_weight, p.gamma0, p.n_cost, p.effort_weight,
                        obs_mean, obs_std, cfg)
    run_dir = create_run_dir(f"patrol_ppo/{ARGS.system}", vars(ARGS), seed=ARGS.seed,
                             tag=ARGS.tag or (ARGS.spec + ("_bc" if bc else "_scratch")),
                             device=jax_device_summary(), tables=level_tables_summary(T))
    print(run_dir, DEVICE, flush=True)
    factory = functools.partial(ppo_networks.make_ppo_networks, policy_hidden_layer_sizes=hidden,
                                value_hidden_layer_sizes=(256, 256, 256))
    common = dict(environment=env, episode_length=ARGS.episode_steps, num_envs=ARGS.num_envs,
                  batch_size=ARGS.batch_size, num_minibatches=ARGS.num_minibatches,
                  unroll_length=ARGS.unroll_length, learning_rate=ARGS.learning_rate,
                  entropy_cost=ARGS.entropy_cost, discounting=0.97, normalize_observations=False,
                  network_factory=factory, seed=ARGS.seed, num_evals=ARGS.num_evals,
                  wrap_env_fn=functools.partial(wrapper.wrap_for_brax_training, full_reset=True),
                  reward_scaling=1.0, num_updates_per_batch=4, clipping_epsilon=0.2)
    _, init_params, _ = ppo.train(num_timesteps=0, **common)
    norm, pol, val = init_params
    if bc is not None:
        pol = jax.tree.map(lambda x: x, pol)
        layers = pol["params"]
        n_layers = len(bc.params)
        act = T.action_size
        s0 = float(np.log(np.expm1(ARGS.init_std)))
        for i in range(n_layers):
            W, b = np.asarray(bc.params[i]["W"]), np.asarray(bc.params[i]["b"])
            key = f"hidden_{i}"
            if i < n_layers - 1:
                layers[key] = {"kernel": jnp.asarray(W), "bias": jnp.asarray(b)}
            else:
                K = np.zeros((W.shape[0], 2 * act), np.float32)
                K[:, :act] = W
                bb = np.full(2 * act, s0, np.float32)
                bb[:act] = b
                layers[key] = {"kernel": jnp.asarray(K), "bias": jnp.asarray(bb)}
        pol = {"params": layers}
    logger = ProgressLogger(run_dir)
    t0 = time.perf_counter()
    make_policy, params, _ = ppo.train(num_timesteps=ARGS.num_timesteps, restore_params=(norm, pol, val),
                                       restore_value_fn=False, progress_fn=logger, **common)
    train_s = time.perf_counter() - t0
    layers = params[1]["params"]
    n_layers = len(layers)
    out = []
    for i in range(n_layers):
        k, b = np.asarray(layers[f"hidden_{i}"]["kernel"]), np.asarray(layers[f"hidden_{i}"]["bias"])
        if i == n_layers - 1:
            k, b = k[:, :T.action_size], b[:T.action_size]
        out.append({"W": jnp.asarray(k), "b": jnp.asarray(b)})
    SegmentPolicy(out, np.asarray(obs_mean), np.asarray(obs_std), hidden, T.reference,
                  T.ref_input_scale).save(run_dir / "policy.npz")
    write_summary(run_dir, {"train_wall_s": train_s, "last_progress": logger.last})
    print(run_dir, logger.last, flush=True)


if __name__ == "__main__":
    main()
