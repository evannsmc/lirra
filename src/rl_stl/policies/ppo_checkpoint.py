"""Config schema of ``experiments/skydio_x2/train_ppo.py`` runs and a loader for their policies.

A run directory holds ``config.yaml`` (the resolved :class:`TrainConfig`) and
``policy_params.pkl`` (Brax ``(normalizer, policy, value)`` params). Reloading the
config requires the exact dataclass shape it was written with, so the schema lives
here and the training script imports it.

Importing this module imports JAX (through the environment config), so call
:func:`rl_stl.utils.device.configure_jax_device` first.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

from rl_stl.envs.skydio_x2.hover import HoverEnvConfig


@dataclasses.dataclass(frozen=True)
class PPOConfig:
    num_timesteps: int = 50_000
    num_evals: int = 3
    reward_scaling: float = 1.0
    episode_length: int = 200
    normalize_observations: bool = True
    action_repeat: int = 1
    unroll_length: int = 10
    num_minibatches: int = 4
    num_updates_per_batch: int = 2
    discounting: float = 0.99
    learning_rate: float = 3e-4
    entropy_cost: float = 1e-3
    num_envs: int = 64
    batch_size: int = 64
    num_eval_envs: int = 64
    num_resets_per_eval: int = 0
    clipping_epsilon: float = 0.2
    max_grad_norm: float | None = 1.0


@dataclasses.dataclass(frozen=True)
class NetworkConfig:
    policy_hidden_layer_sizes: tuple[int, ...] = (64, 64, 64)
    value_hidden_layer_sizes: tuple[int, ...] = (128, 128, 128)


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    seed: int = 0
    device: str = "auto"
    full_reset: bool = True
    wandb: bool = False
    wandb_project: str = "rl-stl"
    final_eval_envs: int = 16
    env: HoverEnvConfig = HoverEnvConfig()
    ppo: PPOConfig = PPOConfig()
    network: NetworkConfig = NetworkConfig()


def load_ppo_run(run_dir: str | Path):
    """Return ``(cfg, env, policy)`` for a trained run.

    ``env`` is the :class:`~rl_stl.envs.skydio_x2.hover.X2Hover` built from the run's own
    env config (so the observation layout matches the network); ``policy(obs, key)`` is the
    deterministic Brax inference function (not jitted).
    """
    import jax
    from brax.io import model as brax_model
    from brax.training.acme import running_statistics
    from brax.training.agents.ppo import networks as ppo_networks

    from rl_stl.envs.skydio_x2.hover import X2Hover, to_config_dict
    from rl_stl.utils.config import load_config

    run_dir = Path(run_dir)
    cfg = load_config(TrainConfig, run_dir / "config.yaml")
    env = X2Hover(to_config_dict(cfg.env))
    obs_shape = env.reset(jax.random.PRNGKey(0)).obs.shape
    preprocess = running_statistics.normalize if cfg.ppo.normalize_observations else (lambda x, y: x)
    network = ppo_networks.make_ppo_networks(
        obs_shape, env.action_size,
        preprocess_observations_fn=preprocess,
        policy_hidden_layer_sizes=cfg.network.policy_hidden_layer_sizes,
        value_hidden_layer_sizes=cfg.network.value_hidden_layer_sizes,
    )
    params = brax_model.load_params(str(run_dir / "policy_params.pkl"))
    return cfg, env, ppo_networks.make_inference_fn(network)(params, deterministic=True)
