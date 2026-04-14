#!/usr/bin/env python3
"""
Compare ODETripodPolicy vs MlpPolicy on the same IMU-matched environment.

- Prints total and export-oriented parameter counts (ODE actor-only excludes critic MLP).
- Optionally evaluates mean return from trained checkpoints (same env, deterministic).

Usage:
  cd e90_repo
  python scripts/compare_tier_policies_imu.py --config configs/config_teensy_imu_compare.yaml
  python scripts/compare_tier_policies_imu.py --config configs/config_teensy_imu_compare.yaml \\
      --ode-checkpoint checkpoints/ode.zip --mlp-checkpoint checkpoints/mlp.zip \\
      --n-eval-episodes 5
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from envs.hexapod_env import make_hexapod_env
from models.ode_gait import ODETripodPolicy  # noqa: F401 — required for load


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def policy_kwargs_ode(ppo_config: dict) -> dict:
    ode_cfg = ppo_config.get("ode_tripod", {}) or {}
    net_arch = ppo_config.get("policy_kwargs", {}).get("net_arch")
    hidden_sizes = (64, 64)
    if isinstance(net_arch, dict) and net_arch.get("pi"):
        hidden_sizes = tuple(int(x) for x in net_arch["pi"])
    elif isinstance(net_arch, list):
        hidden_sizes = tuple(int(x) for x in net_arch)
    fr = ode_cfg.get("freq_hz_range", [0.35, 1.6])
    base = {
        "sim_time_in_obs": bool(ode_cfg.get("sim_time_in_obs", True)),
        "freq_hz_range": (float(fr[0]), float(fr[1])),
        "max_amp_flex": float(ode_cfg.get("max_amp_flex", 0.95)),
        "max_amp_abd": float(ode_cfg.get("max_amp_abd", 0.95)),
        "flex_sign": float(ode_cfg.get("flex_sign", 1.0)),
        "abd_sign": float(ode_cfg.get("abd_sign", 1.0)),
        "residual_scale": float(ode_cfg.get("residual_scale", 0.2)),
        "hidden_sizes": hidden_sizes,
    }
    extra = {k: v for k, v in ppo_config.get("policy_kwargs", {}).items() if k != "net_arch"}
    base.update(extra)
    return base


def count_total_params(policy) -> int:
    return sum(p.numel() for p in policy.parameters())


def count_ode_actor_params(policy) -> int:
    """Parameters needed for deterministic action on Teensy (no value head)."""
    return sum(p.numel() for n, p in policy.named_parameters() if not n.startswith("_value_net"))


def count_mlp_actorish_params(policy) -> int:
    """
    SB3 MlpPolicy shares mlp_extractor between pi and vf; deployment often needs full trunk + action.
    We report total policy params and note shared trunk in the printed summary.
    """
    return sum(p.numel() for p in policy.parameters())


def make_vec_env(config: dict, seed: int = 0) -> DummyVecEnv:
    def _init():
        env = make_hexapod_env(config)
        env.reset(seed=seed)
        return Monitor(env)

    return DummyVecEnv([_init])


def fresh_ppo_ode(env, config: dict, seed: int) -> PPO:
    ppo_config = copy.deepcopy(config["ppo"])
    ppo_config["use_mlp_policy"] = False
    ppo_config["use_ode_tripod_policy"] = True
    pk = policy_kwargs_ode(ppo_config)
    return PPO(
        ODETripodPolicy,
        env,
        learning_rate=3e-4,
        seed=seed,
        policy_kwargs=pk,
        n_steps=128,
        batch_size=32,
    )


def fresh_ppo_mlp(env, config: dict, seed: int) -> PPO:
    ppo_config = copy.deepcopy(config["ppo"])
    ppo_config["use_mlp_policy"] = True
    ppo_config["use_ode_tripod_policy"] = False
    return PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        seed=seed + 1,
        policy_kwargs=dict(ppo_config.get("policy_kwargs", {})),
        n_steps=128,
        batch_size=32,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare ODE-tripod vs MLP on IMU env (params + optional eval).")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config_teensy_imu_compare.yaml",
        help="YAML with shared env/reward (Teensy IMU + sim_time).",
    )
    parser.add_argument("--ode-checkpoint", type=str, default=None, help="Optional PPO zip trained with ODETripodPolicy.")
    parser.add_argument("--mlp-checkpoint", type=str, default=None, help="Optional PPO zip trained with MlpPolicy.")
    parser.add_argument("--n-eval-episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cfg_path = PROJECT_ROOT / args.config if not Path(args.config).is_absolute() else Path(args.config)
    config = load_config(str(cfg_path))
    env = make_vec_env(config, seed=args.seed)

    obs_dim = env.observation_space.shape[0]
    print(f"Config: {cfg_path}")
    print(f"Observation dim: {obs_dim} (expect imu6_prev + phase + sim_time => 21)")
    print()

    model_ode = fresh_ppo_ode(env, config, args.seed)
    model_mlp = fresh_ppo_mlp(env, config, args.seed)

    ode_policy = model_ode.policy
    mlp_policy = model_mlp.policy

    ode_total = count_total_params(ode_policy)
    ode_actor = count_ode_actor_params(ode_policy)
    ode_critic = ode_total - ode_actor

    mlp_total = count_mlp_actorish_params(mlp_policy)

    print("=== Parameter counts (untrained init; architecture from config) ===")
    print(f"ODETripodPolicy: total={ode_total:,}  actor_export≈{ode_actor:,}  critic_only≈{ode_critic:,}")
    print(f"MlpPolicy:       total={mlp_total:,}  (pi/vf share MlpExtractor — export size ≈ total if distilling full policy)")
    print()
    print("Bytes (float32) rough: ODE actor ≈ {:.1f} KiB".format(ode_actor * 4 / 1024))
    print("Bytes (float32) rough: MLP total ≈ {:.1f} KiB".format(mlp_total * 4 / 1024))
    print()

    if args.ode_checkpoint or args.mlp_checkpoint:
        print("=== Evaluation (deterministic) ===")

    if args.ode_checkpoint:
        ck = Path(args.ode_checkpoint)
        if not ck.is_file():
            raise FileNotFoundError(ck)
        loaded = PPO.load(str(ck), env=env)
        mean_r, std_r = evaluate_policy(
            loaded,
            env,
            n_eval_episodes=args.n_eval_episodes,
            deterministic=True,
        )
        print(f"ODE checkpoint {ck.name}: mean_return={mean_r:.3f} +/- {std_r:.3f} ({args.n_eval_episodes} ep)")

    if args.mlp_checkpoint:
        ck = Path(args.mlp_checkpoint)
        if not ck.is_file():
            raise FileNotFoundError(ck)
        loaded = PPO.load(str(ck), env=env)
        mean_r, std_r = evaluate_policy(
            loaded,
            env,
            n_eval_episodes=args.n_eval_episodes,
            deterministic=True,
        )
        print(f"MLP checkpoint {ck.name}: mean_return={mean_r:.3f} +/- {std_r:.3f} ({args.n_eval_episodes} ep)")

    env.close()


if __name__ == "__main__":
    main()
