#!/usr/bin/env python3
"""
Black-box "RL-style" search over open-loop tripod parameters (no neural net).

Evaluates mean episode return in HexapodEnv using the same analytic tripod as
`models/ode_gait/reference_from_phase.py`, with phase time = env._sim_time.

This is evolutionary / policy-search — good for finding Teensy-friendly (f, af, aa)
starting points before or alongside PPO.

Usage:
  cd e90_repo
  python scripts/rl_gait_blackbox_search.py --config configs/config_teensy_imu_compare.yaml --fast
  python scripts/rl_gait_blackbox_search.py --config configs/config_new_body_flat_ode_imu.yaml --iters 40 --episodes 2
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import gymnasium as gym

from envs.hexapod_env import make_hexapod_env

TRIPOD_A_LEGS = (0, 3, 4)
TRIPOD_B_LEGS = (1, 2, 5)
LEFT_LEGS = frozenset((1, 3, 5))


def tripod_actions_np(
    theta_a: float,
    amp_flex: float,
    amp_abd: float,
    flex_sign: float,
    abd_sign: float,
) -> np.ndarray:
    theta_b = theta_a + math.pi
    fs, ads = flex_sign, abd_sign
    flex_a = fs * amp_flex * math.sin(theta_a)
    flex_b = fs * amp_flex * math.sin(theta_b)
    abd_a = ads * (-amp_abd * math.cos(theta_a))
    abd_b = ads * (-amp_abd * math.cos(theta_b))
    ctrl = np.zeros(12, dtype=np.float32)
    for i in TRIPOD_A_LEGS:
        ctrl[i * 2] = abd_a
        ctrl[i * 2 + 1] = flex_a
    for i in TRIPOD_B_LEGS:
        abd_i = -abd_b if i in LEFT_LEGS else abd_b
        ctrl[i * 2] = abd_i
        ctrl[i * 2 + 1] = flex_b
    return np.clip(ctrl, -1.0, 1.0)


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def eval_params(
    env: gym.Env,
    freq_hz: float,
    amp_flex: float,
    amp_abd: float,
    flex_sign: float,
    abd_sign: float,
    phase_off: float,
    episodes: int,
    base_seed: int,
) -> float:
    rewards = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=base_seed + ep)
        total = 0.0
        terminated = truncated = False
        while not (terminated or truncated):
            e = env.unwrapped
            t = float(e._sim_time)
            theta_a = 2 * math.pi * freq_hz * t + phase_off
            action = tripod_actions_np(theta_a, amp_flex, amp_abd, flex_sign, abd_sign)
            obs, r, terminated, truncated, _ = env.step(action)
            total += float(r)
        rewards.append(total)
    return float(np.mean(rewards))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/config_teensy_imu_compare.yaml")
    p.add_argument("--iters", type=int, default=25)
    p.add_argument("--episodes", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fast", action="store_true", help="Shorter episodes (400 steps) for a quick demo")
    p.add_argument("--out", type=str, default="", help="Optional path to write best params YAML")
    args = p.parse_args()

    cfg_path = PROJECT_ROOT / args.config if not Path(args.config).is_absolute() else Path(args.config)
    config = load_config(str(cfg_path))
    env = make_hexapod_env(config)
    if args.fast:
        env = gym.wrappers.TimeLimit(env.unwrapped, max_episode_steps=400)

    rng = np.random.default_rng(args.seed)
    ode = (config.get("ppo") or {}).get("ode_tripod") or {}
    flex_sign = float(ode.get("flex_sign", -1.0))
    abd_sign = float(ode.get("abd_sign", 1.0))

    best = (-1e18, None)
    print(f"Black-box search: {args.iters} samples, {args.episodes} eval eps, config={cfg_path.name}")
    for k in range(args.iters):
        freq = float(rng.uniform(0.35, 1.35))
        amp_flex = float(rng.uniform(0.35, 0.85))
        amp_abd = float(rng.uniform(0.25, 0.65))
        phase_off = float(rng.uniform(-math.pi, math.pi))
        score = eval_params(
            env, freq, amp_flex, amp_abd, flex_sign, abd_sign, phase_off, args.episodes, args.seed + k * 17
        )
        if score > best[0]:
            best = (score, (freq, amp_flex, amp_abd, phase_off))
            print(f"  iter {k+1:3d}  NEW BEST mean_return={score:.1f}  f={freq:.3f} af={amp_flex:.3f} aa={amp_abd:.3f} ph={phase_off:+.3f}")
        elif (k + 1) % 5 == 0:
            print(f"  iter {k+1:3d}  mean_return={score:.1f}")

    env.close()
    score, params = best
    if params is None:
        print("No eval completed.")
        return
    freq, amp_flex, amp_abd, phase_off = params
    print()
    print("=== Best (copy to Teensy Serial: f / af / aa; phase is open-loop offset) ===")
    print(f"  mean_return: {score:.2f}")
    print(f"  f {freq:.4f}")
    print(f"  af {amp_flex:.4f}")
    print(f"  aa {amp_abd:.4f}")
    print(f"  # phase_off_rad {phase_off:.4f}  (Teensy sketch would need this added to theta if you use it)")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        dump = {
            "source_config": str(cfg_path),
            "mean_return": score,
            "freq_hz": freq,
            "amp_flex": amp_flex,
            "amp_abd": amp_abd,
            "phase_off_rad": phase_off,
            "flex_sign": flex_sign,
            "abd_sign": abd_sign,
        }
        with open(out_path, "w", encoding="utf-8") as f:
            yaml.dump(dump, f, default_flow_style=False)
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
