#!/usr/bin/env python3
"""
Load eval_diagnostics.npz and report whether action_mean collapses to constant
and whether log_std is very small (variance collapse).
Usage: python scripts/inspect_eval_diagnostics.py [path/to/eval_diagnostics.npz]
"""

import sys
from pathlib import Path

import numpy as np

def main():
    if len(sys.argv) > 1:
        path = Path(sys.argv[1])
    else:
        path = Path(__file__).resolve().parent.parent / "runs/anti_lazy_v7_no_tilt_term/eval/eval_diagnostics.npz"
    if not path.exists():
        print(f"File not found: {path}")
        sys.exit(1)
    d = np.load(path)
    steps = d["step"]
    action_mean = d["action_mean"]
    action_mean_norm = d["action_mean_norm"]
    log_std = d["log_std"]
    log_std_mean = d["log_std_mean"]
    velocity_reward = d["velocity_reward"]
    forward_velocity = d["forward_velocity"]
    reward = d["reward"]

    n = len(steps)
    print("Eval diagnostics:", path)
    print("  Steps:", n)
    print()
    print("Action mean (policy output):")
    print("  Norm per step: min={:.4f}, max={:.4f}, mean={:.4f}, std={:.4f}".format(
        action_mean_norm.min(), action_mean_norm.max(), action_mean_norm.mean(), action_mean_norm.std()))
    if action_mean_norm.std() < 0.01:
        print("  -> Action mean is effectively CONSTANT (collapse to fixed output).")
    else:
        print("  -> Action mean varies over steps.")
    print()
    print("Log_std (exploration):")
    print("  Mean(log_std) per step: min={:.4f}, max={:.4f}, mean={:.4f}".format(
        log_std_mean.min(), log_std_mean.max(), log_std_mean.mean()))
    if log_std_mean.mean() < -2.0:
        print("  -> Log_std very negative: policy is almost deterministic (variance collapse).")
    else:
        print("  -> Log_std in reasonable range.")
    print()
    print("Reward components (first 10 and last 10 steps):")
    print("  Step  velocity_reward  forward_velocity   reward")
    for i in [0, 1, 2, 5, 9] + ([] if n <= 20 else [n-10, n-5, n-2, n-1]):
        if i < n:
            print("  {:4d}  {:16.4f}  {:17.4f}  {:8.2f}".format(
                int(steps[i]), float(velocity_reward[i]), float(forward_velocity[i]), float(reward[i])))
    print()
    print("  velocity_reward: sum={:.2f}, mean={:.4f}, ever_positive={}".format(
        velocity_reward.sum(), velocity_reward.mean(), (velocity_reward > 0).any()))
    print("  forward_velocity: mean={:.4f}, max={:.4f}".format(forward_velocity.mean(), forward_velocity.max()))

    out_csv = path.with_suffix(".summary.txt")
    with open(out_csv, "w") as f:
        f.write("action_mean_norm_std={}\n".format(float(action_mean_norm.std())))
        f.write("log_std_mean_mean={}\n".format(float(log_std_mean.mean())))
        f.write("velocity_reward_sum={}\n".format(float(velocity_reward.sum())))
        f.write("forward_velocity_mean={}\n".format(float(forward_velocity.mean())))
    print()
    print("Wrote summary to", out_csv)


if __name__ == "__main__":
    main()
