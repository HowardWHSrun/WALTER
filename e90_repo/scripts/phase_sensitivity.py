#!/usr/bin/env python3
"""
Phase sensitivity check: does the policy's action_mean change when only the
phase (sin/cos) in the observation changes? If not, the policy ignores phase
and cannot produce a rhythmic gait from obs alone.

Usage:
  python scripts/phase_sensitivity.py <path_to_model.zip> [--config config.yaml] [--steps 100] [--out output.npz]
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import torch

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from envs.hexapod_env import make_hexapod_env
from stable_baselines3 import PPO


def load_config(config_path: str) -> dict:
    import yaml
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_phase_indices(config: dict) -> int:
    """Return start index of phase in observation vector."""
    dim = 0
    if config.get("env", {}).get("obs_include_joint_pos", True):
        dim += 12
    if config.get("env", {}).get("obs_include_joint_vel", True):
        dim += 12
    if config.get("env", {}).get("obs_include_torso_quat", False):
        dim += 4
    if config.get("env", {}).get("obs_include_torso_vel", True):
        dim += 3
    if config.get("env", {}).get("obs_include_contact", False):
        dim += 6
    if config.get("env", {}).get("obs_include_torso_height", False):
        dim += 1
    return dim  # phase starts here, length 2


def main():
    parser = argparse.ArgumentParser(description="Phase sensitivity: does action_mean change with phase?")
    parser.add_argument("model_path", type=str, help="Path to trained model .zip")
    parser.add_argument("--config", "-c", type=str, default=None, help="Config YAML (default: same dir as model)")
    parser.add_argument("--steps", "-n", type=int, default=100, help="Number of steps to simulate")
    parser.add_argument("--out", "-o", type=str, default=None, help="Save action_mean and phase to this .npz file")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        sys.exit(1)
    config_path = args.config or (model_path.parent / "config.yaml")
    if not Path(config_path).exists():
        config_path = PROJECT_ROOT / "configs" / "config_v7_anti_lazy.yaml"
    config = load_config(config_path)

    # Get obs size from env
    env = make_hexapod_env(config)
    obs_size = int(env.observation_space.shape[0])
    phase_start = get_phase_indices(config)
    env.close()

    # Load model
    model = PPO.load(str(model_path), device=args.device)
    freq_hz = config.get("env", {}).get("obs_phase_frequency_hz", 0.9)
    steps = args.steps

    # Run 1: phase fixed at 0 (sin=0, cos=1)
    obs_fixed = np.zeros((obs_size,), dtype=np.float32)
    obs_fixed[phase_start] = 0.0
    obs_fixed[phase_start + 1] = 1.0
    action_means_fixed = []
    with torch.no_grad():
        for _ in range(steps):
            obs_t = torch.as_tensor(obs_fixed, dtype=torch.float32, device=model.device).unsqueeze(0)
            action_mean, _, _ = model.policy.forward(obs_t, deterministic=True)
            action_means_fixed.append(action_mean.cpu().numpy().squeeze(0))
    action_means_fixed = np.array(action_means_fixed)

    # Run 2: phase advancing with step
    action_means_advancing = []
    with torch.no_grad():
        for step in range(steps):
            phase = 2 * np.pi * freq_hz * step / 50.0  # advance phase over steps
            phase_sin = float(np.sin(phase))
            phase_cos = float(np.cos(phase))
            obs_adv = np.zeros((obs_size,), dtype=np.float32)
            obs_adv[phase_start : phase_start + 2] = [phase_sin, phase_cos]
            obs_t = torch.as_tensor(obs_adv, dtype=torch.float32, device=model.device).unsqueeze(0)
            action_mean, _, _ = model.policy.forward(obs_t, deterministic=True)
            action_means_advancing.append(action_mean.cpu().numpy().squeeze(0))
    action_means_advancing = np.array(action_means_advancing)

    # Compare
    norm_fixed = np.linalg.norm(action_means_fixed, axis=1)
    norm_advancing = np.linalg.norm(action_means_advancing, axis=1)
    std_fixed = np.std(action_means_fixed)
    std_advancing = np.std(action_means_advancing)
    print("Phase sensitivity check")
    print("  Phase fixed (sin=0, cos=1):")
    print(f"    action_mean norm range: [{norm_fixed.min():.4f}, {norm_fixed.max():.4f}], std(all dims): {std_fixed:.4f}")
    print("  Phase advancing with step:")
    print(f"    action_mean norm range: [{norm_advancing.min():.4f}, {norm_advancing.max():.4f}], std(all dims): {std_advancing:.4f}")
    if std_advancing > 1.5 * std_fixed:
        print("  -> Policy output CHANGES with phase (phase is used).")
    else:
        print("  -> Policy output barely changes with phase (phase may be ignored). Consider imitation_action_mix or phase-weighted encoder.")

    if args.out:
        np.savez(
            args.out,
            action_mean_fixed=action_means_fixed,
            action_mean_advancing=action_means_advancing,
            norm_fixed=norm_fixed,
            norm_advancing=norm_advancing,
        )
        print(f"  Saved to {args.out}")


if __name__ == "__main__":
    main()
