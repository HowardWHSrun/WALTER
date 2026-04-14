#!/usr/bin/env python3
"""Interactive MuJoCo viewer for trained hexapod policy.

Mouse controls (MuJoCo native):
- Right drag: zoom
- Left drag: rotate/orbit
- Middle drag / Shift+drag: pan
- Scroll: zoom
"""

import argparse
import time
from pathlib import Path

import numpy as np
import yaml
import mujoco
import mujoco.viewer
from stable_baselines3 import PPO

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Interactive MuJoCo viewer for hexapod policy")
    parser.add_argument("--model", required=True, help="Path to PPO .zip model")
    parser.add_argument("--config", default="configs/config_imu_bno10_natural.yaml", help="Config YAML path")
    parser.add_argument("--realtime", action="store_true", help="Run close to real-time speed")
    args = parser.parse_args()

    from envs.hexapod_env import make_hexapod_env

    config = load_config(str(PROJECT_ROOT / args.config))
    env = make_hexapod_env(config, render_mode=None)
    model = PPO.load(str(PROJECT_ROOT / args.model))

    obs, _ = env.reset(seed=0)

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        viewer.sync()
        while viewer.is_running():
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                obs, _ = env.reset()

            viewer.sync()
            if args.realtime:
                time.sleep(max(0.0, env.model.opt.timestep * env.pd_steps_per_action))

    env.close()


if __name__ == "__main__":
    main()
