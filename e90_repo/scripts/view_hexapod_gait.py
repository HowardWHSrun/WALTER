#!/usr/bin/env python3
"""
Interactive MuJoCo viewer with an open-loop tripod gait (motors driven).

The passive viewer started with only `mujoco.viewer.launch_from_path(...)` does not send any
torques, so ctrl stays zero and the hexapod collapses under gravity. This script uses
`launch_passive` and sets `data.ctrl` each step from the same tripod map as
`models/ode_gait/reference_from_phase.py`.

Usage (from e90_repo):
  python scripts/view_hexapod_gait.py
  python scripts/view_hexapod_gait.py --freq 1.0 --amp-flex 0.7 --amp-abd 0.45

macOS: `launch_passive` requires **mjpython** (not plain `python`):
  mjpython scripts/view_hexapod_gait.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Match models/ode_gait/reference_from_phase.py
TRIPOD_A_LEGS = (0, 3, 4)
TRIPOD_B_LEGS = (1, 2, 5)
LEFT_LEGS = frozenset((1, 3, 5))


def tripod_ctrl(
    sim_time: float,
    freq_hz: float,
    amp_flex: float,
    amp_abd: float,
    flex_sign: float,
    abd_sign: float,
) -> np.ndarray:
    theta_a = 2 * np.pi * freq_hz * sim_time
    theta_b = theta_a + np.pi
    flex_a = flex_sign * amp_flex * np.sin(theta_a)
    flex_b = flex_sign * amp_flex * np.sin(theta_b)
    abd_a = abd_sign * (-amp_abd * np.cos(theta_a))
    abd_b = abd_sign * (-amp_abd * np.cos(theta_b))
    ctrl = np.zeros(12, dtype=np.float64)
    for i in TRIPOD_A_LEGS:
        ctrl[i * 2] = abd_a
        ctrl[i * 2 + 1] = flex_a
    for i in TRIPOD_B_LEGS:
        abd_i = -abd_b if i in LEFT_LEGS else abd_b
        ctrl[i * 2] = abd_i
        ctrl[i * 2 + 1] = flex_b
    return np.clip(ctrl, -1.0, 1.0)


def apply_standing_pose(data: mujoco.MjData, z: float = 0.115) -> None:
    """Root at sensible height, identity quaternion, legs at zero (matches MJCF defaults)."""
    data.qpos[0:3] = 0.0, 0.0, z
    # MuJoCo quat order: [w, x, y, z]
    data.qpos[3:7] = 1.0, 0.0, 0.0, 0.0
    if data.qpos.shape[0] > 7:
        data.qpos[7:] = 0.0


def main() -> None:
    p = argparse.ArgumentParser(description="View hexapod with open-loop tripod gait.")
    p.add_argument("--xml", type=str, default="assets/hexapod.xml", help="Path to MJCF (under e90_repo if relative).")
    p.add_argument("--freq", type=float, default=0.9, help="Gait frequency (Hz).")
    p.add_argument("--amp-flex", type=float, default=0.65, help="Normalized flexion amplitude [0,1].")
    p.add_argument("--amp-abd", type=float, default=0.45, help="Normalized abduction amplitude [0,1].")
    p.add_argument("--flex-sign", type=float, default=-1.0)
    p.add_argument("--abd-sign", type=float, default=1.0)
    p.add_argument("--frame-skip", type=int, default=5, help="Mj steps per control update (same idea as env frame_skip).")
    p.add_argument("--z0", type=float, default=0.115, help="Initial torso height (m).")
    p.add_argument(
        "--realtime",
        type=float,
        default=1.0,
        help="Wall-clock sync: 1.0 = simulation runs at real time; 0.5 = half speed (easier to watch); 0 = no throttling (runs as fast as CPU — looks frantic).",
    )
    args = p.parse_args()

    xml_path = Path(args.xml)
    if not xml_path.is_absolute():
        xml_path = PROJECT_ROOT / xml_path
    if not xml_path.is_file():
        raise FileNotFoundError(xml_path)

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    apply_standing_pose(data, z=args.z0)
    mujoco.mj_forward(model, data)

    dt = model.opt.timestep
    sim_time = 0.0
    min_height = 0.035
    wall_start = time.perf_counter()

    print("Tripod gait viewer — close the window to exit.")
    print("  If motion looks frantic: default is now real-time synced; try --realtime 0.5 for half speed.")
    print("  If the robot falls, try: --freq 0.7 --amp-flex 0.5 --amp-abd 0.35  (gentler gait)")
    rt = args.realtime
    print(
        f"  Using: freq={args.freq} Hz, amp_flex={args.amp_flex}, amp_abd={args.amp_abd}, "
        f"frame_skip={args.frame_skip}, realtime={rt} (1=normal wall clock, 0=uncapped FPS)"
    )

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            ctrl = tripod_ctrl(
                sim_time,
                args.freq,
                args.amp_flex,
                args.amp_abd,
                args.flex_sign,
                args.abd_sign,
            )
            data.ctrl[:] = ctrl
            for _ in range(args.frame_skip):
                mujoco.mj_step(model, data)
            sim_chunk = args.frame_skip * dt
            sim_time += sim_chunk

            if data.qpos[2] < min_height:
                mujoco.mj_resetData(model, data)
                apply_standing_pose(data, z=args.z0)
                sim_time = 0.0
                wall_start = time.perf_counter()

            if rt > 0.0:
                target = wall_start + sim_time / rt
                now = time.perf_counter()
                if target > now:
                    time.sleep(target - now)

            viewer.sync()


if __name__ == "__main__":
    main()
