#!/usr/bin/env python3
"""
Visualization Script for Hexapod in MuJoCo

This script runs the hexapod simulation with visualization.
Supports video recording mode that works on all platforms.

Usage:
    python visualize.py --mode cpg --save-video
    python visualize.py --mode trained --model checkpoints/test_model.zip
"""

import os
import sys
import time
import argparse
from pathlib import Path

import numpy as np
import mujoco

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


class MuJoCoRenderer:
    """Renderer for MuJoCo simulation."""
    
    def __init__(self, model, width=800, height=600):
        self.model = model
        self.width = width
        self.height = height
        self.renderer = mujoco.Renderer(model, height=height, width=width)
        
    def render(self, data, camera=None):
        """Render current frame."""
        if camera:
            self.renderer.update_scene(data, camera=camera)
        else:
            self.renderer.update_scene(data)
        return self.renderer.render()
    
    def close(self):
        self.renderer.close()


def save_video(frames, filename, fps=30):
    """Save frames as video file."""
    try:
        import imageio
        print(f"Saving video to {filename}...")
        imageio.mimsave(filename, frames, fps=fps)
        print(f"Video saved successfully! ({len(frames)} frames)")
        return True
    except ImportError:
        print("Warning: imageio not installed. Saving frames as images...")
        os.makedirs("frames", exist_ok=True)
        for i, frame in enumerate(frames):
            import matplotlib.pyplot as plt
            plt.imsave(f"frames/frame_{i:05d}.png", frame)
        print(f"Saved {len(frames)} frames to ./frames/")
        return True


def run_cpg_demo(duration: float = 10.0, save_video_flag: bool = True):
    """Run simulation with explicit tripod gait: legs 1,4,5 in phase, legs 2,3,6 180 deg out.
    Uses sim time and clear stance/swing split so three legs are visibly up, three down."""
    print("\n" + "="*60)
    print("CPG TRIPOD GAIT DEMO")
    print("="*60)
    
    model_path = PROJECT_ROOT / "assets" / "hexapod.xml"
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    data.qpos[2] = 0.10
    
    renderer = MuJoCoRenderer(model, width=960, height=720)
    frames = []
    camera = mujoco.MjvCamera()
    camera.azimuth = 135
    camera.elevation = -25
    camera.distance = 0.4
    camera.lookat[:] = [0, 0, 0.05]
    
    dt = model.opt.timestep
    frequency = 0.9  # Hz - slower so stance/swing are clearly visible
    amplitude_flex = 0.75  # Strong flexion difference: one tripod up, one down
    amplitude_abd = 0.5
    frame_skip = 5
    tripod_a_legs = [0, 3, 4]   # legs 1, 4, 5
    tripod_b_legs = [1, 2, 5]   # legs 2, 3, 6
    left_legs = [1, 3, 5]       # left side (2, 4, 6): flip abduction for mirror
    
    print(f"Running for {duration} seconds...")
    print("Tripod: legs 1,4,5 vs 2,3,6 with 180 deg phase. Flex = square wave (clear up/down).\n")
    
    start_time = time.time()
    step = 0
    frame_interval = 2
    
    while (time.time() - start_time) < duration:
        # Phase from simulation time so gait is consistent with physics
        sim_time = step * frame_skip * dt
        theta = 2 * np.pi * frequency * sim_time
        theta_a = theta
        theta_b = theta + np.pi
        
        # Square-wave flexion: one tripod clearly "up" (+amp), one "down" (-amp)
        def flex_square(phase):
            return amplitude_flex * (1.0 if np.sin(phase) >= 0 else -1.0)
        flex_a = flex_square(theta_a)
        flex_b = flex_square(theta_b)  # = -flex_a
        
        # Abduction: smooth swing, negated for left legs so both sides push forward
        def abd_smooth(phase):
            return -amplitude_abd * np.cos(phase)
        abd_a = abd_smooth(theta_a)
        abd_b = abd_smooth(theta_b)
        
        ctrl = np.zeros(12)
        for i in tripod_a_legs:
            ctrl[i * 2] = abd_a
            ctrl[i * 2 + 1] = flex_a
        for i in tripod_b_legs:
            ctrl[i * 2] = -abd_b if i in left_legs else abd_b  # left legs: flip abd
            ctrl[i * 2 + 1] = flex_b
        data.ctrl[:] = ctrl
        
        for _ in range(frame_skip):
            mujoco.mj_step(model, data)
        
        if step % frame_interval == 0:
            camera.lookat[0] = data.qpos[0]
            camera.lookat[1] = data.qpos[1]
            frame = renderer.render(data, camera=camera)
            frames.append(frame)
        
        if step % 500 == 0:
            t = time.time() - start_time
            height = data.qpos[2]
            vel_x = data.qvel[0]
            pos_x = data.qpos[0]
            print(f"t={t:.1f}s | Pos X: {pos_x:+.3f}m | Height: {height:.3f}m | Vel X: {vel_x:+.4f}m/s")
        
        step += 1
        
        if data.qpos[2] < 0.03:
            print("Hexapod fell! Resetting...")
            mujoco.mj_resetData(model, data)
            data.qpos[2] = 0.10
    
    renderer.close()
    print(f"\nSimulation completed: {step} steps, {len(frames)} frames")
    if save_video_flag and frames:
        os.makedirs("videos", exist_ok=True)
        out_path = "videos/cpg_demo_v3.mp4"
        save_video(frames, out_path, fps=30)
        print(f"Tripod demo saved as {out_path}")
    return frames


def run_trained_model(model_path: str, duration: float = 10.0, save_video_flag: bool = True):
    """Run simulation with a trained RL model."""
    import torch
    import yaml
    from stable_baselines3 import PPO
    
    print("\n" + "="*60)
    print("TRAINED MODEL VISUALIZATION")
    print("="*60)
    print(f"Loading model: {model_path}")
    
    # Load MuJoCo model
    xml_path = PROJECT_ROOT / "assets" / "hexapod.xml"
    mj_model = mujoco.MjModel.from_xml_path(str(xml_path))
    mj_data = mujoco.MjData(mj_model)
    
    # Load RL model
    rl_model = PPO.load(model_path)
    print("Model loaded successfully!\n")
    
    # Reset
    mujoco.mj_resetData(mj_model, mj_data)
    mj_data.qpos[2] = 0.10
    
    # Setup renderer with better resolution
    renderer = MuJoCoRenderer(mj_model, width=960, height=720)
    frames = []
    
    # Setup tracking camera
    camera = mujoco.MjvCamera()
    camera.azimuth = 135
    camera.elevation = -25
    camera.distance = 0.4
    camera.lookat[:] = [0, 0, 0.05]
    
    def get_obs():
        # Only joint angles (12 values: 6 abduction + 6 flexion)
        obs = mj_data.qpos[7:19].copy()
        return obs.astype(np.float32)
    
    prev_action = np.zeros(12)
    total_reward = 0
    
    print(f"Running for {duration} seconds...")
    
    start_time = time.time()
    step = 0
    frame_interval = 5
    
    while (time.time() - start_time) < duration:
        # Get observation and action
        obs = get_obs()
        action, _ = rl_model.predict(obs, deterministic=True)
        
        # Apply action
        mj_data.ctrl[:] = action
        prev_action = action.copy()
        
        # Step simulation (frame skip = 5)
        for _ in range(5):
            mujoco.mj_step(mj_model, mj_data)
        
        # Calculate reward
        reward = mj_data.qvel[0]
        total_reward += reward
        
        # Capture frame with tracking camera
        if step % frame_interval == 0:
            camera.lookat[0] = mj_data.qpos[0]
            camera.lookat[1] = mj_data.qpos[1]
            frame = renderer.render(mj_data, camera=camera)
            frames.append(frame)
        
        # Print status
        if step % 50 == 0:
            t = time.time() - start_time
            height = mj_data.qpos[2]
            vel_x = mj_data.qvel[0]
            pos_x = mj_data.qpos[0]
            print(f"t={t:.1f}s | Pos X: {pos_x:+.3f}m | Height: {height:.3f}m | "
                  f"Vel X: {vel_x:+.4f}m/s | Reward: {total_reward:.2f}")
        
        step += 1
        
        # Reset if fallen
        if mj_data.qpos[2] < 0.03:
            print("Hexapod fell! Resetting...")
            mujoco.mj_resetData(mj_model, mj_data)
            mj_data.qpos[2] = 0.10
            total_reward = 0
            prev_action = np.zeros(12)
    
    renderer.close()
    
    print(f"\nSimulation completed: {step} steps, {len(frames)} frames")
    print(f"Final total reward: {total_reward:.2f}")
    
    if save_video_flag and frames:
        os.makedirs("videos", exist_ok=True)
        save_video(frames, "videos/trained_model.mp4", fps=30)
    
    return frames


def run_random_demo(duration: float = 10.0, save_video_flag: bool = True):
    """Run simulation with random actions."""
    print("\n" + "="*60)
    print("RANDOM ACTIONS DEMO")
    print("="*60)
    
    model_path = PROJECT_ROOT / "assets" / "hexapod.xml"
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    
    mujoco.mj_resetData(model, data)
    data.qpos[2] = 0.06  # Start lower - legs will push up
    
    renderer = MuJoCoRenderer(model, width=800, height=600)
    frames = []
    
    print(f"Running for {duration} seconds with random actions...\n")
    
    start_time = time.time()
    step = 0
    frame_interval = 2
    
    while (time.time() - start_time) < duration:
        # Random actions
        data.ctrl[:] = np.random.uniform(-0.3, 0.3, model.nu)
        
        mujoco.mj_step(model, data)
        
        if step % frame_interval == 0:
            frame = renderer.render(data)
            frames.append(frame)
        
        if step % 500 == 0:
            t = time.time() - start_time
            height = data.qpos[2]
            vel_x = data.qvel[0]
            print(f"t={t:.1f}s | Height: {height:.3f}m | Vel X: {vel_x:+.4f}m/s")
        
        step += 1
        
        if data.qpos[2] < 0.03:
            print("Hexapod fell! Resetting...")
            mujoco.mj_resetData(model, data)
            data.qpos[2] = 0.06  # Start lower - legs will push up
    
    renderer.close()
    
    print(f"\nSimulation completed: {step} steps, {len(frames)} frames")
    
    if save_video_flag and frames:
        os.makedirs("videos", exist_ok=True)
        save_video(frames, "videos/random_demo.mp4", fps=30)
    
    return frames


def display_frame(frame, title="Hexapod"):
    """Display a single frame using matplotlib."""
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 8))
    plt.imshow(frame)
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig("hexapod_snapshot.png", dpi=150)
    print("Snapshot saved to hexapod_snapshot.png")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Visualize hexapod in MuJoCo")
    parser.add_argument(
        "--mode", "-m",
        type=str,
        default="cpg",
        choices=["random", "trained", "cpg"],
        help="Visualization mode"
    )
    parser.add_argument(
        "--model", "-p",
        type=str,
        default="checkpoints/test_model.zip",
        help="Path to trained model (for 'trained' mode)"
    )
    parser.add_argument(
        "--duration", "-d",
        type=float,
        default=10.0,
        help="Duration in seconds"
    )
    parser.add_argument(
        "--save-video", "-v",
        action="store_true",
        default=True,
        help="Save video"
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show final frame"
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("HEXAPOD MUJOCO VISUALIZATION")
    print("="*60)
    print(f"Mode: {args.mode}")
    print(f"Duration: {args.duration}s")
    print(f"Save video: {args.save_video}")
    
    try:
        if args.mode == "random":
            frames = run_random_demo(args.duration, args.save_video)
        elif args.mode == "trained":
            frames = run_trained_model(args.model, args.duration, args.save_video)
        elif args.mode == "cpg":
            frames = run_cpg_demo(args.duration, args.save_video)
        
        if args.show and frames:
            display_frame(frames[-1], f"Hexapod - {args.mode} mode (final frame)")
            
    except KeyboardInterrupt:
        print("\n\nVisualization stopped by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
