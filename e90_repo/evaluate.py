#!/usr/bin/env python3
"""
Evaluation and Visualization Script for Hexapod sCPG

This script evaluates trained models and provides visualization
of the hexapod locomotion and spike activity.
"""

import os
import sys
import argparse
import hashlib
import yaml
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from stable_baselines3 import PPO

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from envs.hexapod_env import HexapodEnv, make_hexapod_env
from models.encoder import SCPGPolicy, SCPGPolicyV2  # V2 needed when loading tripod-gait checkpoints
from models.ode_gait import ODETripodPolicy  # noqa: F401 — registered for PPO.load of ODE checkpoints


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


class EvaluationMetrics:
    """Collect and compute evaluation metrics."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.episode_rewards = []
        self.episode_lengths = []
        self.forward_velocities = []
        self.torso_heights = []
        self.energy_consumption = []
        self.contact_patterns = []
        self.termination_reasons = []
        # Distance / locomotion (per episode)
        self.episode_forward_delta_x_m = []  # torso x advance vs spawn (same as reward "forward_distance")
        self.episode_xy_displacement_m = []  # straight-line distance start->end in ground plane
        self.episode_path_length_xy_m = []  # sum of |Delta XY| each step (actual path length)
        self.episode_terrain_kinds = []  # e.g. "flat" / "rough" / "steps" / None

    def add_step(self, info: dict, action: np.ndarray):
        """Add metrics from a single step."""
        if 'forward_velocity' in info:
            self.forward_velocities.append(info['forward_velocity'])
        if 'torso_height' in info:
            self.torso_heights.append(info['torso_height'])
        self.energy_consumption.append(np.sum(action ** 2))

    def end_episode(
        self,
        total_reward: float,
        length: int,
        info: dict,
        *,
        forward_delta_x_m: float = 0.0,
        xy_displacement_m: float = 0.0,
        path_length_xy_m: float = 0.0,
        terrain_kind: str = None,
    ):
        """Record episode-level metrics."""
        self.episode_rewards.append(total_reward)
        self.episode_lengths.append(length)
        if 'termination_reason' in info:
            self.termination_reasons.append(info['termination_reason'])
        self.episode_forward_delta_x_m.append(float(forward_delta_x_m))
        self.episode_xy_displacement_m.append(float(xy_displacement_m))
        self.episode_path_length_xy_m.append(float(path_length_xy_m))
        self.episode_terrain_kinds.append(terrain_kind)

    def compute_summary(self) -> dict:
        """Compute summary statistics."""
        n = len(self.episode_rewards)
        fd = self.episode_forward_delta_x_m
        disp = self.episode_xy_displacement_m
        path = self.episode_path_length_xy_m
        out = {
            'mean_reward': float(np.mean(self.episode_rewards)) if n else 0.0,
            'std_reward': float(np.std(self.episode_rewards)) if n else 0.0,
            'mean_length': float(np.mean(self.episode_lengths)) if n else 0.0,
            'std_length': float(np.std(self.episode_lengths)) if n else 0.0,
            'mean_forward_velocity': float(np.mean(self.forward_velocities)) if self.forward_velocities else 0.0,
            'max_forward_velocity': float(np.max(self.forward_velocities)) if self.forward_velocities else 0.0,
            'mean_torso_height': float(np.mean(self.torso_heights)) if self.torso_heights else 0.0,
            'mean_energy': float(np.mean(self.energy_consumption)) if self.energy_consumption else 0.0,
            'num_episodes': n,
            'termination_counts': self._count_terminations(),
            # Distance moved (episode aggregates)
            'mean_forward_delta_x_m': float(np.mean(fd)) if fd else 0.0,
            'std_forward_delta_x_m': float(np.std(fd)) if fd else 0.0,
            'mean_xy_displacement_m': float(np.mean(disp)) if disp else 0.0,
            'std_xy_displacement_m': float(np.std(disp)) if disp else 0.0,
            'mean_path_length_xy_m': float(np.mean(path)) if path else 0.0,
            'std_path_length_xy_m': float(np.std(path)) if path else 0.0,
            'per_episode': self._per_episode_table(),
            'by_terrain': self._by_terrain(),
        }
        return out

    def _per_episode_table(self) -> list:
        rows = []
        for i in range(len(self.episode_rewards)):
            rows.append({
                'episode': i + 1,
                'terrain': self.episode_terrain_kinds[i] if i < len(self.episode_terrain_kinds) else None,
                'reward': float(self.episode_rewards[i]),
                'length': int(self.episode_lengths[i]),
                'forward_delta_x_m': float(self.episode_forward_delta_x_m[i]) if i < len(self.episode_forward_delta_x_m) else 0.0,
                'xy_displacement_m': float(self.episode_xy_displacement_m[i]) if i < len(self.episode_xy_displacement_m) else 0.0,
                'path_length_xy_m': float(self.episode_path_length_xy_m[i]) if i < len(self.episode_path_length_xy_m) else 0.0,
            })
        return rows

    def _by_terrain(self) -> dict:
        """Aggregate distance/reward by terrain label when cycling eval terrains."""
        terrains = set(t for t in self.episode_terrain_kinds if t is not None)
        if not terrains:
            return {}
        by = {}
        for t in sorted(terrains):
            idx = [i for i, tk in enumerate(self.episode_terrain_kinds) if tk == t]
            if not idx:
                continue
            by[t] = {
                'n_episodes': len(idx),
                'mean_reward': float(np.mean([self.episode_rewards[i] for i in idx])),
                'mean_forward_delta_x_m': float(np.mean([self.episode_forward_delta_x_m[i] for i in idx])),
                'mean_xy_displacement_m': float(np.mean([self.episode_xy_displacement_m[i] for i in idx])),
                'mean_path_length_xy_m': float(np.mean([self.episode_path_length_xy_m[i] for i in idx])),
                'mean_length': float(np.mean([self.episode_lengths[i] for i in idx])),
            }
        return by
    
    def _count_terminations(self) -> dict:
        counts = defaultdict(int)
        for reason in self.termination_reasons:
            counts[reason] += 1
        return dict(counts)


class SpikeActivityTracker:
    """Track spike activity for visualization."""
    
    def __init__(self, num_legs: int = 6, neurons_per_leg: int = 32):
        self.num_legs = num_legs
        self.neurons_per_leg = neurons_per_leg
        self.reset()
    
    def reset(self):
        self.spike_history = {f"leg{i+1}": [] for i in range(self.num_legs)}
        self.membrane_history = {f"leg{i+1}": [] for i in range(self.num_legs)}
        self.action_history = []
        self.time_steps = []
    
    def add_step(self, t: int, actions: np.ndarray, spike_activity: dict = None):
        """Record spike activity at time step t."""
        self.time_steps.append(t)
        self.action_history.append(actions.copy())
        
        if spike_activity:
            for key, value in spike_activity.items():
                if key.endswith('_v') and key in self.membrane_history:
                    leg_name = key.replace('_v', '')
                    self.membrane_history[leg_name].append(value.flatten())
    
    def get_raster_data(self) -> dict:
        """Get data formatted for raster plots."""
        return {
            'time': np.array(self.time_steps),
            'actions': np.array(self.action_history),
            'membrane': {k: np.array(v) for k, v in self.membrane_history.items() if v},
        }


def _file_sha256(path: Path) -> str:
    """SHA-256 of file on disk (streaming)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _model_provenance(model_path: str) -> dict:
    """Resolved path, basename, and checksum for audit trail (same policy across terrains)."""
    p = Path(model_path).resolve()
    if not p.is_file():
        return {
            "model_path": str(model_path),
            "model_resolved_path": str(p),
            "model_basename": p.name,
            "model_file_sha256": None,
            "eval_protocol": (
                "Single checkpoint: PPO.load() once; each episode only env.reset(options=terrain_kind). "
                "No retraining and no alternate weights between terrains."
            ),
        }
    return {
        "model_path": str(model_path),
        "model_resolved_path": str(p),
        "model_basename": p.name,
        "model_file_sha256": _file_sha256(p),
        "eval_protocol": (
            "Single checkpoint: PPO.load() once; each episode only env.reset(options=terrain_kind). "
            "No retraining and no alternate weights between terrains."
        ),
    }


def _get_action_mean_and_log_std(model: PPO, obs: np.ndarray):
    """Get policy action_mean and log_std for diagnostics. Returns (action_mean, log_std) as numpy arrays."""
    device = model.device
    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        action_mean, _, _ = model.policy.forward(obs_t, deterministic=True)
        action_mean = action_mean.cpu().numpy().squeeze(0)
    log_std = model.policy.log_std.detach().cpu().numpy()
    return action_mean, log_std


def visualize_episode(
    env: HexapodEnv,
    model: PPO,
    max_steps: int = 500,
    render: bool = True,
    save_video: bool = False,
    video_path: str = None,
    collect_diagnostics: bool = False,
    use_random_actions: bool = False,
    reset_options: dict = None,
    terrain_label: str = None,
    include_trajectory_plot: bool = False,
    video_header_lines: list = None,
) -> tuple:
    """
    Run and visualize a single episode.

    If collect_diagnostics is True, returns a third value: dict with step-wise
    action_mean, log_std, reward_components for debugging.
    If use_random_actions is True, ignores the policy and uses random actions (sanity check).
    reset_options: passed to env.reset(options=...); use e.g. {"terrain_kind": "rough"} to force terrain for eval videos.
    terrain_label: if set, shown on video overlay (e.g. "flat", "rough", "steps").
    video_header_lines: optional lines drawn at top of each frame (e.g. same checkpoint id for all terrains).
    
    Returns:
        metrics: EvaluationMetrics object
        tracker: SpikeActivityTracker object
        diagnostics: optional dict (only if collect_diagnostics=True)
    """
    metrics = EvaluationMetrics()
    tracker = SpikeActivityTracker()
    diagnostics = None
    if collect_diagnostics:
        diagnostics = {
            "step": [],
            "action": [],
            "action_mean": [],
            "action_mean_norm": [],
            "log_std": [],
            "log_std_mean": [],
            "velocity_reward": [],
            "tilt_penalty": [],
            "imitation_penalty": [],
            "idle_penalty": [],
            "forward_velocity": [],
            "reward": [],
        }
    
    obs, info = env.reset(seed=None, options=reset_options)
    done = False
    total_reward = 0
    step = 0
    frames = []
    trajectory_x, trajectory_y, trajectory_dist, trajectory_vel = [], [], [], []
    x0 = float(env.data.qpos[0])
    y0 = float(env.data.qpos[1])
    path_length_xy = 0.0
    prev_xy = (x0, y0)

    # Reset CPG state if available
    if hasattr(model.policy, 'reset_cpg'):
        model.policy.reset_cpg(batch_size=1)
    
    while not done and step < max_steps:
        # Get action from model or random
        if use_random_actions:
            action = np.random.uniform(-1.0, 1.0, size=env.action_space.shape[0]).astype(np.float32)
        else:
            action, _ = model.predict(obs, deterministic=True)
        
        # Collect diagnostics (action_mean and log_std from policy; or placeholder for random)
        if collect_diagnostics:
            diagnostics["step"].append(step)
            diagnostics["action"].append(action.copy())
            if not use_random_actions:
                action_mean_np, log_std_np = _get_action_mean_and_log_std(model, obs)
                diagnostics["action_mean"].append(action_mean_np.copy())
                diagnostics["action_mean_norm"].append(float(np.linalg.norm(action_mean_np)))
                diagnostics["log_std"].append(log_std_np.copy())
                diagnostics["log_std_mean"].append(float(np.mean(log_std_np)))
            else:
                diagnostics["action_mean"].append(action.copy())
                diagnostics["action_mean_norm"].append(float(np.linalg.norm(action)))
                diagnostics["log_std"].append(np.full(env.action_space.shape[0], np.nan))
                diagnostics["log_std_mean"].append(np.nan)
        
        # Track spike activity
        spike_activity = {}
        if hasattr(model.policy, 'cpg_network'):
            spike_activity = model.policy.cpg_network.cpg.get_spike_activity()
        
        tracker.add_step(step, action, spike_activity)
        
        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        step += 1

        # Record trajectory for overlays and plot (position and movement relative to environment)
        x = float(env.data.qpos[0])
        y = float(env.data.qpos[1])
        trajectory_x.append(x)
        trajectory_y.append(y)
        trajectory_dist.append(float(info.get("forward_distance", 0)))
        trajectory_vel.append(float(info.get("forward_velocity", 0)))
        path_length_xy += float(np.hypot(x - prev_xy[0], y - prev_xy[1]))
        prev_xy = (x, y)

        # Record metrics
        metrics.add_step(info, action)
        
        # Reward components for diagnostics
        if collect_diagnostics:
            diagnostics["velocity_reward"].append(float(info.get("velocity_reward", 0)))
            diagnostics["tilt_penalty"].append(float(info.get("tilt_penalty", 0)))
            diagnostics["imitation_penalty"].append(float(info.get("imitation_penalty", 0)))
            diagnostics["idle_penalty"].append(float(info.get("idle_penalty", 0)))
            diagnostics["forward_velocity"].append(float(info.get("forward_velocity", 0)))
            diagnostics["reward"].append(float(reward))
        
        # Render
        if render or save_video:
            frame = env.render()
            if save_video and frame is not None:
                dist = trajectory_dist[-1] if trajectory_dist else 0
                vel = trajectory_vel[-1] if trajectory_vel else 0
                disp2d = float(np.hypot(x - x0, y - y0))
                frame = overlay_metrics_on_frame(
                    frame, step, x, y, dist, vel,
                    terrain_label=terrain_label,
                    x0=x0, y0=y0,
                    path_length_xy=path_length_xy,
                    xy_displacement_m=disp2d,
                    header_lines=video_header_lines,
                )
                frames.append(frame)

    fd_x = float(info.get("forward_distance", trajectory_dist[-1] if trajectory_dist else 0.0))
    x_end = float(env.data.qpos[0])
    y_end = float(env.data.qpos[1])
    xy_disp = float(np.hypot(x_end - x0, y_end - y0))
    metrics.end_episode(
        total_reward, step, info,
        forward_delta_x_m=fd_x,
        xy_displacement_m=xy_disp,
        path_length_xy_m=path_length_xy,
        terrain_kind=terrain_label,
    )

    # Save video if requested
    if save_video and frames and video_path:
        save_video_frames(frames, video_path)
        if include_trajectory_plot and trajectory_x and trajectory_dist:
            save_trajectory_plot(
                trajectory_x, trajectory_y, trajectory_dist, trajectory_vel,
                list(range(len(trajectory_x))), video_path,
            )
    
    if collect_diagnostics:
        for key in list(diagnostics.keys()):
            if key != "step" and isinstance(diagnostics[key], list) and diagnostics[key]:
                diagnostics[key] = np.array(diagnostics[key])
        diagnostics["step"] = np.array(diagnostics["step"])
        return metrics, tracker, diagnostics
    return metrics, tracker


def overlay_metrics_on_frame(
    frame: np.ndarray,
    step: int,
    x: float,
    y: float,
    dist: float,
    vel: float,
    terrain_label: str = None,
    x0: float = None,
    y0: float = None,
    path_length_xy: float = None,
    xy_displacement_m: float = None,
    header_lines: list = None,
) -> np.ndarray:
    """
    Draw step, world XY, locomotion metrics, and optional terrain label.
    dist = forward (+X) progress from episode start (matches reward axis).
    xy_displacement_m / path_length_xy = 2D straight-line / integrated path length in ground plane.
    """
    try:
        import cv2
    except ImportError:
        return frame
    if frame is None or frame.size == 0:
        return frame
    # Frame is RGB; cv2 uses BGR for drawing
    out = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    h, w = out.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    thickness = 1
    color = (255, 255, 255)
    bg = (40, 40, 40)
    line_h = 21
    y_text = 26
    texts = []
    if header_lines:
        for line in header_lines:
            if line:
                texts.append(str(line))
    texts.extend([
        f"Step: {step}",
        f"World X: {x:.3f} m  Y: {y:.3f} m",
        f"Fwd +X: {dist:.3f} m  (advance along X)",
        f"v_x: {vel:.3f} m/s",
    ])
    if xy_displacement_m is not None:
        texts.append(f"2D disp: {xy_displacement_m:.3f} m (start to now)")
    if path_length_xy is not None:
        texts.append(f"2D path: {path_length_xy:.3f} m (integrated)")
    if terrain_label:
        texts.append(f"Terrain: {terrain_label}")
    for i, text in enumerate(texts):
        y_pos = y_text + i * line_h
        # Small background bar for readability
        (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
        cv2.rectangle(out, (8, y_pos - th - 4), (8 + tw + 8, y_pos + 4), bg, -1)
        cv2.putText(out, text, (12, y_pos), font, scale, color, thickness, cv2.LINE_AA)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


def save_trajectory_plot(xs: list, ys: list, dists: list, vels: list, steps: list, path: str):
    """Save a trajectory and velocity plot for the episode (movement relative to environment)."""
    if not xs or not dists:
        return
    path = Path(path)
    path = path.with_name(path.stem + "_trajectory.png")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    steps_arr = np.array(steps)
    # Left: (x, y) trajectory in environment
    axes[0].plot(xs, ys, "b-", alpha=0.7)
    axes[0].plot(xs[0], ys[0], "go", markersize=10, label="Start")
    axes[0].plot(xs[-1], ys[-1], "ro", markersize=10, label="End")
    axes[0].set_xlabel("X (m)")
    axes[0].set_ylabel("Y (m)")
    axes[0].set_title("Trajectory in Environment")
    axes[0].legend()
    axes[0].set_aspect("equal")
    axes[0].grid(True, alpha=0.3)
    # Right: forward distance and velocity over time
    ax2 = axes[1]
    ax2.plot(steps_arr, dists, "b-", label="Forward distance (m)", alpha=0.8)
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Distance (m)")
    ax2.tick_params(axis="y", labelcolor="b")
    ax2b = ax2.twinx()
    ax2b.plot(steps_arr, vels, "orange", alpha=0.7, label="Forward velocity (m/s)")
    ax2b.set_ylabel("Velocity (m/s)")
    ax2b.tick_params(axis="y", labelcolor="orange")
    ax2.set_title("Distance and Velocity Over Time")
    ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Trajectory plot saved to: {path}")


def save_video_frames(frames: list, path: str, fps: int = 30):
    """Save frames as video file."""
    try:
        import imageio
        imageio.mimsave(path, frames, fps=fps)
        print(f"Video saved to: {path}")
    except ImportError:
        print("Warning: imageio not available, cannot save video")


def _terrain_color(terrain: str):
    if terrain == "flat":
        return "#2ca02c"
    if terrain == "rough":
        return "#ff7f0e"
    if terrain == "steps":
        return "#1f77b4"
    return "#7f7f7f"


def plot_evaluation_results(
    metrics: EvaluationMetrics,
    tracker: SpikeActivityTracker,
    save_path: str = None,
    suptitle: str = None,
):
    """Create visualization plots for evaluation results."""
    fig = plt.figure(figsize=(16, 14))
    if suptitle:
        fig.suptitle(suptitle, fontsize=10, y=0.995)
    gs = GridSpec(4, 3, figure=fig, height_ratios=[1, 1, 1, 0.9])
    
    # 1. Forward velocity over time
    ax1 = fig.add_subplot(gs[0, 0])
    if metrics.forward_velocities:
        ax1.plot(metrics.forward_velocities)
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Forward Velocity (m/s)')
        ax1.set_title('Forward Velocity Over Time')
        ax1.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    
    # 2. Torso height over time
    ax2 = fig.add_subplot(gs[0, 1])
    if metrics.torso_heights:
        ax2.plot(metrics.torso_heights)
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Height (m)')
        ax2.set_title('Torso Height Over Time')
        ax2.axhline(y=0.05, color='r', linestyle='--', alpha=0.5, label='Min height')
        ax2.legend()
    
    # 3. Energy consumption over time
    ax3 = fig.add_subplot(gs[0, 2])
    if metrics.energy_consumption:
        ax3.plot(metrics.energy_consumption)
        ax3.set_xlabel('Step')
        ax3.set_ylabel('Energy (sum of torques^2)')
        ax3.set_title('Energy Consumption')
    
    # 4. Action values (joint torques)
    ax4 = fig.add_subplot(gs[1, :])
    raster_data = tracker.get_raster_data()
    if len(raster_data['actions']) > 0:
        actions = raster_data['actions']
        time = raster_data['time']
        
        # Plot as heatmap
        im = ax4.imshow(
            actions.T, 
            aspect='auto', 
            cmap='RdBu_r',
            vmin=-1, vmax=1,
            extent=[time[0], time[-1], 0, actions.shape[1]]
        )
        ax4.set_xlabel('Step')
        ax4.set_ylabel('Joint Index')
        ax4.set_title('Joint Torque Commands')
        plt.colorbar(im, ax=ax4, label='Torque')
        
        # Add joint labels
        joint_labels = []
        for i in range(1, 7):
            joint_labels.extend([f'L{i}_abd', f'L{i}_flex'])
        ax4.set_yticks(np.arange(len(joint_labels)) + 0.5)
        ax4.set_yticklabels(joint_labels, fontsize=8)
    
    # 5. Phase analysis (action pairs per leg)
    ax5 = fig.add_subplot(gs[2, 0])
    if len(raster_data['actions']) > 0:
        actions = raster_data['actions']
        # Plot phase relationship between legs
        for leg_idx in range(3):  # First 3 legs
            abd_idx = leg_idx * 2
            ax5.plot(
                actions[:, abd_idx], 
                actions[:, abd_idx + 1],
                alpha=0.5, 
                label=f'Leg {leg_idx+1}'
            )
        ax5.set_xlabel('Abduction')
        ax5.set_ylabel('Flexion')
        ax5.set_title('Phase Plot (Legs 1-3)')
        ax5.legend()
        ax5.set_xlim(-1.1, 1.1)
        ax5.set_ylim(-1.1, 1.1)
    
    # 6. Gait pattern visualization
    ax6 = fig.add_subplot(gs[2, 1:])
    if len(raster_data['actions']) > 0:
        actions = raster_data['actions']
        # Simple gait indicator: when flexion is positive, leg is lifted
        gait_pattern = np.zeros((6, len(actions)))
        for leg_idx in range(6):
            flex_idx = leg_idx * 2 + 1
            gait_pattern[leg_idx] = actions[:, flex_idx] > 0
        
        ax6.imshow(
            gait_pattern, 
            aspect='auto', 
            cmap='binary',
            extent=[0, len(actions), 0, 6]
        )
        ax6.set_xlabel('Step')
        ax6.set_ylabel('Leg')
        ax6.set_title('Gait Pattern (white = leg lifted)')
        ax6.set_yticks(np.arange(6) + 0.5)
        ax6.set_yticklabels([f'Leg {i+1}' for i in range(6)])

    # 7. Per-episode distance moved (forward +X vs 2D path) — easy to read at a glance
    ax7 = fig.add_subplot(gs[3, :])
    if metrics.episode_forward_delta_x_m:
        n_ep = len(metrics.episode_forward_delta_x_m)
        x_ep = np.arange(n_ep)
        w = 0.35
        fwd = np.array(metrics.episode_forward_delta_x_m, dtype=float)
        pathl = np.array(metrics.episode_path_length_xy_m, dtype=float)
        colors = [
            _terrain_color(metrics.episode_terrain_kinds[i]) if i < len(metrics.episode_terrain_kinds) else "#7f7f7f"
            for i in range(n_ep)
        ]
        ax7.bar(x_ep - w / 2, fwd, width=w, label="Forward +X (m)", color=colors, edgecolor="black", linewidth=0.3)
        ax7.bar(x_ep + w / 2, pathl, width=w, label="2D path length (m)", color=colors, alpha=0.55, edgecolor="black", linewidth=0.3)
        ax7.set_xticks(x_ep)
        ax7.set_xticklabels([f"Ep{i + 1}" + (f"\n{metrics.episode_terrain_kinds[i]}" if i < len(metrics.episode_terrain_kinds) and metrics.episode_terrain_kinds[i] else "") for i in range(n_ep)], fontsize=8)
        ax7.set_ylabel("Meters")
        ax7.set_title(
            "Distance per episode — same checkpoint; only terrain_kind changes between episodes"
        )
        ax7.legend(loc="upper right")
        ax7.grid(True, axis="y", alpha=0.3)
        ax7.axhline(0, color="k", linewidth=0.5)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Plot saved to: {save_path}")
    
    plt.show()


def evaluate(
    model_path: str,
    config_path: str = "config.yaml",
    num_episodes: int = 10,
    max_steps: int = 1000,
    render: bool = False,
    save_video: bool = False,
    video_dir: str = "videos",
    plot: bool = True,
    output_dir: str = None,
    device: str = "auto",
    save_diagnostics: bool = False,
    random_actions_episode: bool = False,
    include_trajectory_plot: bool = False,
    exaggerate_terrain_for_video: bool = None,
):
    """
    Evaluate a trained model.

    save_diagnostics: Run one episode with full step-wise logging of action_mean,
        log_std, and reward components; save to output_dir/eval_diagnostics.npz.
    random_actions_episode: Run one episode with random actions (sanity check)
        before or after normal eval; results printed and optionally saved.
    """
    # Load config
    config = load_config(config_path)
    evaluation_cfg = config.get("evaluation", {})

    # Setup output directory
    if output_dir is None:
        output_dir = Path(model_path).parent / "evaluation"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create environment
    render_mode = "rgb_array" if (render or save_video) else None
    env = make_hexapod_env(config, render_mode=render_mode)

    # Optional: exaggerate rough/step height for *visibility* in video. Default off so eval matches
    # training scales (exaggeration often makes policies look worse than training distribution).
    if exaggerate_terrain_for_video is None:
        exaggerate_terrain_for_video = bool(
            evaluation_cfg.get("exaggerate_terrain_for_video", False)
        )
    if save_video and getattr(env, "_use_terrain_hfield", False) and exaggerate_terrain_for_video:
        env._terrain_rough_min_height = -0.06
        env._terrain_rough_max_height = 0.06
        env._terrain_step_height = 0.05
        env._terrain_step_length = 0.12
        print("Using exaggerated terrain heights for video (evaluation.exaggerate_terrain_for_video: true).")
    elif save_video and getattr(env, "_use_terrain_hfield", False):
        print(
            "Eval terrain scales match training config (set evaluation.exaggerate_terrain_for_video: true "
            "or pass --exaggerate-terrain-for-video for taller bumps in video only)."
        )

    # Load model once — all episodes use this same policy; only terrain resets differ
    provenance = _model_provenance(model_path)
    print("\n" + "=" * 72)
    print("SINGLE CHECKPOINT EVAL (anti-forgetting / generalization protocol)")
    print("  Same PPO weights for every episode. Only env.reset(terrain_kind=...) changes.")
    print(f"  Model file: {provenance.get('model_basename', model_path)}")
    print(f"  Resolved:   {provenance.get('model_resolved_path', '')}")
    if provenance.get("model_file_sha256"):
        h = provenance["model_file_sha256"]
        print(f"  SHA256:     {h}  (verify identical across runs)")
    print(f"  Protocol:   {provenance.get('eval_protocol', '')}")
    print("=" * 72 + "\n")
    print(f"Loading model from: {model_path}")
    model = PPO.load(model_path, device=device)

    video_header_lines = None
    if provenance.get("model_file_sha256"):
        h = provenance["model_file_sha256"]
        video_header_lines = [
            "SAME CHECKPOINT ALL TERRAINS",
            f"{provenance.get('model_basename', 'model.zip')}  SHA256:{h[:16]}...",
        ]
    elif provenance.get("model_basename"):
        video_header_lines = [
            "SAME CHECKPOINT ALL TERRAINS",
            provenance["model_basename"],
        ]

    # Evaluation metrics
    all_metrics = EvaluationMetrics()
    
    # Optional: one episode with random actions (sanity check)
    if random_actions_episode:
        print("\n[Sanity check] Running one episode with RANDOM actions...")
        _, _, diag = visualize_episode(
            env=env,
            model=model,
            max_steps=max_steps,
            render=False,
            save_video=False,
            collect_diagnostics=True,
            use_random_actions=True,
        )
        if diag and len(diag["reward"]) > 0:
            print(f"  Steps: {len(diag['reward'])}, Mean reward: {np.mean(diag['reward']):.2f}")
            print(f"  Mean forward_velocity: {np.mean(diag['forward_velocity']):.4f}")
        np.savez(output_dir / "eval_diagnostics_random.npz", **{k: v for k, v in diag.items() if isinstance(v, np.ndarray)})
        print("  Saved eval_diagnostics_random.npz")
    
    # Run evaluation episodes
    # Cycle through flat / rough / steps when env has heightfield terrain
    terrain_cycle = (
        ["flat", "rough", "steps"]
        if (getattr(env, "_use_terrain_hfield", False) and getattr(env, "_terrain_type", None))
        else None
    )
    ept = evaluation_cfg.get("episodes_per_terrain")
    if terrain_cycle is not None and ept is not None:
        ept = max(1, int(ept))
        terrain_cycle = [t for t in ["flat", "rough", "steps"] for _ in range(ept)]
        need = len(terrain_cycle)
        if num_episodes < need:
            print(f"Note: expanding num_episodes from {num_episodes} to {need} (evaluation.episodes_per_terrain={ept}).")
            num_episodes = need

    print(f"\nEvaluating for {num_episodes} episodes...")
    eval_terrain_seed = evaluation_cfg.get("eval_terrain_base_seed", 0)

    for episode in range(num_episodes):
        print(f"Episode {episode + 1}/{num_episodes}", end=" ")

        video_path = None
        reset_options = None
        terrain_label = None
        if terrain_cycle is not None:
            terrain_label = terrain_cycle[episode % len(terrain_cycle)]
            reset_options = {"terrain_kind": terrain_label}
            # Reproducible but different rough heightfields per episode
            np.random.seed(int(eval_terrain_seed) + episode)
            print(f"[terrain: {terrain_label}] ", end="")

        if save_video:
            video_dir_path = output_dir / video_dir
            video_dir_path.mkdir(exist_ok=True)
            video_path = str(video_dir_path / f"episode_{episode + 1}.mp4")
        
        do_diagnostics = save_diagnostics and (episode == 0)
        result = visualize_episode(
            env=env,
            model=model,
            max_steps=max_steps,
            render=render and (episode == 0),  # Only render first episode
            save_video=save_video,
            video_path=video_path,
            collect_diagnostics=do_diagnostics,
            use_random_actions=False,
            reset_options=reset_options,
            terrain_label=terrain_label,
            include_trajectory_plot=include_trajectory_plot,
            video_header_lines=video_header_lines,
        )
        if do_diagnostics:
            metrics, tracker, diagnostics = result
            np.savez(
                output_dir / "eval_diagnostics.npz",
                **{k: v for k, v in diagnostics.items() if isinstance(v, np.ndarray)},
            )
            print(f" [diagnostics saved to eval_diagnostics.npz]", end="")
        else:
            metrics, tracker = result
        
        # Accumulate metrics
        all_metrics.episode_rewards.extend(metrics.episode_rewards)
        all_metrics.episode_lengths.extend(metrics.episode_lengths)
        all_metrics.forward_velocities.extend(metrics.forward_velocities)
        all_metrics.torso_heights.extend(metrics.torso_heights)
        all_metrics.energy_consumption.extend(metrics.energy_consumption)
        all_metrics.termination_reasons.extend(metrics.termination_reasons)
        all_metrics.episode_forward_delta_x_m.extend(metrics.episode_forward_delta_x_m)
        all_metrics.episode_xy_displacement_m.extend(metrics.episode_xy_displacement_m)
        all_metrics.episode_path_length_xy_m.extend(metrics.episode_path_length_xy_m)
        all_metrics.episode_terrain_kinds.extend(metrics.episode_terrain_kinds)
        
        ep_reward = metrics.episode_rewards[-1] if metrics.episode_rewards else 0
        ep_length = metrics.episode_lengths[-1] if metrics.episode_lengths else 0
        dxf = metrics.episode_forward_delta_x_m[-1] if metrics.episode_forward_delta_x_m else 0.0
        pathm = metrics.episode_path_length_xy_m[-1] if metrics.episode_path_length_xy_m else 0.0
        print(
            f" - Reward: {ep_reward:.2f}, Len: {ep_length}, "
            f"Fwd+X: {dxf:.3f} m, 2D path: {pathm:.3f} m"
        )
    
    # Compute summary (merge model provenance for reproducibility)
    summary = all_metrics.compute_summary()
    summary["model_basename"] = provenance.get("model_basename")
    summary["model_resolved_path"] = provenance.get("model_resolved_path")
    summary["model_file_sha256"] = provenance.get("model_file_sha256")
    summary["eval_protocol"] = provenance.get("eval_protocol")

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY (one policy file on all terrains above)")
    print("=" * 60)
    print(f"Episodes evaluated: {summary['num_episodes']}")
    print(f"Mean reward: {summary['mean_reward']:.2f} +/- {summary['std_reward']:.2f}")
    print(f"Mean episode length: {summary['mean_length']:.1f} +/- {summary['std_length']:.1f}")
    print(f"Mean forward velocity: {summary['mean_forward_velocity']:.4f} m/s")
    print(f"Max forward velocity: {summary['max_forward_velocity']:.4f} m/s")
    print(f"Mean torso height: {summary['mean_torso_height']:.4f} m")
    print(f"Mean energy consumption: {summary['mean_energy']:.4f}")
    print("--- Distance moved (episode means) ---")
    print(f"Mean forward +X: {summary['mean_forward_delta_x_m']:.4f} m  (std {summary['std_forward_delta_x_m']:.4f})")
    print(f"Mean 2D path length: {summary['mean_path_length_xy_m']:.4f} m  (std {summary['std_path_length_xy_m']:.4f})")
    print(f"Mean 2D displacement (start->end): {summary['mean_xy_displacement_m']:.4f} m")
    if summary.get("by_terrain"):
        print("--- By terrain ---")
        for tname, agg in summary["by_terrain"].items():
            print(
                f"  {tname}: n={agg['n_episodes']}  mean_fwd_X={agg['mean_forward_delta_x_m']:.4f} m  "
                f"mean_path={agg['mean_path_length_xy_m']:.4f} m  mean_reward={agg['mean_reward']:.1f}"
            )
    print(f"Termination reasons: {summary['termination_counts']}")
    print("=" * 60)
    
    # Save summary to file
    summary_path = output_dir / "evaluation_summary.yaml"
    with open(summary_path, 'w') as f:
        yaml.dump(summary, f)
    print(f"\nSummary saved to: {summary_path}")
    
    # Create plots
    if plot and (
        len(all_metrics.forward_velocities) > 0 or len(all_metrics.episode_forward_delta_x_m) > 0
    ):
        plot_path = output_dir / "evaluation_plots.png"
        plot_suptitle = None
        if provenance.get("model_file_sha256"):
            plot_suptitle = (
                f"One checkpoint: {provenance.get('model_basename', '')}  "
                f"SHA256 {provenance['model_file_sha256'][:12]}...  "
                f"(same weights; terrain cycled per episode)"
            )
        elif provenance.get("model_basename"):
            plot_suptitle = f"One checkpoint: {provenance['model_basename']} (same weights all episodes)"
        plot_evaluation_results(
            metrics=all_metrics,
            tracker=tracker,  # Use last episode tracker
            save_path=str(plot_path),
            suptitle=plot_suptitle,
        )
    
    # Cleanup
    env.close()
    
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate trained Hexapod sCPG model"
    )
    parser.add_argument(
        "model_path",
        type=str,
        help="Path to trained model (.zip file)"
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--num-episodes", "-n",
        type=int,
        default=10,
        help="Number of evaluation episodes"
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=1000,
        help="Maximum steps per episode"
    )
    parser.add_argument(
        "--render", "-r",
        action="store_true",
        help="Render during evaluation"
    )
    parser.add_argument(
        "--save-video", "-v",
        action="store_true",
        help="Save videos of episodes"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default=None,
        help="Output directory for results"
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Disable plotting"
    )
    parser.add_argument(
        "--device", "-d",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to use"
    )
    parser.add_argument(
        "--save-diagnostics",
        action="store_true",
        help="Save step-wise action_mean, log_std, reward components for first episode to eval_diagnostics.npz",
    )
    parser.add_argument(
        "--random-actions",
        action="store_true",
        help="Run one episode with random actions (sanity check) and save to eval_diagnostics_random.npz",
    )
    parser.add_argument(
        "--trajectory-plots",
        action="store_true",
        help="Save episode trajectory PNGs (trajectory + distance/velocity) when saving video; default is not to save them",
    )
    parser.add_argument(
        "--exaggerate-terrain-for-video",
        action="store_true",
        help="Use taller rough/steps for recordings (easier to see terrain; often harder for the policy).",
    )

    args = parser.parse_args()

    evaluate(
        model_path=args.model_path,
        config_path=args.config,
        num_episodes=args.num_episodes,
        max_steps=args.max_steps,
        render=args.render,
        save_video=args.save_video,
        output_dir=args.output_dir,
        plot=not args.no_plot,
        device=args.device,
        save_diagnostics=args.save_diagnostics,
        random_actions_episode=args.random_actions,
        include_trajectory_plot=args.trajectory_plots,
        exaggerate_terrain_for_video=True if args.exaggerate_terrain_for_video else None,
    )


if __name__ == "__main__":
    main()
