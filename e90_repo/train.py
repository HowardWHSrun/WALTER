#!/usr/bin/env python3
"""
Training Script for Hexapod sCPG RL

This script trains a Spiking Central Pattern Generator (sCPG) controller
for hexapod locomotion using PPO reinforcement learning.
"""

import os
import sys
import argparse
import yaml
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
import json
import time

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    EvalCallback,
    CallbackList,
)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize, sync_envs_normalization
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
from stable_baselines3.common.evaluation import evaluate_policy

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from envs.hexapod_env import HexapodEnv, make_hexapod_env
from models.encoder import SCPGPolicy, SCPGPolicyV2
from models.ode_gait import ODETripodPolicy


def _train_dbg_log(location: str, message: str, data: dict, hypothesis_id: str, run_id: str = "pre-fix"):
    # #region agent log
    try:
        log_path = "/Users/howardwang/Desktop/ValeroLab/E90/RL Temp/.cursor/debug.log"
        payload = {
            "sessionId": "debug-session",
            "runId": run_id,
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data,
            "timestamp": int(time.time() * 1000),
        }
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")
    except Exception:
        pass
    # #endregion


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def make_env(config: dict, rank: int = 0, seed: int = 0):
    """Create a callable that returns a HexapodEnv instance."""
    def _init():
        env = make_hexapod_env(config)
        env.reset(seed=seed + rank)
        return Monitor(env)
    return _init


class SCPGResetCallback:
    """Callback to reset CPG state at episode boundaries."""
    
    def __init__(self, policy):
        self.policy = policy
    
    def __call__(self, locals_, globals_):
        # Check if episode ended
        if 'dones' in locals_:
            dones = locals_['dones']
            if any(dones):
                # Reset CPG for environments that finished
                batch_size = len(dones)
                if hasattr(self.policy, 'reset_cpg'):
                    self.policy.reset_cpg(batch_size)
        return True


def _get_episode_stats_from_vec_env(vec_env):
    """
    Get total episodes and mean episode length from Monitor-wrapped sub-envs.
    Uses env_method so it works with DummyVecEnv and SubprocVecEnv.
    Returns (total_episodes, mean_episode_length) or (0, 0.0) if not available.
    """
    try:
        lengths_per_env = vec_env.env_method("get_episode_lengths")
    except Exception:
        return 0, 0.0
    all_lengths = []
    for lengths in lengths_per_env:
        if lengths is not None:
            all_lengths.extend(lengths)
    total_episodes = len(all_lengths)
    if total_episodes == 0:
        return 0, 0.0
    mean_length = sum(all_lengths) / total_episodes
    return total_episodes, mean_length


class EpisodeCountCallback(BaseCallback):
    """
    Logs total episodes and mean episode length to TensorBoard, and optionally
    writes checkpoints/progress.yaml at each checkpoint interval.
    """

    def __init__(self, output_dir, save_freq, progress_path=None, verbose=0):
        super().__init__(verbose)
        self.output_dir = Path(output_dir)
        self.save_freq = save_freq
        self.progress_path = Path(progress_path) if progress_path else None
        self._progress_entries = []
        self._last_written_timesteps = 0

    def _on_step(self):
        return True

    def _on_rollout_end(self):
        vec_env = self.model.get_env()
        total_episodes, mean_length = _get_episode_stats_from_vec_env(vec_env)
        num_timesteps = self.model.num_timesteps

        if total_episodes > 0:
            self.logger.record("rollout/episodes_total", total_episodes)
            self.logger.record("rollout/mean_episode_length", mean_length)

        if self.progress_path is not None and num_timesteps > 0 and num_timesteps - self._last_written_timesteps >= self.save_freq:
            self._progress_entries.append({
                "timesteps": num_timesteps,
                "episodes": total_episodes,
            })
            self._last_written_timesteps = num_timesteps
            self._write_progress()

        return True

    def _write_progress(self):
        if self.progress_path is None:
            return
        self.progress_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.progress_path, 'w') as f:
            yaml.dump({"checkpoints": self._progress_entries}, f, default_flow_style=False)

    def get_final_episode_stats(self):
        """Get (total_episodes, mean_episode_length) from current env state."""
        vec_env = self.model.get_env()
        return _get_episode_stats_from_vec_env(vec_env)


class TerrainCurriculumCallback(BaseCallback):
    """Updates terrain curriculum stage on HexapodEnv based on num_timesteps and stage thresholds (flat -> flat+rough -> all)."""

    def __init__(self, stage_timesteps, check_freq=5000, verbose=0):
        super().__init__(verbose)
        self.stage_timesteps = list(stage_timesteps)  # e.g. [100000, 200000] -> stage 1 at 100k, stage 2 at 200k
        self.check_freq = check_freq
        self._last_stage = -1

    def _on_training_start(self):
        self._apply_stage(0)
        return True

    def _apply_stage(self, stage):
        vec_env = self.model.get_env()
        for i in range(vec_env.num_envs):
            sub = vec_env.envs[i]
            base = getattr(sub, "unwrapped", sub)
            if hasattr(base, "set_terrain_curriculum_stage"):
                base.set_terrain_curriculum_stage(stage)
        if self.verbose and stage != self._last_stage:
            print(f"Terrain curriculum stage -> {stage} (flat only / flat+rough / all) at {self.model.num_timesteps} timesteps")
        self._last_stage = stage

    def _on_step(self):
        if self.n_calls % self.check_freq != 0:
            return True
        t = self.model.num_timesteps
        stage = 0
        for i, thresh in enumerate(self.stage_timesteps):
            if t >= thresh:
                stage = i + 1
        stage = min(stage, 2)
        if stage == self._last_stage:
            return True
        self._apply_stage(stage)
        return True

    def _on_rollout_end(self):
        return True


class SyncVecNormalizeCallback(BaseCallback):
    """Syncs VecNormalize obs_rms and ret_rms from training env to eval env(s)."""

    def __init__(self, eval_envs, verbose=0):
        super().__init__(verbose)
        if eval_envs is None:
            self.eval_envs = []
        elif isinstance(eval_envs, (list, tuple)):
            self.eval_envs = [e for e in eval_envs if e is not None]
        else:
            self.eval_envs = [eval_envs]

    def _on_rollout_end(self):
        train_env = self.model.get_env()
        for ev in self.eval_envs:
            if isinstance(train_env, VecNormalize) and isinstance(ev, VecNormalize):
                ev.obs_rms = train_env.obs_rms
                ev.ret_rms = train_env.ret_rms
        return True


class PrefixedEvalCallback(EvalCallback):
    """
    Same as EvalCallback but logs TensorBoard scalars under log_prefix/* instead of eval/*
    (avoids collisions when running primary + regression evaluation).
    """

    def __init__(self, *args, log_prefix: str = "regression_eval", **kwargs):
        self._log_prefix = log_prefix.rstrip("/")
        super().__init__(*args, **kwargs)

    def _on_step(self):
        continue_training = True

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            if self.model.get_vec_normalize_env() is not None:
                try:
                    sync_envs_normalization(self.training_env, self.eval_env)
                except AttributeError as e:
                    raise AssertionError(
                        "Training and eval env are not wrapped the same way for VecNormalize; "
                        "see EvalCallback docs."
                    ) from e

            self._is_success_buffer = []

            episode_rewards, episode_lengths = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                render=self.render,
                deterministic=self.deterministic,
                return_episode_rewards=True,
                warn=self.warn,
                callback=self._log_success_callback,
            )

            if self.log_path is not None:
                self.evaluations_timesteps.append(self.num_timesteps)
                self.evaluations_results.append(episode_rewards)
                self.evaluations_length.append(episode_lengths)
                kwargs = {}
                if len(self._is_success_buffer) > 0:
                    self.evaluations_successes.append(self._is_success_buffer)
                    kwargs = dict(successes=self.evaluations_successes)
                np.savez(
                    self.log_path,
                    timesteps=self.evaluations_timesteps,
                    results=self.evaluations_results,
                    ep_lengths=self.evaluations_length,
                    **kwargs,
                )

            mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
            mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
            self.last_mean_reward = float(mean_reward)

            if self.verbose >= 1:
                p = self._log_prefix
                print(
                    f"[{p}] num_timesteps={self.num_timesteps}, "
                    f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}"
                )
                print(f"[{p}] Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")
            self.logger.record(f"{self._log_prefix}/mean_reward", float(mean_reward))
            self.logger.record(f"{self._log_prefix}/mean_ep_length", mean_ep_length)

            if len(self._is_success_buffer) > 0:
                success_rate = np.mean(self._is_success_buffer)
                if self.verbose >= 1:
                    print(f"[{self._log_prefix}] Success rate: {100 * success_rate:.2f}%")
                self.logger.record(f"{self._log_prefix}/success_rate", success_rate)

            self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
            self.logger.dump(self.num_timesteps)

            if mean_reward > self.best_mean_reward:
                if self.verbose >= 1:
                    print(f"[{self._log_prefix}] New best mean reward!")
                if self.best_model_save_path is not None:
                    self.model.save(os.path.join(self.best_model_save_path, "best_model"))
                self.best_mean_reward = float(mean_reward)
                if self.callback_on_new_best is not None:
                    continue_training = self.callback_on_new_best.on_step()

            if self.callback is not None:
                continue_training = continue_training and self._on_event()

        return continue_training


class RewardPhaseCallback(BaseCallback):
    """
    Applies reward weight overrides from config phases based on model.num_timesteps.
    Phases: list of {start_timestep: int, reward: {attr_name: value, ...}}
    """

    def __init__(self, phases, verbose=0):
        super().__init__(verbose)
        self.phases = sorted(phases, key=lambda p: int(p.get("start_timestep", 0)))
        self._applied_index = -1

    def _phase_index_for_timesteps(self, t: int) -> int:
        idx = 0
        for i, p in enumerate(self.phases):
            if t >= int(p.get("start_timestep", 0)):
                idx = i
        return idx

    def _apply_phase(self, idx: int):
        if idx < 0 or idx >= len(self.phases):
            return
        overrides = self.phases[idx].get("reward") or {}
        vec = self.model.get_env()
        for k, v in overrides.items():
            try:
                if isinstance(v, bool):
                    vec.set_attr(k, v)
                elif isinstance(v, (int, float)):
                    vec.set_attr(k, float(v))
                else:
                    vec.set_attr(k, v)
            except Exception as exc:
                if self.verbose:
                    print(f"RewardPhaseCallback: could not set {k}={v!r}: {exc}")
        if self.logger is not None:
            self.logger.record("training/reward_phase_index", idx)
            self.logger.record("training/reward_phase_start_timestep", int(self.phases[idx].get("start_timestep", 0)))

    def _on_step(self):
        return True

    def _on_training_start(self):
        if self.phases:
            self._applied_index = -1
            self._apply_phase(0)
            self._applied_index = 0
        return True

    def _on_rollout_end(self):
        if not self.phases:
            return True
        t = int(self.model.num_timesteps)
        idx = self._phase_index_for_timesteps(t)
        if idx != self._applied_index:
            self._apply_phase(idx)
            self._applied_index = idx
        return True


class ImitationMixCallback(BaseCallback):
    """Linearly decays imitation_action_mix on the training env."""

    def __init__(self, mix_start: float, mix_end: float, decay_steps: int, verbose: int = 0):
        super().__init__(verbose)
        self.mix_start = float(mix_start)
        self.mix_end = float(mix_end)
        self.decay_steps = max(0, int(decay_steps))
        self._last_mix = None

    def _set_mix(self, mix: float):
        env = self.model.get_env()
        try:
            env.set_attr("imitation_action_mix", float(mix))
        except Exception:
            return

    def _compute_mix(self) -> float:
        if self.decay_steps <= 0:
            return self.mix_end
        progress = min(self.model.num_timesteps / self.decay_steps, 1.0)
        return self.mix_start + (self.mix_end - self.mix_start) * progress

    def _on_training_start(self):
        self._last_mix = self.mix_start
        self._set_mix(self.mix_start)
        _train_dbg_log(
            "train.py:ImitationMixCallback",
            "mix_start",
            {"mix": float(self.mix_start), "timesteps": int(self.model.num_timesteps)},
            "H6",
        )
        return True

    def _on_step(self):
        mix = self._compute_mix()
        if self._last_mix is None or abs(mix - self._last_mix) > 1e-4:
            self._set_mix(mix)
            self._last_mix = mix
        if self.model.num_timesteps % 10000 == 0:
            _train_dbg_log(
                "train.py:ImitationMixCallback",
                "mix_update",
                {"mix": float(mix), "timesteps": int(self.model.num_timesteps)},
                "H6",
            )
        return True

def train(
    config_path: str = "config.yaml",
    output_dir: str = None,
    resume_from: str = None,
    seed: int = None,
    num_envs: int = 1,
    device: str = "auto",
    version: int = 1,
):
    """
    Main training function.
    
    Args:
        config_path: Path to configuration YAML file
        output_dir: Directory to save outputs (logs, checkpoints, etc.)
        resume_from: Path to checkpoint to resume from
        seed: Random seed (overrides config if provided)
        num_envs: Number of parallel environments
        device: Device to use for training (cpu, cuda, or auto)
    """
    # Load configuration
    config = load_config(config_path)
    
    # Override seed if provided
    if seed is not None:
        config['training']['seed'] = seed
    
    training_config = config.get('training', {})
    ppo_config = config.get('ppo', {})
    training_seed = training_config.get('seed', 42)
    
    # Setup output directory
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        use_tripod_for_dir = config.get('scpg', {}).get('use_tripod_gait', False) or (version == 2)
        suffix = "_tripod_v2" if use_tripod_for_dir else ""
        output_dir = PROJECT_ROOT / f"runs/hexapod_scpg{suffix}_{timestamp}"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    log_dir = output_dir / "logs"
    checkpoint_dir.mkdir(exist_ok=True)
    log_dir.mkdir(exist_ok=True)
    
    # Save config to output directory
    with open(output_dir / "config.yaml", 'w') as f:
        yaml.dump(config, f)

    # Run manifest: record what this run is and will do
    total_timesteps_target = training_config.get('total_timesteps', 2_000_000)
    run_info = {
        "name": output_dir.name,
        "config_path": str(config_path),
        "total_timesteps": total_timesteps_target,
        "num_envs": num_envs,
        "resume_from": resume_from,
        "version": version,
        "start_time": datetime.now().isoformat(),
    }
    with open(output_dir / "run_info.yaml", 'w') as f:
        yaml.dump(run_info, f, default_flow_style=False)
    
    print(f"Output directory: {output_dir}")
    print(f"Training with seed: {training_seed}")
    
    # Set random seeds
    np.random.seed(training_seed)
    torch.manual_seed(training_seed)
    
    # Create environments
    print(f"Creating {num_envs} parallel environment(s)...")
    
    if num_envs == 1:
        env = DummyVecEnv([make_env(config, 0, training_seed)])
    else:
        env = SubprocVecEnv([
            make_env(config, i, training_seed) 
            for i in range(num_envs)
        ])
    
    # Create evaluation environment
    eval_env = DummyVecEnv([make_env(config, 0, training_seed + 1000)])

    # Optionally wrap with VecNormalize (observation and reward normalization)
    use_vec_normalize = training_config.get("use_vec_normalize", False)
    if use_vec_normalize:
        norm_obs = training_config.get("norm_obs", True)
        norm_reward = training_config.get("norm_reward", True)
        vec_normalize_path = Path(resume_from).parent / "vec_normalize.pkl" if resume_from else None
        if resume_from and vec_normalize_path is not None and vec_normalize_path.exists():
            env = VecNormalize.load(str(vec_normalize_path), env)
            env.training = True
            env.norm_obs = norm_obs
            env.norm_reward = norm_reward
            print(f"Loaded VecNormalize from {vec_normalize_path}")
        else:
            env = VecNormalize(
                env,
                training=True,
                norm_obs=norm_obs,
                norm_reward=norm_reward,
                clip_obs=training_config.get("clip_obs", 10.0),
                clip_reward=training_config.get("clip_reward", 10.0),
                gamma=ppo_config.get("gamma", 0.99),
            )
        eval_env = VecNormalize(
            eval_env,
            training=False,
            norm_obs=norm_obs,
            norm_reward=False,
            clip_obs=training_config.get("clip_obs", 10.0),
        )
        # Sync eval env stats from train env so evaluation uses same normalization
        eval_env.obs_rms = env.obs_rms
        eval_env.ret_rms = env.ret_rms
        print("Using VecNormalize (obs and reward normalization).")

    eval_mix = training_config.get("imitation_action_mix_eval", training_config.get("imitation_action_mix_end"))
    if eval_mix is not None:
        try:
            eval_env.set_attr("imitation_action_mix", float(eval_mix))
        except Exception:
            pass

    # Optional: regression eval on a baseline (e.g. flat) config to monitor catastrophic forgetting
    regression_eval_env = None
    regression_enabled = bool(training_config.get("regression_eval_enabled", False))
    regression_cfg_raw = training_config.get("regression_eval_config_path")
    if regression_enabled and regression_cfg_raw:
        reg_path = Path(regression_cfg_raw)
        if not reg_path.is_file():
            reg_path = PROJECT_ROOT / regression_cfg_raw
        if not reg_path.is_file():
            raise FileNotFoundError(
                f"regression_eval_config_path not found: {regression_cfg_raw} (tried project root)"
            )
        baseline_config = load_config(str(reg_path))
        regression_eval_env = DummyVecEnv([make_env(baseline_config, 0, training_seed + 2000)])
        if use_vec_normalize:
            regression_eval_env = VecNormalize(
                regression_eval_env,
                training=False,
                norm_obs=norm_obs,
                norm_reward=False,
                clip_obs=training_config.get("clip_obs", 10.0),
            )
            regression_eval_env.obs_rms = env.obs_rms
            regression_eval_env.ret_rms = env.ret_rms
        if eval_mix is not None:
            try:
                regression_eval_env.set_attr("imitation_action_mix", float(eval_mix))
            except Exception:
                pass
        print(f"Regression eval env from: {reg_path}")
    elif regression_enabled and not regression_cfg_raw:
        print("Warning: regression_eval_enabled but regression_eval_config_path missing; skipping.")

    # Get dimensions for policy
    obs_space = env.observation_space
    action_space = env.action_space
    
    print(f"Observation space: {obs_space}")
    print(f"Action space: {action_space}")
    
    # Extract CPG parameters from config
    scpg_config = config.get('scpg', {})
    use_tripod = scpg_config.get('use_tripod_gait', False) or (version == 2)
    use_mlp_policy = ppo_config.get('use_mlp_policy', False)
    use_ode_tripod = bool(ppo_config.get('use_ode_tripod_policy', False))

    # Policy and keyword arguments (ODE tripod, MLP baseline, or sCPG v1/v2)
    if use_ode_tripod:
        ode_cfg = ppo_config.get('ode_tripod', {}) or {}
        net_arch = ppo_config.get('policy_kwargs', {}).get('net_arch')
        hidden_sizes = (64, 64)
        if isinstance(net_arch, dict) and net_arch.get('pi'):
            hidden_sizes = tuple(int(x) for x in net_arch['pi'])
        elif isinstance(net_arch, list):
            hidden_sizes = tuple(int(x) for x in net_arch)
        fr = ode_cfg.get('freq_hz_range', [0.35, 1.6])
        policy_class = ODETripodPolicy
        policy_kwargs = {
            'sim_time_in_obs': bool(ode_cfg.get('sim_time_in_obs', True)),
            'freq_hz_range': (float(fr[0]), float(fr[1])),
            'max_amp_flex': float(ode_cfg.get('max_amp_flex', 0.95)),
            'max_amp_abd': float(ode_cfg.get('max_amp_abd', 0.95)),
            'flex_sign': float(ode_cfg.get('flex_sign', 1.0)),
            'abd_sign': float(ode_cfg.get('abd_sign', 1.0)),
            'residual_scale': float(ode_cfg.get('residual_scale', 0.2)),
            'hidden_sizes': hidden_sizes,
        }
        policy_kwargs.update(
            {k: v for k, v in ppo_config.get('policy_kwargs', {}).items() if k != 'net_arch'}
        )
        print(
            "Using ODETripodPolicy: theta=2*pi*f*sim_time+phi from obs; "
            "tripod map + learned residual (stateless phase via env sim time)."
        )
    elif use_mlp_policy:
        policy_class = "MlpPolicy"
        policy_kwargs = dict(ppo_config.get('policy_kwargs', {}))
        print("Using MlpPolicy baseline (same env/reward; for debugging move-once-then-stop).")
    elif use_tripod:
        policy_class = SCPGPolicyV2
        policy_kwargs = {
            "neurons_per_oscillator": scpg_config.get('neurons_per_oscillator', 32),
            "tau_mem": scpg_config.get('tau_mem', 20.0),
            "tau_syn": scpg_config.get('tau_syn', 10.0),
            "num_timesteps": scpg_config.get('num_timesteps', 10),
        }
        print("Using tripod gait policy (SCPGPolicyV2): legs 1,4,5 in phase; 2,3,6 anti-phase.")
    else:
        policy_class = SCPGPolicy
        policy_kwargs = {
            "num_legs": scpg_config.get('num_legs', 6),
            "neurons_per_oscillator": scpg_config.get('neurons_per_oscillator', 32),
            "tau_mem": scpg_config.get('tau_mem', 20.0),
            "tau_syn": scpg_config.get('tau_syn', 10.0),
            "coupling_strength": scpg_config.get('phase_coupling_strength', 0.5),
            "num_timesteps": scpg_config.get('num_timesteps', 10),
        }
    if not use_mlp_policy and not use_ode_tripod:
        policy_kwargs.update(ppo_config.get('policy_kwargs', {}))

    # Learning rate: constant or linear decay (progress_remaining 1 -> 0)
    lr_initial = ppo_config.get('learning_rate', 3e-4)
    lr_schedule_type = ppo_config.get('lr_schedule', 'constant')
    lr_final_frac = ppo_config.get('lr_final_fraction', 0.1)
    if lr_schedule_type == 'linear':
        def lr_schedule(progress_remaining):
            return lr_initial * (lr_final_frac + (1.0 - lr_final_frac) * progress_remaining)
        learning_rate = lr_schedule
        print(f"Using linear LR schedule: {lr_initial} -> {lr_initial * lr_final_frac}")
    else:
        learning_rate = lr_initial

    # Create or load model
    if resume_from is not None:
        print(f"Resuming from checkpoint: {resume_from}")
        model = PPO.load(
            resume_from,
            env=env,
            device=device,
        )
        # Optional: lower LR for continued training (constant or linear decay)
        lr_resume = ppo_config.get('learning_rate_resume')
        if lr_resume is not None:
            if lr_schedule_type == 'linear':
                def lr_schedule_resume(progress_remaining):
                    return lr_resume * (lr_final_frac + (1.0 - lr_final_frac) * progress_remaining)
                model.learning_rate = lr_schedule_resume
                print(f"Set resume LR schedule: {lr_resume} -> {lr_resume * lr_final_frac}")
            else:
                model.learning_rate = lambda _: float(lr_resume)
                print(f"Set resume learning rate to {lr_resume}")
    else:
        _pc = " (MlpPolicy)" if use_mlp_policy else (" (ODETripodPolicy)" if use_ode_tripod else " with sCPG policy")
        print("Creating new PPO model..." + _pc)
        model = PPO(
            policy=policy_class,
            env=env,
            learning_rate=learning_rate,
            n_steps=ppo_config.get('n_steps', 2048),
            batch_size=ppo_config.get('batch_size', 64),
            n_epochs=ppo_config.get('n_epochs', 10),
            gamma=ppo_config.get('gamma', 0.99),
            gae_lambda=ppo_config.get('gae_lambda', 0.95),
            clip_range=ppo_config.get('clip_range', 0.2),
            ent_coef=ppo_config.get('ent_coef', 0.01),
            vf_coef=ppo_config.get('vf_coef', 0.5),
            max_grad_norm=ppo_config.get('max_grad_norm', 0.5),
            policy_kwargs=policy_kwargs,
            verbose=1,
            seed=training_seed,
            device=device,
            tensorboard_log=str(log_dir),
        )
    
    # Setup callbacks
    save_freq = training_config.get('save_freq', 50000) // num_envs
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path=str(checkpoint_dir),
        name_prefix="hexapod_scpg",
        save_replay_buffer=False,
        save_vecnormalize=True,
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(checkpoint_dir / "best"),
        log_path=str(log_dir),
        eval_freq=training_config.get('eval_freq', 10000) // num_envs,
        n_eval_episodes=training_config.get('n_eval_episodes', 5),
        deterministic=True,
        render=False,
    )

    episode_count_callback = EpisodeCountCallback(
        output_dir=str(output_dir),
        save_freq=save_freq,
        progress_path=str(checkpoint_dir / "progress.yaml"),
    )

    callbacks_list = [episode_count_callback, checkpoint_callback, eval_callback]
    reward_phases = training_config.get("reward_phases")
    prefix_callbacks = 0
    if reward_phases:
        callbacks_list.insert(0, RewardPhaseCallback(reward_phases, verbose=1))
        prefix_callbacks += 1
        print(f"Reward phases enabled: {len(reward_phases)} stage(s)")
    mix_start = training_config.get("imitation_action_mix_start")
    mix_end = training_config.get("imitation_action_mix_end")
    mix_decay_steps = training_config.get("imitation_action_mix_decay_steps")
    if mix_start is not None and mix_end is not None and mix_decay_steps is not None:
        callbacks_list.insert(0, ImitationMixCallback(mix_start, mix_end, mix_decay_steps))
        prefix_callbacks += 1
    vec_eval_targets = [eval_env]
    if regression_eval_env is not None:
        vec_eval_targets.append(regression_eval_env)
    if use_vec_normalize:
        # Keep Sync immediately after EpisodeCountCallback (index prefix_callbacks + 1)
        callbacks_list.insert(prefix_callbacks + 1, SyncVecNormalizeCallback(vec_eval_targets))

    if regression_eval_env is not None:
        reg_det = bool(training_config.get("regression_eval_deterministic", True))
        reg_freq = max(1, int(training_config.get("regression_eval_freq", 50000)) // num_envs)
        reg_eps = int(training_config.get("regression_eval_n_episodes", 5))
        reg_verbose = int(training_config.get("regression_eval_verbose", 1))
        reg_prefix = str(training_config.get("regression_eval_log_prefix", "regression_eval"))
        regression_callback = PrefixedEvalCallback(
            regression_eval_env,
            best_model_save_path=str(checkpoint_dir / "regression_best"),
            log_path=str(log_dir / "regression_eval"),
            eval_freq=reg_freq,
            n_eval_episodes=reg_eps,
            deterministic=reg_det,
            render=False,
            verbose=reg_verbose,
            log_prefix=reg_prefix,
        )
        callbacks_list.append(regression_callback)
    env_config = config.get("env", {})
    if env_config.get("terrain_curriculum"):
        stages = env_config.get("terrain_curriculum_stages", [100000, 200000])
        callbacks_list.append(TerrainCurriculumCallback(stages, check_freq=5000, verbose=1))
    callbacks = CallbackList(callbacks_list)
    
    # Train
    total_timesteps = training_config.get('total_timesteps', 2_000_000)
    print(f"\nStarting training for {total_timesteps} timesteps...")
    print("=" * 60)
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            log_interval=training_config.get('log_interval', 10),
            progress_bar=True,
            reset_num_timesteps=resume_from is None,
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")

    # Final episode stats and training summary
    total_episodes, mean_episode_length = episode_count_callback.get_final_episode_stats()
    num_timesteps_done = model.num_timesteps
    training_summary = {
        "total_timesteps": num_timesteps_done,
        "total_episodes": total_episodes,
        "mean_episode_length": round(mean_episode_length, 2) if total_episodes > 0 else None,
        "end_time": datetime.now().isoformat(),
    }
    with open(output_dir / "training_summary.yaml", 'w') as f:
        yaml.dump(training_summary, f, default_flow_style=False)

    # Ensure progress.yaml has final (timesteps, episodes)
    progress_path = checkpoint_dir / "progress.yaml"
    if progress_path.exists():
        with open(progress_path, 'r') as f:
            progress_data = yaml.safe_load(f) or {}
        entries = progress_data.get("checkpoints", [])
    else:
        entries = []
    if not entries or entries[-1].get("timesteps") != num_timesteps_done:
        entries.append({"timesteps": num_timesteps_done, "episodes": total_episodes})
        with open(progress_path, 'w') as f:
            yaml.dump({"checkpoints": entries}, f, default_flow_style=False)
    
    # Save final model
    final_model_path = checkpoint_dir / "final_model"
    model.save(str(final_model_path))
    print(f"\nFinal model saved to: {final_model_path}")
    mean_len_str = f"{mean_episode_length:.1f}" if total_episodes > 0 else "N/A"
    print(f"Training summary: {num_timesteps_done} timesteps, {total_episodes} episodes (mean length: {mean_len_str})")

    # Optional auto-evaluation with video
    if training_config.get("auto_evaluate", True):
        eval_cfg = config.get("evaluation", {})
        eval_num_episodes = int(eval_cfg.get("num_episodes", 1))
        eval_render = bool(eval_cfg.get("render", False))
        eval_save_video = bool(eval_cfg.get("save_video", True))
        eval_video_dir = eval_cfg.get("video_dir", "videos")
        eval_max_steps = int(eval_cfg.get("max_steps", config.get("env", {}).get("max_episode_steps", 1000)))
        eval_output_dir = output_dir / "eval"
        eval_output_dir.mkdir(parents=True, exist_ok=True)
        best_model_path = checkpoint_dir / "best" / "best_model.zip"
        model_path = str(best_model_path if best_model_path.exists() else final_model_path.with_suffix(".zip"))
        try:
            from evaluate import evaluate as run_evaluate
            run_evaluate(
                model_path=model_path,
                config_path=str(config_path),
                num_episodes=eval_num_episodes,
                max_steps=eval_max_steps,
                render=eval_render,
                save_video=eval_save_video,
                output_dir=str(eval_output_dir),
                plot=False,
                device=device,
            )
        except Exception as exc:
            print(f"Auto-evaluation failed: {exc}")
    
    # Cleanup
    env.close()
    eval_env.close()
    if regression_eval_env is not None:
        regression_eval_env.close()
    
    print("\nTraining complete!")
    print(f"Outputs saved to: {output_dir}")
    
    return model


def main():
    parser = argparse.ArgumentParser(
        description="Train Hexapod sCPG controller with PPO"
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default=None,
        help="Output directory for logs and checkpoints"
    )
    parser.add_argument(
        "--resume", "-r",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--seed", "-s",
        type=int,
        default=None,
        help="Random seed"
    )
    parser.add_argument(
        "--num-envs", "-n",
        type=int,
        default=1,
        help="Number of parallel environments"
    )
    parser.add_argument(
        "--device", "-d",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to use for training"
    )
    parser.add_argument(
        "--version", "-v",
        type=int,
        default=1,
        choices=[1, 2],
        help="Policy version: 1=per-leg CPG, 2=tripod gait (three legs in phase each)"
    )
    args = parser.parse_args()

    train(
        config_path=args.config,
        output_dir=args.output_dir,
        resume_from=args.resume,
        seed=args.seed,
        num_envs=args.num_envs,
        device=args.device,
        version=args.version,
    )


if __name__ == "__main__":
    main()
