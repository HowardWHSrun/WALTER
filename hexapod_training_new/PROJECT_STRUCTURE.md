# Project Structure

Detailed explanation of the hexapod training project layout.

---

## Directory tree

```
hexapod_training_new/
├── configs/                    # Training configurations
│   ├── config_v2_train_short.yaml    # 150k steps (quick)
│   ├── config_v2.yaml                # 2M steps (full)
│   ├── config_v2_continue.yaml       # Continue to 350k
│   └── config_v2_145_236_continue.yaml  # Continue with lower LR
│
├── scripts/                    # Helper scripts
│   ├── start_training.sh       # Start new training run
│   ├── check_progress.sh       # Check run status
│   └── resume_training.sh      # Resume from checkpoint
│
├── runs/                       # Training outputs (created on first run)
│   └── run_<timestamp>/        # One folder per run
│       ├── run_info.yaml                  # Run metadata at start
│       ├── config.yaml                    # Copy of config used
│       ├── training_summary.yaml          # Final results (timesteps, episodes)
│       ├── checkpoints/
│       │   ├── progress.yaml              # Timesteps → episodes at each checkpoint
│       │   ├── hexapod_scpg_50000_steps.zip
│       │   ├── hexapod_scpg_100000_steps.zip
│       │   ├── best/
│       │   │   └── best_model.zip         # Best by eval reward
│       │   └── final_model.zip
│       └── logs/
│           ├── evaluations.npz
│           └── PPO_1/
│               └── events.out.tfevents.*   # TensorBoard logs
│
├── envs/                       # MuJoCo environment
│   ├── __init__.py
│   └── hexapod_env.py          # Hexapod Gymnasium env
│
├── models/                     # sCPG policy
│   ├── __init__.py
│   ├── scpg.py                 # CPG with value head (tripod version)
│   └── encoder.py              # sCPG policy for SB3
│
├── assets/                     # MuJoCo model files
│   ├── hexapod.xml             # MJCF model
│   └── meshes/
│       └── *.stl               # 3D meshes
│
├── train.py                    # Main training script
├── evaluate.py                 # Evaluation and video generation
├── visualize.py                # Hand-designed CPG demo (no RL)
├── test_env.py                 # Environment testing
├── requirements.txt            # Python dependencies
│
├── README.md                   # Quick start guide
├── PREVIOUS_TRAINING_TAKEAWAYS.md  # Lessons from prior training
└── PROJECT_STRUCTURE.md        # This file
```

---

## Key files explained

### Training code

- **train.py** – Main training loop with PPO. Accepts `--config`, `--version`, `--output-dir`, `--resume`. Creates run folder, writes `run_info.yaml` at start, `training_summary.yaml` at end. Logs episode counts to TensorBoard.

- **configs/*.yaml** – Training configuration: env settings (termination limits, reward weights, observation), PPO hyperparameters (LR, n_steps, gamma), sCPG settings (neurons, tau), training (total_timesteps, save/eval freq).

### Run outputs (one per training session)

- **run_info.yaml** – Written at start: run name, config path, target timesteps, num_envs, resume_from, start_time. Tells you what the run is supposed to do.

- **training_summary.yaml** – Written at end: `total_timesteps`, `total_episodes`, `mean_episode_length`, end_time. Main file to see **how many episodes** the run had.

- **checkpoints/progress.yaml** – Updated every `save_freq` steps: list of `{timesteps, episodes}` so you know "at 50k steps we had ~200 episodes", etc.

- **checkpoints/*.zip** – PPO model + policy weights at that timestep. Load with `PPO.load(path)` or pass to `--resume`.

- **checkpoints/best/best_model.zip** – Model with highest eval reward during training. Use this for resume or eval.

- **logs/PPO_1/events.out.tfevents.*** – TensorBoard logs. Includes `rollout/episodes_total`, `rollout/mean_episode_length`, reward curves, policy losses. View with `tensorboard --logdir runs/<run>/logs`.

### Environment and policy

- **envs/hexapod_env.py** – Gymnasium environment wrapping MuJoCo. Observation: joint angles + optional (torso height/velocity, prev action). Action: 12 joint torques. Reward: forward distance + velocity + smoothness + stability. Termination: fall, tilt > 35°, flexion > 55°.

- **models/scpg.py** – Spiking CPG network. `CPGWithValueHeadTripod`: one oscillator, output goes to legs 1,4,5; negated output to legs 2,3,6 (tripod gait).

- **models/encoder.py** – SB3-compatible policies. `SCPGPolicyV2` uses tripod CPG for action, separate MLP for value.

- **assets/hexapod.xml** – MuJoCo MJCF: 6 legs × 2 DOF (abduction + flexion), torque control, floor with friction 1.5, leg-leg collision disabled.

### Helper scripts

- **scripts/start_training.sh** – Start training with a config and run name. Outputs go to `runs/<run_name>/`.

- **scripts/check_progress.sh** – Show `run_info.yaml`, `progress.yaml`, `training_summary.yaml` for latest or specified run. Lists checkpoints.

- **scripts/resume_training.sh** – Resume from `checkpoints/best/best_model.zip` of a previous run. Uses continue config by default (can set lower LR).

---

## Episode tracking (implemented)

The training now tracks **episodes** in addition to timesteps:

1. **During training:** Episode count logged to TensorBoard as `rollout/episodes_total` and `rollout/mean_episode_length` every rollout.

2. **At checkpoints:** `checkpoints/progress.yaml` records `{timesteps, episodes}` at each save (e.g. 50k, 100k, 150k).

3. **At end:** `training_summary.yaml` includes `total_episodes` and `mean_episode_length`.

So you can always answer "how many episodes did this run have?" and "at 290k steps, how many episodes were completed?"

---

## Optional features (in configs, no code change)

- **Near-termination penalty:** Set `reward.near_termination_penalty_weight` (e.g. 0.1) and `reward.near_termination_margin_deg` (e.g. 10) to give a soft penalty when close to tilt/flexion limits.

- **Extra observation:** Set `env.obs_include_torso_height: true` or `env.obs_include_torso_vel: true` so policy sees torso state.

- **VecNormalize:** Set `training.use_vec_normalize: true` in config to enable observation and reward normalization (can stabilize PPO). Saved with checkpoints as `vec_normalize.pkl`.

See **PREVIOUS_TRAINING_TAKEAWAYS.md** for full details on PPO levers and config options.
