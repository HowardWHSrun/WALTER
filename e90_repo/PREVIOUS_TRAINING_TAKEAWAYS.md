# Takeaways from Previous Hexapod sCPG PPO Training

Summary of what was done and what to reuse when starting training from scratch.

---

## What the project does

- **Goal:** Train a hexapod walking controller in MuJoCo with PPO. The policy is a spiking CPG (sCPG): one oscillator, tripod A (legs 1,4,5) in phase, tripod B (legs 2,3,6) anti-phase.
- **Training:** PPO (Stable-Baselines3). Reward: forward distance + velocity + smoothness + height stability. Episode ends on fall, tilt > 35°, or leg flexion > 55°.
- **Code location:** Use either `hexapod_scpg_rl/` or `archive/hexapod_scpg_rl/` at the repo root; both have `train.py`, configs, envs, and models.

---

## Run organization (every run has this)

Each run directory (e.g. `runs/hexapod_scpg_tripod_v2_<timestamp>/`) contains:

| File | When | Purpose |
|------|------|--------|
| `run_info.yaml` | Start | Run plan: config path, target timesteps, num_envs, resume path, start time. |
| `training_summary.yaml` | End | **Total timesteps, total episodes, mean episode length**, end time. |
| `checkpoints/progress.yaml` | Every save_freq + end | List of `{timesteps, episodes}` at each checkpoint so you know how many episodes at 50k, 100k, etc. |
| `config.yaml` | Start | Copy of the config used. |
| `checkpoints/` | During | `hexapod_scpg_*_steps.zip`, `best/best_model.zip`, `final_model.zip`. |
| `logs/` | During | TensorBoard logs; includes `rollout/episodes_total` and `rollout/mean_episode_length`. |

So you can always see **how many episodes** a run had and how that maps to timesteps and checkpoints.

---

## Configs to use

| Config | Steps | Use case |
|--------|--------|----------|
| `config_v2_train_short.yaml` | 150k | Quick / sanity-check run. |
| `config_v2.yaml` | 2M | Full training. |
| `config_v2_continue.yaml` | 350k total | Continue a 150k run to 350k. |
| `config_v2_145_236_continue.yaml` | 350k total | Continue with lower LR on resume (`learning_rate_resume: 0.0001`). |

Always use **`--version 2`** for the tripod gait policy.

---

## How to start a new run

From the project root (e.g. `RL Temp/hexapod_scpg_rl` or `RL Temp/archive/hexapod_scpg_rl`):

```bash
cd path/to/hexapod_scpg_rl
python train.py --config config_v2_train_short.yaml --version 2 --output-dir path/to/hexapod_training_new/runs/run_001
```

Or let it create a timestamped dir under `runs/`:

```bash
python train.py --config config_v2_train_short.yaml --version 2
```

New run will appear under `hexapod_scpg_rl/runs/` (or under `--output-dir` if you set it). Check that `run_info.yaml` and then `training_summary.yaml` and `checkpoints/progress.yaml` appear.

---

## How to check if training is running

1. **Process:** `ps aux | grep train.py` or check your IDE’s run/terminal.
2. **Run directory:** Look for a new folder in `runs/` with `run_info.yaml` and a growing `checkpoints/` or `logs/`.
3. **TensorBoard:** `tensorboard --logdir runs/<run_name>/logs` and look for `rollout/episodes_total` and reward curves.
4. **Progress file:** Open `runs/<run_name>/checkpoints/progress.yaml`; it updates every `save_freq` steps.

---

## PPO and environment levers (no code change)

- **Train longer:** Increase `total_timesteps` (e.g. 500k–1M) or use a continue config.
- **Resume with smaller updates:** `--resume runs/<run>/checkpoints/best/best_model.zip` and in config set `learning_rate_resume: 0.0001` (and optionally `lr_schedule: "linear"`).
- **PPO (in config):** `n_steps` (e.g. 4096), `gamma` (e.g. 0.995), `learning_rate`, `clip_range`, `ent_coef`, larger `net_arch` (e.g. [128,128]).
- **Reward:** `smoothness_reward_weight`, `stability_reward_weight`; optional `near_termination_penalty_weight` and `near_termination_margin_deg` for a soft penalty near tilt/flexion limits.
- **Termination:** Relax limits early (e.g. `max_torso_tilt: 45`, `max_flexion_angle: 60`) then tighten in a later phase or in eval only.
- **Observation:** In env config you can set `obs_include_torso_height: true` or `obs_include_torso_vel: true` so the policy sees height/velocity.
- **VecNormalize:** In `training:` set `use_vec_normalize: true` (and optionally `norm_obs` / `norm_reward`) to use observation and reward normalization.

---

## Optional features already implemented

- **Near-termination penalty:** In `reward:` set `near_termination_penalty_weight` (e.g. 0.1) and `near_termination_margin_deg` (e.g. 10). Gives a smooth penalty when close to tilt/flexion limits.
- **Extra observation:** `obs_include_torso_height`, `obs_include_torso_vel`, `obs_include_joint_vel`, etc. in env config.
- **VecNormalize:** `training.use_vec_normalize: true` in config; checkpoint callback saves `vec_normalize.pkl` for resume.

---

## Suggested order when starting from scratch

1. Run 150k with `config_v2_train_short.yaml` and `--version 2`; confirm `run_info.yaml`, `training_summary.yaml`, and `progress.yaml` and that episode counts make sense.
2. If results are good, run longer (e.g. 500k–1M) or resume from best with a continue config and lower LR.
3. Tweak reward weights, termination limits, or PPO hyperparameters as needed; optionally enable VecNormalize or extra obs.

---

## Where things live

- **Training script and configs:** `hexapod_scpg_rl/` or `archive/hexapod_scpg_rl/` (train.py, config_v2*.yaml, envs/, models/).
- **This doc and new run outputs:** `hexapod_training_new/` (this folder). Point `--output-dir` here if you want all new runs under this folder, e.g. `hexapod_training_new/runs/run_001`.
