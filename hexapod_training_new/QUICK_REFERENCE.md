# Quick Reference Card

One-page cheat sheet for hexapod sCPG training.

---

## Start training

```bash
cd hexapod_training_new
./scripts/start_training.sh                           # Uses defaults (config_v2_train_short)
./scripts/start_training.sh config_v2 my_long_run    # Full 2M steps
./scripts/start_training.sh config_v2_train_short_pd.yaml my_pd_run
./scripts/start_training.sh config_v2_train_short_pd_walk.yaml my_walk_run
./scripts/start_training.sh config_v2_train_short_pd_imitation.yaml my_imitation_run
```

---

## Check progress

```bash
./scripts/check_progress.sh              # Latest run
./scripts/check_progress.sh my_run       # Specific run
tensorboard --logdir runs/my_run/logs    # TensorBoard
```

---

## Resume training

```bash
./scripts/resume_training.sh my_run                          # Uses config_v2_continue
./scripts/resume_training.sh my_run config_v2_continue my_continued
```

---

## Manual control

```bash
# Start
python train.py --config configs/config_v2_train_short.yaml --version 2 --output-dir runs/test

# Resume
python train.py --config configs/config_v2_continue.yaml --version 2 \
    --resume runs/test/checkpoints/best/best_model.zip \
    --output-dir runs/test_continued
```

---

## Files to check

| File | What it tells you |
|------|-------------------|
| `runs/<run>/run_info.yaml` | Config, target timesteps, start time |
| `runs/<run>/training_summary.yaml` | **Total timesteps, total episodes, mean length** |
| `runs/<run>/checkpoints/progress.yaml` | Timesteps → episodes at each checkpoint |
| `runs/<run>/logs/` | TensorBoard logs (episode counts, rewards) |

---

## Key configs

- **config_v2_train_short.yaml** – 150k steps (30-60 min)
- **config_v2_train_short_pd.yaml** – 150k steps with PD control + sensors
- **config_v2_train_short_pd_walk.yaml** – 150k steps with anti-stall reward
- **config_v2_train_short_pd_imitation.yaml** – 150k steps with tripod imitation
- **config_v2.yaml** – 2M steps (several hours)
- **config_v2_continue.yaml** – Continue to 350k total

All use **tripod gait** (legs 1,4,5 vs 2,3,6).

---

## Episode tracking

- **TensorBoard:** `rollout/episodes_total`, `rollout/mean_episode_length`
- **Progress file:** `checkpoints/progress.yaml` shows episodes at 50k, 100k, etc.
- **Summary:** `training_summary.yaml` has final episode count

---

## PPO tuning (in config)

```yaml
ppo:
  learning_rate: 0.0003          # Lower to 0.0001 for stability
  n_steps: 2048                  # Increase to 4096 for more stable advantage
  gamma: 0.99                    # Increase to 0.995 for longer horizon
  policy_kwargs:
    net_arch:
      pi: [64, 64]               # Try [128, 128] for more capacity

training:
  total_timesteps: 150000        # Train longer (500k–2M)
  use_vec_normalize: false       # Set true for obs/reward normalization
```

---

## Optional features

```yaml
reward:
  near_termination_penalty_weight: 0.1   # Soft penalty near tilt/flexion limits
  near_termination_margin_deg: 10

env:
  obs_include_torso_height: true         # Add torso height to observation
  obs_include_torso_vel: true            # Add torso velocity
```

---

## Docs

- **README.md** – Quick start, workflow, training commands
- **PROJECT_STRUCTURE.md** – Detailed file/folder explanation
- **PREVIOUS_TRAINING_TAKEAWAYS.md** – Lessons from past training, PPO levers
