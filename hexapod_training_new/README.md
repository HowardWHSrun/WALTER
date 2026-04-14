# Hexapod sCPG PPO Training

**WALTER (project home):** [https://github.com/HowardWHSrun/WALTER](https://github.com/HowardWHSrun/WALTER)

Self-contained training environment for hexapod locomotion with spiking CPG and PPO.

---

## Quick start

**Start new training:**
```bash
./scripts/start_training.sh
```

**Check progress:**
```bash
./scripts/check_progress.sh
```

**Resume from checkpoint:**
```bash
./scripts/resume_training.sh run_20260203_170424
```

---

## Project structure

```
hexapod_training_new/
├── configs/              # All training configs (v2_train_short, v2, continue, etc.)
├── scripts/              # Helper scripts (start_training, check_progress, resume)
├── runs/                 # Training outputs (one folder per run)
│   └── run_<timestamp>/
│       ├── run_info.yaml           # Run plan (config, timesteps, start time)
│       ├── training_summary.yaml   # Final timesteps, episodes, mean length
│       ├── checkpoints/
│       │   ├── progress.yaml       # Timesteps → episodes mapping
│       │   └── *.zip               # Model checkpoints
│       └── logs/                   # TensorBoard logs
├── train.py              # Main training script
├── evaluate.py           # Evaluation script
├── envs/                 # MuJoCo environment
├── models/               # sCPG policy implementation
└── assets/               # Hexapod MJCF model

```

See **PROJECT_STRUCTURE.md** for details.

---

## Training workflow

### 1. Start a short run (150k steps)
```bash
./scripts/start_training.sh config_v2_train_short.yaml my_first_run
```

### 1b. Start a short run with PD control + sensors
```bash
./scripts/start_training.sh config_v2_train_short_pd.yaml my_first_pd_run
```

### 1c. Start a short run with anti-stall reward (recommended)
```bash
./scripts/start_training.sh config_v2_train_short_pd_walk.yaml my_walk_run
```

### 1d. Start a short run with imitation (recommended for continuous gait)
```bash
./scripts/start_training.sh config_v2_train_short_pd_imitation.yaml my_imitation_run
```

### 2. Monitor progress
```bash
# Check latest run
./scripts/check_progress.sh

# Or specify run
./scripts/check_progress.sh my_first_run

# TensorBoard
tensorboard --logdir runs/my_first_run/logs
```

### 3. Resume/continue training
```bash
# Continue with lower learning rate
./scripts/resume_training.sh my_first_run config_v2_continue my_continued_run
```

---

## What you get from each run

Every run folder contains:

| File | Content |
|------|---------|
| `run_info.yaml` | Config used, target timesteps, start time |
| `training_summary.yaml` | **Total timesteps, total episodes, mean episode length** |
| `checkpoints/progress.yaml` | Timesteps and episodes at each checkpoint |
| `checkpoints/*.zip` | Model checkpoints every 50k steps |
| `checkpoints/best/` | Best model by eval reward |
| `logs/` | TensorBoard logs (includes `rollout/episodes_total`) |

You always know **how many episodes** each run had and how that maps to checkpoints.

---

## Key configs

| Config | Steps | Use for |
|--------|-------|---------|
| `config_v2_train_short.yaml` | 150k | Quick test / first run |
| `config_v2.yaml` | 2M | Full training |
| `config_v2_continue.yaml` | 350k total | Continue a 150k run |

All use tripod gait (version 2): legs 1,4,5 in phase, 2,3,6 anti-phase.

---

## Documentation

- **PREVIOUS_TRAINING_TAKEAWAYS.md** – Lessons from prior training (run organization, PPO levers, reward/obs/termination options, VecNormalize)
- **PROJECT_STRUCTURE.md** – Detailed explanation of folder layout and what each file does

---

## Manual training (without scripts)

```bash
python train.py --config configs/config_v2_train_short.yaml --version 2 --output-dir runs/my_run
```
