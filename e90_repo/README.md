# E90 Hexapod RL (sCPG + PPO)

**WALTER (project home):** [https://github.com/HowardWHSrun/WALTER](https://github.com/HowardWHSrun/WALTER)

Self-contained training and evaluation code for hexapod locomotion in MuJoCo, with **spiking central pattern generators (sCPG)** and **PPO** (Stable-Baselines3), plus **MLP** baselines and terrain / IMU variants.

## Where the trained models are (exact paths)

| What | Path on GitHub / in a clone |
|------|-----------------------------|
| **Trained policy weights** (Stable-Baselines3 checkpoints) | **`released_checkpoints/*.zip`** — these are the files you load with `PPO.load(...)` or `evaluate.py`. |
| **Per-model evaluation writeups** | **`released_checkpoints/evaluations/<name>/REPORT.md`** — short report + metrics interpretation. |
| **Per-model metrics (YAML)** | **`released_checkpoints/evaluations/<name>/evaluation_summary.yaml`** — machine-readable means, distances, per-episode table. |
| **Per-model videos (MP4)** | **`released_checkpoints/evaluations/<name>/videos/episode_*.mp4`** — screen captures from the same eval protocol. |
| **Training-run provenance** (timesteps, config path) | **`released_checkpoints/metadata/`** — copies of `run_info.yaml` / `training_summary.yaml` from the original training folders. |
| **Policy / env source code** (not weights) | **`models/`**, **`envs/`** |

**What you did in this project** (sim, training stack, comparisons, ablations) is summarized in **[docs/PROJECT.md](docs/PROJECT.md)**.

**Table of each released `.zip`**, matching config, and links to its report + videos: **[MODELS.md](MODELS.md)**.

To **re-record videos and refresh YAML summaries** after cloning:

```bash
./scripts/regenerate_evaluation_media.sh
```

---

## Teensy 4.1 / sim-to-real (deploy)

- **[docs/TEENSY_DEPLOY.md](docs/TEENSY_DEPLOY.md)** — control period vs sim, IMU observation layout, PD action semantics.
- **[teensy_export/README.md](teensy_export/README.md)** — Tier A (constants) vs Tier B (actor weights); how to run `scripts/export_ode_tripod_teensy.py`.
- **`scripts/compare_tier_policies_imu.py`** — ODE-tripod vs MLP parameter counts (and optional eval) on `configs/config_teensy_imu_compare.yaml`.

---

## Quick start

**Evaluate a released checkpoint** (after `pip install -r requirements.txt`):

```bash
python evaluate.py released_checkpoints/scpg_anti_lazy_1M_best.zip \
  --config configs/config_v7_anti_lazy.yaml --render
```

Other checkpoints and matching configs are listed in [MODELS.md](MODELS.md).

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
./
├── released_checkpoints/   # *.zip weights + evaluations/ (REPORT.md, videos, YAML) + metadata/
├── configs/                # Training / eval YAML
├── scripts/                # start_training, resume, regenerate_evaluation_media.sh, …
├── docs/                   # PROJECT.md (narrative of the E90 work)
├── runs/                   # Created locally when you train (not committed by default)
├── train.py
├── evaluate.py
├── envs/
├── models/
└── assets/
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

- **docs/PROJECT.md** – What this E90 project contains and what was built (sim, policies, experiments).
- **MODELS.md** – Each released checkpoint, config pairing, evaluation report link, and video links.
- **PREVIOUS_TRAINING_TAKEAWAYS.md** – Lessons from prior training (run organization, PPO levers, reward/obs/termination options, VecNormalize)
- **PROJECT_STRUCTURE.md** – Detailed explanation of folder layout and what each file does

---

## Manual training (without scripts)

```bash
python train.py --config configs/config_v2_train_short.yaml --version 2 --output-dir runs/my_run
```
