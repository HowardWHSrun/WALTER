# E90 project: what this repository is

This work is part of an **E90 engineering design** effort on **hexapod locomotion** in the **Valero Lab** context: build a MuJoCo simulation of the robot, train policies with **proximal policy optimization (PPO)**, and compare a **biologically inspired spiking central pattern generator (sCPG)** front-end against **standard MLP** actor–critic baselines.

## What you built (high level)

1. **Simulation stack**  
   - MuJoCo model under `assets/` (`hexapod.xml`, meshes, optional terrain MJCF).  
   - Custom Gymnasium environment `envs/hexapod_env.py`: PD torque control at the joint level, configurable observations (proprioception, phase, path error, IMU-style signals), rewards (forward motion, path following, smoothness, anti-idle / anti-lazy terms), and optional heightfield terrains.

2. **Policy architectures** (`models/`)  
   - **sCPG + PPO**: spiking CPG oscillators coupled to a value head and PPO training (`scpg.py`, `encoder.py`), including **tripod gait** (policy v2).  
   - **MLP baselines**: same PPO loop with `use_mlp_policy: true` in configs for fair comparisons.  
   - **ODE / phase-based gait** experiments (`models/ode_gait/`) for structured gaits.

3. **Training and experiment management**  
   - `train.py` + many YAML configs under `configs/` for ablations: PD vs torque, imitation mixing, velocity windows, terrain curriculum, IMU-only observations, “anti-lazy” termination and reward shaping, etc.  
   - Shell helpers in `scripts/` for starting runs, checking progress, and resuming from checkpoints.  
   - Design notes in `LAZY_AGENT_FIXES.md`, `PREVIOUS_TRAINING_TAKEAWAYS.md`, `PROJECT_STRUCTURE.md`, `QUICK_REFERENCE.md`.

4. **Released artifacts (this GitHub repo)**  
   - **Trained weights** (Stable-Baselines3 `.zip` checkpoints) live only under:

     **`released_checkpoints/*.zip`**

   - For each released checkpoint, this repo includes a **short evaluation report** and **MP4 screen captures** under:

     **`released_checkpoints/evaluations/<checkpoint_name>/`**

     See each folder’s `REPORT.md`, `evaluation_summary.yaml`, and `videos/`.

5. **Evaluation protocol**  
   - `evaluate.py` loads **one** checkpoint and runs episodes (optionally cycling **flat / rough / steps** when the env uses terrain). Metrics and optional videos are written to an output directory. The same protocol was used to generate the bundled reports and videos.

## Repository map (where things live)

| Path | Purpose |
|------|--------|
| `released_checkpoints/*.zip` | **Pre-trained PPO checkpoints** (the “models” in the ML sense). |
| `released_checkpoints/evaluations/` | **Per-checkpoint eval reports + videos** (`REPORT.md`, `evaluation_summary.yaml`, `videos/*.mp4`). |
| `released_checkpoints/metadata/` | Original `run_info.yaml` / `training_summary.yaml` snapshots from training. |
| `models/` | Policy network code (sCPG, encoders, ODE gait helpers). |
| `envs/` | MuJoCo hexapod Gymnasium environment. |
| `configs/` | Training/eval YAML (must match checkpoint when loading). |
| `train.py` / `evaluate.py` | Train and evaluate policies. |

## Regenerating evaluation reports and videos

After cloning and installing dependencies (`pip install -r requirements.txt`), run:

```bash
./scripts/regenerate_evaluation_media.sh
```

This re-runs `evaluate.py` for each released checkpoint with a fixed short protocol (see script). Results will overwrite the corresponding folders under `released_checkpoints/evaluations/`.

## Note on reward scales

Different YAML configs use **different reward combinations and scales**. **Do not compare raw reward numbers across checkpoints** unless they share the same reward definition. Prefer **distance and velocity metrics** in `evaluation_summary.yaml`, and use **videos** for qualitative gait and stability.
