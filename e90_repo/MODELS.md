# Released checkpoints (E90 hexapod RL)

**Trained weights** are only the `.zip` files under **`released_checkpoints/`** at the repository root. They are **Stable-Baselines3 PPO** checkpoints (policy + value + optimizer + optional **VecNormalize**). You must pair each checkpoint with the **same YAML** it was trained with.

---

## Index: weights, report, videos

For every released model there is a folder **`released_checkpoints/evaluations/<same_base_name>/`** containing:

- **`REPORT.md`** — short evaluation narrative and a table of key metrics  
- **`evaluation_summary.yaml`** — full numeric summary from `evaluate.py`  
- **`videos/episode_*.mp4`** — screen captures from that eval run  

| Weights (load this) | Config (required) | Evaluation folder |
|---------------------|-------------------|---------------------|
| [`released_checkpoints/scpg_anti_lazy_1M_best.zip`](released_checkpoints/scpg_anti_lazy_1M_best.zip) | [`configs/config_v7_anti_lazy.yaml`](configs/config_v7_anti_lazy.yaml) | [`evaluations/scpg_anti_lazy_1M_best/`](released_checkpoints/evaluations/scpg_anti_lazy_1M_best/) |
| [`released_checkpoints/scpg_anti_lazy_1M_final.zip`](released_checkpoints/scpg_anti_lazy_1M_final.zip) | [`configs/config_v7_anti_lazy.yaml`](configs/config_v7_anti_lazy.yaml) | [`evaluations/scpg_anti_lazy_1M_final/`](released_checkpoints/evaluations/scpg_anti_lazy_1M_final/) |
| [`released_checkpoints/mlp_clean_run005_best.zip`](released_checkpoints/mlp_clean_run005_best.zip) | [`configs/config_v7_mlp_clean.yaml`](configs/config_v7_mlp_clean.yaml) | [`evaluations/mlp_clean_run005_best/`](released_checkpoints/evaluations/mlp_clean_run005_best/) |
| [`released_checkpoints/imu6_run001_best.zip`](released_checkpoints/imu6_run001_best.zip) | [`configs/config_imu6.yaml`](configs/config_imu6.yaml) | [`evaluations/imu6_run001_best/`](released_checkpoints/evaluations/imu6_run001_best/) |
| [`released_checkpoints/terrain_low_shake_run001_best.zip`](released_checkpoints/terrain_low_shake_run001_best.zip) | [`configs/config_terrain_low_shake.yaml`](configs/config_terrain_low_shake.yaml) | [`evaluations/terrain_low_shake_run001_best/`](released_checkpoints/evaluations/terrain_low_shake_run001_best/) |
| [`released_checkpoints/pd_velocity_window_run018_best.zip`](released_checkpoints/pd_velocity_window_run018_best.zip) | [`configs/config_v6_train_short_pd_ppo_velocity_window.yaml`](configs/config_v6_train_short_pd_ppo_velocity_window.yaml) | [`evaluations/pd_velocity_window_run018_best/`](released_checkpoints/evaluations/pd_velocity_window_run018_best/) |

**Direct links to reports:**  
[scpg anti-lazy best](released_checkpoints/evaluations/scpg_anti_lazy_1M_best/REPORT.md) · [scpg anti-lazy final](released_checkpoints/evaluations/scpg_anti_lazy_1M_final/REPORT.md) · [MLP clean run005](released_checkpoints/evaluations/mlp_clean_run005_best/REPORT.md) · [IMU6](released_checkpoints/evaluations/imu6_run001_best/REPORT.md) · [terrain low shake](released_checkpoints/evaluations/terrain_low_shake_run001_best/REPORT.md) · [PD velocity window](released_checkpoints/evaluations/pd_velocity_window_run018_best/REPORT.md)

---

## Summary table (what each model is)

| File | Policy | Gait / version | Training (approx.) | Role |
|------|--------|----------------|-------------------|------|
| `scpg_anti_lazy_1M_best.zip` | sCPG + value MLP | Tripod (v2) | ~1.0M steps | Primary **sCPG** run; **best** eval checkpoint. Anti-lazy reward/termination design. |
| `scpg_anti_lazy_1M_final.zip` | sCPG + value MLP | Tripod (v2) | ~1.0M steps | Same run; **final** timestep weights vs **best**. |
| `mlp_clean_run005_best.zip` | MLP (256×2) | Path (v1) | 300k | **MLP baseline** on circular path; strong motion on short eval. |
| `imu6_run001_best.zip` | MLP (256×2) | — (v1) | 300k, 2 envs | **IMU-only** observations. |
| `terrain_low_shake_run001_best.zip` | MLP (256×2) | — (v1) | 300k, 2 envs | **Terrain** + smoother regularization; eval videos on flat / rough / steps. |
| `pd_velocity_window_run018_best.zip` | sCPG + value MLP | Tripod (v2) | 300k | **sCPG + PD** with velocity-window reward (different reward scale). |

---

## Quick evaluation (reproduce or extend)

```bash
python evaluate.py released_checkpoints/<CHECKPOINT>.zip --config configs/<CONFIG>.yaml --render
```

Refresh all bundled reports and MP4s (same protocol as in `REPORT.md` files):

```bash
./scripts/regenerate_evaluation_media.sh
```

---

## Architecture notes

- **sCPG**: `models/scpg.py`, `models/encoder.py`; tripod gait uses **v2** policy class in training/eval.  
- **MLP**: `ppo.use_mlp_policy: true` in the corresponding configs.

---

## Provenance

[`released_checkpoints/metadata/`](released_checkpoints/metadata/) retains `run_info.yaml` and `training_summary.yaml` from the original training directories.

---

## Project context

See **[docs/PROJECT.md](docs/PROJECT.md)** for a concise description of **what was built** for E90 (simulation, policies, experiments). Host repo: [github.com/HowardWHSrun/E90](https://github.com/HowardWHSrun/E90).

---

## Reward scale warning

Configs differ in **reward terms and scaling**. **Do not rank policies by raw `mean_reward` across rows** unless the reward definitions match. Prefer **distance / velocity** fields in `evaluation_summary.yaml` and **videos** for behavior.
