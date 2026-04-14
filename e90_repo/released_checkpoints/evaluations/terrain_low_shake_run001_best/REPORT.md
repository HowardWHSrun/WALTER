# Evaluation report: `terrain_low_shake_run001_best`

| Item | Value |
|------|--------|
| **Weights file** | [`../../terrain_low_shake_run001_best.zip`](../../terrain_low_shake_run001_best.zip) |
| **Training config** | [`configs/config_terrain_low_shake.yaml`](../../../configs/config_terrain_low_shake.yaml) |
| **Policy** | MLP (256×256), PPO |
| **Environment** | Heightfield terrains with **low shake** / smoother action regularization vs stiffer terrain configs |

## How this report was produced

Three episodes, one each on **flat**, **rough**, **steps** (eval protocol cycles terrains):

```bash
python evaluate.py released_checkpoints/terrain_low_shake_run001_best.zip \
  --config configs/config_terrain_low_shake.yaml --no-plot --save-video -n 3 --max-steps 300 \
  -o released_checkpoints/evaluations/terrain_low_shake_run001_best
```

Metrics: [`evaluation_summary.yaml`](evaluation_summary.yaml).

## Quantitative summary

| Terrain (1 ep each) | Mean reward | Forward +X | 2D path length |
|---------------------|------------|------------|----------------|
| **flat** | 7740 | **0.47 m** | **0.51 m** |
| **rough** | 7309 | **0.47 m** | **0.52 m** |
| **steps** | 6015 | **0.055 m** | **0.17 m** |

Overall mean over 3 episodes: reward **7022**, mean forward +X **0.33 m** (dominated by flat/rough episodes).

## Videos

| File | Terrain |
|------|---------|
| [`videos/episode_1.mp4`](videos/episode_1.mp4) | **Flat** |
| [`videos/episode_2.mp4`](videos/episode_2.mp4) | **Rough** heightfield |
| [`videos/episode_3.mp4`](videos/episode_3.mp4) | **Steps** |

## Short interpretation

The policy **generalizes well from flat to rough** on this 300-step protocol (similar forward distance). **Steps** are harder: forward progress and path length drop. The videos are the primary artifact for demonstrating **terrain-conditioned behavior** with one fixed checkpoint.
