# Evaluation report: `mlp_clean_run005_best`

| Item | Value |
|------|--------|
| **Weights file** | [`../../mlp_clean_run005_best.zip`](../../mlp_clean_run005_best.zip) |
| **Training config** | [`configs/config_v7_mlp_clean.yaml`](../../../configs/config_v7_mlp_clean.yaml) |
| **Policy** | MLP actor–critic (256×256), PPO |
| **Training run** | `mlp_run_005` — 300k steps; **circular path** task with rich proprioception + phase + path features |

## How this report was produced

```bash
python evaluate.py released_checkpoints/mlp_clean_run005_best.zip \
  --config configs/config_v7_mlp_clean.yaml --no-plot --save-video -n 2 --max-steps 300 \
  -o released_checkpoints/evaluations/mlp_clean_run005_best
```

Metrics: [`evaluation_summary.yaml`](evaluation_summary.yaml).

## Quantitative summary

| Metric | Mean |
|--------|------|
| Episodes | 2 |
| Mean total reward | **6799** (path + smoothness terms — not comparable to sCPG anti-lazy scale) |
| Mean forward +X | **0.18 m** |
| Mean 2D path length | **0.27 m** |
| Mean forward velocity | **0.024 m/s** |
| Max forward velocity | **0.18 m/s** |

## Videos

- [`videos/episode_1.mp4`](videos/episode_1.mp4) — locomotion + path following  
- [`videos/episode_2.mp4`](videos/episode_2.mp4)

## Short interpretation

This is the clearest **flat + path** baseline in the release set: **meaningful forward motion and path length** within 300 steps, consistent with an MLP trained for **circle tracking** and imitation-style shaping. Use it as the main **non-sCPG** reference when explaining controller complexity vs performance in slides or reports.
