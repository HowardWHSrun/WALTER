# Evaluation report: `pd_velocity_window_run018_best`

| Item | Value |
|------|--------|
| **Weights file** | [`../../pd_velocity_window_run018_best.zip`](../../pd_velocity_window_run018_best.zip) |
| **Training config** | [`configs/config_v6_train_short_pd_ppo_velocity_window.yaml`](../../../configs/config_v6_train_short_pd_ppo_velocity_window.yaml) |
| **Policy** | sCPG + value MLP, tripod (v2), **PD** low-level control |
| **Training** | 300k steps; reward uses **velocity window** smoothing |

## How this report was produced

```bash
python evaluate.py released_checkpoints/pd_velocity_window_run018_best.zip \
  --config configs/config_v6_train_short_pd_ppo_velocity_window.yaml --no-plot --save-video -n 2 --max-steps 300 \
  -o released_checkpoints/evaluations/pd_velocity_window_run018_best
```

Metrics: [`evaluation_summary.yaml`](evaluation_summary.yaml).

## Quantitative summary

| Metric | Mean |
|--------|------|
| Episodes | 2 |
| Mean total reward | **~0.8** (this config’s reward is on a **much smaller numeric scale** than MLP path configs) |
| Mean forward +X | **-0.002 m** |
| Mean 2D path length | **0.015 m** |
| Mean forward velocity | **-0.0003 m/s** |
| Mean action energy proxy | **2.86** |

## Videos

- [`videos/episode_1.mp4`](videos/episode_1.mp4)  
- [`videos/episode_2.mp4`](videos/episode_2.mp4)

## Short interpretation

**Do not compare raw reward** to `mlp_clean_run005_best` or terrain runs. This checkpoint is included to document an **sCPG + PD + velocity-window** experiment. On this short eval, net displacement is small; the MP4s show whether the legs remain **coordinated** under PD. Treat this as a **qualitative / archival** artifact unless you re-run longer evals or continue training.
