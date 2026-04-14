# Evaluation report: `scpg_anti_lazy_1M_final`

| Item | Value |
|------|--------|
| **Weights file** | [`../../scpg_anti_lazy_1M_final.zip`](../../scpg_anti_lazy_1M_final.zip) |
| **Training config** | [`configs/config_v7_anti_lazy.yaml`](../../../configs/config_v7_anti_lazy.yaml) |
| **Policy** | sCPG + value MLP, tripod gait (v2) |
| **Training run** | Same as `scpg_anti_lazy_1M_best`, but this is the **final** weights at the end of training (~1M steps), not the best-eval snapshot |

## How this report was produced

```bash
python evaluate.py released_checkpoints/scpg_anti_lazy_1M_final.zip \
  --config configs/config_v7_anti_lazy.yaml --no-plot --save-video -n 2 --max-steps 300 \
  -o released_checkpoints/evaluations/scpg_anti_lazy_1M_final
```

Metrics: [`evaluation_summary.yaml`](evaluation_summary.yaml).

## Quantitative summary

| Metric | Mean |
|--------|------|
| Episodes | 2 |
| Mean total reward | **-735** |
| Mean forward +X | **0.011 m** |
| Mean 2D path length | **0.012 m** |
| Mean forward velocity | **0.0015 m/s** |
| Max forward velocity | **0.16 m/s** |

## Videos

- [`videos/episode_1.mp4`](videos/episode_1.mp4)  
- [`videos/episode_2.mp4`](videos/episode_2.mp4)

## Short interpretation

Behavior on this protocol is **very close** to `scpg_anti_lazy_1M_best` (same config, same horizon). Use **best** vs **final** when you care about **eval checkpoint selection** vs **last iterate** under PPO. Compare videos side-by-side for subtle gait or stability differences.
