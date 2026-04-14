# Evaluation report: `scpg_anti_lazy_1M_best`

| Item | Value |
|------|--------|
| **Weights file** | [`../../scpg_anti_lazy_1M_best.zip`](../../scpg_anti_lazy_1M_best.zip) (also: `released_checkpoints/scpg_anti_lazy_1M_best.zip` from repo root) |
| **Training config** | [`configs/config_v7_anti_lazy.yaml`](../../../configs/config_v7_anti_lazy.yaml) |
| **Policy** | sCPG + value MLP, tripod gait (v2) |
| **Training run** | `anti_lazy_v7_1M` — ~1M timesteps; this file is the **best-by-eval** checkpoint |

## How this report was produced

Fixed short eval for repository documentation (deterministic actions, **300 steps** per episode, **2** episodes, flat ground):

```bash
python evaluate.py released_checkpoints/scpg_anti_lazy_1M_best.zip \
  --config configs/config_v7_anti_lazy.yaml --no-plot --save-video -n 2 --max-steps 300 \
  -o released_checkpoints/evaluations/scpg_anti_lazy_1M_best
```

Machine-readable metrics: [`evaluation_summary.yaml`](evaluation_summary.yaml).

## Quantitative summary (from `evaluation_summary.yaml`)

| Metric | Mean (this run) |
|--------|------------------|
| Episodes | 2 |
| Episode length | 300 (capped) |
| Mean total reward per episode | **-735** (scale set by penalties / imitation / velocity terms in config — not comparable to MLP runs) |
| Mean forward +X displacement | **0.011 m** |
| Mean 2D path length | **0.012 m** |
| Mean forward velocity (step-wise) | **0.0015 m/s** |
| Max forward velocity (step-wise) | **0.16 m/s** |

## Videos

| File | Notes |
|------|--------|
| [`videos/episode_1.mp4`](videos/episode_1.mp4) | First episode, 300 steps |
| [`videos/episode_2.mp4`](videos/episode_2.mp4) | Second episode |

## Short interpretation

On this **short, flat** eval slice, net forward displacement is small and **reward is strongly negative** because this config uses large penalty weights (tilt, fall, idle, lateral velocity, etc.) without early termination on many failure modes — so the scalar reward is not a “0–1 success” score. The videos show whether the tripod CPG produces **rhythmic leg activity** and upright behavior over the horizon. For longer or task-matched evals, increase `--max-steps` and align terrain with training.
