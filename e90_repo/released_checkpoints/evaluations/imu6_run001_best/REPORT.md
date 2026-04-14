# Evaluation report: `imu6_run001_best`

| Item | Value |
|------|--------|
| **Weights file** | [`../../imu6_run001_best.zip`](../../imu6_run001_best.zip) |
| **Training config** | [`configs/config_imu6.yaml`](../../../configs/config_imu6.yaml) |
| **Policy** | MLP (256×256), PPO |
| **Observations** | **IMU-style** 6D (no direct joint angles in obs); noisy accel / gyro in sim |

## How this report was produced

```bash
python evaluate.py released_checkpoints/imu6_run001_best.zip \
  --config configs/config_imu6.yaml --no-plot --save-video -n 2 --max-steps 300 \
  -o released_checkpoints/evaluations/imu6_run001_best
```

Metrics: [`evaluation_summary.yaml`](evaluation_summary.yaml).

## Quantitative summary

| Metric | Mean |
|--------|------|
| Episodes | 2 |
| Mean total reward | **4533** |
| Mean forward +X | **0.0025 m** |
| Mean 2D path length | **0.013 m** |
| Mean forward velocity | **0.0003 m/s** |

## Videos

- [`videos/episode_1.mp4`](videos/episode_1.mp4)  
- [`videos/episode_2.mp4`](videos/episode_2.mp4)

## Short interpretation

High **episode return** can coexist with **small net displacement** here because the reward stack emphasizes penalties and shaping that do not reduce to “meters traveled.” The scientific point of this checkpoint is **partial observability** (IMU only): the policy must infer body motion from noisy inertial data. Watch the videos for stability and stepping despite limited observability; for stronger forward motion, train longer or tune `config_imu6.yaml` / related IMU configs documented in the repo.
