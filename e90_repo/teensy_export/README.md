# Teensy export prototype (ODE tripod)

## Tier A vs Tier B (what this folder supports)

| Tier | Deploy artifact | This repo |
|------|-----------------|-----------|
| **A** | Open-loop or lightly scheduled gait using **constants** only (`ode_tripod_hyper.h`, `ode_tripod_nominal.h`) | Yes: hyperparameters and nominal mean action at `obs=0`, `t=0` as a **hint** for bring-up. Replace nominals with values you trust from sim or logging. |
| **B** | Tier A structure + **small actor MLP** on the Teensy (float32 or int8) | Yes: `ode_tripod_actor_weights.h` lists all Linear weights/biases and `log_std` (ignore `log_std` if you run deterministic mean-only). Implement ReLU + matmul in firmware to match PyTorch. |

**Tier B (quantized int8 / TFLite Micro)** is not auto-generated here: convert the same actor in PyTorch or ONNX with a separate quantization pipeline if flash/RAM requires it.

## Generate headers

From `e90_repo`:

```bash
python scripts/export_ode_tripod_teensy.py --checkpoint path/to/ppo_ode_tripod.zip --output-dir teensy_export
```

Optional: `--config configs/config_teensy_imu_compare.yaml` (reserved for future sanity checks; observation size is taken from the checkpoint).

Outputs:

- `ode_tripod_hyper.h` — frequency bounds, amplitude caps, signs, residual scale, observation dimensions.
- `ode_tripod_actor_weights.h` — `float` arrays for `_actor_core`, `_param_head`, `_residual_head`, `log_std`.
- `ode_tripod_nominal.h` — scalar hints and `ODE_TRIPOD_NOMINAL_ACTION_MEAN[12]` at zero observation.
- `export_meta.yaml` — small manifest.

## Firmware checklist

1. Match **control period** and observation layout to [docs/TEENSY_DEPLOY.md](../docs/TEENSY_DEPLOY.md).
2. Replicate `models/ode_gait/reference_from_phase.py` tripod indexing on the MCU (or call into shared C).
3. Forward pass order: `obs_core` → `_actor_core` (Linear+ReLU) → `_param_head` (4) and `_residual_head` (12); compute `theta = 2*pi*freq*t + phase_off`; `tripod_actions_from_phase` + `residual_scale * tanh(res)`; clip to [-1,1].
4. Keep **safety limits** outside the network path.

## Smoke test checkpoint

Untrained saves used for CI-style checks may be created with:

```bash
python -c "from pathlib import Path; import sys; sys.path.insert(0,'.'); \
from scripts.compare_tier_policies_imu import load_config, make_vec_env, fresh_ppo_ode; \
c=load_config('configs/config_teensy_imu_compare.yaml'); e=make_vec_env(c,0); m=fresh_ppo_ode(e,c,0); \
Path('teensy_export').mkdir(exist_ok=True); m.save('teensy_export/_smoke_ode_ppo.zip'); e.close()"
```

Then run `export_ode_tripod_teensy.py` on `_smoke_ode_ppo.zip`.
