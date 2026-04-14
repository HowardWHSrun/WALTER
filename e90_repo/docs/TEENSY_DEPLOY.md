# Teensy 4.1 deployment: control loop, sensors, and sim parity

This document ties the MuJoCo `HexapodEnv` to what you can run on a **Teensy 4.1** (no training on-device; inference and gait math only).

## 1. Control period (must match training)

Each RL **environment step** advances the simulator by:

`physics_dt_per_env_step = pd_steps_per_action * mj_model.opt.timestep`

Example from many configs: `pd_steps_per_action: 10` and default MuJoCo timestep `0.002` s → **0.02 s per env step** (~50 Hz commands).

On the Teensy:

- Run your **outer policy / gait update** at the **same** period you used in training (e.g. every 20 ms).
- Use `elapsedMillis()` / `micros()` for `t` and for `obs_include_sim_time` parity:  
  `sim_time` in sim is the env’s accumulated physics time; on hardware use **seconds since gait start** (or since reset), advanced by `physics_dt_per_env_step` each tick.

If you change `frame_skip`, `pd_steps_per_action`, or MJCF `timestep` in sim, **recompute and document** the new command rate here and in firmware.

## 2. Action semantics (PD mode)

When `use_pd_control: true` (typical for sim2real):

1. The policy outputs **normalized** actions in **[-1, 1]^12** (abduction/flexion pairs per leg, same joint order as `HexapodEnv`).
2. The env maps them to **target joint positions** (rad) using `action_scale_abd_deg`, `action_scale_flex_deg`, and optionally joint limits (`action_use_joint_limits`).
3. PD computes torques toward those targets.

On the robot:

- If servos are **position-controlled**, send **the same target angles** (after the same scaling and limits as in sim).
- If you only have PWM pulse widths, calibrate once: rad → pulse width per joint.
- **Do not** change scaling between sim training and deploy without retraining or retuning.

## 3. IMU observation parity (`observation_mode`)

MuJoCo sensors (see `assets/hexapod.xml`):

| Sensor name   | Type            | Frame / meaning |
|---------------|-----------------|-----------------|
| `torso_accel` | accelerometer   | Linear accel at `imu_site` (includes gravity in world representation; matches MuJoCo accelerometer semantics). |
| `torso_gyro`  | gyro            | Angular velocity at `imu_site`. |
| `torso_quat`  | framequat       | Orientation of `imu_site` as quaternion (w,x,y,z MuJoCo order). |

Env modes (subset):

| Mode          | Vector layout (before optional `obs_include_sim_time`) |
|---------------|--------------------------------------------------------|
| `imu6`        | `[accel(3), gyro(3)]` |
| `imu6_prev`   | `[accel(3), gyro(3), prev_action(12)]` |
| `imu_bno10`   | `[accel(3), gyro(3), quat(4)]` — BNO055-style 9-DoF fusion can still expose raw accel/gyro + fused quat; align units with sim. |

Optional flags:

- `obs_include_phase: true` (IMU modes): appends `[sin(2π f_phase t), cos(2π f_phase t)]` using env `obs_phase_frequency_hz` and the same `t` as sim time. On Teensy you can replicate with a **software oscillator** at `f_phase`.
- `obs_include_sim_time: true`: appends **one scalar**, **last index**: seconds (must match ODE-tripod policy expectation when `sim_time_in_obs` is true).

**Order on the wire:** Build the observation in the same order as `_get_obs` in `envs/hexapod_env.py` (accel, gyro, quat if any, prev_action if `imu6_prev`, phase sin/cos if enabled, then **sim_time last**).

For **Bosch BNO055** (I2C): map fused or raw readings into the same layout as `imu_bno10` or `imu6`; match **axis conventions** (may require a fixed rotation matrix in firmware).

## 4. Noise and domain randomization

Training may set `imu_noise_std_accel`, `imu_noise_std_gyro`, `imu_noise_std_quat`. On hardware, noise is physical; you do **not** add extra noise at deploy unless doing deliberate robustness tests.

## 5. Safety (firmware responsibility)

Independent of any learned policy:

- Joint limit clamping.
- Max slew rate on targets or torques.
- Watchdog / estop.

## 6. MuJoCo viewer: why the robot “just lays there”

`mujoco.viewer.launch_from_path('assets/hexapod.xml')` only **visualizes** the model; it does **not** drive the motors, so `ctrl` stays zero and the hexapod falls under gravity.

To **see motion**, run the open-loop tripod demo (same tripod map as `models/ode_gait/reference_from_phase.py`):

```bash
cd e90_repo
# macOS: use mjpython so the viewer can open (plain python raises RuntimeError).
mjpython scripts/view_hexapod_gait.py
# Linux / other: often `python scripts/view_hexapod_gait.py` is enough.
```

If it tips over, try a gentler gait, e.g. `mjpython scripts/view_hexapod_gait.py --freq 0.7 --amp-flex 0.5 --amp-abd 0.35`

By default the viewer **throttles to real time** so physics matches wall clock. For slow motion use `--realtime 0.5`; to stress-test uncapped stepping use `--realtime 0`.

The MJCF includes a **keyframe `stand`** so in Simulate you can reset to a standing pose instead of a collapsed heap.

## 7. PPO training and export (headers for Teensy)

Train with **ODETripodPolicy** (same as `configs/config_rl_mini_ode_imu.yaml`: IMU + phase + sim time, velocity reward), then export the **actor** to C headers (Tier B: small MLP weights in `ode_tripod_actor_weights.h`).

```bash
cd e90_repo
# One-shot (train + export); or call train.py / export_ode_tripod_teensy.py separately
chmod +x scripts/train_export_teensy.sh
./scripts/train_export_teensy.sh runs/my_teensy_run configs/config_rl_mini_ode_imu.yaml
```

Example run already produced on this machine: `runs/teensy_ppo_deploy_001/` with `checkpoints/best/best_model.zip` and `teensy_export/` (`obs_dim: 21`, `obs_core_dim: 20`, `sim_time_in_obs: true`). Copy those `.h` files into your Teensy project and implement the forward pass per `teensy_export/README.md` (your current `hexapod_walking_rl` sketch is open-loop phase only until you wire IMU + MLP).

## 8. Related repo artifacts

| Artifact | Purpose |
|----------|---------|
| `configs/config_teensy_imu_compare.yaml` | IMU-matched env + reward for comparing policies in sim. |
| `configs/config_rl_mini_ode_imu.yaml` | Short PPO run, ODE-tripod + IMU; good default for export. |
| `scripts/compare_tier_policies_imu.py` | Parameter counts and optional eval of ODE-tripod vs MLP checkpoints. |
| `scripts/export_ode_tripod_teensy.py` | **Tier A/B** export: hyperparameters + actor weights as C headers. |
| `scripts/train_export_teensy.sh` | Train + export in one step. |
| `teensy_export/README.md` | Export layout and Tier A vs Tier B notes. |
| `scripts/view_hexapod_gait.py` | Interactive viewer **with** tripod motor commands so the robot moves. |

## 9. Servo / protocol (fill in for your PCB)

Record your hardware-specific values below (not inferred by this repo):

- Bus: PWM / UART servo protocol / I2C, etc.
- Command rate: ___ Hz  
- Joint index → channel / ID mapping: ___

Keeping this section accurate is required for reproducible sim-to-real bring-up.
