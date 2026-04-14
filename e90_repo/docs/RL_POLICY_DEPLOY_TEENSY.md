# Deploying the ODE-tripod PPO policy on Teensy 4.1

This is the **end-to-end checklist** for running the same **actor** as `models/ode_gait/ode_tripod_policy.py` on hardware. Training stays on the PC; the Teensy runs **inference only** (no critic, no PPO).

## 1. What you export from the PC

After training an `ODETripodPolicy` checkpoint:

```bash
cd e90_repo
python scripts/export_ode_tripod_teensy.py \
  --checkpoint runs/<your_run>/checkpoints/best/best_model.zip \
  --output-dir <out>/teensy_export \
  --config configs/config_rl_mini_ode_imu.yaml
```

Copy into your Teensy project:

- `ode_tripod_hyper.h` — `freq_hz` bounds, `max_amp_*`, signs, `residual_scale`, `ODE_TRIPOD_OBS_DIM`, etc.
- `ode_tripod_actor_weights.h` — float arrays for `_actor_core` (two `Linear`+`ReLU` blocks), `_param_head` (4 outputs), `_residual_head` (12 outputs). **Ignore** `log_std` if you run **deterministic** control (recommended on hardware).

You do **not** need the value network on the MCU.

## 2. Observation vector (must match training)

For `configs/config_rl_mini_ode_imu.yaml`, the env uses `observation_mode: imu6_prev` with phase + sim time. The layout is **exactly** (21 floats, **last index is time**):

| Index | Content |
|-------|---------|
| 0–2 | Accelerometer (3), same units and frame as MuJoCo sensors `torso_accel` in `assets/hexapod.xml` |
| 3–5 | Gyro (3), same as `torso_gyro` |
| 6–17 | Previous **normalized** action `[-1, 1]` (12), same order as env actuators / sim leg order |
| 18–19 | `sin(2π f_phase t)`, `cos(2π f_phase t)` with `f_phase = obs_phase_frequency_hz` (0.45 Hz in that config) |
| 20 | **Sim time** `t` in **seconds** (monotonic since reset; advances each control step by your deployed `dt`) |

The policy splits: **`obs_core` = indices 0–19**, **`sim_time` = index 20**.

**Axis convention:** If your IMU board uses different axes than the MJCF, apply a **fixed rotation** in firmware and verify against sim (same gravity direction when level).

## 3. Forward pass (same math as PyTorch)

Let `core_dim = ODE_TRIPOD_OBS_CORE_DIM` (20).

1. **MLP on `obs_core`:**  
   `h0 = ReLU(W0 * obs_core + b0)`, `h1 = ReLU(W1 * h0 + b1)` (shapes match exported arrays and `hidden_sizes`, typically 64→64).

2. **Parameter head:**  
   `raw = Wp * h1 + bp` (4 values).  
   - `freq_hz = f_min + sigmoid(raw[0]) * (f_max - f_min)`  
   - `amp_flex = sigmoid(raw[1]) * max_amp_flex`  
   - `amp_abd = sigmoid(raw[2]) * max_amp_abd`  
   - `phase_off = tanh(raw[3]) * π`

3. **Phase:**  
   `theta_a = 2π * freq_hz * sim_time + phase_off`  
   (use the **scalar** `sim_time` from obs index 20, not a separate timer, unless they are identical.)

4. **Residual:**  
   `res = residual_scale * tanh(Wr * h1 + br)` (12 values).

5. **Tripod reference:**  
   Implement `tripod_actions_from_phase(theta_a, amp_flex, amp_abd, flex_sign, abd_sign)` identically to `models/ode_gait/reference_from_phase.py` / your `rl_phase_gait.h` (same leg indexing as sim).

6. **Mean action:**  
   `mean = clip(reference + res, -1, 1)`.

7. **Stochastic policy (optional):**  
   Training used `Normal(mean, exp(log_std))`. On hardware, use **`mean` only** (deterministic) unless you have a reason to sample.

## 4. Control period and time base

Match training:

- Env step time ≈ `pd_steps_per_action * mj_timestep` (e.g. 10 × 0.002 s = **0.02 s** per RL step in many configs).
- Advance **`sim_time`** by that same `dt` each time you compute a new observation and action.
- Run the policy at that rate (or retrain with a different `dt` and document it).

## 5. From normalized action to servos

Training used **`use_pd_control: true`**: actions are **targets in radians** via `action * action_scale` (degrees then rad in env — see `HexapodEnv._action_to_target_positions`).

On the robot you must either:

- **A.** Replicate the same scaling to **target angles**, then your servo/PWM mapping from radians, or  
- **B.** Calibrate a one-time map from **normalized [-1,1]** to your existing **joint command / µs** space (less ideal unless you verify against sim).

Do **not** silently mix the old **piecewise flex-only** Maestro map with RL actions unless you prove equivalence to the sim PD targets.

## 6. Closing the `prev_action` loop

Store the **12 floats you actually send** (or the clipped `mean` before noise) as the next observation’s indices 6–17. First step after reset: use **zeros** or a neutral pose, consistent with the env.

## 7. Safety

Independent of the network:

- Clamp joint targets / PWM to mechanical limits.
- Limit slew rate of targets.
- Watchdog / estop.

## 8. Minimal bring-up without IMU (debug only)

The policy **expects** 21-dim observations trained with IMU. Feeding zeros or fake IMU will not match training. For structure testing only, you can run the **open-loop** path: fix `obs_core` to constants, still advance `sim_time`, and confirm the MLP + tripod produce motion — then add the real IMU.

## 9. Firmware structure (suggested)

- **Setup:** I2C/SPI IMU, Maestro serial, `millis()`/`micros()` for `dt`.
- **Loop each control tick:** read IMU → build 21-dim `obs` → forward pass → normalized action → PD/PWM → save `prev_action` → command servos.

This repo’s `hexapod_walking_rl` sketch is **phase-only** (no MLP). Deploying the RL policy means **adding** the steps above (or a new sketch) alongside your existing Maestro layer.
