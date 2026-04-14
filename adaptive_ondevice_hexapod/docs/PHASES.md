# Implementation roadmap

Four phases, each building on the previous. Every phase produces a working sketch you can flash and test before moving on.

## Phase 1: IMU filter + telemetry

**Goal:** Confirm the IMU works, the filter is stable, and roll/pitch/rates look correct while the robot walks with the unmodified v2 gait.

**Deliverables:**

- New sketch: `firmware/hexapod_imu_test/hexapod_imu_test.ino`
- Wire IMU to Teensy over I2C (SDA/SCL, 400 kHz). Pick a supported board (BNO055, MPU-6050, ICM-20948, etc.) and install its Arduino library.
- Implement complementary filter for roll and pitch at the IMU sample rate.
- Print `roll, pitch, roll_rate, pitch_rate` over USB Serial at ~50 Hz (human-readable CSV).
- **No gait changes.** The v2 tripod runs exactly as before; this sketch only adds IMU reading and printing alongside the walk.
- Verify: tilt the robot by hand while walking; confirm printed angles match physical tilt.

**Key code additions:**

```
#include <Wire.h>
// + IMU library header

float roll, pitch, roll_rate, pitch_rate;

void imuSetup()   { Wire.begin(); Wire.setClock(400000); /* init IMU registers */ }
void imuUpdate(float dt) { /* read accel+gyro, run complementary filter */ }
```

**Exit criterion:** roll and pitch are stable, responsive, and match physical tilts within a few degrees.

## Phase 2: Postural reflexes

**Goal:** The robot visibly levels itself when tilted or placed on a slope. Reflex gains are hard-coded (not learned yet).

**Deliverables:**

- New sketch: `firmware/hexapod_adaptive/hexapod_adaptive.ino` (extends v2 gait + IMU filter from Phase 1).
- `GaitState` struct: runtime-variable versions of `LIFT_FLEX`, `START_FLEX`, `PUSH_FLEX`, stride scale, step counts (see [ARCHITECTURE.md](ARCHITECTURE.md)).
- Reflex layer runs every fast tick:
  - **Behavior 1 (roll/pitch stabilization):** per-leg flex bias from `K_roll * roll` and `K_pitch * pitch`, distributed across left/right and front/rear legs.
  - **Behavior 2 (angular-velocity damping):** add `K_droll * roll_rate` and `K_dpitch * pitch_rate` to the correction.
  - **Behavior 4 (terrain leveling):** shift per-leg `start_flex` from pitch and roll so the chassis stays level on a slope.
- Safety clamp on every adjusted flex value (hard min/max, slew rate).
- Serial commands to tune `K_roll`, `K_pitch`, `K_droll`, `K_dpitch`, `K_level` live. Print current values with `p`.
- EEPROM save/load for reflex gains.

**Key test:** place the robot on a tilted board (~10--15 degrees). It should adjust leg heights to keep the body approximately level while walking.

**Exit criterion:** robot walks on a slope or uneven surface noticeably better than the unmodified v2 gait.

## Phase 3: Cautious mode + push recovery

**Goal:** The robot detects instability and responds: slower/shorter stride when terrain is rough, recovery reflex after a push.

**Deliverables:**

- Extend `firmware/hexapod_adaptive/` with:
  - **Stability score** `S` computed every fast tick (weighted sum of |roll|, |pitch|, rates). Windowed RMS `S_rms` computed every gait cycle.
  - **Behavior 3 (cautious mode):** if `S_rms > S_CAUTIOUS_THRESH` for `N` consecutive cycles, scale down stride, increase interp_delay, increase step height. Hysteresis: restore defaults only after `S_rms < S_NORMAL_THRESH` for `N` cycles.
  - **Behavior 9 (push recovery):** if angular velocity spikes above `PUSH_THRESH`, enter 2-cycle recovery mode (half stride, slower, lower body, more lift). Ramp back over 2 more cycles.
  - **Behavior 5 (step-height adaptation):** if pitch spikes increase during landing, raise step height for the next few steps.
  - **Fall detection:** if |roll| or |pitch| exceeds 45 degrees, freeze servos, print diagnostic, require Serial command to resume.
- Serial commands to tune thresholds (`sthresh`, `pthresh`, `ncycles`).
- Print stability score and current mode (`NORMAL`, `CAUTIOUS`, `RECOVERY`) over Serial.

**Key tests:**

1. Push the robot sideways while walking. It should briefly enter recovery mode and then resume.
2. Walk across a rough surface (books, foam). It should switch to cautious mode (shorter stride, slower).

**Exit criterion:** robot survives moderate pushes and rough terrain that would destabilize the baseline v2 gait.

## Phase 4: Online parameter learning

**Goal:** The robot autonomously improves its reflex gains and gait shape over minutes of walking, minimizing a stability cost.

**Deliverables:**

- Extend `firmware/hexapod_adaptive/` with:
  - **Adaptation parameter vector** `theta[5]`: `K_roll`, `K_pitch`, `stride_scale_base`, `step_height_scale_base`, `freq_scale`.
  - **Cost function** `J` evaluated per window of `M` gait cycles: `J = a * RMS(roll) + b * RMS(pitch) + c * mean_angular_velocity + d * fall_penalty`.
  - **SPSA update:** every `2*M` gait cycles, compare `J` for `theta + delta` vs `theta - delta`; step theta in the improving direction; clamp to safe bounds.
  - **EEPROM persistence:** save `theta_best` whenever a new minimum `J` is found. Load on boot.
  - Print `theta`, `J`, and `J_best` over Serial each update.
- Alternative (simpler): **epsilon-greedy bandit** over a coarse 5-D grid. Pick the best combo seen so far 90% of the time; explore 10%.

**Key tests:**

1. Place robot on flat floor. After 2--5 minutes of SPSA, `J` should decrease and walking should look smoother or faster.
2. Move robot to a slope. `J` initially spikes; over several minutes, theta adapts and `J` decreases again.
3. Power-cycle. Robot boots with `theta_best` from EEPROM and immediately walks better than factory defaults.

**Exit criterion:** measurable reduction in `J` over a 5--10 minute session on at least two different surfaces.

## Summary timeline

| Phase | Depends on | Estimated effort | Key deliverable |
|-------|-----------|------------------|-----------------|
| 1 | IMU wired | 1--2 sessions | Telemetry sketch |
| 2 | Phase 1 | 2--3 sessions | Postural reflexes |
| 3 | Phase 2 | 1--2 sessions | Cautious mode + recovery |
| 4 | Phase 3 | 2--3 sessions | Online learning |

Each phase is independently demonstrable. You can stop after any phase and still have a working, progressively more capable robot.

## The E90 story at each phase

- **After Phase 2:** "Our hexapod uses IMU-based postural reflexes to stabilize locomotion on uneven terrain."
- **After Phase 3:** "Our hexapod detects instability and reflexively adapts its gait, including push recovery."
- **After Phase 4:** "Our hexapod performs online, on-device learning to continuously improve its locomotion, inspired by biological sensorimotor adaptation."
