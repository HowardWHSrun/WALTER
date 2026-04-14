# Hexapod Brain Roadmap (Whiteboard + Current Driving Code)

This version ties the professor's whiteboard plan directly to your working controller in `adaptive_ondevice_hexapod/firmware/hexapod_walking_v2/hexapod_walking_v2.ino`, so the roadmap is realistic for your current stack.

## 1) Current Ground Truth (What Already Works)

Your current firmware is already a solid "brain v0":

- 12-DOF hexapod with explicit channel map and per-servo direction calibration
- Maestro pulse control with hard clamp (`MIN_US`/`MAX_US`)
- Tripod gait as a deterministic state machine:
  - `tripodA_lift -> tripodA_swingB_push -> tripodA_plant`
  - `tripodB_lift -> tripodB_swingA_push -> tripodB_plant`
- Clear gait parameters already exposed as constants:
  - geometry: `FRONT/MID/REAR_*_ABD`
  - body posture: `START_FLEX`, `PUSH_FLEX`, `LIFT_FLEX`, `PLANT_FLEX`
  - timing: `INTERP_DELAY`, `LIFT/SWING/PLANT/PUSH/MID_STEPS`
- Smooth transition layer (`moveSmoothTo`) that already acts like a low-level trajectory executor

This means you do not need to "invent a brain from scratch"; you need to make this parameterized, state-aware, and adaptive.

## 2) Whiteboard Concepts Mapped to Existing Firmware

- **TA/TB clock idea** -> your existing A/B tripod phase sequence
- **P0/P1/P2/P3 control points** -> your per-phase pose targets in `buildPose(...)`
- **Waveform thinking (sin/square/saw)** -> optional next step; keep keyframe tripod first, then add waveform mode as a second generator
- **PPO block** -> output bounded updates to gait parameters, not raw servo pulses
- **Predictor / EOM notes** -> short-horizon stability predictor on top of IMU signals, used for reward/safety

## 3) Practical Target Architecture

Keep the existing deterministic gait engine as the safety backbone, then layer adaptation on top:

1. **Execution Layer (already present)**
   - Maestro writing, clamping, interpolation
2. **Parameterized Gait Layer (next)**
   - Convert fixed `const int` gait knobs into runtime state struct
3. **Fast Reflex Layer (next)**
   - IMU-driven corrections at each interpolation tick
4. **Slow Adaptation Layer (next)**
   - Update baseline gait knobs once per gait cycle/window
5. **Policy Layer (later)**
   - PPO chooses bounded parameter deltas in sim, then deploy to robot safely
6. **Predictor Layer (optional/late)**
   - Stability-risk model for reward shaping and early warning

## 4) Phase Plan Anchored to Your Repo

## Phase 0 - Freeze and Instrument Baseline (2-3 days)

- Keep `hexapod_walking_v2.ino` behavior unchanged
- Add telemetry output (phase, key gait params, optional timing stats)
- Define benchmark protocol: flat terrain, fixed battery level, fixed test duration

**Deliverable:** baseline logs and repeatability metrics

## Phase 1 - Refactor Constants into Runtime State (Week 1)

- Replace hard-coded gait constants with a `GaitState` runtime struct
- Keep defaults equal to current v2 values
- Route all phase builders through that struct
- Add per-cycle bounded parameter update function

**Deliverable:** `hexapod_walking_v3.ino` behavior-matched to v2 but runtime-tunable

## Phase 2 - IMU Reflexes on Fast Tick (Week 2)

- Integrate IMU read + filter + stability score in firmware loop
- Add roll/pitch and roll_rate/pitch_rate reflex corrections
- Apply corrections as small flex/stride adjustments with strict clamps

**Deliverable:** improved disturbance rejection without changing core gait pattern

## Phase 3 - Slow Adaptation Before RL (Weeks 3-4)

- Add cycle-level adaptation (SPSA or coarse bandit) for a small parameter vector
- Candidate learned knobs:
  - stride scale
  - step height scale
  - gait speed scaling (`INTERP_DELAY` / step counts)
  - roll/pitch reflex gains
- Save best parameters (EEPROM) and restore on boot

**Deliverable:** on-robot self-tuning that remains inside safe bounds

## Phase 4 - PPO in Simulation, Safe Transfer to Firmware (Weeks 4-6)

- Train PPO in `sim/` with domain randomization
- Policy outputs only bounded high-level parameters (not servo-level commands)
- Validate against deterministic baseline and reflex-only baseline
- Deploy policy output mapping to firmware as:
  - table lookup, or
  - tiny inference wrapper on host->firmware command path

**Deliverable:** PPO-assisted adaptation that beats baseline on defined terrains/speeds

## Phase 5 - Predictor-Assisted Safety/Reward (Weeks 6-7, optional)

- Build short-horizon instability predictor from IMU + gait phase history
- Use for reward shaping and "cautious mode" trigger

**Deliverable:** fewer falls and faster policy convergence

## 5) Immediate 7-Day Task List (Adjusted)

1. Refactor `hexapod_walking_v2.ino` constants into `GaitState` (no behavior change)
2. Add telemetry schema for:
   - phase id
   - gait parameters
   - roll/pitch and rates (when IMU added)
   - clamp/safety events
3. Define safety bounds and per-cycle max deltas for every tunable parameter
4. Implement a simple "cautious mode" gate from stability score threshold
5. Build one repeatable test script/protocol in `docs/` for before/after comparisons

## 6) Where Each Piece Should Live

- `firmware/hexapod_walking_v2/`  
  - keep as frozen reference
- `firmware/hexapod_walking_v3/`  
  - runtime-parameterized controller, reflexes, safety, adaptation hooks
- `sim/`  
  - PPO training and ablation studies
- `host/`  
  - experiment runner, logging, parameter push/pull, replay tools
- `docs/`  
  - benchmark protocol, milestone criteria, tuning notes

## 7) Milestone Criteria (Professor-Facing)

- **M1:** v3 reproduces v2 gait behavior within tolerance (same stride timing and no new instability)
- **M2:** IMU reflexes reduce tip events versus v2 baseline under push/disturbance test
- **M3:** slow adaptation improves stability score over multi-cycle runs
- **M4:** PPO policy in sim outperforms reflex-only baseline across terrain/speed conditions
- **M5:** safe real-robot demo with bounded adaptation and recovery behavior

## 8) Key Scope Decisions To Confirm

- PPO output type: direct parameter set vs residual from known-safe baseline
- Terrain set for grading/demo (exact surfaces and disturbance cases)
- Priority metric: stability first, speed first, or weighted composite
- Whether predictor is required for this semester milestone or stretch goal
