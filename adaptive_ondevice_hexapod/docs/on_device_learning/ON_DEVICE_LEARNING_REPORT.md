# On-Device Learning Report (Whiteboard Pipeline)

This report translates the whiteboard nodes into a practical pipeline for learning locomotion directly on the robot (Teensy + servo stack), with PPO-style adaptation and safety-first execution.

## 1) What the board is telling us

Interpreting the whiteboard nodes:

- `Clock(t)` is the master timing signal that drives gait phase.
- Periodic bases (`sin`, `square`, `sawtooth`) are candidate waveform primitives.
- `Amp` and `Phase` blocks parameterize each basis to generate leg trajectories.
- `P0, P1, P2, P3` are policy/controller parameter sets (or policy snapshots).
- `V1, V2, V3` and `T1, T2, T3` represent grouped environment/terrain spaces for training diversity.
- The `PPO` node indicates policy improvement from reward feedback.
- The predictor / optimization notes suggest a lightweight model-assisted tuning path (estimate and then optimize gains/shape parameters online).

Core idea: keep a deterministic clocked gait generator, and let learning tune waveform/gain parameters online, rather than learning raw servo outputs from scratch.

## 2) On-device learning architecture

### Runtime stack (fast + slow loops)

1. **Fast control loop (~100 Hz)**
   - Read IMU and compute state features (roll, pitch, angular rates).
   - Evaluate `Clock(t)` and gait bases.
   - Apply current policy parameters (amplitude/phase/gain scales).
   - Generate servo targets and send commands with safety clamps.

2. **Slow learning loop (per gait cycle or small window)**
   - Aggregate stability/performance metrics.
   - Compute reward.
   - Update policy parameters (PPO-style or bounded SPSA fallback).
   - Keep only bounded parameter changes and persist best set.

## 3) State, action, reward for on-device PPO

- **State (`s_t`)**
  - IMU: roll, pitch, roll rate, pitch rate
  - Gait phase from `Clock(t)`
  - Current parameter set id (`P0...Pn`) and recent stability score

- **Action (`a_t`)**
  - Small deltas to:
    - waveform amplitudes
    - phase offsets
    - stride scale / step-height scale
    - stabilization gains (`K_roll`, `K_pitch`, derivative terms)

- **Reward (`r_t`)**
  - Positive for low tilt/oscillation and sustained motion
  - Penalty for high angular rates, instability, or entering recovery mode
  - Hard negative on fall/safety-trigger

## 4) Training strategy from the node groups

- Treat `(V1, V2, V3) x (T1, T2, T3)` as curriculum domains.
- Start from easiest terrain and best-known parameter seed (`P0`).
- Run short rollouts in each domain and update to `P1 -> P2 -> P3`.
- Keep domain tags with each update so we can detect overfitting to one terrain.
- Deploy as **adaptive baseline + reflex layer**, not pure free-running RL output.

## 5) Safety and deployment guardrails

- Clamp all learned parameters to pre-set hard limits.
- Limit per-cycle parameter drift (slew/ramp constraints).
- Freeze or revert to last stable policy on fall detection.
- Keep a fallback static gait profile always available.
- Only write to EEPROM when a policy is consistently better over multiple windows.

## 6) Implementation plan (quick)

1. Refactor gait constants into runtime parameter struct.
2. Add clock/basis module (`Clock(t)`, sin/square/sawtooth primitives).
3. Add policy head that outputs bounded amp/phase/gain deltas.
4. Add reward logger and domain labels (`V*`, `T*`).
5. Add slow-loop optimizer (PPO-lite; SPSA fallback if compute/memory becomes tight).
6. Add persistence and rollback logic.

## 7) Immediate deliverables

- `policy_params.h/.cpp`: parameter bounds, serialization, rollback
- `gait_clock.h/.cpp`: phase and waveform generator
- `learning_loop.h/.cpp`: rollout accounting, reward, updates
- `terrain_profile.h`: `V*` and `T*` profile definitions
- `on_device_learning_log.csv`: cycle-level metrics for offline analysis

## Open questions for you

1. Should we implement full PPO on-device first, or start with SPSA/bandit updates and then migrate to PPO?
2. Do you want `V1..V3` to mean speed regimes, voltage regimes, or separate robot conditions?
3. For `T1..T3`, what exact terrain set should we lock in (flat foam, incline, uneven blocks, etc.)?
4. Should this report be mirrored into `PHASES.md` as a formal milestone plan?
