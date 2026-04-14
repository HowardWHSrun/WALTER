# Adaptive on-device learning (hexapod)

**WALTER — canonical repository:** [https://github.com/HowardWHSrun/WALTER](https://github.com/HowardWHSrun/WALTER)  
*(Walking Alternating Tripod for Evolutionary Research. Consolidate firmware, docs, and CAD here for public release.)*

Bio-inspired hexapod that uses **IMU body-state feedback** to reflexively stabilize locomotion and **gradually adapt its gait** on-device, running entirely on a Teensy 4.1.

## Approach

Three layers, inspired by insect locomotion architecture:

1. **Innate rhythm** -- structured tripod gait (v2 lift/swing/plant keyframes).
2. **Sensory corrections** -- IMU-driven postural reflexes adjust leg targets in real time (roll/pitch stabilization, angular-velocity damping, cautious mode, push recovery).
3. **Online learning** -- slow adaptation of 5 reflex/gait parameters via SPSA or bandit search, minimizing a stability cost computed from IMU signals.

## Documentation

| Document | Contents |
|----------|----------|
| [docs/DESIGN.md](docs/DESIGN.md) | Biological motivation, what the IMU tells us, full behavior catalog (13 behaviors across 3 levels), limitations, references |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Control loop diagram, IMU filter pseudocode, stability score, tuneable gait parameters, reflex rules, adaptation layer (SPSA), timing budget, safety limits |
| [docs/PHASES.md](docs/PHASES.md) | Four-phase implementation roadmap with concrete deliverables and exit criteria per phase |
| [docs/FIRMWARE.md](docs/FIRMWARE.md) | Hardware stack: Teensy 4.1, Pololu Maestro, 12 servos, channel map, Maestro protocol, joint-to-us mapping |

## Repo layout

| Path | Purpose |
|------|---------|
| `docs/` | Design docs, architecture, roadmap, hardware reference |
| `firmware/hexapod_walking_v2/` | Baseline: hard-coded scripted tripod v2 gait (no adaptation) |
| `firmware/hexapod_imu_test/` | Phase 1: IMU filter + telemetry (created during implementation) |
| `firmware/hexapod_adaptive/` | Phases 2--4: reflexes, cautious mode, online learning (created during implementation) |
| `host/` | Optional PC tools (logging, analysis, flash scripts) |
| `sim/` | Optional MuJoCo / Python experiments |

## Hardware

- **MCU:** Teensy 4.1 (600 MHz Cortex-M7, hardware FPU)
- **Servo controller:** Pololu Maestro (UART, 9600 baud on `Serial1`)
- **Servos:** 12 (6 legs x 2 joints: abduction + flexion)
- **Sensor:** 6-axis IMU (I2C) -- BNO055, MPU-6050, or ICM-20948

## Implementation phases

1. **IMU filter + telemetry** -- wire IMU, complementary filter, print roll/pitch/rates. No gait changes.
2. **Postural reflexes** -- per-leg corrections from roll/pitch/rates. Hard-coded gains. Robot levels itself on slopes.
3. **Cautious mode + recovery** -- stability score triggers slower/shorter stride; push recovery reflex on angular velocity spikes.
4. **Online parameter learning** -- 5-scalar SPSA optimization of reflex gains and gait shape, EEPROM persistence.

See [docs/PHASES.md](docs/PHASES.md) for details, exit criteria, and the E90 story at each stage.
