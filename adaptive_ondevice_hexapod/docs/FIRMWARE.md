# Firmware and hardware stack (reference)

This document records how the hexapod is driven today so the **adaptive on-device** work stays aligned with real wiring and protocols. Values match the working **`hexapod_walking_rl`** / **`hexapod_walking`** sketches unless you change calibration.

## Compute

| Item | Detail |
|------|--------|
| **MCU** | Teensy 4.1 |
| **USB Serial** | Host PC / debugging / tuning — typically **115200 baud** |
| **UART to Maestro** | **`Serial1`** (hardware TX/RX to Pololu board) — **9600 baud** in current firmware |

## Actuation chain

```
Teensy 4.1  --UART 9600-->  Pololu Maestro  ----PWM---->  12 servos (6 legs × 2 joints)
```

- **Pololu Maestro** (servo controller): receives **binary protocol** on serial; outputs standard **servo pulse width** commands (microseconds) on **channels 0–11**.
- **12 servos**: six legs, each leg **abduction (abd)** + **flexion (flx)**.

## Maestro channel layout (firmware `ServoID`)

Maestro channels are **0–11** in this order:

| Channel | Leg | Joint |
|---------|-----|--------|
| 0 | FL (front left) | ABD |
| 1 | FL | FLX |
| 2 | FR (front right) | ABD |
| 3 | FR | FLX |
| 4 | ML (mid left) | ABD |
| 5 | ML | FLX |
| 6 | MR (mid right) | ABD |
| 7 | MR | FLX |
| 8 | RL (rear left) | ABD |
| 9 | RL | FLX |
| 10 | RR (rear right) | ABD |
| 11 | RR | FLX |

## Joint command → pulse width

Firmware uses an abstract **joint command** (integer, same convention as the scripted v2 gait), then:

- `pulse_us = clamp( servoCenter[i] + servoDir[i] * jointCommand, MIN_US, MAX_US )`

Typical constants in sketch:

- **`MIN_US` / `MAX_US`**: **600 / 2400** (safety clamp).
- **`servoCenter[]`**: per-channel center in µs — often **1500** everywhere until you calibrate each leg.
- **`servoDir[]`**: **+1** or **-1** per channel so positive “joint command” matches your mechanical convention.

The Maestro **Set Target** command uses **quarter-microsecond** units: `target = pulse_us * 4`, sent as a 3-byte mini packet (Pololu serial protocol).

## Maestro serial setup (typical `setup()`)

1. `Serial1.begin(9600)`
2. Short delay for Maestro boot
3. Send **`0xAA`** (if required for your Maestro mode / “start” behavior in your wiring)
4. Optional: **`0xA1`** repeated (as in existing sketch — keep consistent with your board documentation)

Per-command write pattern for one channel (simplified):

- `0x84`, then channel byte, then 14-bit target in two 7-bit bytes (low bits first) — **target = desired_us × 4**.

## MuJoCo / training order vs Maestro order

Simulation and RL code often use **FR, FL, MR, ML, RR, RL** leg indexing (abd then flex per leg). Your **Maestro** order is **FL, FR, ML, MR, RL, RR**. There is a fixed **permutation** (`MAESTRO_TO_SIM` in `sim_maestro_map.h`): when you produce 12 values in **sim joint order**, reorder before sending to Maestro channels.

Keep this mapping identical anywhere you port policy output → servos.

## Control rate

Existing gait tick uses a fixed period (e.g. **~20 ms** / **50 Hz**) via `millis()`. Adaptive code should either match this or explicitly define a new `dt` and document it.

## Reference: scripted v2 gait (this repo)

The **hard-coded** tripod gait (joint command integers, smoothed keyframes) lives under:

`firmware/hexapod_walking_v2/hexapod_walking_v2.ino`

Use it as the non-adaptive baseline while developing on-device adaptation alongside the same Maestro layer.

## What this project might add later

- IMU on **I2C/SPI** (not documented here until you pick a part and pins).
- EEPROM or flash for saved adaptation parameters (pattern already exists in `hexapod_walking_rl` for gait scalars).
- Same Maestro path: adaptive logic should still output **12 targets** (or deltas) under the same `jointToUs` / safety rules unless you deliberately change calibration.
