# Hexapod scripted tripod gait (v2)

**Hard-coded joint targets** (abstract command integers → `jointToUs` → Pololu Maestro), **lift → swing → plant** sub-phases per tripod group.

- **Board:** Teensy 4.1 (Arduino + Teensyduino).
- **Maestro:** `Serial1` @ 9600 baud — same protocol as [docs/FIRMWARE.md](../../docs/FIRMWARE.md).

This folder is the **baseline reference** for the adaptive on-device project: behavior here is fully scripted, not learned.

## Upload

Open `hexapod_walking_v2.ino` in Arduino IDE (or use `arduino-cli` with FQBN `teensy:avr:teensy41`). Keep **only this sketch** in the folder so the IDE does not merge other `.ino` files.

## Calibrate

Edit `servoCenter[]` and `servoDir[]` for your hardware. Tune `START_FLEX`, `LIFT_FLEX`, `PUSH_FLEX`, and swing/push abd offsets if the stride is too weak or feet drag.

## Note on comments

The file header still mentions early v2 `LIFT_FLEX = -300`; the **active constant** in this copy is `LIFT_FLEX = -120` per your pasted source.
