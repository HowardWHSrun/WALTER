# Open-loop sinusoidal gait (no IMU)

**WALTER (project home):** [https://github.com/HowardWHSrun/WALTER](https://github.com/HowardWHSrun/WALTER)

Firmware implementation of `leg_commands()` from [`hexapod_no_imu_optimization.py`](../../hexapod_brain_roadmap%20copy/ode_optimization_no_imu/hexapod_no_imu_optimization.py): per-leg sinusoidal abduction and flexion with tripod anti-phase (FL/MR/RL vs FR/ML/RR).

## Files

- `hexapod_openloop_sine.ino` — Teensy sketch (Maestro on `Serial1`, same protocol as v2).
- `optimized_gait_params.h` — gait parameters (often generated, not hand-edited).
- `sync_gait_params_from_json.py` — regenerates `optimized_gait_params.h` from `hexapod_brain_roadmap copy/ode_optimization_no_imu/no_imu_optimization_summary.json` after you run `hexapod_no_imu_optimization.py`. Usage: `python3 sync_gait_params_from_json.py` from this folder (or pass path to a summary JSON).
- `OPENLOOP_OPTIMIZATION_REPORT.tex` / `OPENLOOP_OPTIMIZATION_REPORT.pdf` — LaTeX write-up of the no-IMU optimizer and firmware mapping; build with `pdflatex OPENLOOP_OPTIMIZATION_REPORT.tex`.

## Tuning

- **`OPT_LIFT_SWING_DEG_PEAK`** in `optimized_gait_params.h`: extra foot clearance during swing when `sin(abd_phase) > 0` (same as `LIFT_SWING_DEG_PEAK` in Python). Increase if feet drag; set to `0` to disable. If the robot shuffles backward, flip the lift test in the sketch (`sinAbd < 0` instead of `> 0`).
- **`ABD_CMD_PER_DEG` / `FLX_CMD_PER_DEG`** in the `.ino`: map optimizer degrees to abstract joint commands (same convention as [`hexapod_walking_v2`](../hexapod_walking_v2/hexapod_walking_v2.ino)). Adjust on the bench so motion stays within safe pulses and matches expected stride height.
- **`ABD_CMD_*` / `FLX_CMD_*` clamps**: hard limits before `jointToUs`.
- **`FREQ_RAMP_MS`**: linear ramp from 0 Hz to `OPT_FREQUENCY_HZ` after boot (default 3000 ms). Set to `0` to use full frequency from the first tick (`frequencyNow()`).

## Baseline

For discrete tripod keyframes, keep using [`firmware/hexapod_walking_v2`](../hexapod_walking_v2/).
