# No-IMU ODE Optimization

This folder contains a Python version of the "random initialization + ODE simulation + constrained optimization" workflow that you described from the MATLAB `ode45` and `fmincon` example, but adapted for the 12-servo tripod-coupled hexapod.

## What It Optimizes

**Primary objective:** maximize **mean forward velocity** (m/s) over the steady-state portion of the simulation (after an initial transient). Secondary terms penalize large roll/pitch, vertical oscillation, servo overload, and contact imbalance so solutions stay somewhat reasonable.

**Swing lift:** `leg_commands()` applies an extra flex reduction when `sin(abd_phase) > 0` (peak `LIFT_SWING_DEG_PEAK` degrees), analogous to the discrete lift phase in tripod v2 firmware, so simulated feet can clear the ground better than a bare sinusoid.

The optimizer treats the hexapod as a reduced no-IMU body model with:

- 6 abduction joints
- 6 flexion joints
- tripod A / tripod B phase coupling
- one global gait frequency

The decision vector is:

- 6 abduction rest angles
- 6 flexion rest angles
- 6 abduction phase shifts
- 6 flexion phase shifts
- 1 gait frequency

That is 25 optimization variables total.

## Physical Assumptions

The model assumes 12 SG90-like 9g servos and estimates total robot mass from:

- servo mass
- chassis mass
- battery and electronics mass

These values are easy to change in `hexapod_no_imu_optimization.py` if your robot mass is different.

## Run

```bash
python3 hexapod_no_imu_optimization.py
```

Multi-restart runs can be slow because each objective evaluation integrates the ODE. For a quicker exploration (coarser time grid), then deploy with `firmware/hexapod_openloop_sine/sync_gait_params_from_json.py`:

```bash
python3 hexapod_no_imu_optimization.py --restarts 4 --maxiter 80 --max-fun 350 --samples 400 --duration 5
```

Use full `--duration 8`, `--samples 800`, and higher `--max-fun` when you want a more accurate final pass (often with `--restarts 1` from the best seed found above).

Prioritize forward speed more strongly (default weight is 1000):

```bash
python3 hexapod_no_imu_optimization.py --forward-speed-weight 2000 --restarts 12
```

Optional soft floor on mean speed (penalizes solutions that stay below 0.10 m/s):

```bash
python3 hexapod_no_imu_optimization.py --target-speed 0.10 --target-speed-penalty-weight 400
```

## Outputs

Each run writes:

- `*_summary.json`
- `*_summary.md`
- `*_timeseries.csv`

## Notes

- This is a reduced ODE model for optimization and parameter search, not a full rigid-body simulator.
- The script keeps the tripod anti-phase structure from the roadmap and `hexapod_walking_v2_snapshot.ino`.
- The firmware values in the Arduino file look pulse-like rather than directly physical angles, so this Python model uses physically plausible degree-space variables for optimization.
# Hexapod ODE Optimization Without IMU

This folder contains a Python version of a `fmincon` + `ode45` style workflow for the 12-servo hexapod in `hexapod_walking_v2_snapshot.ino`.

## What it does

- models the 12 servos as coupled second-order ODEs
- uses the tripod grouping from the current walking code
- optimizes open-loop gait parameters without IMU feedback
- uses randomized initialization
- simulates with `scipy.integrate.solve_ivp` (Python analogue of `ode45`)
- optimizes with `scipy.optimize.minimize(method="SLSQP")` (Python analogue of `fmincon`)

## Optimized parameters

- front / middle / rear abduction rest angles
- front / middle / rear abduction angle shifts
- flexion rest angle
- flexion angle shift
- gait frequency
- tripod phase split for abduction
- tripod phase split for flexion
- flexion-abduction phase offset

## Modeling assumptions

- no IMU feedback, so this is an open-loop gait-parameter optimizer
- 12 servos total: 6 abduction + 6 flexion
- tripod groups match the existing code:
  - tripod A: `FL`, `MR`, `RL`
  - tripod B: `FR`, `ML`, `RR`
- servo model is a simplified second-order tracking model with damping, coupling, and gravity/load terms
- the robot mass and 9g-servo capability are included as coarse engineering assumptions, not measured hardware truths

## Default hardware assumptions in the script

- robot mass: `0.28 kg`
- 9g servo stall torque: `0.176 N*m`
- effective limb lever arm: `0.05 m`

These should be updated once you have measured values.

## Run

```bash
python3 optimize_hexapod_no_imu.py
```

## Python dependencies

- `numpy`
- `scipy`
- `matplotlib`

## Outputs

The script prints:

- best objective value
- identified dominant oscillation frequency
- optimized gait parameters

It also saves:

- `best_params_no_imu.json`
- `best_response_no_imu.png`

