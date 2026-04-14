# No-IMU Hexapod ODE Optimization

## Robot Assumptions
- Total mass: 0.458 kg
- Servo type: 9g class (SG90-like), stall torque 0.176 N*m
- Optimization objective: maximize mean forward velocity (see script docstring).
- Optional speed floor (soft): 0.0000 m/s

## Best Parameters
- Abduction rest angles (deg): [10.0, 10.0, 0.0, 0.0, -10.0, -10.0]
- Flexion rest angles (deg): [32.0, 32.0, 34.0, 34.0, 32.0, 32.0]
- Abduction phase shifts (deg): [-0.0, -0.0, 0.0, 0.0, -0.0, -0.0]
- Flexion phase shifts (deg): [-90.0, -90.0, -90.0, -90.0, -90.0, -90.0]
- Gait frequency (Hz): 1.8000

## Metrics
- dominant_frequency_hz: 3.537240
- gait_frequency_hz: 1.800000
- mean_forward_speed_mps: 0.233791
- total_displacement_m: 1.863889
- roll_rms_rad: 0.440451
- pitch_rms_rad: 0.000000
- z_rms_m: 0.086256
- peak_servo_usage_ratio: 0.151814
- contact_balance_std: 0.387593
- objective_cost: -228.380390

## Notes
- This is a reduced no-IMU body model used to optimize gait timing and posture before adding reflex feedback.
- Tripod coupling is enforced with a pi phase offset between tripod A and tripod B.
- The optimization variables are the 12 rest angles, 12 per-servo phase shifts, and one global gait frequency.
