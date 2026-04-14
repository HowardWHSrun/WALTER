# With-IMU Optimization

This folder is reserved for the next step of the hexapod model:

- add roll and pitch measurements from the IMU
- add reflex gains to the optimization variables
- compare no-IMU baseline versus IMU-stabilized gait tuning

For now, the implemented runnable optimizer is in `../ode_optimization_no_imu/`.
# Hexapod ODE Optimization With IMU

This folder is reserved for the next stage of the brain stack:

- closed-loop gait correction using IMU signals
- reward shaping from roll, pitch, and angular-rate penalties
- parameter adaptation on top of the open-loop no-IMU model

Suggested next files for this folder:

- `imu_signal_model.py`
- `closed_loop_hexapod_ode.py`
- `imu_reward_definition.md`
- `ppo_parameter_tuning_with_imu.py`

