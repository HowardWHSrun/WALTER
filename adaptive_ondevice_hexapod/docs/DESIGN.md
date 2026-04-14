# Design: IMU-based adaptive locomotion

## Biological motivation

Insect locomotion is modeled as **decentralized, feedback-driven, and layered**: a basic locomotion pattern produces rhythmic output, sensory corrections modulate that pattern in real time, and slower adaptation tunes the corrections over experience. This project follows the same layering on a 12-DOF hexapod with a single IMU as the primary exteroceptive sensor. The approach is grounded in mechanosensory control of locomotion in animals and robots ([Aydin et al., 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC10445419/)) and bio-inspired adaptive hexapod locomotion ([Chen et al., 2021](https://www.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2021.627157/full)).

The most bio-inspired use of the IMU is **not** inverted-pendulum balance. It is: **use body-state feedback to modulate an existing gait** -- the same principle insects use when mechanoreceptors adjust leg timing and force in response to body perturbation.

## What the IMU tells us

From a 6-axis IMU (accelerometer + gyroscope), filtered with a complementary or Mahony attitude estimator, we can reliably obtain:

| Signal | Source | Notes |
|--------|--------|-------|
| Roll angle | Accel + gyro fusion | Trustworthy; no magnetometer needed |
| Pitch angle | Accel + gyro fusion | Trustworthy |
| Roll rate | Gyro (high-pass or direct) | Fast, good for reflexes |
| Pitch rate | Gyro (high-pass or direct) | Fast, good for reflexes |
| Vertical oscillation magnitude | Accel z-axis envelope | Proxy for gait smoothness |
| Sudden disturbance / impact | Accel norm spike or gyro spike | Trigger for recovery reflex |
| Trend: is stability degrading? | Windowed RMS of above signals | Drives adaptation layer |

Roll and pitch are far more reliable and useful than yaw for locomotion when no clean magnetometer is available, since magnetic disturbance and drift primarily corrupt heading estimation ([Valenti et al., 2015](https://www.mdpi.com/1424-8220/15/8/19302)). Hexapod posture-control work specifically uses body orientation and angular velocity as key stability variables on irregular terrain ([Bai et al., 2019](https://www.cambridge.org/core/journals/robotica/article/development-and-implementation-of-a-new-approach-for-posture-control-of-a-hexapod-robot-to-walk-in-irregular-terrains/1B398F52F5E31EF8067E0E7D22BBF2B2)).

## Behavior catalog

### Level 1 -- Must-have (easy, high impact)

| # | Behavior | IMU signals used | Description |
|---|----------|-----------------|-------------|
| 1 | **Roll/pitch stabilization** | roll, pitch | If the body leans, adjust per-leg flex targets to bring it back. Increase stance on the falling side, reduce on the rising side. Clearest postural reflex. |
| 2 | **Angular-velocity damping** | roll rate, pitch rate | React to *how fast* the robot is tipping, not only final angle. Intervene before the angle becomes large. More reflex-like: responds to ongoing destabilization. |
| 3 | **Cautious mode trigger** | stability score (composite) | If the body oscillates too much, switch to a slower or more stable gait: reduce frequency, shorten stride, or switch pattern (tripod to ripple). Sensory state changes the locomotor pattern. |
| 4 | **Terrain compensation / body leveling** | roll, pitch | On a slope or uneven surface, change per-leg extension to keep the body level rather than rigidly following a flat-ground gait. |

### Level 2 -- Good project features (realistic, makes the robot feel intelligent)

| # | Behavior | IMU signals used | Description |
|---|----------|-----------------|-------------|
| 5 | **Step-height adaptation** | pitch spikes, accel impacts | If body shocks or pitch spikes increase during swing/landing, raise step height for the next few steps. Interpretation: "I am hitting stuff." |
| 6 | **Stride-length adaptation** | stability score trend | If roll/pitch instability grows, reduce stride length. Interpretation: "smaller steps are safer." |
| 7 | **Duty-factor adaptation** | stability score | Spend a larger fraction of each cycle in stance when unstable. Keep more legs on the ground longer. |
| 8 | **Frequency adaptation** | stability score, oscillation magnitude | If the body resonates or bounces, reduce gait frequency. Treats locomotion as a dynamic body-controller interaction. |
| 9 | **Push recovery reflex** | angular velocity spike | On sudden angular velocity spike, enter temporary recovery: slower motion, wider support, lower center of mass, longer stance. 1--2 gait cycles. |
| 10 | **Phase reset / timing correction** | body state mid-cycle | If body state becomes bad during a gait cycle, delay or advance the next swing phase instead of blindly following the metronome. |

### Level 3 -- Ambitious (strongest "bio-inspired learning" angle)

| # | Behavior | IMU signals used | Description |
|---|----------|-----------------|-------------|
| 11 | **Learn reflex gains online** | all of the above | Instead of hard-coding correction strength, let the robot learn: how strongly to react to roll, pitch rate, how much to shorten stride, raise step height. Low-dimensional and interpretable. |
| 12 | **Learn gait selection by context** | windowed IMU statistics | Store different "best parameter sets" for flat floor, incline, soft surface, disturbed condition. Select among them from recent IMU statistics. |
| 13 | **Learn stability cost minimization** | composite cost J | Define J = a * RMS(roll) + b * RMS(pitch) + c * angular_velocity - d * forward_progress. Tune gait parameters to reduce J. Clear engineering objective and clean experimental section. |

## What to avoid

- **Do not rely heavily on yaw** unless you have a well-calibrated magnetometer free from motor and wiring interference. Low-cost IMUs need sensor fusion, and magnetic disturbance mainly contaminates heading ([Zhang et al., 2021](https://www.mdpi.com/2072-666X/12/11/1373)).
- **Do not assume the IMU can directly sense foot contact.** It can hint at impacts and instability, but it is not a direct foot sensor.
- **Do not try to learn raw servo commands from scratch.** On this platform, that is harder to stabilize, harder to explain, and less biologically grounded than learning a few meaningful gait parameters ([Chen et al., 2021](https://www.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2021.627157/full)).

## Bio-inspired framing (for the report)

The strongest themes for E90:

- **Postural reflexes** -- the body senses tipping and changes limb output.
- **State-dependent gait modulation** -- the robot changes locomotion based on body state.
- **Layered control** -- innate rhythm first, sensory corrections second, learning third.
- **Embodied adaptation** -- the robot responds to body dynamics as they happen, not purely in software.

These align better with biology than "we used reinforcement learning to control 12 servos directly."

## References

1. Zhang, R. et al. (2021). Attitude Estimation Algorithm of Portable Mobile Robot Based on Complementary Filter. *Micromachines*, 12(11), 1373. [Link](https://www.mdpi.com/2072-666X/12/11/1373)
2. Aydin, Y. et al. (2023). Mechanosensory Control of Locomotion in Animals and Robots: Moving Forward. *PMC*. [Link](https://pmc.ncbi.nlm.nih.gov/articles/PMC10445419/)
3. Bai, L. et al. (2019). Development and implementation of a new approach for posture control of a hexapod robot to walk in irregular terrains. *Robotica*. [Link](https://www.cambridge.org/core/journals/robotica/article/development-and-implementation-of-a-new-approach-for-posture-control-of-a-hexapod-robot-to-walk-in-irregular-terrains/1B398F52F5E31EF8067E0E7D22BBF2B2)
4. Chen, G. et al. (2021). Adaptive Locomotion Control of a Hexapod Robot via Bio-Inspired Learning. *Frontiers in Neurorobotics*. [Link](https://www.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2021.627157/full)
5. Johnson, A. et al. (2010). Disturbance Detection, Identification, and Recovery by Gait Transition in Legged Robots. *IROS 2010*. [Link](https://kodlab.seas.upenn.edu/uploads/Aaron/JohnsonAaronDisturbanceIROS2010.pdf)
6. Valenti, R.G. et al. (2015). Keeping a Good Attitude: A Quaternion-Based Orientation Filter for IMUs and MARGs. *Sensors*, 15(8), 19302. [Link](https://www.mdpi.com/1424-8220/15/8/19302)
