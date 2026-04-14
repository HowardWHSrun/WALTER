# Plan: Reducing Hexapod Shaking

Shaking/jitter in the hexapod comes from fast, high-frequency changes in joint targets and torso motion. The following levers are available in the current codebase. Apply them gradually and compare eval videos.

---

## 1. Action smoothing and rate limiting (env)

These directly limit how quickly the policy can change its output.

| Lever | Current (config_terrain) | Low-shake suggestion | Effect |
|-------|---------------------------|----------------------|--------|
| **action_smooth_alpha** | 0.6 | 0.72–0.78 | Blend more with previous action; reduces step-to-step jerk. |
| **action_rate_limit** | 0.15 | 0.08–0.12 | Cap max change per step; smaller cap = smoother motion. |

**Risk:** Too much smoothing or too tight a rate limit can make the robot sluggish or slow to react on rough/step terrain.

---

## 2. Reward shaping (discourage jerk and fast motion)

The reward already includes smoothness and jerk terms; increasing their weight encourages the policy to avoid shaking.

| Lever | Current | Low-shake suggestion | Effect |
|-------|---------|------------------------|--------|
| **smoothness_reward_weight** | 2.5 | 3.5–4.5 | Stronger penalty for large action deltas (smoother actions). |
| **jerk_penalty_weight** | 0.35 | 0.5–0.7 | Penalize acceleration of actions (less “nervous” control). |
| **joint_velocity_penalty_weight** | 0.04 | 0.06–0.10 | Discourage fast joint motion; reduces limb flutter. |
| **stability_reward_weight** | 1.0 | 1.2–1.5 | Reward staying near target height; can reduce vertical bob. |
| **stance_stability_penalty_weight** | 0.2 | 0.25–0.35 | Stronger penalty for moving when foot is planted (cleaner stance). |
| **stance_action_penalty_weight** | 0.35 | 0.4–0.5 | Same idea; less unnecessary motion during stance. |
| **tilt_penalty_weight** | 2.0 | 2.5–3.0 | Discourage torso tilt; can reduce visible wobble. |

**Risk:** Pushing these too high can favor very slow or stiff gaits; tune in small steps.

---

## 3. Low-level control (PD)

More damping at the joint level can filter out high-frequency oscillations before they become visible motion.

| Lever | Current | Low-shake suggestion | Effect |
|-------|---------|------------------------|--------|
| **pd_kd** | 1.0 | 1.2–1.5 | Higher damping; reduces joint overshoot and oscillation. |
| **pd_kp** | 20.0 | Keep or 18–22 | Slightly lower kp can reduce “twitchiness” if needed. |

**Risk:** Very high kd can make the robot feel sluggish or slow to track targets.

---

## 4. Control frequency (frame_skip)

| Lever | Current | Low-shake suggestion | Effect |
|-------|---------|------------------------|--------|
| **frame_skip** | 5 | 6–8 | Fewer policy updates per second; each action held longer, often smoother visually. |

**Risk:** Too high frame_skip can make the policy too slow to react on uneven terrain.

---

## 5. Velocity reward smoothing

| Lever | Current | Low-shake suggestion | Effect |
|-------|---------|------------------------|--------|
| **velocity_reward_window** | 5 | 7–9 | Velocity reward based on longer window; policy less incentivized to chase instant velocity spikes. |

**Risk:** Very long window can blur the link between action and reward; moderate increase (7–9) is a good first try.

---

## Recommended order of experiments

1. **Quick win (no new training):** Increase **action_smooth_alpha** and/or decrease **action_rate_limit** in the env; reload the same checkpoint and run eval. If the policy was already trained with some smoothing, this only affects execution and can reduce shake immediately.
2. **Config variant:** Train a new run from scratch with a “low-shake” config that:
   - Increases action_smooth_alpha (e.g. 0.75), decreases action_rate_limit (e.g. 0.10)
   - Increases smoothness_reward_weight, jerk_penalty_weight, joint_velocity_penalty_weight
   - Optionally increases pd_kd and velocity_reward_window
   - See `configs/config_terrain_low_shake.yaml`.
3. **A/B comparison:** Compare eval videos (same terrain, same episode length) for:
   - Current terrain_run vs low-shake run
   - Optionally: same policy with different inference-time smoothing (step 1).
4. **Iterate:** If still too shaky, nudge smoothness/jerk/rate limit further. If too sluggish, back off slightly.

---

## Files

- **Env/reward:** `envs/hexapod_env.py` (smoothness, jerk, joint velocity, stance, tilt, velocity window).
- **Config:** `configs/config_terrain.yaml` (current); `configs/config_terrain_low_shake.yaml` (suggested low-shake variant).
- **Camera:** `assets/hexapod_with_terrain.xml` — eval_cam position updated so the camera is a bit further back for clearer viewing.
