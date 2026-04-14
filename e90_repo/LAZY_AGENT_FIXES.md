# Why the Robot Moves Once Then Stops (Lazy Agent) and How to Fix It

## What’s going on

The hexapod is hitting the classic **“Lazy Agent”** (or **local minimum**) in RL locomotion: it learns that **standing still is safer than walking**.

- **Risk:** When it explores by moving, it often tilts or falls.
- **Punishment:** Episodes end on fall/tilt/leg-up (`_check_termination` in `envs/hexapod_env.py`), so it gets a big negative signal and no more rewards.
- **Reward:** With `survival_bonus: 0` you already removed “reward for existing.” But velocity reward is only positive when moving; when still, reward is small or zero, and there was no explicit penalty for standing still.
- **Result:** The policy prefers “move a little once, then freeze” over “keep walking,” because continuing to move increases the chance of crossing a termination condition.

So the robot isn’t broken; it’s exploiting the reward and termination design.

## How this matches your code

1. **Termination** (`_check_termination`): Episode ends on `torso_pos[2] < min_torso_height`, tilt `> max_torso_tilt`, or flexion `> max_flexion_angle`. So “tilting = death” and the agent never gets to learn recovery.
2. **Reward** (`_compute_reward`): You use `forward_velocity_weight * max(0, avg_forward_velocity)` and no survival bonus. There was no penalty for near-zero velocity (`idle_velocity_penalty_weight` was 0 in your run).
3. **Reset** (`reset`): Joint noise was small (0.05 rad) and no initial forward velocity, so the robot always started from a near standstill.
4. **No lateral penalty:** Only backward velocity was penalized; lateral drift wasn’t, so “walk straight” wasn’t strongly encouraged.

## What was implemented (four strategies)

### 1. Softer “fear of falling” (termination logic)

- **New env options:** `terminate_on_fall`, `terminate_on_tilt`, `terminate_on_leg_up` (all default `True`).
- **`terminate_on_fall: false`:** Episode no longer ends when `torso_pos[2] < min_torso_height`. The robot can stay on the ground and still get rewards for forward velocity if it moves (e.g. writhe and scoot).
- **`fall_penalty_weight`:** When not terminating on fall, the env still adds a per-step penalty when `torso_pos[2] < min_torso_height`, so the agent learns to avoid being down without the episode ending.

This reduces the incentive to “do nothing to avoid termination.”

### 2. “Stand still” penalty and lateral drift

- **`idle_velocity_penalty_weight`:** Already in the env; now used in the anti-lazy config. If `|forward_velocity| < idle_velocity_threshold`, the robot gets a penalty each step (e.g. 0.5).
- **`lateral_velocity_penalty_weight`:** New. Penalizes `|torso_lin_vel[1]|` so going straight is preferred.

Together, standing still becomes costly and walking straight is rewarded.

### 3. Initialize in motion (reset)

- **`reset_joint_noise_scale`:** Default 0.05; can be increased (e.g. 0.25) so the robot starts slightly off-balance and must react.
- **`reset_initial_forward_velocity`:** New. Sets `qvel[0]` at reset (e.g. 0.15) so the robot starts with a small forward shove and must use its legs to maintain balance and motion.

So the agent no longer learns only from a dead standstill.

### 4. Phase-based / imitation (already in the codebase)

- **`obs_include_phase`**, **`imitation_weight`**, **`imitation_action_mix`**, **`_tripod_reference_action`** already implement a phase signal and optional tripod imitation. You can enable these in config for “training wheels” and then reduce or turn them off later.

## Config to try first: anti-lazy

**`configs/config_v7_anti_lazy.yaml`** applies the above:

- `terminate_on_fall: false`, `fall_penalty_weight: 2.0`
- `idle_velocity_penalty_weight: 0.5`, `lateral_velocity_penalty_weight: 1.0`
- `reset_joint_noise_scale: 0.25`, `reset_initial_forward_velocity: 0.15`
- Same PD and velocity-only reward as your v5-style run (e.g. `forward_velocity_weight: 50`).

Train with:

```bash
python train.py --config configs/config_v7_anti_lazy.yaml --version 2 --output-dir runs/anti_lazy_001
```

If the robot still freezes after a few steps, try:

- Increasing `idle_velocity_penalty_weight` (e.g. 1.0).
- Enabling phase-based imitation: set `imitation_weight` and/or `imitation_action_mix` in the same config so the policy gets a strong tripod signal early, then decay it in a later run.

## References (online)

- Reward shaping and “lazy” local optima in locomotion: agents can exploit reward structure so that inaction is preferred (e.g. Isaac Lab / JetBot reward exploration docs, barrier-style reward shaping for locomotion).
- Letting the robot recover instead of terminating on fall: research on fall recovery and time-varying rewards suggests not always ending the episode on first fall so the agent can learn to recover and keep moving; your `terminate_on_fall: false` + `fall_penalty_weight` follows that idea.
- Phase / clock input for periodic gaits: adding a phase (or imitation) signal is a standard way to help PPO find a rhythm (e.g. tripod) instead of discovering it from scratch; your code already supports this.

---

**Summary:** The robot “moves once then stops” because the current reward and termination make standing still a relatively safe, high-return strategy. The env now supports optional no-termination-on-fall with a fall penalty, idle and lateral penalties, and reset-in-motion. Use `config_v7_anti_lazy.yaml` first; if needed, add phase-based imitation next.

## Post-run: anti_lazy_v7_1M (still not moving)

After 1M steps, mean_episode_length stayed ~35. Cause: **tilt death trap** — with only `terminate_on_tilt: true` left, the robot hit 45 deg tilt in ~35 steps and died; the policy learned to avoid motion. Fix: set **`terminate_on_tilt: false`** and **`tilt_penalty_weight: 2.0`** so episodes run to 1000 steps and the agent pays for tilt per step instead of terminating. Re-train with updated config_v7_anti_lazy.yaml.
