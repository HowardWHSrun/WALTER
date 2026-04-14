"""
Hexapod MuJoCo Environment for RL Training

This environment wraps the hexapod MuJoCo model and provides a Gymnasium-compatible
interface for reinforcement learning with spiking CPG controllers.
"""

import os
from collections import deque
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco

# #region agent log
def _dbg_log(location: str, message: str, data: dict, hypothesis_id: str, run_id: str = "pre-fix"):
    try:
        import json, time
        payload = {
            "sessionId": "debug-session",
            "runId": run_id,
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data,
            "timestamp": int(time.time() * 1000),
        }
        with open("/Users/howardwang/Desktop/ValeroLab/E90/RL Temp/.cursor/debug.log", "a") as f:
            f.write(json.dumps(payload) + "\n")
    except Exception:
        pass
# #endregion


class HexapodEnv(gym.Env):
    """
    Gymnasium environment for hexapod locomotion with MuJoCo physics.
    
    The hexapod has 6 legs, each with 2 DOF (abduction and flexion),
    totaling 12 actuated joints.
    
    Observation Space:
        - Joint positions (12)
        - Joint velocities (12)
        - Torso orientation quaternion (4)
        - Torso linear velocity (3)
        - Torso angular velocity (3)
        - Foot contact states (6)
        - Previous action (12)
        Total: 52 dimensions
    
    Action Space:
        - Joint torques (12), normalized to [-1, 1]
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}
    
    def __init__(
        self,
        xml_path: str = None,
        frame_skip: int = 5,
        max_episode_steps: int = 1000,
        render_mode: str = None,
        # Observation flags
        obs_include_joint_pos: bool = True,
        obs_include_joint_vel: bool = True,
        obs_include_torso_quat: bool = True,
        obs_include_torso_vel: bool = True,
        obs_include_contact: bool = True,
        obs_include_prev_action: bool = True,
        obs_include_torso_height: bool = False,
        obs_use_sensors: bool = False,
        obs_include_phase: bool = False,
        obs_phase_frequency_hz: float = 0.9,
        obs_include_sim_time: bool = False,
        # Termination conditions
        min_torso_height: float = 0.05,
        max_torso_tilt: float = 35.0,
        max_flexion_angle: float = 55.0,
        terminate_on_fall: bool = True,
        terminate_on_tilt: bool = True,
        terminate_on_leg_up: bool = True,
        fall_penalty_weight: float = 0.0,
        # Gradual tripod: action smoothing and reward shaping
        action_smooth_alpha: float = 0.0,
        smoothness_reward_weight: float = 0.0,
        stability_reward_weight: float = 0.0,
        target_torso_height: float = 0.10,
        action_warmup_steps: int = 0,
        # Near-termination (optional)
        near_termination_penalty_weight: float = 0.0,
        near_termination_margin_deg: float = 10.0,
        # Reward shaping (optional)
        forward_distance_weight: float = 15.0,
        forward_velocity_weight: float = 10.0,
        idle_velocity_threshold: float = 0.02,
        idle_velocity_penalty_weight: float = 0.0,
        idle_penalty_ramp_weight: float = 0.0,
        idle_penalty_ramp_max: float = 4.0,
        velocity_reward_window: int = 1,
        survival_bonus: float = 0.01,
        backward_penalty_weight: float = 2.0,
        lateral_velocity_penalty_weight: float = 0.0,
        stall_penalty_threshold: float = 0.1,
        stall_penalty_weight: float = 0.0,
        control_cost_weight: float = 0.0,
        velocity_reward_uncapped: bool = False,
        height_penalty_weight: float = 1.0,
        height_penalty_threshold: float = 0.06,
        tilt_penalty_weight: float = 0.0,
        yaw_rate_penalty_weight: float = 0.0,
        # Imitation of tripod CPG (optional)
        imitation_weight: float = 0.0,
        imitation_frequency_hz: float = 0.9,
        imitation_amp_flex: float = 0.75,
        imitation_amp_abd: float = 0.5,
        imitation_action_mix: float = 0.0,
        imitation_flex_sign: float = 1.0,
        imitation_abd_sign: float = 1.0,
        # PD control (optional): NN outputs target positions, PD computes torques
        use_pd_control: bool = False,
        pd_kp: float = 5.0,
        pd_kd: float = 0.5,
        pd_steps_per_action: int = 5,
        action_use_joint_limits: bool = True,
        action_scale_abd_deg: float = 45.0,
        action_scale_flex_deg: float = 90.0,
        reset_joint_noise_scale: float = 0.05,
        reset_initial_forward_velocity: float = 0.0,
        # Target velocity (slower, steadier): reward proximity to this speed when > 0
        target_forward_velocity: float = 0.0,
        target_velocity_weight: float = 0.0,
        target_velocity_scale: float = 0.1,
        # Path following: waypoints or circle
        path_type: str = None,
        path_waypoints: list = None,
        path_radius: float = 0.5,
        path_center: tuple = (0.0, 0.0),
        path_waypoint_reach_dist: float = 0.15,
        path_heading_reward_weight: float = 0.0,
        path_proximity_reward_weight: float = 0.0,
        path_proximity_tolerance: float = 0.1,
        obs_include_path_heading_error: bool = False,
        jerk_penalty_weight: float = 0.0,
        joint_velocity_penalty_weight: float = 0.0,
        action_rate_limit: float = 0.0,
        action_change_deadzone: float = 0.0,
        stance_stability_penalty_weight: float = 0.0,
        stance_action_penalty_weight: float = 0.0,
        # Terrain: flat | rough | steps | random (use heightfield; overwrite at reset)
        terrain_type: str = None,
        terrain_rough_scale: float = 0.4,
        terrain_rough_min_height: float = -0.02,
        terrain_rough_max_height: float = 0.02,
        terrain_step_height: float = 0.02,
        terrain_step_length: float = 0.08,
        terrain_num_steps: int = 5,
        # Observation mode: "full" | "imu6" (accel+gyro, 6D) | "imu6_prev" (IMU + previous action) | "imu_bno10" (accel+gyro+quat, 10D)
        observation_mode: str = "full",
        imu_noise_std_accel: float = 0.0,
        imu_noise_std_gyro: float = 0.0,
        imu_noise_std_quat: float = 0.0,
    ):
        super().__init__()

        # IMU-only modes: partial observability — no global path in observation (disable path following)
        _obs_mode = (observation_mode or "full").strip().lower()
        if _obs_mode in ("imu6", "imu6_prev", "imu_bno10"):
            path_type = None
            path_waypoints = None
            obs_include_path_heading_error = False

        assets_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets")
        # Use terrain XML when terrain_type is set (flat/rough/steps/random)
        if terrain_type is not None and xml_path is None:
            xml_path = os.path.join(assets_dir, "hexapod_with_terrain.xml")
        if xml_path is None:
            xml_path = os.path.join(assets_dir, "hexapod.xml")
        
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        self.observation_mode = _obs_mode
        self._imu_only_mode = _obs_mode in ("imu6", "imu6_prev", "imu_bno10")
        self._imu_include_prev_action = _obs_mode == "imu6_prev"
        self._imu_include_quat = _obs_mode == "imu_bno10"
        self._imu_noise_std_accel = float(imu_noise_std_accel)
        self._imu_noise_std_gyro = float(imu_noise_std_gyro)
        self._imu_noise_std_quat = float(imu_noise_std_quat)
        if self._imu_only_mode:
            acc_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "torso_accel")
            gyro_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "torso_gyro")
            if acc_id < 0 or gyro_id < 0:
                raise ValueError(
                    "observation_mode imu6/imu6_prev/imu_bno10 requires sensors 'torso_accel' and 'torso_gyro' in the MJCF"
                )
            self._imu_accel_adr = int(self.model.sensor_adr[acc_id])
            self._imu_gyro_adr = int(self.model.sensor_adr[gyro_id])
            if int(self.model.sensor_dim[acc_id]) != 3 or int(self.model.sensor_dim[gyro_id]) != 3:
                raise ValueError("torso_accel and torso_gyro must each have sensor dimension 3")
            if self._imu_include_quat:
                quat_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "torso_quat")
                if quat_id < 0:
                    raise ValueError("observation_mode=imu_bno10 requires sensor 'torso_quat' in the MJCF")
                self._imu_quat_adr = int(self.model.sensor_adr[quat_id])
                if int(self.model.sensor_dim[quat_id]) != 4:
                    raise ValueError("torso_quat must have sensor dimension 4")

        # Terrain: if model has heightfield, we can overwrite at reset
        self._terrain_type = terrain_type
        self._terrain_rough_scale = float(terrain_rough_scale)
        self._terrain_rough_min_height = float(terrain_rough_min_height)
        self._terrain_rough_max_height = float(terrain_rough_max_height)
        self._terrain_step_height = float(terrain_step_height)
        self._terrain_step_length = float(terrain_step_length)
        self._terrain_num_steps = int(terrain_num_steps)
        self._use_terrain_hfield = getattr(self.model, "nhfield", 0) > 0
        if self._use_terrain_hfield:
            self._hfield_id = 0
            self._hfield_nrow = int(self.model.hfield_nrow[self._hfield_id])
            self._hfield_ncol = int(self.model.hfield_ncol[self._hfield_id])
            self._hfield_adr = int(self.model.hfield_adr[self._hfield_id])
            self._hfield_size = self.model.hfield_size[self._hfield_id].copy()
            self._hfield_data_len = self._hfield_nrow * self._hfield_ncol
        # Optional curriculum: 0=flat only, 1=flat+rough, 2=flat+rough+steps (set by callback)
        self._terrain_curriculum_stage = None
        self._terrain_changed = False
        
        # Environment parameters
        self.frame_skip = frame_skip
        self._max_episode_steps = max_episode_steps
        self.render_mode = render_mode
        
        # Observation flags
        self.obs_include_joint_pos = obs_include_joint_pos
        self.obs_include_joint_vel = obs_include_joint_vel
        self.obs_include_torso_quat = obs_include_torso_quat
        self.obs_include_torso_vel = obs_include_torso_vel
        self.obs_include_contact = obs_include_contact
        self.obs_include_prev_action = obs_include_prev_action
        self.obs_include_torso_height = obs_include_torso_height
        self.obs_use_sensors = obs_use_sensors
        if self._imu_only_mode:
            self.obs_use_sensors = False
        self.obs_include_phase = obs_include_phase
        self.obs_phase_frequency_hz = float(obs_phase_frequency_hz)
        self.obs_include_sim_time = bool(obs_include_sim_time)
        # Clock signal for IMU modes (no joint encoders): helps periodic gaits
        self._imu_include_phase = self._imu_only_mode and bool(obs_include_phase)
        
        # Termination conditions
        self.min_torso_height = min_torso_height
        self.max_torso_tilt = np.deg2rad(max_torso_tilt)
        self.max_flexion_angle = np.deg2rad(max_flexion_angle)
        self.terminate_on_fall = bool(terminate_on_fall)
        self.terminate_on_tilt = bool(terminate_on_tilt)
        self.terminate_on_leg_up = bool(terminate_on_leg_up)
        self.fall_penalty_weight = float(fall_penalty_weight)
        # Gradual tripod: smooth back-and-forth
        self.action_smooth_alpha = float(np.clip(action_smooth_alpha, 0.0, 1.0))
        self.smoothness_reward_weight = smoothness_reward_weight
        self.stability_reward_weight = stability_reward_weight
        self.target_torso_height = target_torso_height
        self.action_warmup_steps = max(0, int(action_warmup_steps))
        self.near_termination_penalty_weight = float(near_termination_penalty_weight)
        self.near_termination_margin_rad = np.deg2rad(float(near_termination_margin_deg))
        self.forward_distance_weight = float(forward_distance_weight)
        self.forward_velocity_weight = float(forward_velocity_weight)
        self.idle_velocity_threshold = float(idle_velocity_threshold)
        self.idle_velocity_penalty_weight = float(idle_velocity_penalty_weight)
        self.idle_penalty_ramp_weight = float(idle_penalty_ramp_weight)
        self.idle_penalty_ramp_max = float(idle_penalty_ramp_max)
        self.velocity_reward_window = max(1, int(velocity_reward_window))
        self._velocity_window = deque(maxlen=self.velocity_reward_window)
        self.survival_bonus = float(survival_bonus)
        self.backward_penalty_weight = float(backward_penalty_weight)
        self.lateral_velocity_penalty_weight = float(lateral_velocity_penalty_weight)
        self.stall_penalty_threshold = float(stall_penalty_threshold)
        self.stall_penalty_weight = float(stall_penalty_weight)
        self.control_cost_weight = float(control_cost_weight)
        self.velocity_reward_uncapped = bool(velocity_reward_uncapped)
        self.height_penalty_weight = float(height_penalty_weight)
        self.tilt_penalty_weight = float(tilt_penalty_weight)
        self.reset_joint_noise_scale = float(reset_joint_noise_scale)
        self.reset_initial_forward_velocity = float(reset_initial_forward_velocity)
        self.height_penalty_threshold = float(height_penalty_threshold)
        self.yaw_rate_penalty_weight = float(yaw_rate_penalty_weight)
        self.imitation_weight = float(imitation_weight)
        self.imitation_frequency_hz = float(imitation_frequency_hz)
        self.imitation_amp_flex = float(imitation_amp_flex)
        self.imitation_amp_abd = float(imitation_amp_abd)
        self.imitation_action_mix = float(imitation_action_mix)
        self.imitation_flex_sign = float(imitation_flex_sign)
        self.imitation_abd_sign = float(imitation_abd_sign)
        self.use_pd_control = bool(use_pd_control)
        self.pd_kp = float(pd_kp)
        self.pd_kd = float(pd_kd)
        self.pd_steps_per_action = max(1, int(pd_steps_per_action))
        self.action_use_joint_limits = bool(action_use_joint_limits)
        self.action_scale_abd = np.deg2rad(float(action_scale_abd_deg))
        self.action_scale_flex = np.deg2rad(float(action_scale_flex_deg))
        self.reset_joint_noise_scale = float(reset_joint_noise_scale)
        self.reset_initial_forward_velocity = float(reset_initial_forward_velocity)
        self.target_forward_velocity = float(target_forward_velocity)
        self.target_velocity_weight = float(target_velocity_weight)
        self.target_velocity_scale = float(target_velocity_scale)
        self.path_type = path_type or None
        self.path_waypoints = list(path_waypoints) if path_waypoints else None
        self.path_radius = float(path_radius)
        self.path_center = tuple(float(x) for x in path_center)
        self.path_waypoint_reach_dist = float(path_waypoint_reach_dist)
        self.path_heading_reward_weight = float(path_heading_reward_weight)
        self.path_proximity_reward_weight = float(path_proximity_reward_weight)
        self.path_proximity_tolerance = float(path_proximity_tolerance)
        self.obs_include_path_heading_error = bool(obs_include_path_heading_error)
        self.jerk_penalty_weight = float(jerk_penalty_weight)
        self.joint_velocity_penalty_weight = float(joint_velocity_penalty_weight)
        self.action_rate_limit = float(action_rate_limit)
        self.action_change_deadzone = float(action_change_deadzone)
        self.stance_stability_penalty_weight = float(stance_stability_penalty_weight)
        self.stance_action_penalty_weight = float(stance_action_penalty_weight)
        self._path_waypoints = None  # normalized list of (x, y) for current path
        self._path_waypoint_idx = 0
        if self.path_type == "circle":
            self._path_waypoints = self._build_circle_waypoints()
        elif self.path_waypoints and len(self.path_waypoints) > 0:
            self._path_waypoints = [(float(p[0]), float(p[1])) for p in self.path_waypoints]
        self._path_active = bool(self._path_waypoints and len(self._path_waypoints) > 0)
        # Joint and actuator info
        self.num_joints = 12
        self.num_actuators = 12
        self.num_legs = 6
        
        # Get joint IDs (skip the freejoint)
        self.joint_names = [
            f"leg{i}_abduction" for i in range(1, 7)
        ] + [
            f"leg{i}_flexion" for i in range(1, 7)
        ]
        # Reorder to match leg order: [leg1_abd, leg1_flex, leg2_abd, leg2_flex, ...]
        self.joint_names = []
        for i in range(1, 7):
            self.joint_names.append(f"leg{i}_abduction")
            self.joint_names.append(f"leg{i}_flexion")
        self.joint_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in self.joint_names
        ]
        self.joint_qpos_addr = [self.model.jnt_qposadr[jid] for jid in self.joint_ids]
        self.joint_dof_addr = [self.model.jnt_dofadr[jid] for jid in self.joint_ids]
        self.joint_ranges = self.model.jnt_range[self.joint_ids].copy()
        self._action_scale = np.array(
            [self.action_scale_abd, self.action_scale_flex] * 6, dtype=np.float32
        )
        
        # State tracking
        self._prev_action = np.zeros(self.num_actuators)
        self._prev_prev_action = np.zeros(self.num_actuators)
        self._prev_torso_pos = None
        self._initial_torso_x = None  # Track initial X position for distance reward
        self._step_count = 0
        self._sim_time = 0.0
        self._consecutive_idle_steps = 0
        
        # Define observation and action spaces
        obs_dim = self._get_obs_dim()
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.num_actuators,), dtype=np.float32
        )
        
        # Rendering: use zoomed eval_cam if present (e.g. in hexapod_with_terrain.xml)
        self.renderer = None
        self.viewer = None
        self._eval_cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "eval_cam") if self.model.ncam > 0 else -1
        if self._eval_cam_id < 0:
            self._eval_cam_id = -1
        
    def _get_obs_dim(self) -> int:
        """Observation dimension: joint positions, optional torso/velocity/contact/height, optional previous action."""
        if self._imu_only_mode:
            n = 6
            if self._imu_include_prev_action:
                n += 12
            if self._imu_include_quat:
                n += 4
            if self._imu_include_phase:
                n += 2
            if self.obs_include_sim_time:
                n += 1
            return n
        if self.obs_use_sensors:
            return int(self.model.nsensordata)
        dim = self.num_joints if self.obs_include_joint_pos else 0
        if self.obs_include_joint_vel:
            dim += self.num_joints
        if self.obs_include_torso_quat:
            dim += 4
        if self.obs_include_torso_vel:
            dim += 3
        if self.obs_include_contact:
            dim += self.num_legs
        if self.obs_include_torso_height:
            dim += 1
        if self.obs_include_phase:
            dim += 2
        if self.obs_include_prev_action:
            dim += self.num_actuators
        if self.obs_include_path_heading_error and self._path_active:
            dim += 1
        if self.obs_include_sim_time:
            dim += 1
        return dim

    def _get_obs(self) -> np.ndarray:
        """Construct observation: joint angles, optional torso/velocity/contact/height, optional previous action."""
        if self._imu_only_mode:
            mujoco.mj_forward(self.model, self.data)
            adr_a = self._imu_accel_adr
            adr_g = self._imu_gyro_adr
            acc = np.array(self.data.sensordata[adr_a : adr_a + 3], dtype=np.float32)
            gyro = np.array(self.data.sensordata[adr_g : adr_g + 3], dtype=np.float32)
            obs = np.concatenate([acc, gyro], axis=0)
            if self._imu_noise_std_accel > 0.0:
                obs[0:3] += np.random.randn(3).astype(np.float32) * self._imu_noise_std_accel
            if self._imu_noise_std_gyro > 0.0:
                obs[3:6] += np.random.randn(3).astype(np.float32) * self._imu_noise_std_gyro
            if self._imu_include_quat:
                quat = np.array(self.data.sensordata[self._imu_quat_adr : self._imu_quat_adr + 4], dtype=np.float32)
                if self._imu_noise_std_quat > 0.0:
                    quat += np.random.randn(4).astype(np.float32) * self._imu_noise_std_quat
                # Renormalize noisy quaternion for physical consistency
                qn = np.linalg.norm(quat)
                if qn > 1e-8:
                    quat = quat / qn
                obs = np.concatenate([obs, quat], axis=0)
            if self._imu_include_prev_action:
                obs = np.concatenate([obs, self._prev_action.copy().astype(np.float32)], axis=0)
            if self._imu_include_phase:
                phase = 2 * np.pi * self.obs_phase_frequency_hz * self._sim_time
                obs = np.concatenate(
                    [obs, np.array([np.sin(phase), np.cos(phase)], dtype=np.float32)], axis=0
                )
            if self.obs_include_sim_time:
                obs = np.concatenate([obs, np.array([np.float32(self._sim_time)], dtype=np.float32)], axis=0)
            return obs
        if self.obs_use_sensors:
            return self.data.sensordata.copy().astype(np.float32)
        parts = []
        if self.obs_include_joint_pos:
            parts.append(self.data.qpos[7:7+self.num_joints].copy())
        if self.obs_include_joint_vel:
            parts.append(self.data.qvel[6:6+self.num_joints].copy())
        if self.obs_include_torso_quat:
            parts.append(self.data.qpos[3:7].copy())
        if self.obs_include_torso_vel:
            parts.append(self.data.qvel[0:3].copy())
        if self.obs_include_contact:
            parts.append(self._get_foot_contacts())
        if self.obs_include_torso_height:
            parts.append(np.array([self.data.qpos[2]], dtype=np.float32))
        if self.obs_include_phase:
            phase = 2 * np.pi * self.obs_phase_frequency_hz * self._sim_time
            parts.append(np.array([np.sin(phase), np.cos(phase)], dtype=np.float32))
        if self.obs_include_prev_action:
            parts.append(self._prev_action.copy())
        if self.obs_include_path_heading_error and self._path_active:
            torso_pos, torso_quat, _, _ = self._get_torso_state()
            desired_heading, _, _ = self._get_path_state(torso_pos)
            yaw = self._quat_to_yaw(torso_quat)
            heading_error = desired_heading - yaw
            heading_error = float(np.arctan2(np.sin(heading_error), np.cos(heading_error)))
            parts.append(np.array([heading_error], dtype=np.float32))
        if self.obs_include_sim_time:
            parts.append(np.array([np.float32(self._sim_time)], dtype=np.float32))
        if not parts:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(parts, axis=0).astype(np.float32)

    def _action_to_target_positions(self, action: np.ndarray) -> np.ndarray:
        """Map normalized actions [-1, 1] to target joint positions (radians)."""
        if self.action_use_joint_limits:
            lo = self.joint_ranges[:, 0]
            hi = self.joint_ranges[:, 1]
            return lo + (action + 1.0) * 0.5 * (hi - lo)
        return action * self._action_scale

    def _tripod_reference_action(self, sim_time: float) -> np.ndarray:
        """Compute a tripod CPG reference action for imitation."""
        frequency = self.imitation_frequency_hz
        theta = 2 * np.pi * frequency * sim_time
        theta_a = theta
        theta_b = theta + np.pi

        def flex_sine(phase):
            return self.imitation_flex_sign * self.imitation_amp_flex * np.sin(phase)

        flex_a = flex_sine(theta_a)
        flex_b = flex_sine(theta_b)

        def abd_smooth(phase):
            return self.imitation_abd_sign * -self.imitation_amp_abd * np.cos(phase)

        abd_a = abd_smooth(theta_a)
        abd_b = abd_smooth(theta_b)

        ctrl = np.zeros(12, dtype=np.float32)
        tripod_a_legs = [0, 3, 4]   # legs 1,4,5
        tripod_b_legs = [1, 2, 5]   # legs 2,3,6
        left_legs = [1, 3, 5]       # left side: 2,4,6
        for i in tripod_a_legs:
            ctrl[i * 2] = abd_a
            ctrl[i * 2 + 1] = flex_a
        for i in tripod_b_legs:
            ctrl[i * 2] = -abd_b if i in left_legs else abd_b
            ctrl[i * 2 + 1] = flex_b
        return np.clip(ctrl, -1.0, 1.0)
    
    def _generate_terrain_heights(self, terrain_kind: str) -> np.ndarray:
        """Generate heightfield data for flat, rough, or steps. Returns 1D array in MuJoCo data units (elevation = z_bias + z_scale * data)."""
        nrow, ncol = self._hfield_nrow, self._hfield_ncol
        z_scale = float(self._hfield_size[2])
        z_bias = float(self._hfield_size[3])
        if z_scale <= 0:
            z_scale = 0.05
        # Convert elevation in meters to hfield data: data = (elevation_m - z_bias) / z_scale
        def to_data(elev_m):
            return np.float32((elev_m - z_bias) / z_scale)
        out = np.zeros((nrow * ncol,), dtype=np.float32)
        if terrain_kind == "flat":
            return out
        if terrain_kind == "rough":
            scale = self._terrain_rough_scale
            lo, hi = self._terrain_rough_min_height, self._terrain_rough_max_height
            small_n = max(4, int(min(nrow, ncol) * scale))
            small = np.random.uniform(lo, hi, (small_n, small_n)).astype(np.float32)
            for i in range(nrow):
                for j in range(ncol):
                    si = (i / max(1, nrow - 1)) * (small_n - 1)
                    sj = (j / max(1, ncol - 1)) * (small_n - 1)
                    i0, j0 = int(np.clip(si, 0, small_n - 2)), int(np.clip(sj, 0, small_n - 2))
                    di, dj = si - i0, sj - j0
                    v = (1 - di) * (1 - dj) * small[i0, j0] + (1 - di) * dj * small[i0, j0 + 1]
                    v += di * (1 - dj) * small[i0 + 1, j0] + di * dj * small[i0 + 1, j0 + 1]
                    out[i * ncol + j] = to_data(v)
            return out
        if terrain_kind == "steps":
            step_h = self._terrain_step_height
            step_l = self._terrain_step_length
            num_steps = self._terrain_num_steps
            half_x = float(self._hfield_size[0])
            dx = 2.0 * half_x / ncol
            for i in range(nrow):
                for j in range(ncol):
                    x = -half_x + (j + 0.5) * dx
                    step_index = int(x / step_l) if step_l > 0 else 0
                    step_index = np.clip(step_index, 0, num_steps - 1)
                    out[i * ncol + j] = to_data(step_index * step_h)
            return out
        return out

    def _get_foot_contacts(self) -> np.ndarray:
        """Get binary contact states for each foot."""
        contacts = np.zeros(self.num_legs)
        
        # Check contacts from MuJoCo contact list
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            geom1_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1)
            geom2_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2)
            
            # Check if any foot is in contact with the floor (plane or hfield geom named "floor")
            for leg_idx in range(1, 7):
                foot_name = f"leg{leg_idx}_foot_geom"
                if foot_name in [geom1_name, geom2_name]:
                    if "floor" in [geom1_name, geom2_name]:
                        contacts[leg_idx - 1] = 1.0
        
        return contacts
    
    def _get_torso_state(self):
        """Get torso position, orientation, and velocity."""
        torso_pos = self.data.qpos[0:3].copy()
        torso_quat = self.data.qpos[3:7].copy()
        torso_lin_vel = self.data.qvel[0:3].copy()
        torso_ang_vel = self.data.qvel[3:6].copy()
        return torso_pos, torso_quat, torso_lin_vel, torso_ang_vel

    @staticmethod
    def _quat_to_yaw(quat: np.ndarray) -> float:
        """Extract yaw (rotation around world Z) from quaternion (w, x, y, z)."""
        w, x, y, z = quat[0], quat[1], quat[2], quat[3]
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return float(np.arctan2(siny_cosp, cosy_cosp))

    def _build_circle_waypoints(self, num_points: int = 64) -> list:
        """Build waypoints along a circle for path_type='circle'."""
        cx, cy = self.path_center[0], self.path_center[1]
        R = self.path_radius
        waypoints = []
        for i in range(num_points):
            t = 2 * np.pi * i / num_points
            waypoints.append((cx + R * np.cos(t), cy + R * np.sin(t)))
        return waypoints

    def _get_path_state(self, torso_pos: np.ndarray) -> tuple:
        """
        Compute desired heading (world frame) and distance to path.
        Returns (desired_heading_rad, distance_to_path, tangent_heading_rad).
        Updates _path_waypoint_idx when within path_waypoint_reach_dist.
        """
        if not self._path_active or not self._path_waypoints:
            return 0.0, 0.0, 0.0
        x, y = float(torso_pos[0]), float(torso_pos[1])
        n = len(self._path_waypoints)
        idx = self._path_waypoint_idx % n
        wx, wy = self._path_waypoints[idx][0], self._path_waypoints[idx][1]
        dx, dy = wx - x, wy - y
        dist_to_waypoint = np.sqrt(dx * dx + dy * dy)
        if dist_to_waypoint < 1e-6:
            desired_heading = 0.0
        else:
            desired_heading = float(np.arctan2(dy, dx))
        if dist_to_waypoint < self.path_waypoint_reach_dist:
            self._path_waypoint_idx = (idx + 1) % n
        next_idx = self._path_waypoint_idx % n
        nwx, nwy = self._path_waypoints[next_idx][0], self._path_waypoints[next_idx][1]
        tangent_dx, tangent_dy = nwx - wx, nwy - wy
        tangent_norm = np.sqrt(tangent_dx * tangent_dx + tangent_dy * tangent_dy)
        if tangent_norm < 1e-9:
            tangent_heading = desired_heading
        else:
            tangent_heading = float(np.arctan2(tangent_dy, tangent_dx))
        dist_to_path = dist_to_waypoint
        return desired_heading, dist_to_path, tangent_heading

    def _compute_reward(self, action: np.ndarray, raw_action: np.ndarray | None = None) -> tuple:
        """Compute reward based on forward distance, velocity, and optional smoothness/stability."""
        torso_pos, torso_quat, torso_lin_vel, torso_ang_vel = self._get_torso_state()
        
        if self._initial_torso_x is None:
            self._initial_torso_x = torso_pos[0]
        
        forward_distance = torso_pos[0] - self._initial_torso_x
        forward_velocity = torso_lin_vel[0]
        self._velocity_window.append(float(forward_velocity))
        avg_forward_velocity = float(np.mean(self._velocity_window)) if self._velocity_window else float(forward_velocity)
        
        distance_reward = self.forward_distance_weight * forward_distance
        if self.velocity_reward_uncapped:
            velocity_reward = self.forward_velocity_weight * avg_forward_velocity
        else:
            velocity_reward = self.forward_velocity_weight * max(0, avg_forward_velocity)
        target_velocity_reward = 0.0
        if self.target_forward_velocity > 0 and self.target_velocity_weight > 0:
            scale = max(1e-6, self.target_velocity_scale)
            target_velocity_reward = self.target_velocity_weight * np.exp(
                -0.5 * ((avg_forward_velocity - self.target_forward_velocity) / scale) ** 2
            )
        survival_bonus = self.survival_bonus
        backward_penalty = self.backward_penalty_weight * min(0, avg_forward_velocity)
        stall_penalty = self.stall_penalty_weight if avg_forward_velocity < self.stall_penalty_threshold else 0.0
        control_cost = self.control_cost_weight * float(np.sum(np.square(action)))
        lateral_velocity = torso_lin_vel[1]
        lateral_penalty = self.lateral_velocity_penalty_weight * abs(lateral_velocity)
        # Yaw rate (angular velocity around world Z): penalize turning to encourage straight-line walking
        yaw_rate = torso_ang_vel[2]
        yaw_rate_penalty = self.yaw_rate_penalty_weight * abs(yaw_rate)
        height_penalty = -self.height_penalty_weight if torso_pos[2] < self.height_penalty_threshold else 0.0
        # When not terminating on fall, penalize being below min height so the agent still learns to avoid it
        fall_penalty = 0.0
        if not self.terminate_on_fall and self.fall_penalty_weight > 0 and torso_pos[2] < self.min_torso_height:
            fall_penalty = self.fall_penalty_weight
        
        # Gradual tripod: reward smooth action changes (small delta = good)
        action_delta_sq = np.sum((action - self._prev_action) ** 2)
        smoothness_reward = -self.smoothness_reward_weight * action_delta_sq
        
        # Stability: reward keeping torso height near target (gradual, differentiable)
        height_error = torso_pos[2] - self.target_torso_height
        stability_reward = self.stability_reward_weight * np.exp(-10.0 * height_error ** 2)

        # Tilt: continuous penalty so agent learns to stay upright without terminating (avoids "tilt death trap")
        tilt_angle = 2 * np.arccos(np.clip(np.abs(torso_quat[0]), 0, 1))
        tilt_penalty = self.tilt_penalty_weight * tilt_angle

        # Near-termination: soft penalty when close to tilt/flexion limits
        near_termination_penalty = 0.0
        if self.near_termination_penalty_weight > 0:
            tilt_limit_rad = self.max_torso_tilt
            if tilt_angle > tilt_limit_rad - self.near_termination_margin_rad:
                excess = (tilt_angle - (tilt_limit_rad - self.near_termination_margin_rad)) / self.near_termination_margin_rad
                near_termination_penalty += self.near_termination_penalty_weight * min(1.0, excess)
            flexion = self.data.qpos[7 + np.arange(6) * 2 + 1]
            max_flex = np.max(np.abs(flexion))
            flex_limit_rad = self.max_flexion_angle
            if max_flex > flex_limit_rad - self.near_termination_margin_rad:
                excess = (max_flex - (flex_limit_rad - self.near_termination_margin_rad)) / self.near_termination_margin_rad
                near_termination_penalty += self.near_termination_penalty_weight * min(1.0, excess)
        idle_penalty = 0.0
        if abs(forward_velocity) < self.idle_velocity_threshold:
            self._consecutive_idle_steps += 1
            if self.idle_velocity_penalty_weight > 0:
                ramp = 1.0
                if self.idle_penalty_ramp_weight > 0:
                    ramp = min(
                        self.idle_penalty_ramp_max,
                        1.0 + self.idle_penalty_ramp_weight * self._consecutive_idle_steps,
                    )
                idle_penalty = self.idle_velocity_penalty_weight * ramp
        else:
            self._consecutive_idle_steps = 0

        imitation_penalty = 0.0
        if self.imitation_weight > 0:
            ref_action = self._tripod_reference_action(self._sim_time)
            penalty_action = raw_action if raw_action is not None else action
            imitation_penalty = np.mean((penalty_action - ref_action) ** 2)

        path_heading_reward = 0.0
        path_proximity_penalty = 0.0
        if self._path_active and (self.path_heading_reward_weight > 0 or self.path_proximity_reward_weight > 0):
            desired_heading, dist_to_path, _ = self._get_path_state(torso_pos)
            yaw = self._quat_to_yaw(torso_quat)
            heading_error = desired_heading - yaw
            heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))
            speed = np.sqrt(torso_lin_vel[0] ** 2 + torso_lin_vel[1] ** 2)
            if self.path_heading_reward_weight > 0 and speed > 1e-4:
                path_heading_reward = self.path_heading_reward_weight * np.cos(heading_error) * min(1.0, speed / 0.1)
            else:
                path_heading_reward = self.path_heading_reward_weight * np.cos(heading_error)
            if self.path_proximity_reward_weight > 0:
                excess = max(0.0, dist_to_path - self.path_proximity_tolerance)
                path_proximity_penalty = self.path_proximity_reward_weight * excess

        jerk_penalty = 0.0
        if self.jerk_penalty_weight > 0:
            jerk_vec = action - 2.0 * self._prev_action + self._prev_prev_action
            jerk_sq = float(np.sum(jerk_vec ** 2))
            jerk_penalty = self.jerk_penalty_weight * jerk_sq

        joint_velocity_penalty = 0.0
        if self.joint_velocity_penalty_weight > 0:
            joint_vel = self.data.qvel[6 : 6 + self.num_joints]
            joint_velocity_penalty = self.joint_velocity_penalty_weight * float(np.sum(joint_vel ** 2))

        stance_stability_penalty = 0.0
        stance_action_penalty = 0.0
        if self.stance_stability_penalty_weight > 0 or self.stance_action_penalty_weight > 0:
            contact = self._get_foot_contacts()
            joint_vel = self.data.qvel[6 : 6 + self.num_joints]
            for L in range(self.num_legs):
                if contact[L] > 0.5:
                    if self.stance_stability_penalty_weight > 0:
                        stance_stability_penalty += self.stance_stability_penalty_weight * (
                            joint_vel[2 * L] ** 2 + joint_vel[2 * L + 1] ** 2
                        )
                    if self.stance_action_penalty_weight > 0:
                        stance_action_penalty += self.stance_action_penalty_weight * (
                            (action[2 * L] - self._prev_action[2 * L]) ** 2
                            + (action[2 * L + 1] - self._prev_action[2 * L + 1]) ** 2
                        )

        total_reward = (
            distance_reward + velocity_reward + target_velocity_reward + survival_bonus
            + backward_penalty - lateral_penalty - yaw_rate_penalty + height_penalty
            + smoothness_reward + stability_reward
            - near_termination_penalty
            - tilt_penalty
            - idle_penalty
            - stall_penalty
            - control_cost
            - fall_penalty
            - self.imitation_weight * imitation_penalty
            + path_heading_reward
            - path_proximity_penalty
            - jerk_penalty
            - joint_velocity_penalty
            - stance_stability_penalty
            - stance_action_penalty
        )

        reward_info = {
            "distance_reward": distance_reward,
            "velocity_reward": velocity_reward,
            "target_velocity_reward": target_velocity_reward,
            "path_heading_reward": path_heading_reward,
            "path_proximity_penalty": path_proximity_penalty,
            "jerk_penalty": jerk_penalty,
            "joint_velocity_penalty": joint_velocity_penalty,
            "stance_stability_penalty": stance_stability_penalty,
            "stance_action_penalty": stance_action_penalty,
            "survival_bonus": survival_bonus,
            "backward_penalty": backward_penalty,
            "lateral_penalty": lateral_penalty,
            "yaw_rate_penalty": yaw_rate_penalty,
            "tilt_penalty": tilt_penalty,
            "stall_penalty": stall_penalty,
            "control_cost": control_cost,
            "height_penalty": height_penalty,
            "fall_penalty": fall_penalty,
            "smoothness_reward": smoothness_reward,
            "stability_reward": stability_reward,
            "near_termination_penalty": near_termination_penalty,
            "idle_penalty": idle_penalty,
            "imitation_penalty": imitation_penalty,
            "forward_distance": forward_distance,
            "forward_velocity": forward_velocity,
            "avg_forward_velocity": avg_forward_velocity,
            "torso_height": torso_pos[2],
        }
        
        return total_reward, reward_info
    
    def _check_termination(self) -> tuple:
        """Check if episode should terminate (respects terminate_on_* flags for lazy-agent mode)."""
        torso_pos, torso_quat, _, _ = self._get_torso_state()
        
        # Check torso height (fallen)
        if torso_pos[2] < self.min_torso_height and self.terminate_on_fall:
            _dbg_log("envs/hexapod_env.py:termination", "terminated_fallen", {
                "step": self._step_count,
                "torso_height": float(torso_pos[2]),
                "min_torso_height": float(self.min_torso_height),
            }, "H5")
            return True, {"termination_reason": "fallen"}
        
        # Check torso tilt (using quaternion)
        tilt_angle = 2 * np.arccos(np.clip(np.abs(torso_quat[0]), 0, 1))
        if tilt_angle > self.max_torso_tilt and self.terminate_on_tilt:
            _dbg_log("envs/hexapod_env.py:termination", "terminated_tilted", {
                "step": self._step_count,
                "tilt_angle_rad": float(tilt_angle),
                "max_torso_tilt_rad": float(self.max_torso_tilt),
            }, "H5")
            return True, {"termination_reason": "tilted"}
        # Check leg flexion
        flexion = self.data.qpos[7 + np.arange(6) * 2 + 1]
        if np.any(np.abs(flexion) > self.max_flexion_angle) and self.terminate_on_leg_up:
            _dbg_log("envs/hexapod_env.py:termination", "terminated_leg_up", {
                "step": self._step_count,
                "max_flexion_abs_rad": float(np.max(np.abs(flexion))),
                "max_flexion_angle_rad": float(self.max_flexion_angle),
            }, "H5")
            return True, {"termination_reason": "leg_up"}
        return False, {}
    
    def _check_truncation(self) -> bool:
        """Check if episode should be truncated (max steps)."""
        return self._step_count >= self._max_episode_steps

    def set_terrain_curriculum_stage(self, stage: int) -> None:
        """Set curriculum stage for terrain (0=flat, 1=flat+rough, 2=flat+rough+steps). Used by TerrainCurriculumCallback."""
        self._terrain_curriculum_stage = max(0, min(2, int(stage)))
    
    def reset(self, seed=None, options=None):
        """Reset the environment to initial state. options may include terrain_kind: 'flat'|'rough'|'steps' to force terrain for this episode (e.g. for eval videos)."""
        super().reset(seed=seed)
        
        # Reset MuJoCo state
        mujoco.mj_resetData(self.model, self.data)
        
        # Terrain: overwrite heightfield data for this episode
        if self._use_terrain_hfield and self._terrain_type is not None:
            terrain_kind = None
            if options and isinstance(options, dict) and "terrain_kind" in options:
                terrain_kind = options["terrain_kind"]
                if terrain_kind not in ("flat", "rough", "steps"):
                    terrain_kind = None
            if terrain_kind is None:
                terrain_kind = self._terrain_type
                if terrain_kind == "random" or self._terrain_curriculum_stage is not None:
                    if self._terrain_curriculum_stage is not None:
                        allowed = [["flat"], ["flat", "rough"], ["flat", "rough", "steps"]][
                            min(self._terrain_curriculum_stage, 2)
                        ]
                        terrain_kind = np.random.choice(allowed)
                    else:
                        terrain_kind = np.random.choice(["flat", "rough", "steps"])
            heights = self._generate_terrain_heights(terrain_kind)
            adr = self._hfield_adr
            end = adr + self._hfield_data_len
            self.model.hfield_data[adr:end] = heights
            # Force renderer to be recreated so it picks up new terrain geometry
            self._terrain_changed = True
        
        # Add small random noise to initial joint positions
        if seed is not None:
            np.random.seed(seed)
        
        # Set initial position slightly above ground
        self.data.qpos[2] = 0.10  # torso height
        
        # Random perturbation to joint positions (larger scale can force initial movement)
        noise_scale = self.reset_joint_noise_scale
        self.data.qpos[7:7+self.num_joints] += np.random.uniform(
            -noise_scale, noise_scale, self.num_joints
        )
        if self.reset_initial_forward_velocity != 0.0:
            self.data.qvel[0] = self.reset_initial_forward_velocity
        
        # Forward simulate to settle
        mujoco.mj_forward(self.model, self.data)
        
        # Reset state tracking
        self._prev_action = np.zeros(self.num_actuators)
        self._prev_prev_action = np.zeros(self.num_actuators)
        self._prev_torso_pos = self.data.qpos[0:3].copy()
        self._initial_torso_x = self.data.qpos[0].copy()  # Track initial X for distance reward
        self._step_count = 0
        self._sim_time = 0.0
        self._velocity_window.clear()
        self._consecutive_idle_steps = 0
        self._path_waypoint_idx = 0

        obs = self._get_obs()
        info = {"initial_height": self.data.qpos[2]}

        return obs, info

    def step(self, action: np.ndarray):
        """Execute one environment step."""
        action = np.clip(action, -1.0, 1.0)
        raw_action = action.copy()
        ref_action = None
        # #region agent log
        if self._step_count % 50 == 0:
            _dbg_log("envs/hexapod_env.py:step:raw_action", "raw_action_in", {
                "step": self._step_count,
                "raw_min": float(np.min(raw_action)),
                "raw_max": float(np.max(raw_action)),
                "raw_mean": float(np.mean(raw_action)),
                "imitation_action_mix": float(self.imitation_action_mix),
            }, "H1")
        # #endregion
        if self.imitation_action_mix > 0.0:
            ref_action = self._tripod_reference_action(self._sim_time)
            mix = float(np.clip(self.imitation_action_mix, 0.0, 1.0))
            action = np.clip((1.0 - mix) * action + mix * ref_action, -1.0, 1.0)
        # #region agent log
        if self._step_count % 50 == 0:
            _dbg_log("envs/hexapod_env.py:step:action", "action_in", {
                "step": self._step_count,
                "action_min": float(np.min(action)),
                "action_max": float(np.max(action)),
                "action_mean": float(np.mean(action)),
                "use_pd_control": self.use_pd_control,
                "imitation_action_mix": float(self.imitation_action_mix),
                "ref_min": float(np.min(ref_action)) if ref_action is not None else None,
                "ref_max": float(np.max(ref_action)) if ref_action is not None else None,
            }, "H1")
        # #endregion
        # Gradual tripod: blend with previous action for smoother back-and-forth
        if self.action_smooth_alpha > 0:
            applied = (1.0 - self.action_smooth_alpha) * action + self.action_smooth_alpha * self._prev_action
            applied = np.clip(applied, -1.0, 1.0)
        else:
            applied = action
        # Ramp up action over first N steps so legs do not slam into each other at start
        if self.action_warmup_steps > 0 and self._step_count < self.action_warmup_steps:
            scale = (self._step_count + 1) / self.action_warmup_steps
            applied = applied * scale
        # Action change deadzone: zero out tiny changes (suppress servo-impossible micro-wiggles)
        delta = applied - self._prev_action
        if self.action_change_deadzone > 0:
            delta = np.where(np.abs(delta) < self.action_change_deadzone, 0.0, delta)
            applied = np.clip(self._prev_action + delta, -1.0, 1.0)
            delta = applied - self._prev_action
        # Action rate limit: cap change per step (servo max speed)
        if self.action_rate_limit > 0:
            delta = np.clip(delta, -self.action_rate_limit, self.action_rate_limit)
            applied = np.clip(self._prev_action + delta, -1.0, 1.0)
        # #region agent log
        if self._step_count % 50 == 0:
            _dbg_log("envs/hexapod_env.py:step:applied", "applied_action", {
                "step": self._step_count,
                "applied_min": float(np.min(applied)),
                "applied_max": float(np.max(applied)),
                "applied_mean": float(np.mean(applied)),
                "warmup_steps": int(self.action_warmup_steps),
                "smooth_alpha": float(self.action_smooth_alpha),
            }, "H1")
        # #endregion
        if self.use_pd_control:
            target_pos = self._action_to_target_positions(applied)
            ctrlrange = self.model.actuator_ctrlrange
            # #region agent log
            if self._step_count % 50 == 0:
                _dbg_log("envs/hexapod_env.py:step:pd", "pd_targets", {
                    "step": self._step_count,
                    "target_min": float(np.min(target_pos)),
                    "target_max": float(np.max(target_pos)),
                    "target_mean": float(np.mean(target_pos)),
                    "ctrlrange_min": float(ctrlrange[:, 0].min()) if ctrlrange is not None else None,
                    "ctrlrange_max": float(ctrlrange[:, 1].max()) if ctrlrange is not None else None,
                }, "H2")
            # #endregion
            for _ in range(self.pd_steps_per_action):
                current_pos = self.data.qpos[self.joint_qpos_addr]
                current_vel = self.data.qvel[self.joint_dof_addr]
                torques = self.pd_kp * (target_pos - current_pos) - self.pd_kd * current_vel
                if ctrlrange is not None and ctrlrange.shape[0] == self.num_actuators:
                    torques = np.clip(torques, ctrlrange[:, 0], ctrlrange[:, 1])
                self.data.ctrl[:] = torques
                mujoco.mj_step(self.model, self.data)
            # #region agent log
            if self._step_count % 50 == 0:
                _dbg_log("envs/hexapod_env.py:step:torque", "pd_torques", {
                    "step": self._step_count,
                    "torque_min": float(np.min(torques)),
                    "torque_max": float(np.max(torques)),
                    "torque_mean": float(np.mean(torques)),
                }, "H2")
            # #endregion
        else:
            self.data.ctrl[:] = applied
            # Step simulation
            for _ in range(self.frame_skip):
                mujoco.mj_step(self.model, self.data)
        
        # Get observation
        obs = self._get_obs()
        # #region agent log
        if self._step_count % 50 == 0 and isinstance(obs, np.ndarray) and obs.size > 0:
            _dbg_log("envs/hexapod_env.py:step:obs", "obs_stats", {
                "step": self._step_count,
                "obs_mean": float(np.mean(obs)),
                "obs_std": float(np.std(obs)),
                "obs_min": float(np.min(obs)),
                "obs_max": float(np.max(obs)),
            }, "H4")
        # #endregion
        
        # Compute reward
        reward, reward_info = self._compute_reward(applied, raw_action=raw_action)
        # #region agent log
        if self._step_count % 50 == 0:
            _dbg_log("envs/hexapod_env.py:step:reward", "reward_metrics", {
                "step": self._step_count,
                "reward": float(reward),
                "forward_velocity": float(reward_info.get("forward_velocity", 0.0)),
                "forward_distance": float(reward_info.get("forward_distance", 0.0)),
                "idle_penalty": float(reward_info.get("idle_penalty", 0.0)),
                "imitation_penalty": float(reward_info.get("imitation_penalty", 0.0)),
            }, "H3")
        # #endregion
        
        # Check termination and truncation
        terminated, term_info = self._check_termination()
        truncated = self._check_truncation()
        
        # Update state tracking (store applied so smoothing and smoothness reward use same reference)
        self._prev_prev_action = self._prev_action.copy()
        self._prev_action = applied.copy()
        self._prev_torso_pos = self.data.qpos[0:3].copy()
        self._step_count += 1
        # PD control uses pd_steps_per_action substeps per RL step; match sim_time to physics
        if self.use_pd_control:
            self._sim_time += self.pd_steps_per_action * self.model.opt.timestep
        else:
            self._sim_time += self.frame_skip * self.model.opt.timestep
        # #region agent log
        if self._step_count % 50 == 0:
            _dbg_log("envs/hexapod_env.py:step:state", "state_post_step", {
                "step": self._step_count,
                "torso_height": float(self.data.qpos[2]),
                "torso_vel_x": float(self.data.qvel[0]),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
            }, "H4")
        # #endregion
        
        # Combine info
        info = {**reward_info, **term_info, "step": self._step_count}
        
        return obs, reward, terminated, truncated, info
    
    def render(self):
        """Render the environment."""
        if self.render_mode is None:
            return None
        
        # Recreate renderer after terrain change so heightfield geometry is updated
        if getattr(self, "_terrain_changed", False):
            if self.renderer is not None:
                self.renderer.close()
                self.renderer = None
            self._terrain_changed = False
        if self.renderer is None:
            from mujoco import Renderer
            self.renderer = Renderer(self.model, height=480, width=640)
        cam_id = getattr(self, "_eval_cam_id", -1)
        self.renderer.update_scene(self.data, camera=cam_id if cam_id >= 0 else -1)
        
        if self.render_mode == "rgb_array":
            return self.renderer.render()
        elif self.render_mode == "human":
            import cv2
            img = self.renderer.render()
            cv2.imshow("Hexapod", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)
            return img
    
    def close(self):
        """Clean up resources."""
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None


def make_hexapod_env(config: dict = None, render_mode: str = None) -> HexapodEnv:
    """Factory function to create HexapodEnv from config dict."""
    if config is None:
        config = {}

    env_config = config.get("env", {})
    reward_config = config.get("reward", {})

    return HexapodEnv(
        xml_path=env_config.get("xml_path"),
        frame_skip=env_config.get("frame_skip", 5),
        max_episode_steps=env_config.get("max_episode_steps", 1000),
        render_mode=render_mode,
        obs_include_joint_pos=env_config.get("obs_include_joint_pos", True),
        obs_include_joint_vel=env_config.get("obs_include_joint_vel", True),
        obs_include_torso_quat=env_config.get("obs_include_torso_quat", True),
        obs_include_torso_vel=env_config.get("obs_include_torso_vel", False),
        obs_include_contact=env_config.get("obs_include_contact", False),
        obs_include_prev_action=env_config.get("obs_include_prev_action", False),
        obs_include_torso_height=env_config.get("obs_include_torso_height", False),
        obs_use_sensors=env_config.get("obs_use_sensors", False),
        obs_include_phase=env_config.get("obs_include_phase", False),
        obs_phase_frequency_hz=env_config.get("obs_phase_frequency_hz", 0.9),
        obs_include_sim_time=env_config.get("obs_include_sim_time", False),
        min_torso_height=env_config.get("min_torso_height", 0.05),
        max_torso_tilt=env_config.get("max_torso_tilt", 35.0),
        max_flexion_angle=env_config.get("max_flexion_angle", 55.0),
        terminate_on_fall=env_config.get("terminate_on_fall", True),
        terminate_on_tilt=env_config.get("terminate_on_tilt", True),
        terminate_on_leg_up=env_config.get("terminate_on_leg_up", True),
        fall_penalty_weight=reward_config.get("fall_penalty_weight", 0.0),
        action_smooth_alpha=env_config.get("action_smooth_alpha", 0.0),
        smoothness_reward_weight=reward_config.get("smoothness_reward_weight", 0.0),
        stability_reward_weight=reward_config.get("stability_reward_weight", 0.0),
        target_torso_height=env_config.get("target_torso_height", 0.10),
        action_warmup_steps=env_config.get("action_warmup_steps", 0),
        near_termination_penalty_weight=reward_config.get("near_termination_penalty_weight", 0.0),
        near_termination_margin_deg=reward_config.get("near_termination_margin_deg", 10.0),
        forward_distance_weight=reward_config.get("forward_distance_weight", 15.0),
        forward_velocity_weight=reward_config.get("forward_velocity_weight", 10.0),
        idle_velocity_threshold=reward_config.get("idle_velocity_threshold", 0.02),
        idle_velocity_penalty_weight=reward_config.get("idle_velocity_penalty_weight", 0.0),
        velocity_reward_window=reward_config.get("velocity_reward_window", 1),
        survival_bonus=reward_config.get("survival_bonus", 0.01),
        backward_penalty_weight=reward_config.get("backward_penalty_weight", 2.0),
        lateral_velocity_penalty_weight=reward_config.get("lateral_velocity_penalty_weight", 0.0),
        stall_penalty_threshold=reward_config.get("stall_penalty_threshold", 0.1),
        stall_penalty_weight=reward_config.get("stall_penalty_weight", 0.0),
        control_cost_weight=reward_config.get("control_cost_weight", 0.0),
        velocity_reward_uncapped=reward_config.get("velocity_reward_uncapped", False),
        height_penalty_weight=reward_config.get("height_penalty_weight", 1.0),
        height_penalty_threshold=reward_config.get("height_penalty_threshold", 0.06),
        tilt_penalty_weight=reward_config.get("tilt_penalty_weight", 0.0),
        yaw_rate_penalty_weight=reward_config.get("yaw_rate_penalty_weight", 0.0),
        imitation_weight=reward_config.get("imitation_weight", 0.0),
        imitation_frequency_hz=reward_config.get("imitation_frequency_hz", 0.9),
        imitation_amp_flex=reward_config.get("imitation_amp_flex", 0.75),
        imitation_amp_abd=reward_config.get("imitation_amp_abd", 0.5),
        imitation_action_mix=reward_config.get("imitation_action_mix", 0.0),
        imitation_flex_sign=reward_config.get("imitation_flex_sign", 1.0),
        imitation_abd_sign=reward_config.get("imitation_abd_sign", 1.0),
        use_pd_control=env_config.get("use_pd_control", False),
        pd_kp=env_config.get("pd_kp", 5.0),
        pd_kd=env_config.get("pd_kd", 0.5),
        pd_steps_per_action=env_config.get("pd_steps_per_action", 5),
        action_use_joint_limits=env_config.get("action_use_joint_limits", True),
        action_scale_abd_deg=env_config.get("action_scale_abd_deg", 45.0),
        action_scale_flex_deg=env_config.get("action_scale_flex_deg", 90.0),
        reset_joint_noise_scale=env_config.get("reset_joint_noise_scale", 0.05),
        reset_initial_forward_velocity=env_config.get("reset_initial_forward_velocity", 0.0),
        target_forward_velocity=reward_config.get("target_forward_velocity", 0.0),
        target_velocity_weight=reward_config.get("target_velocity_weight", 0.0),
        target_velocity_scale=reward_config.get("target_velocity_scale", 0.1),
        path_type=env_config.get("path_type"),
        path_waypoints=env_config.get("path_waypoints"),
        path_radius=env_config.get("path_radius", 0.5),
        path_center=tuple(env_config.get("path_center", [0.0, 0.0])),
        path_waypoint_reach_dist=env_config.get("path_waypoint_reach_dist", 0.15),
        path_heading_reward_weight=reward_config.get("path_heading_reward_weight", 0.0),
        path_proximity_reward_weight=reward_config.get("path_proximity_reward_weight", 0.0),
        path_proximity_tolerance=reward_config.get("path_proximity_tolerance", 0.1),
        obs_include_path_heading_error=env_config.get("obs_include_path_heading_error", False),
        jerk_penalty_weight=reward_config.get("jerk_penalty_weight", 0.0),
        joint_velocity_penalty_weight=reward_config.get("joint_velocity_penalty_weight", 0.0),
        action_rate_limit=env_config.get("action_rate_limit", 0.0),
        action_change_deadzone=env_config.get("action_change_deadzone", 0.0),
        stance_stability_penalty_weight=reward_config.get("stance_stability_penalty_weight", 0.0),
        stance_action_penalty_weight=reward_config.get("stance_action_penalty_weight", 0.0),
        terrain_type=env_config.get("terrain_type"),
        terrain_rough_scale=env_config.get("terrain_rough_scale", 0.4),
        terrain_rough_min_height=env_config.get("terrain_rough_min_height", -0.02),
        terrain_rough_max_height=env_config.get("terrain_rough_max_height", 0.02),
        terrain_step_height=env_config.get("terrain_step_height", 0.02),
        terrain_step_length=env_config.get("terrain_step_length", 0.08),
        terrain_num_steps=env_config.get("terrain_num_steps", 5),
        observation_mode=env_config.get("observation_mode", "full"),
        imu_noise_std_accel=env_config.get("imu_noise_std_accel", 0.0),
        imu_noise_std_gyro=env_config.get("imu_noise_std_gyro", 0.0),
        imu_noise_std_quat=env_config.get("imu_noise_std_quat", 0.0),
    )


# Register environment with Gymnasium
gym.register(
    id="Hexapod-v0",
    entry_point="envs.hexapod_env:HexapodEnv",
    max_episode_steps=1000,
)
