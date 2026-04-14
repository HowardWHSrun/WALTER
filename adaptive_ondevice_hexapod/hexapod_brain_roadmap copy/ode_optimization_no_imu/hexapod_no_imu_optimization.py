from __future__ import annotations

"""
No-IMU hexapod gait optimization.

Primary goal: maximize mean forward velocity (steady-state segment of the ODE solution).
Secondary: keep roll/pitch/z oscillations and servo load within reasonable bounds.
"""

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import Bounds, minimize
from scipy.signal import periodogram


LEG_ORDER = ("FL", "FR", "ML", "MR", "RL", "RR")
TRIPOD_PHASE = {
    "FL": 0.0,
    "MR": 0.0,
    "RL": 0.0,
    "FR": np.pi,
    "ML": np.pi,
    "RR": np.pi,
}
BODY_X_M = {
    "FL": 0.105,
    "FR": 0.105,
    "ML": 0.0,
    "MR": 0.0,
    "RL": -0.105,
    "RR": -0.105,
}
BODY_Y_M = {
    "FL": 0.075,
    "FR": -0.075,
    "ML": 0.085,
    "MR": -0.085,
    "RL": 0.075,
    "RR": -0.075,
}
DEFAULT_ABD_REST_DEG = np.array([10.0, 10.0, 0.0, 0.0, -10.0, -10.0])
DEFAULT_FLX_REST_DEG = np.array([32.0, 32.0, 34.0, 34.0, 32.0, 32.0])
DEFAULT_ABD_AMP_DEG = np.array([14.0, 14.0, 10.0, 10.0, 14.0, 14.0])
DEFAULT_FLX_AMP_DEG = np.array([18.0, 18.0, 20.0, 20.0, 18.0, 18.0])
DEFAULT_ABD_SHIFT_DEG = np.zeros(6)
DEFAULT_FLX_SHIFT_DEG = np.full(6, -90.0)

# Swing-phase foot clearance (like v2 discrete lift): subtract flex when sin(abd_phase) > 0.
# Peak reduction in flex *degrees* at sin==1. Set to 0.0 to disable. Match OPT_LIFT_SWING_DEG_PEAK in firmware.
LIFT_SWING_DEG_PEAK = 12.0


@dataclass(frozen=True)
class RobotSpec:
    servo_mass_kg: float = 0.009
    servo_count: int = 12
    chassis_mass_kg: float = 0.24
    battery_and_electronics_mass_kg: float = 0.11
    sg90_stall_torque_nm: float = 0.176
    support_gain_n_per_rad: float = 32.0
    support_damping_n_per_mps: float = 7.5
    ground_stiffness_n_per_m: float = 180.0
    stride_gain_n_per_rad_per_s: float = 0.34
    contact_sharpness: float = 5.5
    body_longitudinal_damping_n_per_mps: float = 6.0
    body_vertical_damping_n_per_mps: float = 8.0
    roll_damping_nm_per_rad_s: float = 0.06
    pitch_damping_nm_per_rad_s: float = 0.07
    roll_inertia_kgm2: float = 0.0034
    pitch_inertia_kgm2: float = 0.0048
    torque_proxy_gain_nm_per_rad: float = 0.058

    @property
    def total_mass_kg(self) -> float:
        return (
            self.chassis_mass_kg
            + self.battery_and_electronics_mass_kg
            + self.servo_mass_kg * self.servo_count
        )


@dataclass(frozen=True)
class OptimizationConfig:
    """Tuning knobs for the optimizer."""

    # Primary: cost includes -forward_speed_weight * mean_forward_speed_mps (maximize forward speed).
    forward_speed_weight: float = 1000.0
    # Strong penalty if mean forward speed is negative (net backward motion).
    backward_motion_penalty_weight: float = 50000.0
    # Secondary stability / actuator terms (keep small relative to forward_speed_weight * typical speed).
    roll_pitch_penalty_weight: float = 25.0
    vertical_penalty_weight: float = 35.0
    servo_overuse_penalty_weight: float = 25.0
    contact_balance_penalty_weight: float = 2.0
    regularization_weight: float = 0.3
    # Optional soft floor: extra penalty if mean speed stays below this (pushes exploration upward).
    target_forward_speed_mps: float = 0.0
    target_speed_floor_penalty_weight: float = 200.0
    # Legacy field kept for summary / CLI compatibility (not a primary objective term).
    target_resonant_frequency_hz: float = 1.8
    sim_duration_s: float = 8.0
    sample_count: int = 800
    restarts: int = 8
    seed: int = 7
    maxiter: int = 180
    # Upper bound on objective evaluations per restart (each eval runs solve_ivp).
    max_fun: int = 1200


def default_parameter_vector() -> np.ndarray:
    return np.concatenate(
        [
            DEFAULT_ABD_REST_DEG,
            DEFAULT_FLX_REST_DEG,
            DEFAULT_ABD_SHIFT_DEG,
            DEFAULT_FLX_SHIFT_DEG,
            np.array([1.8]),
        ]
    )


def parameter_bounds() -> Bounds:
    lower = np.concatenate(
        [
            np.full(6, -25.0),
            np.full(6, 10.0),
            np.full(6, -120.0),
            np.full(6, -120.0),
            np.array([0.4]),
        ]
    )
    upper = np.concatenate(
        [
            np.full(6, 25.0),
            np.full(6, 65.0),
            np.full(6, 120.0),
            np.full(6, 120.0),
            np.array([4.0]),
        ]
    )
    return Bounds(lower, upper)


def unpack_parameters(params: np.ndarray) -> dict[str, np.ndarray | float]:
    abd_rest_deg = params[0:6]
    flx_rest_deg = params[6:12]
    abd_shift_deg = params[12:18]
    flx_shift_deg = params[18:24]
    frequency_hz = float(params[24])
    return {
        "abd_rest_rad": np.deg2rad(abd_rest_deg),
        "flx_rest_rad": np.deg2rad(flx_rest_deg),
        "abd_shift_rad": np.deg2rad(abd_shift_deg),
        "flx_shift_rad": np.deg2rad(flx_shift_deg),
        "frequency_hz": frequency_hz,
    }


def contact_weights(flx_cmd: np.ndarray, flx_rest: np.ndarray, spec: RobotSpec) -> np.ndarray:
    amplitude = np.deg2rad(DEFAULT_FLX_AMP_DEG)
    if flx_cmd.ndim > 1:
        amplitude = amplitude[:, None]
    normalized = (flx_cmd - flx_rest) / amplitude
    return 1.0 / (1.0 + np.exp(-spec.contact_sharpness * normalized))


def body_ode(t: float, state: np.ndarray, params: np.ndarray, spec: RobotSpec) -> np.ndarray:
    x, x_dot, z, z_dot, roll, roll_dot, pitch, pitch_dot = state
    del x

    abd_cmd, flx_cmd, abd_vel, _flx_vel = leg_commands(t, params)
    p = unpack_parameters(params)
    contacts = contact_weights(flx_cmd, p["flx_rest_rad"], spec)
    active_contacts = max(float(np.sum(contacts)), 1.0)

    gravity = 9.81
    static_support = spec.total_mass_kg * gravity / active_contacts
    leg_forces = (
        contacts
        * (
            static_support
            + spec.support_gain_n_per_rad * (flx_cmd - p["flx_rest_rad"])
            - (spec.support_damping_n_per_mps / active_contacts) * z_dot
        )
    )
    leg_forces = np.clip(leg_forces, 0.0, None)

    ground_restoring = -spec.ground_stiffness_n_per_m * z

    thrust_forces = -contacts * spec.stride_gain_n_per_rad_per_s * abd_vel
    total_thrust = float(np.sum(thrust_forces))
    total_vertical = float(np.sum(leg_forces)) + ground_restoring

    body_y = np.array([BODY_Y_M[leg] for leg in LEG_ORDER])
    body_x = np.array([BODY_X_M[leg] for leg in LEG_ORDER])
    roll_torque = float(
        np.sum(body_y * leg_forces)
        - spec.roll_damping_nm_per_rad_s * roll_dot
    )
    pitch_torque = float(
        np.sum(body_x * leg_forces)
        - spec.pitch_damping_nm_per_rad_s * pitch_dot
    )

    x_ddot = (
        total_thrust - spec.body_longitudinal_damping_n_per_mps * x_dot
    ) / spec.total_mass_kg
    z_ddot = (
        total_vertical
        - spec.total_mass_kg * gravity
        - spec.body_vertical_damping_n_per_mps * z_dot
    ) / spec.total_mass_kg
    roll_ddot = roll_torque / spec.roll_inertia_kgm2
    pitch_ddot = pitch_torque / spec.pitch_inertia_kgm2

    return np.array([x_dot, x_ddot, z_dot, z_ddot, roll_dot, roll_ddot, pitch_dot, pitch_ddot])


def simulate_hexapod(
    params: np.ndarray,
    spec: RobotSpec,
    config: OptimizationConfig,
) -> dict[str, np.ndarray | float]:
    t_eval = np.linspace(0.0, config.sim_duration_s, config.sample_count)
    sol = solve_ivp(
        lambda t, y: body_ode(t, y, params, spec),
        t_span=(0.0, config.sim_duration_s),
        y0=np.zeros(8),
        method="RK45",
        t_eval=t_eval,
        rtol=1e-4,
        atol=1e-6,
    )
    if not sol.success:
        raise RuntimeError(sol.message)

    abd_cmd, flx_cmd, _, _ = leg_commands(sol.t, params)  # type: ignore[arg-type]
    return {
        "t": sol.t,
        "x": sol.y[0],
        "x_dot": sol.y[1],
        "z": sol.y[2],
        "z_dot": sol.y[3],
        "roll": sol.y[4],
        "roll_dot": sol.y[5],
        "pitch": sol.y[6],
        "pitch_dot": sol.y[7],
        "abd_cmd": abd_cmd,
        "flx_cmd": flx_cmd,
    }


def leg_commands(t: np.ndarray | float, params: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    t_arr = np.atleast_1d(t).astype(float)
    p = unpack_parameters(params)
    omega = 2.0 * np.pi * p["frequency_hz"]
    abd_cmd = np.zeros((6, t_arr.size))
    flx_cmd = np.zeros((6, t_arr.size))
    abd_vel = np.zeros((6, t_arr.size))
    flx_vel = np.zeros((6, t_arr.size))

    for idx, leg in enumerate(LEG_ORDER):
        phase = omega * t_arr + TRIPOD_PHASE[leg]
        abd_phase = phase + p["abd_shift_rad"][idx]
        flx_phase = phase + p["flx_shift_rad"][idx]
        abd_amp = np.deg2rad(DEFAULT_ABD_AMP_DEG[idx])
        flx_amp = np.deg2rad(DEFAULT_FLX_AMP_DEG[idx])

        abd_cmd[idx, :] = p["abd_rest_rad"][idx] + abd_amp * np.sin(abd_phase)
        flx_cmd[idx, :] = p["flx_rest_rad"][idx] + flx_amp * np.sin(flx_phase)
        abd_vel[idx, :] = abd_amp * omega * np.cos(abd_phase)
        flx_vel[idx, :] = flx_amp * omega * np.cos(flx_phase)

        sin_abd = np.sin(abd_phase)
        pos_lift = np.maximum(sin_abd, 0.0)
        lift_rad = np.deg2rad(LIFT_SWING_DEG_PEAK) * pos_lift
        flx_cmd[idx, :] -= lift_rad
        dpos_dt = np.where(sin_abd > 0.0, np.cos(abd_phase) * omega, 0.0)
        flx_vel[idx, :] -= np.deg2rad(LIFT_SWING_DEG_PEAK) * dpos_dt

    if np.isscalar(t):
        return abd_cmd[:, 0], flx_cmd[:, 0], abd_vel[:, 0], flx_vel[:, 0]
    return abd_cmd, flx_cmd, abd_vel, flx_vel


def dominant_frequency_hz(signal: np.ndarray, sample_spacing_s: float) -> float:
    if signal.size < 4:
        return 0.0
    time_index = np.arange(signal.size, dtype=float)
    trend = np.polyval(np.polyfit(time_index, signal, 1), time_index)
    detrended = signal - trend
    freq, power = periodogram(detrended, fs=1.0 / sample_spacing_s)
    valid = (freq >= 0.2) & (freq <= 6.0)
    if not np.any(valid):
        return 0.0
    idx = np.argmax(power[valid])
    return float(freq[valid][idx])


def metrics_from_simulation(
    sim: dict[str, np.ndarray | float],
    params: np.ndarray,
    spec: RobotSpec,
    config: OptimizationConfig,
) -> dict[str, float]:
    t = sim["t"]
    x = sim["x"]
    x_dot = sim["x_dot"]
    z = sim["z"]
    roll = sim["roll"]
    pitch = sim["pitch"]
    p = unpack_parameters(params)
    contacts = contact_weights(sim["flx_cmd"], p["flx_rest_rad"][:, None], spec)

    steady_start = int(0.4 * len(z))
    steady_slice = slice(steady_start, None)
    dominant_hz = dominant_frequency_hz(sim["z_dot"][steady_slice], float(t[1] - t[0]))

    mean_speed = float(np.mean(x_dot[steady_start:]))
    total_displacement = float(x[-1] - x[0])
    roll_rms = float(np.sqrt(np.mean(roll[steady_start:] ** 2)))
    pitch_rms = float(np.sqrt(np.mean(pitch[steady_start:] ** 2)))
    z_rms = float(np.sqrt(np.mean(z[steady_start:] ** 2)))

    _, _, abd_vel, flx_vel = leg_commands(t, params)
    angular_excursion = np.abs(sim["flx_cmd"] - p["flx_rest_rad"][:, None]) + 0.2 * np.abs(
        sim["abd_cmd"] - p["abd_rest_rad"][:, None]
    )
    angular_rate = np.abs(flx_vel) + 0.1 * np.abs(abd_vel)
    torque_proxy = spec.torque_proxy_gain_nm_per_rad * (angular_excursion + 0.03 * angular_rate)
    peak_servo_usage = float(np.max(torque_proxy) / spec.sg90_stall_torque_nm)
    contact_balance = float(np.mean(np.std(contacts, axis=0)))

    return {
        "dominant_frequency_hz": dominant_hz,
        "gait_frequency_hz": float(params[24]),
        "mean_forward_speed_mps": mean_speed,
        "total_displacement_m": total_displacement,
        "roll_rms_rad": roll_rms,
        "pitch_rms_rad": pitch_rms,
        "z_rms_m": z_rms,
        "peak_servo_usage_ratio": peak_servo_usage,
        "contact_balance_std": contact_balance,
    }


def objective_function(
    params: np.ndarray,
    spec: RobotSpec,
    config: OptimizationConfig,
) -> float:
    try:
        sim = simulate_hexapod(params, spec, config)
        metrics = metrics_from_simulation(sim, params, spec, config)
    except Exception:
        return 1e6

    speed = metrics["mean_forward_speed_mps"]
    servo_overuse = max(0.0, metrics["peak_servo_usage_ratio"] - 0.85)
    baseline = default_parameter_vector()
    regularization = float(np.mean(((params - baseline) / (parameter_bounds().ub - parameter_bounds().lb)) ** 2))

    # Primary: maximize mean forward velocity (minimize negative weighted speed).
    cost = -config.forward_speed_weight * speed

    # Penalize net backward motion (negative mean forward speed).
    backward = max(0.0, -speed)
    cost += config.backward_motion_penalty_weight * backward**2

    # Optional soft floor on speed (only if target > 0): penalize falling short of target mean speed.
    if config.target_forward_speed_mps > 0.0:
        shortfall = max(0.0, config.target_forward_speed_mps - speed)
        cost += config.target_speed_floor_penalty_weight * shortfall**2

    cost += (
        config.roll_pitch_penalty_weight * metrics["roll_rms_rad"] ** 2
        + config.roll_pitch_penalty_weight * metrics["pitch_rms_rad"] ** 2
        + config.vertical_penalty_weight * metrics["z_rms_m"] ** 2
        + config.servo_overuse_penalty_weight * servo_overuse**2
        + config.contact_balance_penalty_weight * metrics["contact_balance_std"] ** 2
        + config.regularization_weight * regularization
    )
    return float(cost)


def random_initial_guess(rng: np.random.Generator, bounds: Bounds) -> np.ndarray:
    return rng.uniform(bounds.lb, bounds.ub)


def optimize_parameters(
    spec: RobotSpec,
    config: OptimizationConfig,
) -> tuple[np.ndarray, dict[str, float]]:
    rng = np.random.default_rng(config.seed)
    bounds = parameter_bounds()
    best_cost = np.inf
    best_params = default_parameter_vector()

    candidates = [default_parameter_vector()]
    for _ in range(config.restarts - 1):
        candidates.append(random_initial_guess(rng, bounds))

    for i, x0 in enumerate(candidates):
        print(f"  restart {i + 1}/{len(candidates)} ...", flush=True)
        result = minimize(
            objective_function,
            x0=x0,
            args=(spec, config),
            method="L-BFGS-B",
            bounds=bounds,
                       options={
                "maxiter": config.maxiter,
                "ftol": 1e-5,
                "maxfun": config.max_fun,
            },
        )
        sim_r = simulate_hexapod(result.x, spec, config)
        m_r = metrics_from_simulation(sim_r, result.x, spec, config)
        print(
            f"    cost={result.fun:.4f}  mean_forward_speed={m_r['mean_forward_speed_mps']:.4f} m/s",
            flush=True,
        )
        if result.fun < best_cost:
            best_cost = float(result.fun)
            best_params = result.x.copy()

    final_sim = simulate_hexapod(best_params, spec, config)
    final_metrics = metrics_from_simulation(final_sim, best_params, spec, config)
    final_metrics["objective_cost"] = best_cost
    return best_params, final_metrics


def make_summary(best_params: np.ndarray, metrics: dict[str, float], spec: RobotSpec) -> dict[str, object]:
    return {
        "robot_spec": asdict(spec) | {"total_mass_kg": spec.total_mass_kg},
        "optimized_parameters": {
            "abd_rest_deg": best_params[0:6].tolist(),
            "flx_rest_deg": best_params[6:12].tolist(),
            "abd_shift_deg": best_params[12:18].tolist(),
            "flx_shift_deg": best_params[18:24].tolist(),
            "frequency_hz": float(best_params[24]),
            "lift_swing_deg_peak": float(LIFT_SWING_DEG_PEAK),
        },
        "leg_order": list(LEG_ORDER),
        "metrics": metrics,
    }


def write_outputs(
    output_dir: Path,
    prefix: str,
    best_params: np.ndarray,
    metrics: dict[str, float],
    spec: RobotSpec,
    config: OptimizationConfig,
) -> None:
    sim = simulate_hexapod(best_params, spec, config)
    summary = make_summary(best_params, metrics, spec)

    summary_path = output_dir / f"{prefix}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    time_series = np.column_stack(
        [
            sim["t"],
            sim["x"],
            sim["x_dot"],
            sim["z"],
            sim["z_dot"],
            sim["roll"],
            sim["pitch"],
        ]
    )
    np.savetxt(
        output_dir / f"{prefix}_timeseries.csv",
        time_series,
        delimiter=",",
        header="t_s,x_m,x_dot_mps,z_m,z_dot_mps,roll_rad,pitch_rad",
        comments="",
    )

    markdown = [
        "# No-IMU Hexapod ODE Optimization",
        "",
        "## Robot Assumptions",
        f"- Total mass: {spec.total_mass_kg:.3f} kg",
        f"- Servo type: 9g class (SG90-like), stall torque {spec.sg90_stall_torque_nm:.3f} N*m",
        f"- Optimization objective: maximize mean forward velocity (see script docstring).",
        f"- Optional speed floor (soft): {config.target_forward_speed_mps:.4f} m/s",
        "",
        "## Best Parameters",
        f"- Abduction rest angles (deg): {np.round(best_params[0:6], 3).tolist()}",
        f"- Flexion rest angles (deg): {np.round(best_params[6:12], 3).tolist()}",
        f"- Abduction phase shifts (deg): {np.round(best_params[12:18], 3).tolist()}",
        f"- Flexion phase shifts (deg): {np.round(best_params[18:24], 3).tolist()}",
        f"- Gait frequency (Hz): {best_params[24]:.4f}",
        "",
        "## Metrics",
        *(f"- {key}: {value:.6f}" for key, value in metrics.items()),
        "",
        "## Notes",
        "- This is a reduced no-IMU body model used to optimize gait timing and posture before adding reflex feedback.",
        "- Tripod coupling is enforced with a pi phase offset between tripod A and tripod B.",
        "- The optimization variables are the 12 rest angles, 12 per-servo phase shifts, and one global gait frequency.",
    ]
    (output_dir / f"{prefix}_summary.md").write_text("\n".join(markdown) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optimize a no-IMU 12-servo hexapod gait: primary objective is mean forward velocity."
    )
    parser.add_argument(
        "--forward-speed-weight",
        type=float,
        default=1000.0,
        help="Weight on mean forward speed (higher = prioritize velocity more vs stability penalties).",
    )
    parser.add_argument(
        "--target-speed",
        type=float,
        default=0.0,
        help="Optional soft floor on mean forward speed (m/s). 0 disables. Adds penalty if simulated speed stays below this.",
    )
    parser.add_argument(
        "--target-speed-penalty-weight",
        type=float,
        default=200.0,
        help="Multiplier for falling below --target-speed when target-speed > 0.",
    )
    parser.add_argument("--target-frequency", type=float, default=1.8, help="Recorded in summary only (not objective).")
    parser.add_argument("--duration", type=float, default=8.0, help="Simulation duration in seconds.")
    parser.add_argument("--samples", type=int, default=800, help="Number of samples for solve_ivp output.")
    parser.add_argument("--restarts", type=int, default=8, help="Number of random initializations before selecting the best run.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for restart initialization.")
    parser.add_argument("--maxiter", type=int, default=180, help="Maximum L-BFGS-B iterations per restart.")
    parser.add_argument(
        "--max-fun",
        type=int,
        default=1200,
        help="Max objective evaluations per restart (each runs ODE integration). Lower for faster multi-restart runs.",
    )
    parser.add_argument("--output-prefix", type=str, default="no_imu_optimization", help="Prefix for the output files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spec = RobotSpec()
    config = OptimizationConfig(
        forward_speed_weight=args.forward_speed_weight,
        target_forward_speed_mps=args.target_speed,
        target_speed_floor_penalty_weight=args.target_speed_penalty_weight,
        target_resonant_frequency_hz=args.target_frequency,
        sim_duration_s=args.duration,
        sample_count=args.samples,
        restarts=args.restarts,
        seed=args.seed,
        maxiter=args.maxiter,
        max_fun=args.max_fun,
    )
    best_params, metrics = optimize_parameters(spec, config)
    output_dir = Path(__file__).resolve().parent
    write_outputs(output_dir, args.output_prefix, best_params, metrics, spec, config)

    print("Optimization finished (primary objective: maximize mean forward velocity).")
    print(f"Total mass: {spec.total_mass_kg:.3f} kg")
    print(f"Forward speed weight: {config.forward_speed_weight}")
    print(f"Optimized gait frequency: {best_params[24]:.4f} Hz")
    print(f"Mean forward speed: {metrics['mean_forward_speed_mps']:.4f} m/s")
    print(f"Total displacement: {metrics['total_displacement_m']:.4f} m")
    print(f"Roll RMS: {metrics['roll_rms_rad']:.4f} rad")
    print(f"Pitch RMS: {metrics['pitch_rms_rad']:.4f} rad")
    print(f"Vertical RMS: {metrics['z_rms_m']:.4f} m")
    print(f"Peak servo usage: {metrics['peak_servo_usage_ratio']:.4f}")


if __name__ == "__main__":
    main()
