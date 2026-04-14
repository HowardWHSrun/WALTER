from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize


LEGS = ["FL", "FR", "ML", "MR", "RL", "RR"]
ABD_SERVOS = ["FL_ABD", "FR_ABD", "ML_ABD", "MR_ABD", "RL_ABD", "RR_ABD"]
FLX_SERVOS = ["FL_FLX", "FR_FLX", "ML_FLX", "MR_FLX", "RL_FLX", "RR_FLX"]
TRIPOD_A = {"FL", "MR", "RL"}
TRIPOD_B = {"FR", "ML", "RR"}
FRONT_LEGS = {"FL", "FR"}
MID_LEGS = {"ML", "MR"}
REAR_LEGS = {"RL", "RR"}


@dataclass
class ModelConfig:
    target_frequency_hz: float = 1.2
    sim_duration_s: float = 8.0
    samples_per_second: int = 200
    robot_mass_kg: float = 0.28
    gravity_mps2: float = 9.81
    lever_arm_m: float = 0.05
    servo_stall_torque_nm: float = 0.176
    servo_nat_freq_hz: float = 6.0
    servo_damping_ratio: float = 0.85
    same_leg_coupling: float = 6.0
    same_tripod_coupling: float = 2.5
    max_angle_deg: float = 55.0
    n_random_starts: int = 8
    seed: int = 7


PARAM_NAMES = [
    "abd_rest_front_deg",
    "abd_rest_mid_deg",
    "abd_rest_rear_deg",
    "abd_shift_front_deg",
    "abd_shift_mid_deg",
    "abd_shift_rear_deg",
    "flex_rest_deg",
    "flex_shift_deg",
    "gait_frequency_hz",
    "abd_tripod_split_rad",
    "flex_tripod_split_rad",
    "flex_abd_offset_rad",
]

LOWER_BOUNDS = np.array(
    [
        -25.0,
        -15.0,
        -25.0,
        5.0,
        5.0,
        5.0,
        5.0,
        5.0,
        0.4,
        2.4,
        2.4,
        -np.pi / 2.0,
    ]
)

UPPER_BOUNDS = np.array(
    [
        25.0,
        15.0,
        25.0,
        35.0,
        30.0,
        35.0,
        45.0,
        35.0,
        2.8,
        3.9,
        3.9,
        np.pi / 2.0,
    ]
)


def leg_group(leg: str) -> str:
    if leg in FRONT_LEGS:
        return "front"
    if leg in MID_LEGS:
        return "mid"
    return "rear"


def coupling_matrix() -> np.ndarray:
    n = 12
    mat = np.zeros((n, n))
    leg_pairs = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11)]
    for a, b in leg_pairs:
        mat[a, b] = 1.0
        mat[b, a] = 1.0

    tripod_servo_groups = [
        [0, 6, 8],   # ABD on tripod A: FL, MR, RL
        [2, 4, 10],  # ABD on tripod B: FR, ML, RR
        [1, 7, 9],   # FLX on tripod A
        [3, 5, 11],  # FLX on tripod B
    ]
    for group in tripod_servo_groups:
        for i in group:
            for j in group:
                if i != j:
                    mat[i, j] += 1.0
    return mat


COUPLING_TEMPLATE = coupling_matrix()


def unpack_params(x: np.ndarray) -> dict[str, float]:
    return dict(zip(PARAM_NAMES, x))


def target_angle_deg(servo_idx: int, t: float, params: dict[str, float]) -> float:
    if servo_idx % 2 == 0:
        leg = ABD_SERVOS[servo_idx // 2].split("_")[0]
        tripod_phase = 0.0 if leg in TRIPOD_A else params["abd_tripod_split_rad"]
        group = leg_group(leg)
        rest = params[f"abd_rest_{group}_deg"]
        shift = params[f"abd_shift_{group}_deg"]
        return rest + shift * np.sin(2.0 * np.pi * params["gait_frequency_hz"] * t + tripod_phase)

    leg = FLX_SERVOS[servo_idx // 2].split("_")[0]
    tripod_phase = 0.0 if leg in TRIPOD_A else params["flex_tripod_split_rad"]
    phase = tripod_phase + params["flex_abd_offset_rad"]
    return params["flex_rest_deg"] + params["flex_shift_deg"] * np.sin(
        2.0 * np.pi * params["gait_frequency_hz"] * t + phase
    )


def servo_load_term_radps2(servo_idx: int, theta_rad: float, cfg: ModelConfig) -> float:
    per_leg_weight = cfg.robot_mass_kg * cfg.gravity_mps2 / 6.0
    abd_share = 0.30
    flx_share = 0.70
    force = per_leg_weight * (abd_share if servo_idx % 2 == 0 else flx_share)
    torque = force * cfg.lever_arm_m * np.sin(theta_rad)
    # Normalize by servo torque capability to get a dimensionless penalty-scaled acceleration term.
    return 4.0 * torque / max(cfg.servo_stall_torque_nm, 1e-6)


def ode_rhs(t: float, state: np.ndarray, x: np.ndarray, cfg: ModelConfig) -> np.ndarray:
    params = unpack_params(x)
    theta = state[:12]
    omega = state[12:]

    wn = 2.0 * np.pi * cfg.servo_nat_freq_hz
    zeta = cfg.servo_damping_ratio

    dtheta = omega.copy()
    domega = np.zeros_like(omega)

    for i in range(12):
        target_deg = target_angle_deg(i, t, params)
        target_rad = np.deg2rad(target_deg)
        same_leg_term = cfg.same_leg_coupling * np.sum(COUPLING_TEMPLATE[i] * (theta[i] - theta))
        load_term = servo_load_term_radps2(i, theta[i], cfg)
        domega[i] = (
            wn**2 * (target_rad - theta[i])
            - 2.0 * zeta * wn * omega[i]
            - same_leg_term
            - load_term
        )

    return np.concatenate([dtheta, domega])


def simulate_response(x: np.ndarray, cfg: ModelConfig) -> tuple[np.ndarray, np.ndarray]:
    t_eval = np.linspace(0.0, cfg.sim_duration_s, int(cfg.sim_duration_s * cfg.samples_per_second))
    y0 = np.zeros(24)
    sol = solve_ivp(
        lambda t, y: ode_rhs(t, y, x, cfg),
        (0.0, cfg.sim_duration_s),
        y0,
        t_eval=t_eval,
        method="RK45",
        rtol=1e-5,
        atol=1e-7,
    )
    return sol.t, sol.y


def dominant_frequency_hz(signal: np.ndarray, dt: float) -> float:
    centered = signal - np.mean(signal)
    fft = np.fft.rfft(centered)
    freqs = np.fft.rfftfreq(centered.size, d=dt)
    if centered.size < 4:
        return 0.0
    if freqs.size <= 1:
        return 0.0
    peak_idx = np.argmax(np.abs(fft[1:])) + 1
    return float(freqs[peak_idx])


def objective(x: np.ndarray, cfg: ModelConfig) -> float:
    t, y = simulate_response(x, cfg)
    theta = y[:12]
    flex_mean = np.mean(theta[[1, 3, 5, 7, 9, 11]], axis=0)
    dt = t[1] - t[0]
    dom_freq = dominant_frequency_hz(flex_mean, dt)

    freq_penalty = (dom_freq - cfg.target_frequency_hz) ** 2

    tripod_a = np.mean(theta[[1, 7, 9]], axis=0)
    tripod_b = np.mean(theta[[3, 5, 11]], axis=0)
    symmetry_penalty = np.mean((tripod_a + tripod_b) ** 2)

    angle_limit_rad = np.deg2rad(cfg.max_angle_deg)
    saturation_penalty = np.mean(np.clip(np.abs(theta) - angle_limit_rad, 0.0, None) ** 2)

    velocity_penalty = 1e-3 * np.mean(y[12:] ** 2)

    params = unpack_params(x)
    static_torque = cfg.robot_mass_kg * cfg.gravity_mps2 * cfg.lever_arm_m
    torque_ratio = static_torque / max(cfg.servo_stall_torque_nm * 6.0, 1e-6)
    load_penalty = max(0.0, torque_ratio - 0.8) ** 2

    freq_tracking_penalty = 0.05 * (params["gait_frequency_hz"] - cfg.target_frequency_hz) ** 2

    return (
        15.0 * freq_penalty
        + 2.0 * symmetry_penalty
        + 40.0 * saturation_penalty
        + velocity_penalty
        + 5.0 * load_penalty
        + freq_tracking_penalty
    )


def random_guess(rng: np.random.Generator) -> np.ndarray:
    return rng.uniform(LOWER_BOUNDS, UPPER_BOUNDS)


def optimize(cfg: ModelConfig) -> tuple[np.ndarray, float]:
    rng = np.random.default_rng(cfg.seed)
    bounds = list(zip(LOWER_BOUNDS, UPPER_BOUNDS))
    best_x = None
    best_f = np.inf

    for _ in range(cfg.n_random_starts):
        x0 = random_guess(rng)
        result = minimize(
            objective,
            x0,
            args=(cfg,),
            method="SLSQP",
            bounds=bounds,
            options={"maxiter": 160, "ftol": 1e-6, "disp": False},
        )
        if result.fun < best_f:
            best_f = float(result.fun)
            best_x = result.x.copy()

    if best_x is None:
        raise RuntimeError("Optimization failed to produce a solution.")
    return best_x, best_f


def save_outputs(x: np.ndarray, cfg: ModelConfig, out_dir: Path) -> None:
    params = unpack_params(x)
    t, y = simulate_response(x, cfg)
    theta_deg = np.rad2deg(y[:12])
    flex_mean = np.mean(theta_deg[[1, 3, 5, 7, 9, 11]], axis=0)
    abd_mean = np.mean(theta_deg[[0, 2, 4, 6, 8, 10]], axis=0)
    dom_freq = dominant_frequency_hz(np.deg2rad(flex_mean), t[1] - t[0])

    payload = {
        "model_config": asdict(cfg),
        "best_parameters": params,
        "objective_value": float(objective(x, cfg)),
        "dominant_frequency_hz": dom_freq,
    }
    with (out_dir / "best_params_no_imu.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    plt.figure(figsize=(10, 6))
    plt.plot(t, flex_mean, label="Mean flexion angle (deg)")
    plt.plot(t, abd_mean, label="Mean abduction angle (deg)")
    plt.xlabel("Time (s)")
    plt.ylabel("Angle (deg)")
    plt.title(f"Best no-IMU response, dominant frequency = {dom_freq:.3f} Hz")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "best_response_no_imu.png", dpi=180)
    plt.close()


def main() -> None:
    cfg = ModelConfig()
    out_dir = Path(__file__).resolve().parent
    best_x, best_f = optimize(cfg)
    save_outputs(best_x, cfg, out_dir)

    params = unpack_params(best_x)
    t, y = simulate_response(best_x, cfg)
    flex_mean = np.mean(y[[1, 3, 5, 7, 9, 11]], axis=0)
    dom_freq = dominant_frequency_hz(flex_mean, t[1] - t[0])

    print("Best objective:", best_f)
    print("Dominant frequency (Hz):", round(dom_freq, 4))
    print("Optimized parameters:")
    for name in PARAM_NAMES:
        value = params[name]
        print(f"  {name}: {value:.6f}")


if __name__ == "__main__":
    main()
