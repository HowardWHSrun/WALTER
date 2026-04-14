"""
Tripod gait reference in normalized action space [-1, 1]^12.

Matches leg indexing and abduction sign rules in envs/hexapod_env.HexapodEnv._tripod_reference_action.
"""

from __future__ import annotations

import math
import torch

# Leg indices 0..5 = legs 1..6
TRIPOD_A_LEGS = (0, 3, 4)
TRIPOD_B_LEGS = (1, 2, 5)
LEFT_LEGS = frozenset((1, 3, 5))


def tripod_actions_from_phase(
    theta_a: torch.Tensor,
    amp_flex: torch.Tensor,
    amp_abd: torch.Tensor,
    flex_sign: float = 1.0,
    abd_sign: float = 1.0,
) -> torch.Tensor:
    """
    Args:
        theta_a: Phase for tripod A (batch,) radians.
        amp_flex, amp_abd: Amplitudes (batch,) in [0, 1] scale (same as imitation amps).
        flex_sign, abd_sign: Scalar signs (fixed defaults match typical imitation).

    Returns:
        (batch, 12) actions in [-1, 1] (clipped).
    """
    theta_b = theta_a + math.pi
    fs = float(flex_sign)
    ads = float(abd_sign)

    flex_a = fs * amp_flex * torch.sin(theta_a)
    flex_b = fs * amp_flex * torch.sin(theta_b)
    abd_a = ads * (-amp_abd * torch.cos(theta_a))
    abd_b = ads * (-amp_abd * torch.cos(theta_b))

    batch = int(theta_a.shape[0])
    device = theta_a.device
    dtype = theta_a.dtype
    ctrl = torch.zeros(batch, 12, device=device, dtype=dtype)

    for i in TRIPOD_A_LEGS:
        ctrl[:, i * 2] = abd_a
        ctrl[:, i * 2 + 1] = flex_a
    for i in TRIPOD_B_LEGS:
        abd_i = torch.where(
            torch.full((batch,), i in LEFT_LEGS, device=device, dtype=torch.bool),
            -abd_b,
            abd_b,
        )
        ctrl[:, i * 2] = abd_i
        ctrl[:, i * 2 + 1] = flex_b

    return torch.clamp(ctrl, -1.0, 1.0)
