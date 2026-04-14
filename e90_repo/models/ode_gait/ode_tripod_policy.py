"""
PPO policy: continuous-time tripod phase theta = 2*pi*f*t + phi with t from env observation.

Stateless in the sense that phase is reconstructed from sim_time in obs (last dimension),
so rollout collection and PPO evaluate_actions stay consistent.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.policies import ActorCriticPolicy

from .reference_from_phase import tripod_actions_from_phase


class ODETripodPolicy(ActorCriticPolicy):
    """
    Actor outputs mean action = tripod_ODE_reference(theta) + residual_scale * tanh(res),
    with theta = 2*pi*freq*sim_time + phase_offset; freq, amps, offsets, residual from MLP(obs_core).
    Last observation dimension must be simulation time (seconds) when sim_time_in_obs is True.
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule,
        sim_time_in_obs: bool = True,
        freq_hz_range: Tuple[float, float] = (0.35, 1.6),
        max_amp_flex: float = 0.95,
        max_amp_abd: float = 0.95,
        flex_sign: float = 1.0,
        abd_sign: float = 1.0,
        residual_scale: float = 0.2,
        hidden_sizes: Tuple[int, ...] = (64, 64),
        **kwargs,
    ):
        self.sim_time_in_obs = bool(sim_time_in_obs)
        self.freq_hz_min = float(freq_hz_range[0])
        self.freq_hz_max = float(freq_hz_range[1])
        self.max_amp_flex = float(max_amp_flex)
        self.max_amp_abd = float(max_amp_abd)
        self.flex_sign = float(flex_sign)
        self.abd_sign = float(abd_sign)
        self.residual_scale = float(residual_scale)
        self.hidden_sizes = tuple(hidden_sizes)

        kwargs.pop("features_extractor_class", None)
        kwargs.pop("features_extractor_kwargs", None)
        kwargs.pop("share_features_extractor", None)

        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            **kwargs,
        )

    def _build_mlp_extractor(self) -> None:
        obs_dim = int(np.prod(self.observation_space.shape))
        if self.sim_time_in_obs:
            if obs_dim < 2:
                raise ValueError("observation_space must include at least obs_core + sim_time")
            core_dim = obs_dim - 1
        else:
            core_dim = obs_dim

        act_dim = int(np.prod(self.action_space.shape))
        layers = []
        in_d = core_dim
        for h in self.hidden_sizes:
            layers.append(nn.Linear(in_d, h))
            layers.append(nn.ReLU())
            in_d = h
        self._actor_core = nn.Sequential(*layers)
        self._param_head = nn.Linear(in_d, 4)
        self._residual_head = nn.Linear(in_d, act_dim)
        self.mlp_extractor = nn.Identity()

        vlayers = []
        vd = obs_dim
        for h in self.hidden_sizes:
            vlayers.append(nn.Linear(vd, h))
            vlayers.append(nn.ReLU())
            vd = h
        vlayers.append(nn.Linear(vd, 1))
        self._value_net = nn.Sequential(*vlayers)

    def _build(self, lr_schedule) -> None:
        self._build_mlp_extractor()
        self.action_net = nn.Identity()
        self.value_net = nn.Identity()
        action_dim = int(np.prod(self.action_space.shape))
        self.log_std = nn.Parameter(torch.zeros(action_dim), requires_grad=True)
        self.optimizer = self.optimizer_class(
            self.parameters(),
            lr=lr_schedule(1),
            **self.optimizer_kwargs,
        )

    def _split_obs(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.sim_time_in_obs:
            obs_core = obs[..., :-1]
            sim_time = obs[..., -1]
            return obs_core, sim_time
        zeros = torch.zeros(obs.shape[0], device=obs.device, dtype=obs.dtype)
        return obs, zeros

    def _params_from_obs(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        obs_core, sim_time = self._split_obs(obs)
        h = self._actor_core(obs_core)
        raw = self._param_head(h)
        res_raw = self._residual_head(h)

        freq = self.freq_hz_min + torch.sigmoid(raw[:, 0]) * (self.freq_hz_max - self.freq_hz_min)
        amp_flex = torch.sigmoid(raw[:, 1]) * self.max_amp_flex
        amp_abd = torch.sigmoid(raw[:, 2]) * self.max_amp_abd
        phase_off = torch.tanh(raw[:, 3]) * np.pi

        theta_a = 2 * np.pi * freq * sim_time + phase_off
        residual = self.residual_scale * torch.tanh(res_raw)
        return theta_a, amp_flex, amp_abd, residual, sim_time

    def action_mean_from_obs(self, obs: torch.Tensor) -> torch.Tensor:
        theta_a, amp_flex, amp_abd, residual, _ = self._params_from_obs(obs)
        base = tripod_actions_from_phase(
            theta_a, amp_flex, amp_abd, flex_sign=self.flex_sign, abd_sign=self.abd_sign
        )
        return torch.clamp(base + residual, -1.0, 1.0)

    def reset_cpg(self, batch_size: int = 1) -> None:
        """Compatibility with evaluate.py / SCPG reset hook (no internal phase buffer)."""
        return

    def forward(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        obs_flat = obs.view(obs.shape[0], -1)
        action_mean = self.action_mean_from_obs(obs_flat)
        values = self._value_net(obs_flat).squeeze(-1)
        action_std = torch.exp(self.log_std)
        dist = torch.distributions.Normal(action_mean, action_std)
        if deterministic:
            actions = action_mean
        else:
            actions = dist.sample()
        actions = torch.clamp(actions, -1.0, 1.0)
        log_prob = dist.log_prob(actions).sum(dim=-1)
        return actions, values, log_prob

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        obs_flat = obs.view(obs.shape[0], -1)
        action_mean = self.action_mean_from_obs(obs_flat)
        values = self._value_net(obs_flat).squeeze(-1)
        action_std = torch.exp(self.log_std)
        dist = torch.distributions.Normal(action_mean, action_std)
        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return values, log_prob, entropy

    def get_distribution(self, obs: torch.Tensor):
        obs_flat = obs.view(obs.shape[0], -1)
        action_mean = self.action_mean_from_obs(obs_flat)
        action_std = torch.exp(self.log_std)
        return torch.distributions.Normal(action_mean, action_std)

    def _predict(self, observation: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        actions, _, _ = self.forward(observation, deterministic=deterministic)
        return actions

    def predict_values(self, obs: torch.Tensor) -> torch.Tensor:
        obs_flat = obs.view(obs.shape[0], -1)
        return self._value_net(obs_flat).squeeze(-1)
