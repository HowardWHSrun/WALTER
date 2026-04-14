"""
Spike Encoding and Decoding Modules

This module provides encoding schemes to convert continuous observations
to spike trains and decoding schemes to convert spike outputs to
continuous action values.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, Dict, Any, List
from gymnasium import spaces

from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from .scpg import SpikingCPG, CPGWithValueHead, SpikingCPGTripod, CPGWithValueHeadTripod


class SpikeEncoder(nn.Module):
    """
    Encode continuous observations into spike trains.
    
    Supports multiple encoding schemes:
    - Rate coding: Higher values -> higher firing rates
    - Latency coding: Higher values -> earlier spike times
    - Population coding: Multiple neurons per input dimension
    """
    
    def __init__(
        self,
        input_size: int,
        encoding_type: str = "rate",
        num_timesteps: int = 10,
        population_size: int = 1,
        max_rate: float = 100.0,  # Hz
        dt: float = 1.0,  # ms
    ):
        super().__init__()
        self.input_size = input_size
        self.encoding_type = encoding_type
        self.num_timesteps = num_timesteps
        self.population_size = population_size
        self.max_rate = max_rate
        self.dt = dt
        
        # For population coding
        if encoding_type == "population":
            # Gaussian receptive fields
            self.centers = nn.Parameter(
                torch.linspace(-1, 1, population_size).unsqueeze(0).repeat(input_size, 1),
                requires_grad=False
            )
            self.sigma = 2.0 / population_size
        
        # Learnable gain for rate coding
        self.gain = nn.Parameter(torch.ones(input_size))
        self.bias = nn.Parameter(torch.zeros(input_size))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode continuous values to spike trains.
        
        Args:
            x: Input tensor (batch, input_size)
        
        Returns:
            spikes: Spike train tensor (batch, timesteps, encoded_size)
        """
        batch_size = x.shape[0]
        
        # Normalize input to [0, 1] range
        x_norm = torch.sigmoid(self.gain * x + self.bias)
        
        if self.encoding_type == "rate":
            return self._rate_encoding(x_norm)
        elif self.encoding_type == "latency":
            return self._latency_encoding(x_norm)
        elif self.encoding_type == "population":
            return self._population_encoding(x)
        else:
            raise ValueError(f"Unknown encoding type: {self.encoding_type}")
    
    def _rate_encoding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Rate coding: probability of spike proportional to input value.
        """
        batch_size = x.shape[0]
        device = x.device
        
        # Convert to firing probability per timestep
        prob = x * self.max_rate * self.dt / 1000.0
        prob = torch.clamp(prob, 0, 1)
        
        # Generate spikes through Poisson process
        spikes = torch.zeros(batch_size, self.num_timesteps, self.input_size, device=device)
        
        for t in range(self.num_timesteps):
            spikes[:, t, :] = (torch.rand_like(x) < prob).float()
        
        return spikes
    
    def _latency_encoding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Latency coding: higher values spike earlier.
        """
        batch_size = x.shape[0]
        device = x.device
        
        # Convert to spike times (inverse: higher value = earlier spike)
        spike_times = ((1 - x) * self.num_timesteps).long()
        spike_times = torch.clamp(spike_times, 0, self.num_timesteps - 1)
        
        # Generate one-hot spike train
        spikes = torch.zeros(batch_size, self.num_timesteps, self.input_size, device=device)
        
        for b in range(batch_size):
            for i in range(self.input_size):
                t = spike_times[b, i].item()
                spikes[b, t, i] = 1.0
        
        return spikes
    
    def _population_encoding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Population coding with Gaussian receptive fields.
        """
        batch_size = x.shape[0]
        device = x.device
        
        # Compute activations for each population neuron
        # x: (batch, input_size), centers: (input_size, population_size)
        x_expanded = x.unsqueeze(-1)  # (batch, input_size, 1)
        activations = torch.exp(-((x_expanded - self.centers) ** 2) / (2 * self.sigma ** 2))
        # activations: (batch, input_size, population_size)
        
        # Flatten to (batch, input_size * population_size)
        activations = activations.view(batch_size, -1)
        
        # Convert to spikes using rate coding
        prob = activations * self.max_rate * self.dt / 1000.0
        prob = torch.clamp(prob, 0, 1)
        
        spikes = torch.zeros(
            batch_size, self.num_timesteps, self.input_size * self.population_size, 
            device=device
        )
        
        for t in range(self.num_timesteps):
            spikes[:, t, :] = (torch.rand_like(activations) < prob).float()
        
        return spikes
    
    @property
    def output_size(self) -> int:
        """Return the size of encoded output."""
        if self.encoding_type == "population":
            return self.input_size * self.population_size
        return self.input_size


class SpikeDecoder(nn.Module):
    """
    Decode spike train outputs to continuous action values.
    
    Supports:
    - Rate decoding: spike count -> continuous value
    - Weighted decoding: learned weights on spike counts
    - Membrane potential decoding: use final membrane potentials
    """
    
    def __init__(
        self,
        input_size: int,
        output_size: int,
        decoding_type: str = "rate",
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.decoding_type = decoding_type
        
        # Linear projection
        self.linear = nn.Linear(input_size, output_size)
        
        # Learnable scaling
        self.scale = nn.Parameter(torch.ones(output_size))
        self.bias = nn.Parameter(torch.zeros(output_size))
    
    def forward(
        self, 
        spikes: torch.Tensor,
        membrane_potentials: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Decode spikes to continuous values.
        
        Args:
            spikes: Spike train tensor (batch, timesteps, neurons) or (batch, neurons)
            membrane_potentials: Optional final membrane potentials
        
        Returns:
            output: Continuous output (batch, output_size)
        """
        if self.decoding_type == "rate":
            return self._rate_decoding(spikes)
        elif self.decoding_type == "weighted":
            return self._weighted_decoding(spikes)
        elif self.decoding_type == "membrane":
            if membrane_potentials is None:
                return self._rate_decoding(spikes)
            return self._membrane_decoding(membrane_potentials)
        else:
            raise ValueError(f"Unknown decoding type: {self.decoding_type}")
    
    def _rate_decoding(self, spikes: torch.Tensor) -> torch.Tensor:
        """Decode based on spike counts."""
        # Sum over time if 3D
        if spikes.dim() == 3:
            spike_counts = spikes.sum(dim=1)
        else:
            spike_counts = spikes
        
        # Normalize by max possible count
        if spikes.dim() == 3:
            max_count = spikes.shape[1]
            spike_rates = spike_counts / max_count
        else:
            spike_rates = spike_counts
        
        # Project and scale
        output = self.linear(spike_rates)
        output = output * self.scale + self.bias
        
        return torch.tanh(output)
    
    def _weighted_decoding(self, spikes: torch.Tensor) -> torch.Tensor:
        """Decode with time-weighted spike counts."""
        if spikes.dim() == 3:
            batch_size, num_timesteps, num_neurons = spikes.shape
            device = spikes.device
            
            # Time weights (later spikes weighted more)
            time_weights = torch.linspace(0.5, 1.5, num_timesteps, device=device)
            time_weights = time_weights.view(1, -1, 1)
            
            weighted_spikes = (spikes * time_weights).sum(dim=1)
            weighted_spikes = weighted_spikes / num_timesteps
        else:
            weighted_spikes = spikes
        
        output = self.linear(weighted_spikes)
        output = output * self.scale + self.bias
        
        return torch.tanh(output)
    
    def _membrane_decoding(self, membrane_potentials: torch.Tensor) -> torch.Tensor:
        """Decode from membrane potentials directly."""
        output = self.linear(membrane_potentials)
        output = output * self.scale + self.bias
        return torch.tanh(output)


class SCPGFeaturesExtractor(BaseFeaturesExtractor):
    """
    Custom feature extractor that processes observations through
    spike encoding for use with Stable-Baselines3.
    """
    
    def __init__(
        self,
        observation_space: spaces.Box,
        features_dim: int = 64,
        encoding_type: str = "rate",
        num_timesteps: int = 10,
    ):
        super().__init__(observation_space, features_dim)
        
        self.obs_size = int(np.prod(observation_space.shape))
        
        # Spike encoder
        self.encoder = SpikeEncoder(
            input_size=self.obs_size,
            encoding_type=encoding_type,
            num_timesteps=num_timesteps,
        )
        
        # Feature projection
        self.feature_net = nn.Sequential(
            nn.Linear(self.obs_size, features_dim),
            nn.ReLU(),
        )
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # Flatten observations
        obs_flat = observations.view(observations.shape[0], -1)
        
        # For now, just use the observation directly
        # The spike encoding is handled in the CPG itself
        features = self.feature_net(obs_flat)
        
        return features


class SCPGPolicy(ActorCriticPolicy):
    """
    Custom Actor-Critic policy using Spiking CPG for action generation.
    
    This policy wraps the SpikingCPG network to be compatible with
    Stable-Baselines3's PPO algorithm.
    """
    
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule,
        # CPG parameters
        num_legs: int = 6,
        neurons_per_oscillator: int = 32,
        tau_mem: float = 20.0,
        tau_syn: float = 10.0,
        coupling_strength: float = 0.5,
        num_timesteps: int = 10,
        # Standard policy kwargs
        **kwargs,
    ):
        # Store CPG parameters before calling super().__init__
        self.num_legs = num_legs
        self.neurons_per_oscillator = neurons_per_oscillator
        self.tau_mem = tau_mem
        self.tau_syn = tau_syn
        self.coupling_strength = coupling_strength
        self.num_timesteps_snn = num_timesteps
        
        # Remove features_extractor_class if present to avoid conflict
        kwargs.pop("features_extractor_class", None)
        kwargs.pop("features_extractor_kwargs", None)
        
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            **kwargs,
        )
    
    def _build_mlp_extractor(self) -> None:
        """Build the MLP extractor - we override this to use our CPG."""
        obs_size = int(np.prod(self.observation_space.shape))
        
        # Create CPG with value head
        self.cpg_network = CPGWithValueHead(
            obs_size=obs_size,
            num_legs=self.num_legs,
            neurons_per_oscillator=self.neurons_per_oscillator,
            tau_mem=self.tau_mem,
            tau_syn=self.tau_syn,
            coupling_strength=self.coupling_strength,
            num_timesteps=self.num_timesteps_snn,
        )
        
        # Dummy mlp_extractor to satisfy parent class
        self.mlp_extractor = nn.Identity()
    
    def _build(self, lr_schedule) -> None:
        """Build networks."""
        self._build_mlp_extractor()
        
        # Action and value networks are part of CPG
        action_dim = int(np.prod(self.action_space.shape))
        
        # Action net outputs mean
        self.action_net = nn.Identity()
        self.value_net = nn.Identity()
        
        # Log std parameter
        self.log_std = nn.Parameter(
            torch.zeros(action_dim), requires_grad=True
        )
        
        # Setup optimizer
        self.optimizer = self.optimizer_class(
            self.parameters(),
            lr=lr_schedule(1),
            **self.optimizer_kwargs,
        )
    
    def reset_cpg(self, batch_size: int = 1):
        """Reset CPG state."""
        if hasattr(self, 'cpg_network'):
            device = next(self.parameters()).device
            self.cpg_network.reset(batch_size, device)
    
    def forward(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the policy.
        
        Args:
            obs: Observation tensor
            deterministic: If True, return mean action
        
        Returns:
            actions: Sampled or mean actions
            values: Value estimates
            log_prob: Log probability of actions
        """
        obs_flat = obs.view(obs.shape[0], -1)
        
        # Get action mean and value from CPG
        action_mean, values = self.cpg_network(obs_flat)
        
        # Create action distribution
        action_std = torch.exp(self.log_std)
        distribution = torch.distributions.Normal(action_mean, action_std)
        
        if deterministic:
            actions = action_mean
        else:
            actions = distribution.sample()
        
        # Clip actions
        actions = torch.clamp(actions, -1.0, 1.0)
        
        log_prob = distribution.log_prob(actions).sum(dim=-1)
        
        return actions, values.squeeze(-1), log_prob
    
    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Evaluate actions according to the current policy.
        
        Args:
            obs: Observation tensor
            actions: Action tensor
        
        Returns:
            values: Value estimates
            log_prob: Log probability of actions
            entropy: Entropy of distribution
        """
        obs_flat = obs.view(obs.shape[0], -1)
        
        # Get action mean and value
        action_mean, values = self.cpg_network(obs_flat)
        
        # Create distribution
        action_std = torch.exp(self.log_std)
        distribution = torch.distributions.Normal(action_mean, action_std)
        
        log_prob = distribution.log_prob(actions).sum(dim=-1)
        entropy = distribution.entropy().sum(dim=-1)
        
        return values.squeeze(-1), log_prob, entropy
    
    def get_distribution(self, obs: torch.Tensor):
        """Get action distribution for given observations."""
        obs_flat = obs.view(obs.shape[0], -1)
        action_mean, _ = self.cpg_network(obs_flat)
        action_std = torch.exp(self.log_std)
        return torch.distributions.Normal(action_mean, action_std)
    
    def _predict(
        self,
        observation: torch.Tensor,
        deterministic: bool = False,
    ) -> torch.Tensor:
        """
        Get the action according to the policy for a given observation.
        
        Args:
            observation: Observation tensor
            deterministic: If True, return mean action
        
        Returns:
            actions: The actions to take
        """
        actions, _, _ = self.forward(observation, deterministic=deterministic)
        return actions
    
    def predict_values(self, obs: torch.Tensor) -> torch.Tensor:
        """Predict value for observations."""
        obs_flat = obs.view(obs.shape[0], -1)
        _, values = self.cpg_network(obs_flat)
        return values


class SCPGPolicyV2(ActorCriticPolicy):
    """
    Actor-Critic policy using tripod-gait Spiking CPG (version 2).
    Three legs (1, 4, 5) share one phase; legs (2, 3, 6) share the anti-phase.
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule,
        neurons_per_oscillator: int = 32,
        tau_mem: float = 20.0,
        tau_syn: float = 10.0,
        num_timesteps: int = 10,
        **kwargs,
    ):
        self.neurons_per_oscillator = neurons_per_oscillator
        self.tau_mem = tau_mem
        self.tau_syn = tau_syn
        self.num_timesteps_snn = num_timesteps
        kwargs.pop("features_extractor_class", None)
        kwargs.pop("features_extractor_kwargs", None)
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            **kwargs,
        )

    def _build_mlp_extractor(self) -> None:
        obs_size = int(np.prod(self.observation_space.shape))
        self.cpg_network = CPGWithValueHeadTripod(
            obs_size=obs_size,
            neurons_per_oscillator=self.neurons_per_oscillator,
            tau_mem=self.tau_mem,
            tau_syn=self.tau_syn,
            num_timesteps=self.num_timesteps_snn,
        )
        self.mlp_extractor = nn.Identity()

    def _build(self, lr_schedule) -> None:
        self._build_mlp_extractor()
        action_dim = int(np.prod(self.action_space.shape))
        self.action_net = nn.Identity()
        self.value_net = nn.Identity()
        self.log_std = nn.Parameter(torch.zeros(action_dim), requires_grad=True)
        self.optimizer = self.optimizer_class(
            self.parameters(),
            lr=lr_schedule(1),
            **self.optimizer_kwargs,
        )

    def reset_cpg(self, batch_size: int = 1):
        if hasattr(self, "cpg_network"):
            device = next(self.parameters()).device
            self.cpg_network.reset(batch_size, device)

    def forward(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        obs_flat = obs.view(obs.shape[0], -1)
        action_mean, values = self.cpg_network(obs_flat)
        action_std = torch.exp(self.log_std)
        distribution = torch.distributions.Normal(action_mean, action_std)
        if deterministic:
            actions = action_mean
        else:
            actions = distribution.sample()
        actions = torch.clamp(actions, -1.0, 1.0)
        log_prob = distribution.log_prob(actions).sum(dim=-1)
        return actions, values.squeeze(-1), log_prob

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        obs_flat = obs.view(obs.shape[0], -1)
        action_mean, values = self.cpg_network(obs_flat)
        action_std = torch.exp(self.log_std)
        distribution = torch.distributions.Normal(action_mean, action_std)
        log_prob = distribution.log_prob(actions).sum(dim=-1)
        entropy = distribution.entropy().sum(dim=-1)
        return values.squeeze(-1), log_prob, entropy

    def get_distribution(self, obs: torch.Tensor):
        obs_flat = obs.view(obs.shape[0], -1)
        action_mean, _ = self.cpg_network(obs_flat)
        action_std = torch.exp(self.log_std)
        return torch.distributions.Normal(action_mean, action_std)

    def _predict(
        self,
        observation: torch.Tensor,
        deterministic: bool = False,
    ) -> torch.Tensor:
        actions, _, _ = self.forward(observation, deterministic=deterministic)
        return actions

    def predict_values(self, obs: torch.Tensor) -> torch.Tensor:
        obs_flat = obs.view(obs.shape[0], -1)
        _, values = self.cpg_network(obs_flat)
        return values
