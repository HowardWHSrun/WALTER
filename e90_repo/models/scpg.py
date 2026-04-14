"""
Spiking Central Pattern Generator (sCPG) Network

This module implements a biologically-inspired spiking neural network
that generates rhythmic locomotion patterns for hexapod walking.
The CPG uses Leaky Integrate-and-Fire (LIF) neurons with recurrent
connections to produce oscillatory outputs.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, List

try:
    import norse.torch as norse
    from norse.torch.module.lif import LIFCell, LIFParameters
    from norse.torch.functional.lif import LIFState
    NORSE_AVAILABLE = True
except ImportError:
    NORSE_AVAILABLE = False
    print("Warning: Norse not available, using fallback LIF implementation")


class LIFNeuron(nn.Module):
    """
    Leaky Integrate-and-Fire neuron implementation.
    Fallback implementation when Norse is not available.
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        tau_mem: float = 20.0,
        tau_syn: float = 10.0,
        threshold: float = 1.0,
        reset: float = 0.0,
        dt: float = 1.0,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.tau_mem = tau_mem
        self.tau_syn = tau_syn
        self.threshold = threshold
        self.reset = reset
        self.dt = dt
        
        # Decay constants
        self.alpha = np.exp(-dt / tau_mem)
        self.beta = np.exp(-dt / tau_syn)
        
        # Learnable weights
        self.input_weights = nn.Linear(input_size, hidden_size, bias=False)
        self.recurrent_weights = nn.Linear(hidden_size, hidden_size, bias=False)
        
        # Initialize weights
        nn.init.xavier_uniform_(self.input_weights.weight)
        nn.init.orthogonal_(self.recurrent_weights.weight, gain=0.5)
    
    def forward(
        self,
        x: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through LIF neuron layer.
        
        Args:
            x: Input tensor of shape (batch, input_size)
            state: Tuple of (membrane_potential, synaptic_current)
        
        Returns:
            spikes: Output spikes (batch, hidden_size)
            new_state: Updated (membrane_potential, synaptic_current)
        """
        batch_size = x.shape[0]
        
        if state is None:
            v = torch.zeros(batch_size, self.hidden_size, device=x.device)
            i_syn = torch.zeros(batch_size, self.hidden_size, device=x.device)
        else:
            v, i_syn = state
        
        # Input current
        i_input = self.input_weights(x)
        
        # Recurrent current (from previous spikes, use previous membrane potential as proxy)
        spikes_prev = (v > self.threshold).float()
        i_rec = self.recurrent_weights(spikes_prev)
        
        # Update synaptic current
        i_syn = self.beta * i_syn + i_input + i_rec
        
        # Update membrane potential
        v = self.alpha * v + (1 - self.alpha) * i_syn
        
        # Generate spikes
        spikes = (v > self.threshold).float()
        
        # Reset neurons that spiked
        v = v * (1 - spikes) + self.reset * spikes
        
        return spikes, (v, i_syn)


class CPGOscillator(nn.Module):
    """
    Single CPG oscillator unit for one leg.
    
    Each oscillator contains flexor and extensor neuron populations
    with reciprocal inhibition to generate alternating patterns.
    """
    
    def __init__(
        self,
        input_size: int,
        neurons_per_population: int = 16,
        tau_mem: float = 20.0,
        tau_syn: float = 10.0,
        output_size: int = 2,  # abduction and flexion
    ):
        super().__init__()
        self.input_size = input_size
        self.neurons_per_population = neurons_per_population
        self.output_size = output_size
        
        # Two populations: flexor and extensor
        total_neurons = neurons_per_population * 2
        
        if NORSE_AVAILABLE:
            # Use Norse LIF cells
            self.lif_params = LIFParameters(
                tau_mem_inv=torch.tensor(1.0 / tau_mem),
                tau_syn_inv=torch.tensor(1.0 / tau_syn),
                v_th=torch.tensor(1.0),
                v_reset=torch.tensor(0.0),
            )
            self.input_layer = nn.Linear(input_size, total_neurons)
            self.lif_cell = LIFCell(p=self.lif_params)
            self.recurrent = nn.Linear(total_neurons, total_neurons, bias=False)
        else:
            # Use fallback implementation
            self.lif = LIFNeuron(
                input_size=input_size,
                hidden_size=total_neurons,
                tau_mem=tau_mem,
                tau_syn=tau_syn,
            )
        
        # Output projection (population coding to joint commands)
        self.output_layer = nn.Linear(total_neurons, output_size)
        
        # Reciprocal inhibition mask
        self._setup_inhibition_mask(neurons_per_population)
        
        self.state = None
    
    def _setup_inhibition_mask(self, n: int):
        """Setup reciprocal inhibition between flexor and extensor populations."""
        # Create inhibition pattern
        mask = torch.ones(2 * n, 2 * n)
        # Flexor inhibits extensor and vice versa
        mask[:n, n:] = -0.5  # Flexor -> Extensor inhibition
        mask[n:, :n] = -0.5  # Extensor -> Flexor inhibition
        self.register_buffer("inhibition_mask", mask)
    
    def reset_state(self, batch_size: int = 1, device: torch.device = None):
        """Reset oscillator state."""
        self.state = None
    
    def forward(
        self,
        x: torch.Tensor,
        coupling_input: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through oscillator.
        
        Args:
            x: Input tensor (batch, input_size)
            coupling_input: Input from coupled oscillators (batch, neurons)
        
        Returns:
            output: Joint commands (batch, output_size)
            spikes: Spike activity (batch, total_neurons)
        """
        if NORSE_AVAILABLE:
            # Transform input
            h = self.input_layer(x)
            
            # Add coupling if provided
            if coupling_input is not None:
                h = h + coupling_input
            
            # LIF dynamics
            if self.state is None:
                self.state = None  # Norse handles None state internally
            
            spikes, self.state = self.lif_cell(h, self.state)
            
            # Recurrent connection with inhibition
            rec = self.recurrent(spikes)
            rec = rec * torch.diagonal(self.inhibition_mask).unsqueeze(0)
        else:
            # Fallback implementation
            if coupling_input is not None:
                x_combined = torch.cat([x, coupling_input], dim=-1)
                # Pad to match expected input size
                x_combined = x[:, :self.input_size]  # Just use original input
            else:
                x_combined = x
            
            spikes, self.state = self.lif(x_combined, self.state)
        
        # Decode to joint commands
        output = self.output_layer(spikes)
        output = torch.tanh(output)  # Normalize to [-1, 1]
        
        return output, spikes


class SpikingCPG(nn.Module):
    """
    Complete Spiking Central Pattern Generator network for hexapod locomotion.
    
    The network consists of 6 coupled oscillators (one per leg) that generate
    coordinated rhythmic patterns for walking gaits.
    """
    
    def __init__(
        self,
        obs_size: int,
        num_legs: int = 6,
        neurons_per_oscillator: int = 32,
        tau_mem: float = 20.0,
        tau_syn: float = 10.0,
        coupling_strength: float = 0.5,
        num_timesteps: int = 10,
    ):
        super().__init__()
        self.obs_size = obs_size
        self.num_legs = num_legs
        self.neurons_per_oscillator = neurons_per_oscillator
        self.coupling_strength = coupling_strength
        self.num_timesteps = num_timesteps
        
        # Input encoder (shared across oscillators)
        self.input_encoder = nn.Sequential(
            nn.Linear(obs_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
        )
        
        # Create oscillators for each leg
        self.oscillators = nn.ModuleList([
            CPGOscillator(
                input_size=32,
                neurons_per_population=neurons_per_oscillator // 2,
                tau_mem=tau_mem,
                tau_syn=tau_syn,
                output_size=2,  # 2 DOF per leg
            )
            for _ in range(num_legs)
        ])
        
        # Inter-oscillator coupling weights (learnable phase relationships)
        self.coupling_weights = nn.Parameter(
            torch.zeros(num_legs, num_legs, neurons_per_oscillator)
        )
        self._init_coupling_weights()
        
        # Phase bias for each oscillator (learnable)
        self.phase_bias = nn.Parameter(torch.zeros(num_legs))
        
        # Output smoothing
        self.output_smoother = nn.Sequential(
            nn.Linear(num_legs * 2, num_legs * 2),
            nn.Tanh(),
        )
    
    def _init_coupling_weights(self):
        """Initialize coupling weights for tripod gait pattern."""
        # Tripod gait: legs 1,4,5 move together, legs 2,3,6 move together (alternating)
        tripod_a = [0, 3, 4]  # Legs 1, 4, 5 (0-indexed)
        tripod_b = [1, 2, 5]  # Legs 2, 3, 6
        
        with torch.no_grad():
            # Same group: positive coupling (synchronize)
            for group in [tripod_a, tripod_b]:
                for i in group:
                    for j in group:
                        if i != j:
                            self.coupling_weights[i, j] = 0.3
            
            # Different groups: negative coupling (anti-phase)
            for i in tripod_a:
                for j in tripod_b:
                    self.coupling_weights[i, j] = -0.3
                    self.coupling_weights[j, i] = -0.3
    
    def reset(self, batch_size: int = 1, device: torch.device = None):
        """Reset all oscillator states."""
        for osc in self.oscillators:
            osc.reset_state(batch_size, device)
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Generate locomotion commands from observations.
        
        Args:
            obs: Observation tensor (batch, obs_size)
        
        Returns:
            actions: Joint torque commands (batch, 12)
        """
        batch_size = obs.shape[0]
        device = obs.device
        
        # Reset oscillator states for new batch (handles batch size changes)
        for oscillator in self.oscillators:
            oscillator.state = None
        
        # Encode observations
        encoded = self.input_encoder(obs)
        
        # Store spike activities for coupling
        all_spikes = []
        
        # Run SNN for multiple timesteps
        accumulated_outputs = torch.zeros(batch_size, self.num_legs, 2, device=device)
        
        for t in range(self.num_timesteps):
            leg_outputs = []
            current_spikes = []
            
            for leg_idx, oscillator in enumerate(self.oscillators):
                # Compute coupling input from other oscillators
                coupling_input = None
                if len(all_spikes) > 0:
                    prev_spikes = all_spikes[-1]
                    coupling_input = torch.zeros(
                        batch_size, self.neurons_per_oscillator, device=device
                    )
                    for other_idx in range(self.num_legs):
                        if other_idx != leg_idx:
                            weight = self.coupling_weights[leg_idx, other_idx]
                            coupling_input += (
                                self.coupling_strength * 
                                prev_spikes[other_idx] * 
                                weight.unsqueeze(0)
                            )
                
                # Add phase bias
                phase_input = encoded + self.phase_bias[leg_idx]
                
                # Forward through oscillator
                output, spikes = oscillator(phase_input, coupling_input)
                
                leg_outputs.append(output)
                current_spikes.append(spikes)
            
            # Stack outputs and spikes
            outputs = torch.stack(leg_outputs, dim=1)  # (batch, num_legs, 2)
            accumulated_outputs += outputs
            all_spikes.append(current_spikes)
        
        # Average over timesteps
        accumulated_outputs /= self.num_timesteps
        
        # Flatten to action space (batch, 12)
        actions = accumulated_outputs.view(batch_size, -1)
        
        # Smooth output
        actions = self.output_smoother(actions)
        
        return actions
    
    def get_spike_activity(self) -> dict:
        """Return spike activity statistics for visualization."""
        activity = {}
        for leg_idx, osc in enumerate(self.oscillators):
            if osc.state is not None:
                if NORSE_AVAILABLE:
                    activity[f"leg{leg_idx+1}_v"] = osc.state.v.detach().cpu().numpy()
                else:
                    v, i_syn = osc.state
                    activity[f"leg{leg_idx+1}_v"] = v.detach().cpu().numpy()
        return activity


class CPGWithValueHead(nn.Module):
    """
    CPG network with additional value function head for actor-critic RL.
    """
    
    def __init__(
        self,
        obs_size: int,
        num_legs: int = 6,
        neurons_per_oscillator: int = 32,
        tau_mem: float = 20.0,
        tau_syn: float = 10.0,
        coupling_strength: float = 0.5,
        num_timesteps: int = 10,
    ):
        super().__init__()
        
        # Actor (CPG)
        self.cpg = SpikingCPG(
            obs_size=obs_size,
            num_legs=num_legs,
            neurons_per_oscillator=neurons_per_oscillator,
            tau_mem=tau_mem,
            tau_syn=tau_syn,
            coupling_strength=coupling_strength,
            num_timesteps=num_timesteps,
        )
        
        # Critic (value function)
        self.value_net = nn.Sequential(
            nn.Linear(obs_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        
        # Action log std (learnable)
        self.log_std = nn.Parameter(torch.zeros(num_legs * 2))
    
    def reset(self, batch_size: int = 1, device: torch.device = None):
        """Reset CPG state."""
        self.cpg.reset(batch_size, device)
    
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning action mean and value.
        
        Args:
            obs: Observation tensor
        
        Returns:
            action_mean: Mean action from CPG
            value: State value estimate
        """
        action_mean = self.cpg(obs)
        value = self.value_net(obs)
        return action_mean, value
    
    def get_action(
        self, 
        obs: torch.Tensor, 
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample action from policy.
        
        Args:
            obs: Observation tensor
            deterministic: If True, return mean action
        
        Returns:
            action: Sampled or mean action
            log_prob: Log probability of action
            value: State value estimate
        """
        action_mean, value = self.forward(obs)
        
        if deterministic:
            return action_mean, None, value
        
        # Sample from Gaussian
        std = torch.exp(self.log_std)
        dist = torch.distributions.Normal(action_mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        
        # Clamp action to valid range
        action = torch.clamp(action, -1.0, 1.0)
        
        return action, log_prob, value


# --- Tripod gait (version 2): two phase groups, three legs per group ---

# Tripod A: legs 1, 4, 5 (indices 0, 3, 4). Tripod B: legs 2, 3, 6 (indices 1, 2, 5). Alternating for stability.
TRIPOD_A_LEGS = [0, 3, 4]
TRIPOD_B_LEGS = [1, 2, 5]


class SpikingCPGTripod(nn.Module):
    """
    Spiking CPG with explicit tripod gait (version 2).
    
    One oscillator drives tripod A (legs 1, 4, 5); the same waveform
    negated (anti-phase) drives tripod B (legs 2, 3, 6). All legs in
    each tripod receive the same command, so the gait is fixed as
    tripod rather than learned per-leg.
    """

    def __init__(
        self,
        obs_size: int,
        neurons_per_oscillator: int = 32,
        tau_mem: float = 20.0,
        tau_syn: float = 10.0,
        num_timesteps: int = 10,
    ):
        super().__init__()
        self.obs_size = obs_size
        self.neurons_per_oscillator = neurons_per_oscillator
        self.num_timesteps = num_timesteps
        self.num_legs = 6
        self.dof_per_leg = 2

        self.input_encoder = nn.Sequential(
            nn.Linear(obs_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
        )

        # Single oscillator for tripod A; tripod B = -output (anti-phase)
        self.oscillator = CPGOscillator(
            input_size=32,
            neurons_per_population=neurons_per_oscillator // 2,
            tau_mem=tau_mem,
            tau_syn=tau_syn,
            output_size=self.dof_per_leg,
        )
        # Smoother applied to 2-DOF tripod command so symmetry is preserved
        self.output_smoother = nn.Sequential(
            nn.Linear(self.dof_per_leg, self.dof_per_leg),
            nn.Tanh(),
        )

    def reset(self, batch_size: int = 1, device: torch.device = None):
        """Reset oscillator state."""
        self.oscillator.reset_state(batch_size, device)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Generate 12-D actions from observations.
        Tripod A (legs 1,4,5) gets oscillator output; tripod B (legs 2,3,6) gets -output.
        """
        batch_size = obs.shape[0]
        device = obs.device
        self.oscillator.state = None

        encoded = self.input_encoder(obs)
        accumulated = torch.zeros(batch_size, self.dof_per_leg, device=device)

        for _ in range(self.num_timesteps):
            out, _ = self.oscillator(encoded, coupling_input=None)
            accumulated += out

        accumulated /= self.num_timesteps
        tripod_a_cmd = self.output_smoother(accumulated)
        tripod_b_cmd = -tripod_a_cmd

        # Build (batch, 6, 2): legs 1,4,5 = tripod_a_cmd; legs 2,3,6 = tripod_b_cmd
        actions_per_leg = torch.zeros(batch_size, self.num_legs, self.dof_per_leg, device=device)
        for i in TRIPOD_A_LEGS:
            actions_per_leg[:, i, :] = tripod_a_cmd
        for i in TRIPOD_B_LEGS:
            actions_per_leg[:, i, :] = tripod_b_cmd

        return actions_per_leg.view(batch_size, -1)

    def get_spike_activity(self) -> dict:
        """Return spike activity for visualization."""
        activity = {}
        if self.oscillator.state is not None:
            if NORSE_AVAILABLE:
                activity["tripod_oscillator_v"] = self.oscillator.state.v.detach().cpu().numpy()
            else:
                v, _ = self.oscillator.state
                activity["tripod_oscillator_v"] = v.detach().cpu().numpy()
        return activity


class CPGWithValueHeadTripod(nn.Module):
    """Tripod CPG with value head for actor-critic RL."""

    def __init__(
        self,
        obs_size: int,
        neurons_per_oscillator: int = 32,
        tau_mem: float = 20.0,
        tau_syn: float = 10.0,
        num_timesteps: int = 10,
    ):
        super().__init__()
        self.cpg = SpikingCPGTripod(
            obs_size=obs_size,
            neurons_per_oscillator=neurons_per_oscillator,
            tau_mem=tau_mem,
            tau_syn=tau_syn,
            num_timesteps=num_timesteps,
        )
        self.value_net = nn.Sequential(
            nn.Linear(obs_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.log_std = nn.Parameter(torch.zeros(self.cpg.num_legs * self.cpg.dof_per_leg))

    def reset(self, batch_size: int = 1, device: torch.device = None):
        self.cpg.reset(batch_size, device)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        action_mean = self.cpg(obs)
        value = self.value_net(obs)
        return action_mean, value
