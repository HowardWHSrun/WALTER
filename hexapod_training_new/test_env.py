#!/usr/bin/env python3
"""
Test script to verify the hexapod environment setup.

This script tests:
1. MuJoCo model loading
2. Environment creation
3. Random action execution
4. Observation and action spaces
"""

import sys
from pathlib import Path

import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


def test_mujoco_model():
    """Test that the MuJoCo model loads correctly."""
    print("=" * 60)
    print("Testing MuJoCo Model Loading")
    print("=" * 60)
    
    import mujoco
    
    model_path = PROJECT_ROOT / "assets" / "hexapod.xml"
    print(f"Loading model from: {model_path}")
    
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    
    print(f"Model loaded successfully!")
    print(f"  - Number of bodies: {model.nbody}")
    print(f"  - Number of joints: {model.njnt}")
    print(f"  - Number of actuators: {model.nu}")
    print(f"  - Number of sensors: {model.nsensor}")
    print(f"  - Timestep: {model.opt.timestep}")
    
    # List joints
    print("\nJoints:")
    for i in range(model.njnt):
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        print(f"  {i}: {joint_name}")
    
    # List actuators
    print("\nActuators:")
    for i in range(model.nu):
        actuator_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        print(f"  {i}: {actuator_name}")
    
    return True


def test_environment():
    """Test the Gymnasium environment."""
    print("\n" + "=" * 60)
    print("Testing Gymnasium Environment")
    print("=" * 60)
    
    from envs.hexapod_env import HexapodEnv
    
    env = HexapodEnv()
    
    print(f"Environment created successfully!")
    print(f"  - Observation space: {env.observation_space}")
    print(f"  - Action space: {env.action_space}")
    
    # Test reset
    print("\nTesting reset...")
    obs, info = env.reset(seed=42)
    print(f"  - Initial observation shape: {obs.shape}")
    print(f"  - Initial info: {info}")
    
    # Test stepping with random actions
    print("\nTesting random actions (100 steps)...")
    total_reward = 0
    
    for i in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if terminated or truncated:
            print(f"  Episode ended at step {i+1}")
            print(f"  Termination info: {info}")
            obs, info = env.reset()
    
    print(f"  - Total reward (100 steps): {total_reward:.2f}")
    print(f"  - Final observation shape: {obs.shape}")
    
    env.close()
    return True


def test_imu6_environment():
    """IMU-only (6D) observation from config_imu6.yaml."""
    print("\n" + "=" * 60)
    print("Testing IMU6 (accel + gyro) observation mode")
    print("=" * 60)

    import yaml
    from envs.hexapod_env import make_hexapod_env

    cfg_path = PROJECT_ROOT / "configs" / "config_imu6.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    env = make_hexapod_env(config)
    assert env.observation_space.shape == (6,), f"expected obs (6,), got {env.observation_space.shape}"

    obs, info = env.reset(seed=0)
    assert obs.shape == (6,), f"expected obs (6,), got {obs.shape}"
    print(f"  - Observation space: {env.observation_space}")
    print(f"  - Initial observation shape: {obs.shape}")

    for i in range(20):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs.shape == (6,), f"step {i}: expected obs (6,), got {obs.shape}"
        if terminated or truncated:
            obs, info = env.reset()

    print("  - IMU6 mode: OK")
    env.close()
    return True


def test_imu6_prev_environment():
    """IMU + previous action (18D): observation_mode imu6_prev."""
    print("\n" + "=" * 60)
    print("Testing imu6_prev (IMU 6D + previous action 12D)")
    print("=" * 60)

    import yaml
    from envs.hexapod_env import make_hexapod_env

    cfg_path = PROJECT_ROOT / "configs" / "config_imu6_prev.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    env = make_hexapod_env(config)
    ecfg = config.get("env", {})
    expected = 18
    if ecfg.get("obs_include_phase"):
        expected += 2
    assert env.observation_space.shape == (expected,), (
        f"expected obs ({expected},), got {env.observation_space.shape}"
    )

    obs, info = env.reset(seed=0)
    assert obs.shape == (expected,), f"expected obs ({expected},), got {obs.shape}"
    print(f"  - Observation space: {env.observation_space}")
    print(f"  - Initial observation shape: {obs.shape}")

    for i in range(20):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs.shape == (expected,), f"step {i}: expected obs ({expected},), got {obs.shape}"
        if terminated or truncated:
            obs, info = env.reset()

    print("  - imu6_prev mode: OK")
    env.close()
    return True




def test_imu_bno10_environment():
    """BNO-like IMU mode (accel+gyro+quat = 10D)."""
    print("\n" + "=" * 60)
    print("Testing imu_bno10 observation mode")
    print("=" * 60)

    import yaml
    from envs.hexapod_env import make_hexapod_env

    cfg_path = PROJECT_ROOT / "configs" / "config_imu_bno10.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    env = make_hexapod_env(config)
    assert env.observation_space.shape == (10,), f"expected obs (10,), got {env.observation_space.shape}"

    obs, info = env.reset(seed=0)
    assert obs.shape == (10,), f"expected obs (10,), got {obs.shape}"
    print(f"  - Observation space: {env.observation_space}")
    print(f"  - Initial observation shape: {obs.shape}")

    for i in range(20):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs.shape == (10,), f"step {i}: expected obs (10,), got {obs.shape}"
        if terminated or truncated:
            obs, info = env.reset()

    print("  - imu_bno10 mode: OK")
    env.close()
    return True


def test_cpg_network():
    """Test the Spiking CPG network."""
    print("\n" + "=" * 60)
    print("Testing Spiking CPG Network")
    print("=" * 60)
    
    import torch
    from models.scpg import SpikingCPG, CPGWithValueHead
    
    obs_size = 52  # From environment
    batch_size = 4
    
    # Test SpikingCPG
    print("\nTesting SpikingCPG...")
    cpg = SpikingCPG(
        obs_size=obs_size,
        num_legs=6,
        neurons_per_oscillator=32,
        num_timesteps=10,
    )
    
    # Create dummy observation
    obs = torch.randn(batch_size, obs_size)
    
    # Reset and forward
    cpg.reset(batch_size)
    actions = cpg(obs)
    
    print(f"  - Input shape: {obs.shape}")
    print(f"  - Output shape: {actions.shape}")
    print(f"  - Action range: [{actions.min().item():.3f}, {actions.max().item():.3f}]")
    
    # Test CPGWithValueHead
    print("\nTesting CPGWithValueHead...")
    cpg_av = CPGWithValueHead(
        obs_size=obs_size,
        num_legs=6,
        neurons_per_oscillator=32,
        num_timesteps=10,
    )
    
    cpg_av.reset(batch_size)
    action_mean, value = cpg_av(obs)
    
    print(f"  - Action mean shape: {action_mean.shape}")
    print(f"  - Value shape: {value.shape}")
    
    # Test action sampling
    action, log_prob, value = cpg_av.get_action(obs, deterministic=False)
    print(f"  - Sampled action shape: {action.shape}")
    print(f"  - Log prob shape: {log_prob.shape}")
    
    return True


def test_spike_encoding():
    """Test spike encoding and decoding."""
    print("\n" + "=" * 60)
    print("Testing Spike Encoding/Decoding")
    print("=" * 60)
    
    import torch
    from models.encoder import SpikeEncoder, SpikeDecoder
    
    input_size = 52
    output_size = 12
    batch_size = 4
    num_timesteps = 10
    
    # Test encoder
    print("\nTesting SpikeEncoder (rate coding)...")
    encoder = SpikeEncoder(
        input_size=input_size,
        encoding_type="rate",
        num_timesteps=num_timesteps,
    )
    
    x = torch.randn(batch_size, input_size)
    spikes = encoder(x)
    
    print(f"  - Input shape: {x.shape}")
    print(f"  - Spike train shape: {spikes.shape}")
    print(f"  - Mean spike rate: {spikes.mean().item():.3f}")
    
    # Test decoder
    print("\nTesting SpikeDecoder...")
    decoder = SpikeDecoder(
        input_size=input_size,
        output_size=output_size,
        decoding_type="rate",
    )
    
    output = decoder(spikes)
    print(f"  - Output shape: {output.shape}")
    print(f"  - Output range: [{output.min().item():.3f}, {output.max().item():.3f}]")
    
    return True


def test_full_pipeline():
    """Test the full training pipeline components."""
    print("\n" + "=" * 60)
    print("Testing Full Pipeline")
    print("=" * 60)
    
    import torch
    from envs.hexapod_env import HexapodEnv
    from models.scpg import CPGWithValueHead
    
    # Create environment
    env = HexapodEnv()
    obs, info = env.reset(seed=42)
    
    # Create network
    obs_size = env.observation_space.shape[0]
    network = CPGWithValueHead(
        obs_size=obs_size,
        num_legs=6,
        neurons_per_oscillator=32,
        num_timesteps=10,
    )
    
    print(f"Running 50 steps with sCPG controller...")
    
    total_reward = 0
    network.reset(batch_size=1)
    
    for i in range(50):
        # Convert observation to tensor
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
        
        # Get action from network
        with torch.no_grad():
            action, log_prob, value = network.get_action(obs_tensor, deterministic=True)
        
        action_np = action.squeeze(0).numpy()
        
        # Step environment
        obs, reward, terminated, truncated, info = env.step(action_np)
        total_reward += reward
        
        if terminated or truncated:
            print(f"  Episode ended at step {i+1}")
            break
    
    print(f"  - Steps completed: {min(i+1, 50)}")
    print(f"  - Total reward: {total_reward:.2f}")
    print(f"  - Final forward velocity: {info.get('forward_velocity', 'N/A')}")
    print(f"  - Final torso height: {info.get('torso_height', 'N/A')}")
    
    env.close()
    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("HEXAPOD sCPG ENVIRONMENT TESTS")
    print("=" * 60)
    
    tests = [
        ("MuJoCo Model", test_mujoco_model),
        ("Gymnasium Environment", test_environment),
        ("IMU6 observation mode", test_imu6_environment),
        ("imu6_prev observation mode", test_imu6_prev_environment),
        ("imu_bno10 observation mode", test_imu_bno10_environment),
        ("Spiking CPG Network", test_cpg_network),
        ("Spike Encoding", test_spike_encoding),
        ("Full Pipeline", test_full_pipeline),
    ]
    
    results = []
    
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success, None))
        except Exception as e:
            results.append((name, False, str(e)))
            print(f"\nERROR in {name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    for name, success, error in results:
        status = "PASSED" if success else "FAILED"
        print(f"  {name}: {status}")
        if error:
            print(f"    Error: {error}")
    
    passed = sum(1 for _, s, _ in results if s)
    total = len(results)
    print(f"\nTotal: {passed}/{total} tests passed")
    
    return all(s for _, s, _ in results)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
