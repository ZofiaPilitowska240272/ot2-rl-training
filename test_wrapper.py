"""
Test script for OT2GymEnv wrapper.

This script validates the Gymnasium environment implementation by running
1000 steps with random actions and collecting performance statistics.
"""

import numpy as np
import time
from ot2_gym_wrapper import OT2GymEnv
import matplotlib.pyplot as plt
from collections import defaultdict


def test_environment_basic():
    """Test basic environment functionality."""
    print("TEST 1: Basic Environment Functionality")
    
    env = OT2GymEnv(use_gui=False, max_steps=100)
    
    # Test spaces
    print("\n1. Testing action and observation spaces")
    print(f"   Action space: {env.action_space}")
    print(f"   Action space shape: {env.action_space.shape}")
    print(f"   Observation space: {env.observation_space}")
    print(f"   Observation space shape: {env.observation_space.shape}")
    
    # Test reset
    print("\n2. Testing reset functionality")
    obs, info = env.reset(seed=42)
    print(f"   Observation shape: {obs.shape}")
    print(f"   Initial distance: {info['initial_distance']:.4f}m")
    print(f"   Target position: {info['target_position']}")
    assert env.observation_space.contains(obs), "Observation not in observation space!"
    print("   Reset test passed")
    
    # Test step
    print("\n3. Testing step functionality")
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"   Action: {action}")
    print(f"   Reward: {reward:.4f}")
    print(f"   Terminated: {terminated}, Truncated: {truncated}")
    print(f"   Distance: {info['distance']:.4f}m")
    assert env.observation_space.contains(obs), "Observation not in observation space!"
    assert env.action_space.contains(action), "Action not in action space!"
    print("   Step test passed")
    
    env.close()
    print("\nEnd of basics tests\n")


def test_random_policy(n_steps=1000, n_episodes=10):
    """
    Test environment with random policy for specified number of steps.
    
    Args:
        n_steps: Total number of steps to run
        n_episodes: Number of episodes to run
    """

    print(f"TEST 2: Random Policy - {n_steps} steps over {n_episodes} episodes")
    
    env = OT2GymEnv(use_gui=False, max_steps=100)
    
    # Statistics tracking
    stats = {
        'episode_lengths': [],
        'episode_rewards': [],
        'success_count': 0,
        'out_of_bounds_count': 0,
        'distances': [],
        'actions': [],
        'step_times': []
    }
    
    total_steps = 0
    episode = 0
    
    start_time = time.time()
    
    while episode < n_episodes and total_steps < n_steps:
        obs, info = env.reset(seed=episode)
        episode_reward = 0
        episode_length = 0
        done = False
        
        while not done and total_steps < n_steps:
            step_start = time.time()
            
            # Random action
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            step_time = time.time() - step_start
            stats['step_times'].append(step_time)
            
            episode_reward += reward
            episode_length += 1
            total_steps += 1
            done = terminated or truncated
            
            # Track statistics
            stats['distances'].append(info['distance'])
            stats['actions'].append(action)
            
        
        # Episode finished
        stats['episode_lengths'].append(episode_length)
        stats['episode_rewards'].append(episode_reward)
        
        if info['success']:
            stats['success_count'] += 1
        
        if not env._is_within_bounds(info['position']):
            stats['out_of_bounds_count'] += 1
        
        episode += 1
    
    total_time = time.time() - start_time
    env.close()
    
    
    print("RESULTS")
    
    print(f"\nExecution Statistics:")
    print(f"  Total steps: {total_steps}")
    print(f"  Total episodes: {episode}")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Steps per second: {total_steps / total_time:.1f}")
    print(f"  Avg step time: {np.mean(stats['step_times']) * 1000:.2f}ms")
    
    print(f"\nEpisode Statistics:")
    print(f"  Avg episode length: {np.mean(stats['episode_lengths']):.1f} steps")
    print(f"  Std episode length: {np.std(stats['episode_lengths']):.1f} steps")
    print(f"  Min/Max episode length: {min(stats['episode_lengths'])}/{max(stats['episode_lengths'])}")
    
    print(f"\nReward Statistics:")
    print(f"  Avg episode reward: {np.mean(stats['episode_rewards']):.2f}")
    print(f"  Std episode reward: {np.std(stats['episode_rewards']):.2f}")
    print(f"  Min/Max episode reward: {min(stats['episode_rewards']):.2f}/{max(stats['episode_rewards']):.2f}")
    
    print(f"\nPerformance Statistics:")
    print(f"  Success rate: {stats['success_count'] / episode * 100:.1f}% ({stats['success_count']}/{episode})")
    print(f"  Out of bounds rate: {stats['out_of_bounds_count'] / episode * 100:.1f}% ({stats['out_of_bounds_count']}/{episode})")
    print(f"  Avg distance to target: {np.mean(stats['distances']):.4f}m")
    print(f"  Min distance achieved: {min(stats['distances']):.4f}m")
    
    print(f"\nAction Statistics:")
    actions_array = np.array(stats['actions'])
    print(f"  Action mean: {np.mean(actions_array, axis=0)}")
    print(f"  Action std: {np.std(actions_array, axis=0)}")
    print(f"  Action min: {np.min(actions_array, axis=0)}")
    print(f"  Action max: {np.max(actions_array, axis=0)}")
    
    return stats


def test_specific_scenarios():
    """Test specific scenarios and edge cases."""
    print("TEST 3: Specific Scenarios")
    
    env = OT2GymEnv(use_gui=False, max_steps=50)
    
    # Test 1: Target at workspace center
    print("\n1. Target at workspace center")
    target = np.array([0.03, 0.02, 0.20])
    obs, info = env.reset(options={'target_position': target})
    print(f"   Target: {target}")
    print(f"   Initial distance: {info['initial_distance']:.4f}m")
    
    # Test 2: Target at workspace corner
    print("\n2. Target at workspace corner")
    target = np.array([0.15, 0.15, 0.15]) 
    obs, info = env.reset(options={'target_position': target})
    print(f"   Target: {target}")
    print(f"   Initial distance: {info['initial_distance']:.4f}m")
    
    # Test 3: Zero action (should not move much due to velocity)
    print("\n3. Testing zero action")
    obs, info = env.reset()
    initial_pos = obs[:3].copy()  #  Get CURRENT position from observation
    action = np.zeros(3)
    for _ in range(5):  # Test over multiple steps
        obs, reward, terminated, truncated, info = env.step(action)
    final_pos = info['position']
    movement = np.linalg.norm(final_pos - initial_pos)
    print(f"   Position change: {movement:.6f}m")

    # Test 4: Maximum action
    print("\n4. Testing maximum action")
    obs, info = env.reset()
    action = np.array([1.0, 1.0, 1.0])
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"   Action: {action}")
    print(f"   Reward: {reward:.4f}")
    
    env.close()
    print("\nScenario tests completed!\n")


def plot_statistics(stats):
    """
    Create visualization plots for test statistics.
    
    Args:
        stats: Dictionary containing test statistics
    """
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('OT2GymEnv Test Statistics', fontsize=16, fontweight='bold')
    
    # Plot 1: Episode lengths
    axes[0, 0].hist(stats['episode_lengths'], bins=20, edgecolor='black', alpha=0.7)
    axes[0, 0].set_xlabel('Episode Length (steps)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Distribution of Episode Lengths')
    axes[0, 0].axvline(np.mean(stats['episode_lengths']), color='r', 
                       linestyle='--', label=f'Mean: {np.mean(stats["episode_lengths"]):.1f}')
    axes[0, 0].legend()
    
    # Plot 2: Episode rewards
    axes[0, 1].hist(stats['episode_rewards'], bins=20, edgecolor='black', alpha=0.7, color='green')
    axes[0, 1].set_xlabel('Episode Reward')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Distribution of Episode Rewards')
    axes[0, 1].axvline(np.mean(stats['episode_rewards']), color='r', 
                       linestyle='--', label=f'Mean: {np.mean(stats["episode_rewards"]):.1f}')
    axes[0, 1].legend()
    
    # Plot 3: Distance over time
    axes[0, 2].plot(stats['distances'], alpha=0.6, linewidth=0.5)
    axes[0, 2].set_xlabel('Step')
    axes[0, 2].set_ylabel('Distance to Target (m)')
    axes[0, 2].set_title('Distance to Target Over Time')
    axes[0, 2].axhline(0.005, color='r', linestyle='--', label='Success threshold')
    axes[0, 2].legend()
    
    # Plot 4: Action distribution (X component)
    actions_array = np.array(stats['actions'])
    axes[1, 0].hist(actions_array[:, 0], bins=30, edgecolor='black', alpha=0.7)
    axes[1, 0].set_xlabel('Action X')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Distribution of X Actions')
    
    # Plot 5: Action distribution (Y component)
    axes[1, 1].hist(actions_array[:, 1], bins=30, edgecolor='black', alpha=0.7, color='orange')
    axes[1, 1].set_xlabel('Action Y')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Distribution of Y Actions')
    
    # Plot 6: Action distribution (Z component)
    axes[1, 2].hist(actions_array[:, 2], bins=30, edgecolor='black', alpha=0.7, color='purple')
    axes[1, 2].set_xlabel('Action Z')
    axes[1, 2].set_ylabel('Frequency')
    axes[1, 2].set_title('Distribution of Z Actions')
    
    plt.tight_layout()
    plt.savefig('test_wrapper_results.png', dpi=150)
    print("Plots saved as 'test_wrapper_results.png'\n")


def main():
    """Run all tests."""
    print("OT2 GYMNASIUM ENVIRONMENT TEST SUITE")
    print(f"Testing environment with random policy\n")
    
    # Run tests
    try:
        # Test 1: Basic functionality
        test_environment_basic()
        
        # Test 2: Random policy for 1000 steps
        stats = test_random_policy(n_steps=1000, n_episodes=10)
        
        # Test 3: Specific scenarios
        test_specific_scenarios()
        
        # Generate plots
        plot_statistics(stats)
        
        print("=" * 60)
        print("ALL TESTS COMPLETED SUCCESSFULLY! ✓")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())