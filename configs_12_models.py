"""
12 Hand-Picked Configurations for OT-2 RL Training
---------------------------------------------------
These configs are designed to explore the hyperparameter space systematically.

Author: Zofia Pilitowska (240272)
"""

# 12 carefully selected configurations
CONFIGS = [
    # ========== CONSERVATIVE CONFIGS (Lower LR, More Stable) ==========
    {
        'name': 'ppo_256x2_lr5e5_b256',
        'description': 'Stable baseline - small network, safe learning rate',
        'algorithm': 'PPO',
        'total_timesteps': 2000000,
        'learning_rate': 5e-5,
        'batch_size': 256,
        'n_steps': 2048,
        'net_arch': '256 256',
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_range': 0.2,
        'ent_coef': 0.001,
        'vf_coef': 0.5,
        'max_velocity': 0.03,
        'reward_scale': 0.5,
        'max_steps': 700,
        'n_envs': 4,
        'seed': 42
    },
    
    {
        'name': 'ppo_512x2_lr5e5_b512_vel025',
        'description': 'Very slow movement (0.025 m/s), large net, long episodes',
        'algorithm': 'PPO',
        'total_timesteps': 2000000,
        'learning_rate': 5e-5,
        'batch_size': 512,
        'n_steps': 2048,
        'net_arch': '512 512',
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_range': 0.2,
        'ent_coef': 0.001,
        'vf_coef': 0.5,
        'max_velocity': 0.025,
        'reward_scale': 0.5,
        'max_steps': 1000,
        'n_envs': 4,
        'seed': 42
    },
    
    {
        'name': 'ppo_256x3_lr3e5_g995',
        'description': 'Deep 3-layer net, highest gamma for long-term precision',
        'algorithm': 'PPO',
        'total_timesteps': 2000000,
        'learning_rate': 3e-5,
        'batch_size': 256,
        'n_steps': 4096,
        'net_arch': '256 256 256',
        'gamma': 0.995,
        'gae_lambda': 0.95,
        'clip_range': 0.15,
        'ent_coef': 0.0005,
        'vf_coef': 0.5,
        'max_velocity': 0.03,
        'reward_scale': 0.8,
        'max_steps': 700,
        'n_envs': 4,
        'seed': 123
    },
    
    # ========== MODERATE CONFIGS (Balanced) ==========
    {
        'name': 'ppo_512x2_lr1e4_env8',
        'description': 'Balanced config with 8 parallel envs for speed',
        'algorithm': 'PPO',
        'total_timesteps': 2000000,
        'learning_rate': 1e-4,
        'batch_size': 256,
        'n_steps': 2048,
        'net_arch': '512 512',
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_range': 0.2,
        'ent_coef': 0.005,
        'vf_coef': 0.5,
        'max_velocity': 0.04,
        'reward_scale': 0.5,
        'max_steps': 500,
        'n_envs': 8,
        'seed': 42
    },
    
    {
        'name': 'ppo_512x2_lr1e4_vel02_s1000',
        'description': 'RECOMMENDED: Slowest velocity (0.02), longest episodes, precision-focused',
        'algorithm': 'PPO',
        'total_timesteps': 2000000,
        'learning_rate': 1e-4,
        'batch_size': 512,
        'n_steps': 2048,
        'net_arch': '512 512',
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_range': 0.2,
        'ent_coef': 0.001,
        'vf_coef': 0.5,
        'max_velocity': 0.02,
        'reward_scale': 1.0,
        'max_steps': 1000,
        'n_envs': 4,
        'seed': 456
    },
    
    {
        'name': 'ppo_256x2_lr8e5_clip25',
        'description': 'Well-balanced reliable config, good all-around performance',
        'algorithm': 'PPO',
        'total_timesteps': 2000000,
        'learning_rate': 8e-5,
        'batch_size': 256,
        'n_steps': 4096,
        'net_arch': '256 256',
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_range': 0.25,
        'ent_coef': 0.003,
        'vf_coef': 0.5,
        'max_velocity': 0.035,
        'reward_scale': 0.7,
        'max_steps': 700,
        'n_envs': 4,
        'seed': 42
    },
    
    # ========== AGGRESSIVE CONFIGS (Higher LR, Faster Learning) ==========
    {
        'name': 'ppo_512x2_lr3e4_env8',
        'description': 'Fastest learning rate (3e-4), fastest velocity, 8 envs',
        'algorithm': 'PPO',
        'total_timesteps': 2000000,
        'learning_rate': 3e-4,
        'batch_size': 512,
        'n_steps': 2048,
        'net_arch': '512 512',
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_range': 0.3,
        'ent_coef': 0.01,
        'vf_coef': 0.5,
        'max_velocity': 0.05,
        'reward_scale': 0.3,
        'max_steps': 500,
        'n_envs': 8,
        'seed': 42
    },
    
    {
        'name': 'ppo_512x3_lr2e4_b256',
        'description': 'Widest & deepest network (512x3), good for complex control',
        'algorithm': 'PPO',
        'total_timesteps': 2000000,
        'learning_rate': 2e-4,
        'batch_size': 256,
        'n_steps': 2048,
        'net_arch': '512 512 512',
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_range': 0.2,
        'ent_coef': 0.005,
        'vf_coef': 0.5,
        'max_velocity': 0.03,
        'reward_scale': 0.5,
        'max_steps': 700,
        'n_envs': 4,
        'seed': 123
    },
    
    {
        'name': 'ppo_256x2_lr2e4_ent01_env8',
        'description': 'Maximum exploration (high entropy), parallel training',
        'algorithm': 'PPO',
        'total_timesteps': 2000000,
        'learning_rate': 2e-4,
        'batch_size': 256,
        'n_steps': 4096,
        'net_arch': '256 256',
        'gamma': 0.98,
        'gae_lambda': 0.9,
        'clip_range': 0.25,
        'ent_coef': 0.01,
        'vf_coef': 0.3,
        'max_velocity': 0.04,
        'reward_scale': 0.5,
        'max_steps': 500,
        'n_envs': 8,
        'seed': 456
    },
    
    # ========== SPECIALIZED CONFIGS (Edge Cases) ==========
    {
        'name': 'ppo_512x2_lr3e5_vel02_g995',
        'description': 'MAXIMUM PRECISION: slowest movement, highest gamma, minimal entropy',
        'algorithm': 'PPO',
        'total_timesteps': 2000000,
        'learning_rate': 3e-5,
        'batch_size': 512,
        'n_steps': 4096,
        'net_arch': '512 512',
        'gamma': 0.995,
        'gae_lambda': 0.98,
        'clip_range': 0.15,
        'ent_coef': 0.0001,
        'vf_coef': 0.5,
        'max_velocity': 0.02,
        'reward_scale': 1.0,
        'max_steps': 1000,
        'n_envs': 4,
        'seed': 42
    },
    
    {
        'name': 'ppo_512x2_lr15e4_b1024_env8',
        'description': 'Largest batch size (1024), 8 parallel envs, fast throughput',
        'algorithm': 'PPO',
        'total_timesteps': 2000000,
        'learning_rate': 1.5e-4,
        'batch_size': 1024,
        'n_steps': 2048,
        'net_arch': '512 512',
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_range': 0.2,
        'ent_coef': 0.005,
        'vf_coef': 0.5,
        'max_velocity': 0.035,
        'reward_scale': 0.5,
        'max_steps': 700,
        'n_envs': 8,
        'seed': 789
    },
    
    {
        'name': 'ppo_256x3_lr8e5_step8192_g995',
        'description': 'Longest horizon planning (8192 steps), highest gamma, deep net',
        'algorithm': 'PPO',
        'total_timesteps': 2000000,
        'learning_rate': 8e-5,
        'batch_size': 256,
        'n_steps': 8192,
        'net_arch': '256 256 256',
        'gamma': 0.995,
        'gae_lambda': 0.97,
        'clip_range': 0.2,
        'ent_coef': 0.002,
        'vf_coef': 0.5,
        'max_velocity': 0.03,
        'reward_scale': 0.8,
        'max_steps': 1000,
        'n_envs': 4,
        'seed': 42
    }
]


def print_configs_summary():
    """Print a summary of all 12 configurations."""
    print("\n" + "="*100)
    print("12 RL CONFIGURATIONS - TECHNICAL NAMES")
    print("="*100)
    
    categories = [
        ("CONSERVATIVE (Stable, Safe Learning)", 0, 3),
        ("MODERATE (Balanced, Recommended)", 3, 6),
        ("AGGRESSIVE (Fast Learning, Higher Risk)", 6, 9),
        ("SPECIALIZED (Optimized for Specific Goals)", 9, 12)
    ]
    
    for cat_name, start, end in categories:
        print(f"\n{cat_name}")
        print("-" * 100)
        for i in range(start, end):
            cfg = CONFIGS[i]
            print(f"{i}. {cfg['name']}")
            print(f"   {cfg['description']}")
    
    print("\n" + "="*100)
    print("QUICK START:")
    print("  Recommended:  python launch_experiments.py --config 4 --use-gpu")
    print("  Best 3:       python launch_experiments.py --config 1 4 9 --use-gpu")
    print("  All 12:       python launch_experiments.py --all --use-gpu")
    print("="*100 + "\n")


def get_config(index):
    """Get configuration by index (0-11)."""
    if 0 <= index < len(CONFIGS):
        return CONFIGS[index]
    else:
        raise ValueError(f"Config index must be 0-11, got {index}")


def get_config_by_name(name):
    """Get configuration by name."""
    for cfg in CONFIGS:
        if cfg['name'] == name:
            return cfg
    raise ValueError(f"Config '{name}' not found")


if __name__ == "__main__":
    print_configs_summary()