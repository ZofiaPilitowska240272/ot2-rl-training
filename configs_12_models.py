"""
12 Hand-Picked Configurations for OT-2 RL Training
---------------------------------------------------
These configs are designed to explore the hyperparameter space systematically.

Author: Zofia Pilitowska (240272)
"""

# 12 SPEED-OPTIMIZED configurations
CONFIGS = [
    # ========== FAST PPO - HIGH PARALLELISM (Run on CPU) ==========
    {
        'name': 'ppo_256x2_lr1e4_env16',
        'description': 'Fast baseline - 16 parallel envs',
        'algorithm': 'PPO',
        'total_timesteps': 1000000,
        'learning_rate': 1e-4,
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
        'max_steps': 400,
        'n_envs': 16,
        'seed': 42
    },
    
    {
        'name': 'ppo_512x2_lr1e4_env16_vel02',
        'description': 'BEST BET: 16 envs, slow velocity (0.02), large network',
        'algorithm': 'PPO',
        'total_timesteps': 1000000,
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
        'max_steps': 500,
        'n_envs': 16,
        'seed': 42
    },
    
    {
        'name': 'ppo_256x3_lr5e5_env24_g995',
        'description': '24 envs! Deep network, high gamma for precision',
        'algorithm': 'PPO',
        'total_timesteps': 1000000,
        'learning_rate': 5e-5,
        'batch_size': 256,
        'n_steps': 2048,
        'net_arch': '256 256 256',
        'gamma': 0.995,
        'gae_lambda': 0.95,
        'clip_range': 0.15,
        'ent_coef': 0.0005,
        'vf_coef': 0.5,
        'max_velocity': 0.03,
        'reward_scale': 0.8,
        'max_steps': 400,
        'n_envs': 24,
        'seed': 123
    },
    
    {
        'name': 'ppo_512x2_lr8e5_env24',
        'description': '24 envs, balanced, very fast training',
        'algorithm': 'PPO',
        'total_timesteps': 1000000,
        'learning_rate': 8e-5,
        'batch_size': 512,
        'n_steps': 2048,
        'net_arch': '512 512',
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_range': 0.2,
        'ent_coef': 0.003,
        'vf_coef': 0.5,
        'max_velocity': 0.035,
        'reward_scale': 0.7,
        'max_steps': 400,
        'n_envs': 24,
        'seed': 456
    },
    
    # ========== ULTRA-FAST PPO (32 envs) ==========
    {
        'name': 'ppo_256x2_lr2e4_env32',
        'description': 'FASTEST: 32 parallel envs, aggressive learning',
        'algorithm': 'PPO',
        'total_timesteps': 1000000,
        'learning_rate': 2e-4,
        'batch_size': 512,
        'n_steps': 1024,
        'net_arch': '256 256',
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_range': 0.3,
        'ent_coef': 0.01,
        'vf_coef': 0.5,
        'max_velocity': 0.04,
        'reward_scale': 0.5,
        'max_steps': 300,
        'n_envs': 32,
        'seed': 42
    },
    
    {
        'name': 'ppo_512x2_lr1e4_env32_bal',
        'description': '32 envs, balanced LR, excellent speed',
        'algorithm': 'PPO',
        'total_timesteps': 1000000,
        'learning_rate': 1e-4,
        'batch_size': 512,
        'n_steps': 1024,
        'net_arch': '512 512',
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_range': 0.2,
        'ent_coef': 0.005,
        'vf_coef': 0.5,
        'max_velocity': 0.03,
        'reward_scale': 0.5,
        'max_steps': 400,
        'n_envs': 32,
        'seed': 123
    },
    
    # ========== PRECISION PPO (16 envs) ==========
    {
        'name': 'ppo_512x2_lr5e5_env16_vel025',
        'description': 'Very slow velocity (0.025), 16 envs, stable',
        'algorithm': 'PPO',
        'total_timesteps': 1000000,
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
        'max_steps': 500,
        'n_envs': 16,
        'seed': 789
    },
    
    {
        'name': 'ppo_512x2_lr3e5_env16_vel02_g995',
        'description': 'PRECISION: slowest velocity (0.02), highest gamma, 16 envs',
        'algorithm': 'PPO',
        'total_timesteps': 1000000,
        'learning_rate': 3e-5,
        'batch_size': 512,
        'n_steps': 2048,
        'net_arch': '512 512',
        'gamma': 0.995,
        'gae_lambda': 0.98,
        'clip_range': 0.15,
        'ent_coef': 0.0001,
        'vf_coef': 0.5,
        'max_velocity': 0.02,
        'reward_scale': 1.0,
        'max_steps': 500,
        'n_envs': 16,
        'seed': 42
    },
    
    # ========== SAC - GPU OPTIMIZED ==========
    {
        'name': 'sac_512x2_lr3e4_gpu',
        'description': 'SAC on GPU, fast learning, good for continuous control',
        'algorithm': 'SAC',
        'total_timesteps': 1000000,
        'learning_rate': 3e-4,
        'batch_size': 512,
        'n_steps': 2048,
        'net_arch': '512 512',
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_range': 0.2,
        'ent_coef': 'auto',
        'vf_coef': 0.5,
        'tau': 0.005,
        'buffer_size': 200000,
        'max_velocity': 0.03,
        'reward_scale': 1.0,
        'max_steps': 500,
        'n_envs': 1,
        'seed': 42
    },
    
    {
        'name': 'sac_512x2_lr1e4_vel02_gpu',
        'description': 'SAC precision config, slow velocity, GPU optimized',
        'algorithm': 'SAC',
        'total_timesteps': 1000000,
        'learning_rate': 1e-4,
        'batch_size': 512,
        'n_steps': 2048,
        'net_arch': '512 512',
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_range': 0.2,
        'ent_coef': 'auto',
        'vf_coef': 0.5,
        'tau': 0.005,
        'buffer_size': 300000,
        'max_velocity': 0.02,
        'reward_scale': 1.0,
        'max_steps': 500,
        'n_envs': 1,
        'seed': 456
    },
    
    {
        'name': 'sac_256x3_lr2e4_gpu',
        'description': 'SAC deep network, GPU, good capacity',
        'algorithm': 'SAC',
        'total_timesteps': 1000000,
        'learning_rate': 2e-4,
        'batch_size': 256,
        'n_steps': 2048,
        'net_arch': '256 256 256',
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_range': 0.2,
        'ent_coef': 'auto',
        'vf_coef': 0.5,
        'tau': 0.005,
        'buffer_size': 200000,
        'max_velocity': 0.035,
        'reward_scale': 0.7,
        'max_steps': 500,
        'n_envs': 1,
        'seed': 123
    }
]


def print_configs_summary():
    """Print a summary of all 12 configurations."""
    print("\n" + "="*100)
    print("12 SPEED-OPTIMIZED RL CONFIGURATIONS")
    print("="*100)
    
    categories = [
        ("FAST PPO - HIGH PARALLELISM (16-24 envs, CPU)", 0, 4),
        ("ULTRA-FAST PPO (32 envs, CPU)", 4, 6),
        ("PRECISION PPO (16 envs, CPU)", 6, 8),
        ("SAC - GPU OPTIMIZED (1 env, GPU)", 8, 12)
    ]
    
    for cat_name, start, end in categories:
        print(f"\n{cat_name}")
        print("-" * 100)
        for i in range(start, end):
            cfg = CONFIGS[i]
            device = "GPU" if cfg['algorithm'] == 'SAC' else "CPU"
            timesteps = f"{cfg['total_timesteps']/1e6:.1f}M"
            print(f"{i}. {cfg['name']} [{device}, {timesteps}, {cfg['n_envs']} envs]")
            print(f"   {cfg['description']}")
    
    print("\n" + "="*100)
    print("SPEED IMPROVEMENTS:")
    print("  • PPO: 16-32 parallel envs (was 4-8) → 4-8x faster!")
    print("  • All: 1M timesteps (was 2M) → 2x faster!")
    print("  • Shorter episodes (300-500 steps) → faster iterations")
    print("  • SAC uses GPU properly")
    print("\nESTIMATED TIME PER CONFIG:")
    print("  • PPO (16 envs): 1-2 hours (was 15+ hours)")
    print("  • PPO (24 envs): 45-90 min")
    print("  • PPO (32 envs): 30-60 min")
    print("  • SAC (GPU): 2-3 hours")
    print("\nTOTAL FOR ALL 12: 12-20 hours (was 100+ hours)")
    print("="*100)
    print("\nQUICK START:")
    print("  Best 3:  python launch_experiments.py --config 1 7 9")
    print("  All 12:  python queue_all_configs.py")
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