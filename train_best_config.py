"""
Train final model using best hyperparameters from sweep.

Usage:
    python train_best_config.py --config best_config_<sweep_id>.yaml
    
    # Or manually specify parameters
    python train_best_config.py \
        --algorithm PPO \
        --learning_rate 0.0002 \
        --batch_size 256 \
        --net_arch 512 512 \
        --total_timesteps 5000000
        
Author: Zofia Pilitowska (240272)
"""

import argparse
import yaml
from train_rl_sweep import train
import wandb


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def parse_args():
    parser = argparse.ArgumentParser(description="Train with best config")
    
    # Config file option
    parser.add_argument('--config', type=str, help='Path to best config YAML')
    
    # Or manual specification (same as train_rl_sweep.py)
    parser.add_argument('--algorithm', type=str, default='PPO', choices=['PPO', 'SAC', 'TD3'])
    parser.add_argument('--total_timesteps', type=int, default=5000000)
    parser.add_argument('--n_envs', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--n_steps', type=int, default=2048)
    parser.add_argument('--net_arch', nargs='+', type=int, default=[512, 512])
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--gae_lambda', type=float, default=0.95)
    parser.add_argument('--clip_range', type=float, default=0.2)
    parser.add_argument('--ent_coef', type=float, default=0.01)
    parser.add_argument('--vf_coef', type=float, default=0.5)
    parser.add_argument('--tau', type=float, default=0.005)
    parser.add_argument('--buffer_size', type=int, default=200000)
    parser.add_argument('--max_steps', type=int, default=500)
    parser.add_argument('--success_threshold', type=float, default=0.005)
    parser.add_argument('--reward_scale', type=float, default=1.0)
    parser.add_argument('--max_velocity', type=float, default=0.05)
    parser.add_argument('--seed', type=int, default=42)
    
    # Training settings
    parser.add_argument('--run_name', type=str, default=None)
    parser.add_argument('--save_dir', type=str, default='models/final')
    parser.add_argument('--wandb_project', type=str, default='ot2_rl_final')
    parser.add_argument('--use_gpu', action='store_true')
    parser.add_argument('--use_clearml', action='store_true')
    
    # Additional runs for robustness
    parser.add_argument('--n_runs', type=int, default=1, help='Number of training runs with different seeds')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load config from file if provided
    if args.config:
        print(f"Loading configuration from: {args.config}")
        config_dict = load_config(args.config)
        
        # Update args with config
        for key, value in config_dict.items():
            if hasattr(args, key) and value is not None:
                setattr(args, key, value)
        
        print(f"✓ Configuration loaded")
        print(f"  Algorithm: {args.algorithm}")
        print(f"  Learning Rate: {args.learning_rate:.2e}")
        print(f"  Batch Size: {args.batch_size}")
        print(f"  Network: {args.net_arch}")
        print()
    
    # Run training with best config
    base_seed = args.seed
    
    for run_idx in range(args.n_runs):
        print(f"\n{'='*80}")
        print(f"TRAINING RUN {run_idx + 1}/{args.n_runs}")
        print(f"{'='*80}\n")
        
        # Set unique seed for each run
        args.seed = base_seed + run_idx * 100
        
        # Set unique run name
        if args.run_name is None:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            args.run_name = f"best_{args.algorithm}_{timestamp}_seed{args.seed}"
        else:
            args.run_name = f"{args.run_name}_run{run_idx}_seed{args.seed}"
        
        # Convert args to dict for train function
        config_dict = vars(args)
        
        # Train
        try:
            train(config=config_dict)
        except Exception as e:
            print(f"\n❌ Run {run_idx + 1} failed with error: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        print(f"\n✓ Run {run_idx + 1}/{args.n_runs} completed")
    
    print(f"\n{'='*80}")
    print(f"ALL TRAINING RUNS COMPLETED")
    print(f"{'='*80}")
    print(f"Completed {args.n_runs} run(s)")
    print(f"Models saved to: {args.save_dir}")
    print(f"Check W&B project: {args.wandb_project}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()