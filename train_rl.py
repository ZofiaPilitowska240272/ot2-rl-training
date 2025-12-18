"""
OT-2 RL Training Script with ClearML + W&B Integration
------------------------------------------------------
Train RL agents (PPO, SAC, TD3) to control the Opentrons OT-2 pipette system.

Author: Zofia Pilitowska (240272)
Course: Applied Data Science and Artificial Intelligence
Institution: Breda University of Applied Sciences

Usage:
    # Single run
    python train_rl.py --algorithm PPO --total_timesteps 2000000 \
        --learning_rate 1e-4 --net_arch "512 512" --use_gpu
    
    # With launcher
    python launch_experiments.py --config 4 --use-gpu
"""

import os
import argparse
import numpy as np
from datetime import datetime
import json

import gymnasium as gym
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback

import wandb
from clearml import Task
from ot2_gym_wrapper import OT2GymEnv


# ===================== Custom Evaluation Callback =====================
class DetailedEvalCallback(BaseCallback):
    """Evaluate agent and log detailed metrics to W&B and ClearML."""
    
    def __init__(self, eval_env, eval_freq=10000, n_eval_episodes=10, 
                 save_path="models/best", verbose=1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.save_path = save_path
        self.best_success_rate = -np.inf
        self.best_mean_reward = -np.inf
        self.best_distance = np.inf

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq != 0:
            return True

        if self.verbose > 0:
            print(f"\n{'='*60}")
            print(f"Evaluation at timestep {self.num_timesteps}")
            print(f"{'='*60}")

        # Run evaluation episodes
        successes = []
        distances = []
        lengths = []
        rewards = []

        for _ in range(self.n_eval_episodes):
            obs, info = self.eval_env.reset()
            episode_reward = 0
            episode_length = 0
            done = False

            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.eval_env.step(action)
                episode_reward += reward
                episode_length += 1
                done = terminated or truncated

            successes.append(info.get('success', False))
            distances.append(info.get('distance', 1.0) * 1000)  # to mm
            lengths.append(episode_length)
            rewards.append(episode_reward)

        # Compute stats
        success_rate = np.mean(successes)
        mean_distance = np.mean(distances)
        std_distance = np.std(distances)
        min_distance = np.min(distances)
        mean_length = np.mean(lengths)
        mean_reward = np.mean(rewards)

        # Print results
        if self.verbose > 0:
            print(f"Success rate: {success_rate*100:.1f}% ({sum(successes)}/{self.n_eval_episodes})")
            print(f"Distance: {mean_distance:.2f} ± {std_distance:.2f} mm (min: {min_distance:.2f})")
            print(f"Mean length: {mean_length:.1f} steps")
            print(f"Mean reward: {mean_reward:.2f}")
            print(f"{'='*60}\n")

        # Log to W&B
        if wandb.run is not None:
            wandb.log({
                'eval/success_rate': success_rate,
                'eval/mean_distance_mm': mean_distance,
                'eval/std_distance_mm': std_distance,
                'eval/min_distance_mm': min_distance,
                'eval/mean_length': mean_length,
                'eval/mean_reward': mean_reward,
                'timesteps': self.num_timesteps
            })

        # Save best model (prioritize distance, then success rate)
        improved = False
        if mean_distance < self.best_distance:
            improved = True
        elif mean_distance == self.best_distance and success_rate > self.best_success_rate:
            improved = True
        
        if improved:
            self.best_distance = mean_distance
            self.best_success_rate = success_rate
            self.best_mean_reward = mean_reward
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
            self.model.save(self.save_path)
            if self.verbose > 0:
                print(f"🏆 New best! Distance: {mean_distance:.2f}mm, Success: {success_rate*100:.1f}%")
                print(f"   Saved to: {self.save_path}.zip\n")

        return True


# ===================== Environment Factory =====================
def make_env(seed=0, **kwargs):
    """Create monitored environment."""
    def _init():
        env = OT2GymEnv(**kwargs)
        env = Monitor(env)
        env.reset(seed=seed)
        return env
    set_random_seed(seed)
    return _init


# ===================== Argument Parser =====================
def parse_args():
    parser = argparse.ArgumentParser(description="Train RL agent for OT-2 pipette")
    
    # Core parameters
    parser.add_argument('--algorithm', type=str, default='PPO', 
                       choices=['PPO', 'SAC', 'TD3'])
    parser.add_argument('--total_timesteps', type=int, default=2000000)
    parser.add_argument('--n_envs', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--n_steps', type=int, default=2048)  # PPO only
    parser.add_argument('--seed', type=int, default=42)
    
    # PPO-specific hyperparameters
    parser.add_argument('--clip_range', type=float, default=0.2)
    parser.add_argument('--ent_coef', type=float, default=0.01)
    parser.add_argument('--gae_lambda', type=float, default=0.95)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--vf_coef', type=float, default=0.5)
    
    # SAC/TD3-specific
    parser.add_argument('--tau', type=float, default=0.005)
    parser.add_argument('--buffer_size', type=int, default=200000)
    
    # Environment parameters
    parser.add_argument('--max_steps', type=int, default=500)
    parser.add_argument('--success_threshold', type=float, default=0.005)
    parser.add_argument('--reward_scale', type=float, default=1.0)
    parser.add_argument('--max_velocity', type=float, default=0.05)
    
    # Network architecture (string format: "256 256" or list)
    parser.add_argument('--net_arch', type=str, default='256 256',
                       help='Network architecture as space-separated values (e.g., "256 256")')
    
    # Logging
    parser.add_argument('--run_name', type=str, default=None)
    parser.add_argument('--save_dir', type=str, default='models')
    parser.add_argument('--wandb_project', type=str, default='ot2_rl_final')
    
    # Other
    parser.add_argument('--use_gpu', action='store_true')
    
    return parser.parse_args()


# ===================== Model Creation =====================
def create_model(algorithm, env, args):
    """Create RL model with proper hyperparameters."""
    
    # Parse network architecture
    if isinstance(args.net_arch, str):
        # Convert "256 256" -> [256, 256]
        net_arch = [int(x) for x in args.net_arch.split()]
    else:
        net_arch = args.net_arch
    
    policy_kwargs = {'net_arch': net_arch}
    
    common_params = {
        'policy': 'MlpPolicy',
        'env': env,
        'learning_rate': args.learning_rate,
        'verbose': 1,
        'seed': args.seed,
        'device': 'cuda' if args.use_gpu else 'auto',
        'tensorboard_log': None,
        'policy_kwargs': policy_kwargs
    }

    if algorithm == 'PPO':
        return PPO(
            **common_params,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=10,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_range=args.clip_range,
            ent_coef=args.ent_coef,
            vf_coef=args.vf_coef,
            max_grad_norm=0.5
        )
    elif algorithm == 'SAC':
        return SAC(
            **common_params,
            buffer_size=args.buffer_size,
            batch_size=args.batch_size,
            tau=args.tau,
            gamma=args.gamma,
            learning_starts=10000,
            train_freq=1,
            gradient_steps=1,
            ent_coef='auto'
        )
    elif algorithm == 'TD3':
        return TD3(
            **common_params,
            buffer_size=args.buffer_size,
            batch_size=args.batch_size,
            tau=args.tau,
            gamma=args.gamma,
            learning_starts=10000,
            train_freq=1,
            gradient_steps=1
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")


# ===================== Training Function =====================
def train(args):
    """Main training function with ClearML and W&B."""
    
    # Generate run name if not provided
    if args.run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.run_name = f"{args.algorithm}_{args.total_timesteps//1000}k_{timestamp}"

    # ===================== ClearML Initialization =====================
    # Set W&B API key in environment (before Task.init so ClearML captures it)
    os.environ['WANDB_API_KEY'] = '914ae068e68f7aa8e0bef176d0a165d47f9b4c7f'
    
    task = Task.init(
        project_name='Mentor Group - Uther/Group 2',
        task_name=args.run_name,
        reuse_last_task_id=False
    )
    
    # Configure ClearML
    task.set_base_docker('deanis/2023y2b-rl:latest')
    task.connect(vars(args))
    
    # Execute remotely
    task.execute_remotely(queue_name="default")
    
    print(f"\n{'='*80}")
    print("OT-2 RL CONTROLLER TRAINING")
    print(f"{'='*80}")
    print(f"Author: Zofia Pilitowska (240272)")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Run: {args.run_name}")
    print(f"Config:")
    print(f"  Algorithm: {args.algorithm}")
    print(f"  LR: {args.learning_rate:.2e}")
    print(f"  Network: {args.net_arch}")
    print(f"  Batch: {args.batch_size}")
    print(f"  Velocity: {args.max_velocity}")
    print(f"  Gamma: {args.gamma}")
    print(f"{'='*80}\n")

    # ===================== W&B Initialization =====================
    config = {
        'algorithm': args.algorithm,
        'total_timesteps': args.total_timesteps,
        'n_envs': args.n_envs,
        'learning_rate': args.learning_rate,
        'batch_size': args.batch_size,
        'n_steps': args.n_steps,
        'net_arch': args.net_arch,
        'gamma': args.gamma,
        'gae_lambda': args.gae_lambda,
        'clip_range': args.clip_range,
        'ent_coef': args.ent_coef,
        'vf_coef': args.vf_coef,
        'success_threshold': args.success_threshold,
        'reward_scale': args.reward_scale,
        'max_velocity': args.max_velocity,
        'max_steps': args.max_steps,
        'student_id': '240272',
        'seed': args.seed
    }
    
    run = wandb.init(
        project=args.wandb_project,
        name=args.run_name,
        config=config,
        sync_tensorboard=False,
        monitor_gym=True
    )
    print(f"✓ W&B initialized: {run.url}\n")

    # ===================== Environment Setup =====================
    print("Creating environments...")
    
    env_kwargs = {
        'max_steps': args.max_steps,
        'success_threshold': args.success_threshold,
        'reward_scale': args.reward_scale,
        'max_velocity': args.max_velocity,
        'use_gui': False
    }
    
    # Training environment (vectorized)
    env = DummyVecEnv([
        make_env(seed=args.seed + i, **env_kwargs) 
        for i in range(args.n_envs)
    ])
    
    # Evaluation environment
    eval_env = Monitor(OT2GymEnv(**env_kwargs))
    
    print(f"✓ Created {args.n_envs} training environment(s)")
    print(f"✓ Created evaluation environment\n")

    # ===================== Model Setup =====================
    print(f"Initializing {args.algorithm} model...")
    model = create_model(args.algorithm, env, args)
    
    n_params = sum(p.numel() for p in model.policy.parameters())
    print(f"✓ Model initialized")
    print(f"  Network: {args.net_arch}")
    print(f"  Parameters: {n_params:,}\n")

    # ===================== Callbacks Setup =====================
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(f"{args.save_dir}/best", exist_ok=True)
    os.makedirs(f"{args.save_dir}/checkpoints", exist_ok=True)
    
    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=max(50000 // args.n_envs, 1),
        save_path=f"{args.save_dir}/checkpoints",
        name_prefix=f"{args.run_name}",
        verbose=1
    )
    
    # Evaluation callback
    eval_callback = DetailedEvalCallback(
        eval_env=eval_env,
        eval_freq=max(10000 // args.n_envs, 1),
        n_eval_episodes=10,
        save_path=f"{args.save_dir}/best/{args.run_name}_best",
        verbose=1
    )
    
    callbacks = CallbackList([checkpoint_callback, eval_callback])
    
    print("✓ Callbacks configured\n")

    # ===================== Training =====================
    print("="*80)
    print("STARTING TRAINING")
    print("="*80)
    print(f"Total timesteps: {args.total_timesteps:,}")
    print(f"Expected duration: ~{args.total_timesteps / 35 / 3600:.1f} hours")
    print("="*80 + "\n")

    try:
        start_time = datetime.now()
        
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=callbacks,
            log_interval=10,
            progress_bar=True
        )
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        # ===================== Save Models =====================
        print("\n" + "="*80)
        print("SAVING MODELS")
        print("="*80)
        
        # Save final model
        final_path = f"{args.save_dir}/{args.run_name}_final"
        model.save(final_path)
        print(f"✓ Final model: {final_path}.zip")
        
        # Upload to ClearML
        task.upload_artifact('final_model', f"{final_path}.zip")
        
        best_path = f"{args.save_dir}/best/{args.run_name}_best.zip"
        if os.path.exists(best_path):
            task.upload_artifact('best_model', best_path)
            print(f"✓ Best model: {best_path}")
        
        print(f"✓ Models uploaded to ClearML")
        
        # Upload to W&B
        wandb.save(f"{final_path}.zip")
        if os.path.exists(best_path):
            wandb.save(best_path)
        print(f"✓ Models uploaded to W&B\n")

        # ===================== Training Summary =====================
        print("="*80)
        print("TRAINING COMPLETED!")
        print("="*80)
        print(f"Duration: {duration/3600:.2f} hours")
        print(f"Steps/sec: {args.total_timesteps/duration:.1f}")
        print(f"Best distance: {eval_callback.best_distance:.2f} mm")
        print(f"Best success rate: {eval_callback.best_success_rate*100:.1f}%")
        print("="*80 + "\n")
        
        # Save summary
        summary = {
            'run_name': args.run_name,
            'algorithm': args.algorithm,
            'total_timesteps': args.total_timesteps,
            'duration_hours': duration / 3600,
            'best_distance_mm': eval_callback.best_distance,
            'best_success_rate': eval_callback.best_success_rate,
            'hyperparameters': vars(args),
            'clearml_task_id': task.id,
            'wandb_url': run.url,
            'timestamp': datetime.now().isoformat()
        }
        
        summary_path = f"{args.save_dir}/training_summary_{args.run_name}.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        task.upload_artifact('training_summary', summary_path)

    except KeyboardInterrupt:
        print("\n⚠️  TRAINING INTERRUPTED")
        interrupted_path = f"{args.save_dir}/{args.run_name}_interrupted"
        model.save(interrupted_path)
        task.upload_artifact('interrupted_model', f"{interrupted_path}.zip")
        print(f"✓ Interrupted model saved: {interrupted_path}.zip")

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Cleanup
        env.close()
        eval_env.close()
        wandb.finish()
        print("✓ Training completed\n")


# ===================== Main =====================
def main():
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()