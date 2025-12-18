"""
OT-2 RL Training Script with W&B Sweep Support
-----------------------------------------------
Train RL agents with hyperparameter optimization using W&B sweeps.

Author: Zofia Pilitowska (240272)
Course: Applied Data Science and Artificial Intelligence
Institution: Breda University of Applied Sciences

Usage:
    # Single run:
    python train_rl_sweep.py --algorithm PPO --total_timesteps 2000000
    
    # Sweep agent:
    wandb agent <sweep_id>
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
    """Evaluate agent and log detailed metrics to W&B."""
    
    def __init__(self, eval_env, eval_freq=10000, n_eval_episodes=10, 
                 save_path="models/best", verbose=1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.save_path = save_path
        self.best_success_rate = -np.inf
        self.best_mean_reward = -np.inf
        self.best_mean_distance = np.inf

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
        under_1mm = []
        under_2mm = []
        under_5mm = []

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

            dist_m = info.get('distance', 1.0)
            dist_mm = dist_m * 1000
            
            successes.append(info.get('success', False))
            distances.append(dist_mm)
            lengths.append(episode_length)
            rewards.append(episode_reward)
            
            # Track distance thresholds
            under_1mm.append(dist_mm < 1.0)
            under_2mm.append(dist_mm < 2.0)
            under_5mm.append(dist_mm < 5.0)

        # Compute stats
        success_rate = np.mean(successes)
        mean_distance = np.mean(distances)
        std_distance = np.std(distances)
        min_distance = np.min(distances)
        max_distance = np.max(distances)
        mean_length = np.mean(lengths)
        mean_reward = np.mean(rewards)
        
        # Success rates at different thresholds
        success_1mm = np.mean(under_1mm)
        success_2mm = np.mean(under_2mm)
        success_5mm = np.mean(under_5mm)

        # Print results
        if self.verbose > 0:
            print(f"Success rate (5mm): {success_rate*100:.1f}% ({sum(successes)}/{self.n_eval_episodes})")
            print(f"Success rate (1mm): {success_1mm*100:.1f}%")
            print(f"Success rate (2mm): {success_2mm*100:.1f}%")
            print(f"Mean distance: {mean_distance:.2f} ± {std_distance:.2f} mm")
            print(f"Min/Max distance: {min_distance:.2f} / {max_distance:.2f} mm")
            print(f"Mean length: {mean_length:.1f} steps")
            print(f"Mean reward: {mean_reward:.2f}")
            print(f"{'='*60}\n")

        # Log to W&B
        if wandb.run is not None:
            wandb.log({
                'eval/success_rate': success_rate,
                'eval/success_rate_1mm': success_1mm,
                'eval/success_rate_2mm': success_2mm,
                'eval/success_rate_5mm': success_5mm,
                'eval/mean_distance_mm': mean_distance,
                'eval/std_distance_mm': std_distance,
                'eval/min_distance_mm': min_distance,
                'eval/max_distance_mm': max_distance,
                'eval/mean_length': mean_length,
                'eval/mean_reward': mean_reward,
                'timesteps': self.num_timesteps
            })

        # Save best model (prioritize success rate, then distance)
        improved = False
        if success_rate > self.best_success_rate:
            improved = True
        elif success_rate == self.best_success_rate and mean_distance < self.best_mean_distance:
            improved = True
        
        if improved:
            self.best_success_rate = success_rate
            self.best_mean_distance = mean_distance
            self.best_mean_reward = mean_reward
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
            self.model.save(self.save_path)
            if self.verbose > 0:
                print(f"🏆 New best model! Success: {success_rate*100:.1f}%, Distance: {mean_distance:.2f}mm")
                print(f"   Saved to: {self.save_path}.zip\n")
            
            # Log best metrics
            if wandb.run is not None:
                wandb.run.summary['best_success_rate'] = success_rate
                wandb.run.summary['best_mean_distance'] = mean_distance
                wandb.run.summary['best_timestep'] = self.num_timesteps

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
    parser.add_argument('--algorithm', type=str, default='PPO', choices=['PPO', 'SAC', 'TD3'])
    parser.add_argument('--total_timesteps', type=int, default=1000000)
    parser.add_argument('--n_envs', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--n_steps', type=int, default=2048)
    parser.add_argument('--seed', type=int, default=42)
    
    # PPO specific
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--gae_lambda', type=float, default=0.95)
    parser.add_argument('--clip_range', type=float, default=0.2)
    parser.add_argument('--ent_coef', type=float, default=0.01)
    parser.add_argument('--vf_coef', type=float, default=0.5)
    
    # SAC/TD3 specific
    parser.add_argument('--tau', type=float, default=0.005)
    parser.add_argument('--buffer_size', type=int, default=200000)
    
    # Environment parameters
    parser.add_argument('--max_steps', type=int, default=500)
    parser.add_argument('--success_threshold', type=float, default=0.005)
    parser.add_argument('--reward_scale', type=float, default=1.0)
    parser.add_argument('--max_velocity', type=float, default=0.05)
    
    # Network
    parser.add_argument('--net_arch', nargs='+', type=int, default=[256, 256])
    
    # Logging
    parser.add_argument('--run_name', type=str, default=None)
    parser.add_argument('--save_dir', type=str, default='models')
    parser.add_argument('--wandb_project', type=str, default='ot2_rl_sweep')
    parser.add_argument('--use_clearml', action='store_true', help='Use ClearML tracking')
    
    # Other
    parser.add_argument('--use_gpu', action='store_true')
    
    return parser.parse_args()


# ===================== Model Creation =====================
def create_model(algorithm, env, args):
    """Create RL model with proper hyperparameters."""
    policy_kwargs = {'net_arch': args.net_arch}
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
def train(config=None):
    """Main training function with W&B sweep support."""
    
    # Initialize W&B (this will use sweep config if running as sweep agent)
    with wandb.init(config=config) as run:
        # Get config from W&B (either sweep or default)
        config = wandb.config
        
        # Convert config to args object
        args = argparse.Namespace()
        
        # Set defaults first
        defaults = parse_args()
        for key, value in vars(defaults).items():
            setattr(args, key, value)
        
        # Override with sweep config
        for key, value in config.items():
            if hasattr(args, key):
                setattr(args, key, value)
        
        # Handle net_arch if it's passed as tuple or list from sweep
        if isinstance(args.net_arch, (tuple, list)) and len(args.net_arch) > 0:
            if not isinstance(args.net_arch[0], int):
                args.net_arch = list(args.net_arch)
        
        # Generate run name
        if args.run_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            args.run_name = f"{args.algorithm}_{args.total_timesteps//1000}k_{timestamp}"
        
        # Set W&B run name
        wandb.run.name = args.run_name
        
        print(f"\n{'='*80}")
        print("OT-2 RL CONTROLLER TRAINING - SWEEP MODE")
        print(f"{'='*80}")
        print(f"Author: Zofia Pilitowska (240272)")
        print(f"Run: {args.run_name}")
        print(f"Algorithm: {args.algorithm}")
        print(f"Learning Rate: {args.learning_rate:.2e}")
        print(f"Batch Size: {args.batch_size}")
        print(f"Network: {args.net_arch}")
        print(f"{'='*80}\n")

        # ===================== ClearML (optional) =====================
        task = None
        if args.use_clearml:
            os.environ['WANDB_API_KEY'] = '914ae068e68f7aa8e0bef176d0a165d47f9b4c7f'
            task = Task.init(
                project_name='Mentor Group - Uther/Group 2',
                task_name=args.run_name,
                reuse_last_task_id=False
            )
            task.set_base_docker('deanis/2023y2b-rl:latest')
            task.connect(vars(args))
            task.execute_remotely(queue_name="default")

        # ===================== Environment Setup =====================
        print("Creating environments...")
        
        env_kwargs = {
            'max_steps': args.max_steps,
            'success_threshold': args.success_threshold,
            'reward_scale': args.reward_scale,
            'max_velocity': args.max_velocity,
            'use_gui': False
        }
        
        # Training environment
        env = DummyVecEnv([
            make_env(seed=args.seed + i, **env_kwargs) 
            for i in range(args.n_envs)
        ])
        
        # Evaluation environment
        eval_env = Monitor(OT2GymEnv(**env_kwargs))
        
        print(f"✓ Created {args.n_envs} training environment(s)\n")

        # ===================== Model Setup =====================
        print(f"Initializing {args.algorithm} model...")
        model = create_model(args.algorithm, env, args)
        
        n_params = sum(p.numel() for p in model.policy.parameters())
        print(f"✓ Model initialized ({n_params:,} parameters)\n")
        
        # Log model info to W&B
        wandb.config.update({
            'n_parameters': n_params,
            'student_id': '240272'
        }, allow_val_change=True)

        # ===================== Callbacks Setup =====================
        os.makedirs(args.save_dir, exist_ok=True)
        os.makedirs(f"{args.save_dir}/best", exist_ok=True)
        
        # Evaluation callback
        eval_callback = DetailedEvalCallback(
            eval_env=eval_env,
            eval_freq=max(10000 // args.n_envs, 1),
            n_eval_episodes=10,
            save_path=f"{args.save_dir}/best/{wandb.run.id}_best",
            verbose=1
        )
        
        callbacks = CallbackList([eval_callback])

        # ===================== Training =====================
        print("="*80)
        print("STARTING TRAINING")
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

            # ===================== Final Evaluation =====================
            print("\n" + "="*80)
            print("FINAL EVALUATION")
            print("="*80)
            
            # Run extended final evaluation
            n_final_eval = 50
            successes = []
            distances = []
            
            for _ in range(n_final_eval):
                obs, info = eval_env.reset()
                done = False
                while not done:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = eval_env.step(action)
                    done = terminated or truncated
                
                dist_mm = info.get('distance', 1.0) * 1000
                successes.append(info.get('success', False))
                distances.append(dist_mm)
            
            final_success_rate = np.mean(successes)
            final_mean_dist = np.mean(distances)
            final_std_dist = np.std(distances)
            
            print(f"Final Success Rate: {final_success_rate*100:.1f}%")
            print(f"Final Mean Distance: {final_mean_dist:.2f} ± {final_std_dist:.2f} mm")
            print(f"Training Duration: {duration/3600:.2f} hours")
            print("="*80 + "\n")
            
            # Log final results
            wandb.run.summary['final_success_rate'] = final_success_rate
            wandb.run.summary['final_mean_distance'] = final_mean_dist
            wandb.run.summary['final_std_distance'] = final_std_dist
            wandb.run.summary['training_duration_hours'] = duration / 3600
            
            # Save final model
            final_path = f"{args.save_dir}/{wandb.run.id}_final"
            model.save(final_path)
            wandb.save(f"{final_path}.zip")
            
            if task:
                task.upload_artifact('final_model', f"{final_path}.zip")

        except KeyboardInterrupt:
            print("\n⚠️  TRAINING INTERRUPTED")
            
        except Exception as e:
            print(f"\n❌ ERROR: {e}")
            import traceback
            traceback.print_exc()

        finally:
            env.close()
            eval_env.close()
            print("✓ Training completed\n")


# ===================== Main =====================
def main():
    """Main entry point - can be called by sweep agent or standalone."""
    # Check if running as part of a sweep
    if wandb.run is None:
        # Not in a sweep, parse args and run normally
        args = parse_args()
        config_dict = vars(args)
        train(config=config_dict)
    else:
        # Already in a sweep (called by wandb agent)
        train()


if __name__ == "__main__":
    main()