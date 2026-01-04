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


# ===================== Evaluation Callback =====================
class DetailedEvalCallback(BaseCallback):
    """
    Evaluate agent and log detailed metrics to W&B and ClearML.
    
    Runs dedicated evaluation episodes periodically to measure:
    - Success rate (% within threshold)
    - Mean distance to target (mm)
    - Episode efficiency
    
    Automatically saves the best model based on distance.
    """
    
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

        # Compute statistics
        success_rate = np.mean(successes)
        mean_distance = np.mean(distances)
        std_distance = np.std(distances)
        min_distance = np.min(distances)
        max_distance = np.max(distances)
        mean_length = np.mean(lengths)
        mean_reward = np.mean(rewards)

        # Print results
        if self.verbose > 0:
            print(f"Success rate: {success_rate*100:.1f}% ({sum(successes)}/{self.n_eval_episodes})")
            print(f"Distance: {mean_distance:.2f} ± {std_distance:.2f} mm")
            print(f"  Min: {min_distance:.2f} mm, Max: {max_distance:.2f} mm")
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
                'eval/max_distance_mm': max_distance,
                'eval/mean_length': mean_length,
                'eval/mean_reward': mean_reward,
                'timesteps': self.num_timesteps
            })

        # Save best model (prioritize distance)
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
                print(f"🏆 New best model!")
                print(f"   Distance: {mean_distance:.2f} mm")
                print(f"   Success rate: {success_rate*100:.1f}%")
                print(f"   Saved to: {self.save_path}.zip\n")

        return True


# ===================== Environment Factory =====================
def make_env(seed=0, **kwargs):
    """
    Create a monitored environment for training.
    
    Each environment is wrapped with Monitor for episode statistics.
    Seeds are set for reproducibility.
    """
    def _init():
        env = OT2GymEnv(**kwargs)
        env = Monitor(env)
        env.reset(seed=seed)
        return env
    set_random_seed(seed)
    return _init


# ===================== Argument Parser =====================
def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train RL agent for OT-2 pipette positioning",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Core parameters
    parser.add_argument('--algorithm', type=str, default='PPO', 
                       choices=['PPO', 'SAC', 'TD3'],
                       help='RL algorithm to use')
    parser.add_argument('--total_timesteps', type=int, default=1500000,
                       help='Total training timesteps')
    parser.add_argument('--n_envs', type=int, default=16,
                       help='Number of parallel environments (16-32 recommended)')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate (use 1e-4 to 3e-4)')
    parser.add_argument('--batch_size', type=int, default=256,
                       help='Batch size for training')
    parser.add_argument('--n_steps', type=int, default=2048,
                       help='Steps per update (PPO only)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    # PPO-specific hyperparameters
    parser.add_argument('--clip_range', type=float, default=0.2,
                       help='PPO clip range')
    parser.add_argument('--ent_coef', type=float, default=0.005,
                       help='Entropy coefficient')
    parser.add_argument('--gae_lambda', type=float, default=0.95,
                       help='GAE lambda')
    parser.add_argument('--gamma', type=float, default=0.99,
                       help='Discount factor (0.99 standard, 0.995 for precision)')
    parser.add_argument('--vf_coef', type=float, default=0.5,
                       help='Value function coefficient')
    
    # SAC/TD3-specific
    parser.add_argument('--tau', type=float, default=0.005,
                       help='Target network update rate')
    parser.add_argument('--buffer_size', type=int, default=300000,
                       help='Replay buffer size')
    
    # Environment parameters
    parser.add_argument('--max_steps', type=int, default=500,
                       help='Max steps per episode')
    parser.add_argument('--success_threshold', type=float, default=0.005,
                       help='Success threshold in meters (0.005 = 5mm)')
    parser.add_argument('--reward_scale', type=float, default=1.0,
                       help='Reward scaling factor (keep at 1.0)')
    parser.add_argument('--max_velocity', type=float, default=0.02,
                       help='Max velocity in m/s (0.015-0.03 for precision)')
    
    # Network architecture
    parser.add_argument('--net_arch', type=str, default='512 512',
                       help='Network architecture (e.g., "256 256" or "512 512")')
    
    # Logging and saving
    parser.add_argument('--run_name', type=str, default=None,
                       help='Custom run name (auto-generated if not provided)')
    parser.add_argument('--save_dir', type=str, default='models',
                       help='Directory to save models')
    parser.add_argument('--wandb_project', type=str, default='ot2_rl_final',
                       help='W&B project name')
    
    # Execution
    parser.add_argument('--use_gpu', action='store_true',
                       help='Use GPU (recommended for SAC, not PPO)')
    parser.add_argument('--no_wandb', action='store_true',
                       help='Disable W&B tracking')
    parser.add_argument('--no_clearml', action='store_true',
                       help='Disable ClearML tracking')
    
    return parser.parse_args()


# ===================== Model Creation =====================
def create_model(algorithm, env, args):
    """
    Create RL model with appropriate hyperparameters.
    
    Automatically selects optimal device:
    - PPO: CPU (faster with parallel envs)
    - SAC/TD3: GPU (if available and requested)
    """
    
    # Parse network architecture
    if isinstance(args.net_arch, str):
        net_arch = [int(x) for x in args.net_arch.split()]
    else:
        net_arch = args.net_arch
    
    policy_kwargs = {'net_arch': net_arch}
    
    # Smart device selection
    if algorithm == 'PPO':
        # PPO with parallel envs is faster on CPU
        device = 'cpu'
        if args.use_gpu:
            print("⚠️  Note: PPO runs faster on CPU with parallel environments")
    else:
        # SAC/TD3 benefit from GPU
        device = 'cuda' if args.use_gpu else 'cpu'
    
    common_params = {
        'policy': 'MlpPolicy',
        'env': env,
        'learning_rate': args.learning_rate,
        'verbose': 1,
        'seed': args.seed,
        'device': device,
        'tensorboard_log': None,
        'policy_kwargs': policy_kwargs
    }

    if algorithm == 'PPO':
        model = PPO(
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
        model = SAC(
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
        model = TD3(
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
    
    return model


# ===================== Training Function =====================
def train(args):
    """Main training function with ClearML and W&B integration."""
    
    # Generate run name if not provided
    if args.run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.run_name = f"{args.algorithm}_{args.total_timesteps//1000}k_{timestamp}"

    # ===================== ClearML Setup =====================
    task = None
    if not args.no_clearml:
        os.environ['WANDB_API_KEY'] = '914ae068e68f7aa8e0bef176d0a165d47f9b4c7f'
        
        task = Task.init(
            project_name='Mentor Group - Uther/Group 2',
            task_name=args.run_name,
            reuse_last_task_id=False
        )
        
        task.set_base_docker('deanis/2023y2b-rl:latest')
        task.connect(vars(args))
        task.execute_remotely(queue_name="default")
    
    # ===================== Print Configuration =====================
    print(f"\n{'='*80}")
    print("OT-2 RL CONTROLLER TRAINING")
    print(f"{'='*80}")
    print(f"Author: Zofia Pilitowska (240272)")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Run: {args.run_name}")
    print(f"\nConfiguration:")
    print(f"  Algorithm: {args.algorithm}")
    print(f"  Learning Rate: {args.learning_rate:.2e}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Network: {args.net_arch}")
    print(f"  N_envs: {args.n_envs}")
    print(f"  Total Timesteps: {args.total_timesteps:,}")
    print(f"\nEnvironment:")
    print(f"  Max Velocity: {args.max_velocity} m/s")
    print(f"  Max Steps: {args.max_steps}")
    print(f"  Success Threshold: {args.success_threshold*1000:.1f} mm")
    print(f"  Gamma: {args.gamma}")
    print(f"{'='*80}\n")

    # ===================== W&B Setup =====================
    wandb_run = None
    if not args.no_wandb:
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
        
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.run_name,
            config=config,
            sync_tensorboard=False,
            monitor_gym=True
        )
        print(f"✓ W&B initialized: {wandb_run.url}\n")

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
    print(f"✓ Model initialized on {model.device}")
    print(f"  Network: {args.net_arch}")
    print(f"  Parameters: {n_params:,}\n")

    # ===================== Callbacks Setup =====================
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(f"{args.save_dir}/best", exist_ok=True)
    os.makedirs(f"{args.save_dir}/checkpoints", exist_ok=True)
    
    # Checkpoint callback (saves every 50k steps)
    checkpoint_callback = CheckpointCallback(
        save_freq=max(50000 // args.n_envs, 1),
        save_path=f"{args.save_dir}/checkpoints",
        name_prefix=f"{args.run_name}",
        verbose=1
    )
    
    # Evaluation callback (evaluates every 10k steps)
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
    
    # Estimate training time
    if args.algorithm == 'PPO' and args.n_envs >= 16:
        est_speed = 25 * (args.n_envs / 16)  # Scale with n_envs
        est_time = args.total_timesteps / est_speed / 3600
    elif args.algorithm in ['SAC', 'TD3']:
        est_speed = 15
        est_time = args.total_timesteps / est_speed / 3600
    else:
        est_time = args.total_timesteps / 10 / 3600
    
    print(f"Expected duration: ~{est_time:.1f} hours")
    print("="*80 + "\n")

    try:
        start_time = datetime.now()
        
        # Train the model
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
        if task:
            task.upload_artifact('final_model', f"{final_path}.zip")
            
            best_path = f"{args.save_dir}/best/{args.run_name}_best.zip"
            if os.path.exists(best_path):
                task.upload_artifact('best_model', best_path)
                print(f"✓ Best model: {best_path}")
            
            print(f"✓ Models uploaded to ClearML")
        
        # Upload to W&B
        if wandb_run:
            wandb.save(f"{final_path}.zip")
            best_path = f"{args.save_dir}/best/{args.run_name}_best.zip"
            if os.path.exists(best_path):
                wandb.save(best_path)
            print(f"✓ Models uploaded to W&B")

        # ===================== Training Summary =====================
        print("\n" + "="*80)
        print("TRAINING COMPLETED!")
        print("="*80)
        print(f"Duration: {duration/3600:.2f} hours")
        print(f"Steps/sec: {args.total_timesteps/duration:.1f}")
        print(f"Best distance: {eval_callback.best_distance:.2f} mm")
        print(f"Best success rate: {eval_callback.best_success_rate*100:.1f}%")
        
        if eval_callback.best_distance < 5.0:
            print(f"\n🎉 SUCCESS! Achieved <5mm target!")
        elif eval_callback.best_distance < 10.0:
            print(f"\n✓ Good result! Close to target. Extend training for <5mm.")
        elif eval_callback.best_distance < 30.0:
            print(f"\n⚠️  Model is learning. Consider extending to 3-5M timesteps.")
        else:
            print(f"\n❌ Model needs improvement. Check hyperparameters.")
        
        print("="*80 + "\n")
        
        # Save training summary
        summary = {
            'run_name': args.run_name,
            'algorithm': args.algorithm,
            'total_timesteps': args.total_timesteps,
            'duration_hours': duration / 3600,
            'steps_per_second': args.total_timesteps / duration,
            'best_distance_mm': float(eval_callback.best_distance),
            'best_success_rate': float(eval_callback.best_success_rate),
            'hyperparameters': {
                'learning_rate': args.learning_rate,
                'batch_size': args.batch_size,
                'n_envs': args.n_envs,
                'gamma': args.gamma,
                'max_velocity': args.max_velocity,
                'net_arch': args.net_arch
            },
            'timestamp': datetime.now().isoformat()
        }
        
        if task:
            summary['clearml_task_id'] = task.id
        if wandb_run:
            summary['wandb_url'] = wandb_run.url
        
        summary_path = f"{args.save_dir}/training_summary_{args.run_name}.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"✓ Training summary saved: {summary_path}")
        
        if task:
            task.upload_artifact('training_summary', summary_path)

    except KeyboardInterrupt:
        print("\n⚠️  TRAINING INTERRUPTED BY USER")
        interrupted_path = f"{args.save_dir}/{args.run_name}_interrupted"
        model.save(interrupted_path)
        print(f"✓ Interrupted model saved: {interrupted_path}.zip")
        
        if task:
            task.upload_artifact('interrupted_model', f"{interrupted_path}.zip")

    except Exception as e:
        print(f"\n❌ ERROR DURING TRAINING: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Cleanup
        try:
            env.close()
            eval_env.close()
            if wandb_run:
                wandb.finish()
            print("\n✓ Cleanup completed")
        except:
            pass


# ===================== Main Entry Point =====================
def main():
    """Main entry point."""
    args = parse_args()
    
    # Validate arguments
    if args.learning_rate > 0.001:
        print(f"⚠️  WARNING: Learning rate {args.learning_rate} is very high!")
        print(f"   Recommended range: 1e-5 to 3e-4")
        print(f"   Training may be unstable.\n")
    
    if args.n_envs == 1 and args.algorithm == 'PPO':
        print(f"⚠️  WARNING: Using only 1 environment with PPO is very slow!")
        print(f"   Recommended: --n_envs 16 or higher")
        print(f"   This will take a long time.\n")
    
    if args.max_velocity > 0.05:
        print(f"⚠️  WARNING: Max velocity {args.max_velocity} is high for precision!")
        print(f"   For <5mm target, use 0.015-0.03 m/s\n")
    
    # Start training
    train(args)


if __name__ == "__main__":
    main()