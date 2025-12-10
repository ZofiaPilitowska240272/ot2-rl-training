"""
Reinforcement Learning Training Script for OT-2 Pipette Controller

This script trains RL agents (PPO, SAC, TD3) to control the Opentrons OT-2
pipette positioning system with comprehensive hyperparameter search support.


Usage:
    # Quick test (10k steps)
    python train_rl.py --algorithm PPO --total_timesteps 10000 --n_envs 4
    
    # Full training (2M steps)
    python train_rl.py --algorithm PPO --total_timesteps 2000000 --n_envs 8 \
        --learning_rate 3e-4 --run_name ppo_final_240272
    
    # Hyperparameter search
    python train_rl.py --algorithm PPO --learning_rate 1e-4 --batch_size 256 \
        --success_threshold 0.01 --run_name search_1_240272
"""

import os
import argparse
import numpy as np
from datetime import datetime
from typing import Callable, Optional
import json

import gymnasium as gym
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import (
    EvalCallback, 
    CheckpointCallback, 
    CallbackList,
    BaseCallback
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.logger import configure

import wandb
from wandb.integration.sb3 import WandbCallback

from ot2_gym_wrapper import OT2GymEnv


# =========================== WEIGHTS & BIASES LOGGING =======================================================
class DetailedEvalCallback(BaseCallback):
    """
    Custom callback for detailed evaluation metrics and W&B logging.

    """
    
    def __init__(
        self,
        eval_env,
        eval_freq: int = 10000,
        n_eval_episodes: int = 10,
        log_path: Optional[str] = None,
        best_model_save_path: Optional[str] = None,
        deterministic: bool = True,
        verbose: int = 1
    ):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.log_path = log_path
        self.best_model_save_path = best_model_save_path
        self.deterministic = deterministic
        
        self.best_mean_reward = -np.inf
        self.best_success_rate = 0.0
        self.evaluations_timesteps = []
        self.evaluations_results = []
        

    def _on_step(self) -> bool:
        """Called after every step. Returns True to continue training."""
        
        if self.n_calls % self.eval_freq != 0:
            return True
        
        if self.verbose > 0:
            print(f"\n{'='*60}")
            print(f"Evaluation at timestep {self.num_timesteps}")
            print(f"{'='*60}")
        
        # Run evaluation episodes
        episode_rewards = []
        episode_lengths = []
        success_count = 0
        final_distances = []
        initial_distances = []
        action_stds = []
        
        for episode in range(self.n_eval_episodes):
            obs, info = self.eval_env.reset()
            done = False
            episode_reward = 0
            episode_length = 0
            episode_actions = []
            initial_distances.append(info['initial_distance'])
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=self.deterministic)
                obs, reward, terminated, truncated, info = self.eval_env.step(action)
                
                episode_reward += reward
                episode_length += 1
                episode_actions.append(action)
                done = terminated or truncated
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
            if info['success']:
                success_count += 1
            
            final_distances.append(info['distance'])
            
            # Calculate action smoothness (lower is better)
            if len(episode_actions) > 1:
                action_std = np.std(np.array(episode_actions), axis=0).mean()
                action_stds.append(action_std)
        
        # Compute statistics
        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        success_rate = success_count / self.n_eval_episodes
        mean_length = np.mean(episode_lengths)
        mean_final_distance = np.mean(final_distances) * 1000  # Convert to mm
        std_final_distance = np.std(final_distances) * 1000
        min_final_distance = np.min(final_distances) * 1000
        mean_initial_distance = np.mean(initial_distances) * 1000
        mean_smoothness = np.mean(action_stds) if action_stds else 0.0
        
        # Calculate distance improvement percentage
        distance_improvements = [
            (init - final) / init * 100
            for init, final in zip(initial_distances, final_distances)
        ]
        mean_distance_improvement = np.mean(distance_improvements)
        
        # Store results
        self.evaluations_timesteps.append(self.num_timesteps)
        eval_result = {
            'timesteps': self.num_timesteps,
            'mean_reward': float(mean_reward),
            'std_reward': float(std_reward),
            'success_rate': float(success_rate),
            'mean_length': float(mean_length),
            'mean_final_distance_mm': float(mean_final_distance),
            'std_final_distance_mm': float(std_final_distance),
            'min_final_distance_mm': float(min_final_distance),
            'mean_smoothness': float(mean_smoothness),
            'mean_distance_improvement': float(mean_distance_improvement)
        }
        self.evaluations_results.append(eval_result)
        
        # Print results
        if self.verbose > 0:
            print(f"Episodes: {self.n_eval_episodes}")
            print(f"Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
            print(f"Success rate: {success_rate*100:.1f}% ({success_count}/{self.n_eval_episodes})")
            print(f"Mean episode length: {mean_length:.1f} steps")
            print(f"Mean final distance: {mean_final_distance:.2f} ± {std_final_distance:.2f} mm")
            print(f"Min final distance: {min_final_distance:.2f} mm")
            print(f"Distance improvement: {mean_distance_improvement:.1f}%")
            print(f"Action smoothness: {mean_smoothness:.4f}")
        
        # Log to W&B
        if wandb.run is not None:
            wandb.log({
                'eval/mean_reward': mean_reward,
                'eval/std_reward': std_reward,
                'eval/success_rate': success_rate,
                'eval/success_count': success_count,
                'eval/mean_episode_length': mean_length,
                'eval/mean_final_distance_mm': mean_final_distance,
                'eval/std_final_distance_mm': std_final_distance,
                'eval/min_final_distance_mm': min_final_distance,
                'eval/mean_initial_distance_mm': mean_initial_distance,
                'eval/action_smoothness': mean_smoothness,
                'eval/distance_improvement_pct': mean_distance_improvement,
                'eval/timesteps': self.num_timesteps
            })
        
        # Save best model based on success rate (primary) and reward (secondary)
        is_best = False
        if success_rate > self.best_success_rate:
            is_best = True
            self.best_success_rate = success_rate
            self.best_mean_reward = mean_reward
        elif success_rate == self.best_success_rate and mean_reward > self.best_mean_reward:
            is_best = True
            self.best_mean_reward = mean_reward
        
        if is_best and self.best_model_save_path is not None:
            model_path = os.path.join(self.best_model_save_path, 'best_model')
            self.model.save(model_path)
            if self.verbose > 0:
                print(f"\n New best model! Success rate: {success_rate*100:.1f}%, Reward: {mean_reward:.2f}")
                print(f"   Saved to: {model_path}.zip")
        
        # Save evaluation results
        if self.log_path is not None:
            os.makedirs(self.log_path, exist_ok=True)
            results_path = os.path.join(self.log_path, 'evaluations.json')
            with open(results_path, 'w') as f:
                json.dump({
                    'timesteps': self.evaluations_timesteps,
                    'results': self.evaluations_results
                }, f, indent=2)
        
        if self.verbose > 0:
            print(f"{'='*60}\n")
        
        return True



# ==================================================================================
def make_env(rank: int, seed: int = 0, **env_kwargs) -> Callable:
    """
    Create a function that initializes a gym environment.
    
    
    Args:
        rank: Index of the environment (for parallel training)
        seed: Random seed for reproducibility
        **env_kwargs: Additional environment arguments
        
    Returns:
        Function that creates and returns an environment instance
    """
    def _init():
        env = OT2GymEnv(**env_kwargs)
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    
    set_random_seed(seed)
    return _init

# ============================ PARSER ARUGMENTS ==========================================================================

def parse_args():
    """
    Parse command line arguments for training configuration.
    
    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description='Train RL agent for OT-2 pipette controller',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # ============ Algorithm Selection ============
    parser.add_argument(
        '--algorithm',
        type=str,
        default='PPO',
        choices=['PPO', 'SAC', 'TD3'],
        help='RL algorithm to use'
    )
    
    # ============ Training Parameters ============
    parser.add_argument(
        '--total_timesteps',
        type=int,
        default=100000,
        help='Total training timesteps '
    )
    parser.add_argument(
        '--n_envs',
        type=int,
        default=8,
        help='Number of parallel environments'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=3e-4,
        help='Learning rate'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    
    # ============ PPO-Specific Parameters ============
    parser.add_argument(
        '--batch_size',
        type=int,
        default=128,
        help='Batch size for training '
    )
    parser.add_argument(
        '--n_steps',
        type=int,
        default=2048,
        help='Number of steps per update '
    )
    parser.add_argument(
        '--n_epochs',
        type=int,
        default=10,
        help='Number of epochs per update (PPO only)'
    )
    parser.add_argument(
        '--gamma',
        type=float,
        default=0.99,
        help='Discount factor'
    )
    parser.add_argument(
        '--gae_lambda',
        type=float,
        default=0.95,
        help='GAE lambda (PPO only)'
    )
    parser.add_argument(
        '--clip_range',
        type=float,
        default=0.2,
        help='Clip range (PPO only)'
    )
    parser.add_argument(
        '--ent_coef',
        type=float,
        default=0.01,
        help='Entropy coefficient'
    )
    
    # ============ SAC/TD3-Specific Parameters ============
    parser.add_argument(
        '--buffer_size',
        type=int,
        default=200000,
        help='Replay buffer size (SAC/TD3 only)'
    )
    parser.add_argument(
        '--learning_starts',
        type=int,
        default=10000,
        help='Steps before learning starts (SAC/TD3 only)'
    )
    parser.add_argument(
        '--tau',
        type=float,
        default=0.005,
        help='Target network update rate (SAC/TD3 only)'
    )
    
    # ============ Network Architecture ============
    parser.add_argument(
        '--net_arch',
        type=int,
        nargs='+',
        default=[256, 256],
        help='Network architecture'
    )
    
    # ============ Environment Parameters ============
    parser.add_argument(
        '--max_velocity',
        type=float,
        default=0.05,
        help='Maximum pipette velocity (m/s)'
    )
    parser.add_argument(
        '--success_threshold',
        type=float,
        default=0.005,
        help='Success distance threshold in meters '
    )
    parser.add_argument(
        '--max_steps',
        type=int,
        default=500,
        help='Maximum steps per episode'
    )
    parser.add_argument(
        '--reward_scale',
        type=float,
        default=1.0,
        help='Reward scaling factor '
    )
    
    # ============ Logging and Checkpointing ============
    parser.add_argument(
        '--wandb_project',
        type=str,
        default='ot2_rl',
        help='W&B project name'
    )
    parser.add_argument(
        '--wandb_entity',
        type=str,
        default=None,
        help='W&B entity/username'
    )
    parser.add_argument(
        '--run_name',
        type=str,
        default=None,
        help='Name for this training run'
    )
    parser.add_argument(
        '--save_dir',
        type=str,
        default='models',
        help='Directory to save models'
    )
    parser.add_argument(
        '--eval_freq',
        type=int,
        default=10000,
        help='Evaluation frequency (in steps)'
    )
    parser.add_argument(
        '--n_eval_episodes',
        type=int,
        default=10,
        help='Number of episodes per evaluation'
    )
    parser.add_argument(
        '--checkpoint_freq',
        type=int,
        default=50000,
        help='Checkpoint save frequency (in steps)'
    )
    parser.add_argument(
        '--log_interval',
        type=int,
        default=10,
        help='Log interval for training metrics'
    )
    
    # ============ ClearML Integration ============
    parser.add_argument(
        '--use_clearml',
        action='store_true',
        help='Use ClearML for experiment tracking'
    )
    parser.add_argument(
        '--clearml_project',
        type=str,
        default='OT2_RL_Controller',
        help='ClearML project name'
    )
    parser.add_argument(
        '--clearml_task',
        type=str,
        default=None,
        help='ClearML task name'
    )
    
    

    parser.add_argument('--no_wandb', action='store_true',
                    help='Disable W&B logging')
    parser.add_argument('--use_gpu', action='store_true',
                    help='Force GPU usage')
    
    return parser.parse_args()

#  ===================================================MODEL CREATION =================================================================================================


def create_model(algorithm: str, env, args):
    """
    Create and configure RL model based on algorithm choice.
    
    Args:
        algorithm: Algorithm name ('PPO', 'SAC', 'TD3')
        env: Vectorized training environment
        args: Command line arguments
        
    Returns:
        Configured RL model ready for training
    """
    # Common parameters for all algorithms
    common_params = {
        'policy': 'MlpPolicy',
        'env': env,
        'learning_rate': args.learning_rate,
        'verbose': 1,
        'seed': args.seed,
        'device': 'cuda' if args.use_gpu else 'auto',
        'tensorboard_log': f"./tensorboard_logs/{args.run_name}",
    }
    
    # Policy network architecture
    policy_kwargs = {
        'net_arch': args.net_arch
    }
    
    if algorithm == 'PPO':
        model = PPO(
            **common_params,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_range=args.clip_range,
            ent_coef=args.ent_coef,
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs=policy_kwargs
        )
    
    elif algorithm == 'SAC':
        model = SAC(
            **common_params,
            buffer_size=args.buffer_size,
            learning_starts=args.learning_starts,
            batch_size=args.batch_size,
            tau=args.tau,
            gamma=args.gamma,
            train_freq=1,
            gradient_steps=1,
            ent_coef='auto',
            policy_kwargs=policy_kwargs
        )
    
    elif algorithm == 'TD3':
        model = TD3(
            **common_params,
            buffer_size=args.buffer_size,
            learning_starts=args.learning_starts,
            batch_size=args.batch_size,
            tau=args.tau,
            gamma=args.gamma,
            train_freq=1,
            gradient_steps=1,
            policy_kwargs=policy_kwargs
        )
    
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    return model

# ==========================================MODEL TRAINING =======================================================
def train(args):
    """
    Main training function with ClearML and W&B integration.
    
    This function:
    1. Initializes ClearML for remote execution
    2. Sets up W&B logging
    3. Creates training and evaluation environments
    4. Initializes the RL model
    5. Sets up callbacks for evaluation and checkpointing
    6. Trains the model
    7. Saves and uploads models to ClearML artifacts
    
    Args:
        args: Command line arguments
    """
    
    # Generate run name if not provided
    if args.run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.run_name = f"{args.algorithm}_{timestamp}_240272"
    
    # ============================================================
    # CLEARML INITIALIZATION
    from clearml import Task

    task = Task.init(
        project_name='OT2-RL-Controller/Zofia_240272',
        task_name=args.run_name
    )

    # Set base docker image
    task.set_base_docker('deanis/2023y2b-rl:latest')

    # Connect hyperparameters
    task.connect(vars(args))

    # Execute remotely
    task.execute_remotely(queue_name="default")

    # Print header
    print("OT-2 RL CONTROLLER TRAINING")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Run name: {args.run_name}")

    
    # Print configuration
    print("\nTraining Configuration:")
    print(f"  Algorithm: {args.algorithm}")
    print(f"  Total timesteps: {args.total_timesteps:,}")
    print(f"  Parallel environments: {args.n_envs}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Batch size: {args.batch_size}")
    if args.algorithm == 'PPO':
        print(f"  N steps: {args.n_steps}")
        print(f"  N epochs: {args.n_epochs}")
    print(f"  Network architecture: {args.net_arch}")
    print(f"  Success threshold: {args.success_threshold*1000:.1f}mm")
    print(f"  Reward scale: {args.reward_scale}")
    print("="*80 + "\n")
    
    # Initialize W&B
    if not args.no_wandb:
        config = {
            'algorithm': args.algorithm,
            'total_timesteps': args.total_timesteps,
            'n_envs': args.n_envs,
            'learning_rate': args.learning_rate,
            'batch_size': args.batch_size,
            'n_steps': args.n_steps if args.algorithm == 'PPO' else None,
            'network_arch': args.net_arch,
            'success_threshold': args.success_threshold,
            'reward_scale': args.reward_scale,
            'max_velocity': args.max_velocity,
            'max_steps_per_episode': args.max_steps,
            'student_id': '240272',
            'seed': args.seed
        }
        
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.run_name,
            config=config,
            sync_tensorboard=True,
            monitor_gym=True,
            save_code=True
        )
        print(" Weights & Biases initialized")
        print(f"  Dashboard: {run.url}\n")
    else:
        run = None
        print(" W&B logging disabled\n")
    
    # Set random seed
    set_random_seed(args.seed)
    
    # Environment parameters
    env_kwargs = {
        'max_velocity': args.max_velocity,
        'success_threshold': args.success_threshold,
        'max_steps': args.max_steps,
        'reward_scale': args.reward_scale,
        'use_gui': False
    }
    
    print("Creating environments...")
    
    # Create vectorized training environments
    if args.n_envs > 1:
        env = SubprocVecEnv([
            make_env(i, args.seed, **env_kwargs) 
            for i in range(args.n_envs)
        ])
        print(f" Created {args.n_envs} parallel training environments")
    else:
        env = DummyVecEnv([make_env(0, args.seed, **env_kwargs)])
        print(f" Created 1 training environment")
    
    # Create evaluation environment
    eval_env = OT2GymEnv(**env_kwargs)
    eval_env = Monitor(eval_env)
    print(f" Created evaluation environment\n")
    
    # Create model
    print(f"Initializing {args.algorithm} model...")
    model = create_model(args.algorithm, env, args)
    print(f" Model initialized")
    print(f"  Policy network: {args.net_arch}")
    print(f"  Total parameters: ~{sum(p.numel() for p in model.policy.parameters()):,}\n")
    
    # Create directories
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(f"{args.save_dir}/checkpoints", exist_ok=True)
    os.makedirs(f"{args.save_dir}/best", exist_ok=True)
    os.makedirs(f"{args.save_dir}/logs", exist_ok=True)
    
    # Setup callbacks
    print("Setting up training callbacks...")
    
    # 1. Checkpoint callback - saves model periodically
    checkpoint_callback = CheckpointCallback(
        save_freq=max(args.checkpoint_freq // args.n_envs, 1),
        save_path=f"{args.save_dir}/checkpoints",
        name_prefix=f"{args.algorithm}_{args.run_name}",
        save_replay_buffer=True,
        save_vecnormalize=True,
        verbose=1
    )
    
    # 2. Detailed evaluation callback
    eval_callback = DetailedEvalCallback(
        eval_env=eval_env,
        eval_freq=max(args.eval_freq // args.n_envs, 1),
        n_eval_episodes=args.n_eval_episodes,
        log_path=f"{args.save_dir}/logs",
        best_model_save_path=f"{args.save_dir}/best",
        deterministic=True,
        verbose=1
    )
    
    # 3. W&B callback (if enabled)
    callback_list = [checkpoint_callback, eval_callback]
    
    if not args.no_wandb:
        wandb_callback = WandbCallback(
            model_save_path=f"{args.save_dir}/wandb",
            verbose=2,
        )
        callback_list.append(wandb_callback)
    
    callbacks = CallbackList(callback_list)
    
    print(" Callbacks configured:")
    print(f"  - Checkpoints every {args.checkpoint_freq:,} steps")
    print(f"  - Evaluation every {args.eval_freq:,} steps")
    print(f"  - Best model saving enabled")
    if not args.no_wandb:
        print(f"  - W&B logging enabled\n")
    
    # Train the model
    print("STARTING TRAINING")
    print(f"Training will run for {args.total_timesteps:,} timesteps")
    print(f"Expected time: ~{args.total_timesteps / (args.n_envs * 600) / 60:.1f}-{args.total_timesteps / (args.n_envs * 400) / 60:.1f} hours")

    
    try:
        start_time = datetime.now()
        
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=callbacks,
            log_interval=args.log_interval,
            progress_bar=True
        )
        
        end_time = datetime.now()
        training_duration = (end_time - start_time).total_seconds()
        
        # ============================================================
        # SAVE AND UPLOAD MODELS TO CLEARML
        print("SAVING MODELS")
        
        # 1. Save final model
        final_model_path = f"{args.save_dir}/{args.algorithm}_{args.run_name}_final"
        model.save(final_model_path)
        print(f" Final model saved: {final_model_path}.zip")
        
        # 2. Upload final model to ClearML artifacts
        task.upload_artifact(
            name='final_model',
            artifact_object=f"{final_model_path}.zip"
        )
        print(f" Final model uploaded to ClearML artifacts")
        
        # 3. Upload best model to ClearML artifacts
        best_model_path = f"{args.save_dir}/best/best_model.zip"
        if os.path.exists(best_model_path):
            task.upload_artifact(
                name='best_model',
                artifact_object=best_model_path
            )
            print(f" Best model uploaded to ClearML artifacts")
        
        # 4. Upload evaluation results to ClearML
        eval_results_path = f"{args.save_dir}/logs/evaluations.json"
        if os.path.exists(eval_results_path):
            task.upload_artifact(
                name='evaluation_results',
                artifact_object=eval_results_path
            )
            print(f" Evaluation results uploaded to ClearML")
        
        # ============================================================
        
        
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print(f"Training duration: {training_duration/3600:.2f} hours")
        print(f"Steps per second: {args.total_timesteps/training_duration:.1f}")
        print(f"\nModels saved locally:")
        print(f"  - Final model: {final_model_path}.zip")
        print(f"  - Best model: {best_model_path}")
        print(f"  - Checkpoints: {args.save_dir}/checkpoints/")
        print(f"\nModels uploaded to ClearML artifacts:")
        print(f"  - final_model")
        print(f"  - best_model")
        print(f"  - evaluation_results")
        print("="*80 + "\n")
        
        # Save training summary
        summary = {
            'run_name': args.run_name,
            'algorithm': args.algorithm,
            'total_timesteps': args.total_timesteps,
            'training_duration_hours': training_duration / 3600,
            'steps_per_second': args.total_timesteps / training_duration,
            'final_model_path': f"{final_model_path}.zip",
            'best_model_path': best_model_path,
            'hyperparameters': vars(args),
            'timestamp': datetime.now().isoformat(),
            'clearml_task_id': task.id
        }
        
        summary_path = f"{args.save_dir}/logs/training_summary_{args.run_name}.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Training summary saved to: {summary_path}\n")
        
        # Upload summary to ClearML
        task.upload_artifact(
            name='training_summary',
            artifact_object=summary_path
        )
        
        # Upload to W&B
        if not args.no_wandb:
            wandb.save(f"{final_model_path}.zip")
            wandb.save(best_model_path)
            print(" Models uploaded to W&B\n")
        
    except KeyboardInterrupt:
        print("\n\n" + "="*80)
        print("TRAINING INTERRUPTED BY USER")
        print("="*80)
        
        # Save interrupted model
        interrupted_path = f"{args.save_dir}/{args.algorithm}_{args.run_name}_interrupted"
        model.save(interrupted_path)
        print(f"Model saved to: {interrupted_path}.zip")
        
        # Upload to ClearML
        task.upload_artifact(
            name='interrupted_model',
            artifact_object=f"{interrupted_path}.zip"
        )
        print(f" Interrupted model uploaded to ClearML")
        print("="*80 + "\n")
    
    except Exception as e:
        print(f"\n\n❌ ERROR DURING TRAINING: {e}")
        import traceback
        traceback.print_exc()
        
        # Try to save model anyway
        try:
            error_path = f"{args.save_dir}/{args.algorithm}_{args.run_name}_error"
            model.save(error_path)
            print(f"\nModel saved despite error: {error_path}.zip")
            
            # Upload to ClearML
            task.upload_artifact(
                name='error_model',
                artifact_object=f"{error_path}.zip"
            )
            print(f" Error model uploaded to ClearML")
        except:
            pass
    
    finally:
        # Cleanup
        print("\nCleaning up...")
        env.close()
        eval_env.close()
        
        if run is not None:
            run.finish()
            print(" W&B run finished")
        
        print(" Environments closed")
        print("\nTraining script completed.\n")


# =================================================================================================

def main():
    """Entry point for training script."""
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()