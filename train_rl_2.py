"""
OT-2 RL Training - IMPROVED VERSION
"""

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from clearml import Task
import argparse
from datetime import datetime
import numpy as np
import os
from ot2_gym_wrapper import OT2GymEnv


# Generate timestamp for unique task name and model filename
timestamp = datetime.now().strftime("%y%m%d.%H%M")

# ============================================================================
# Custom Callback for OT2-Specific Metrics
# ============================================================================
class OT2Callback(BaseCallback):
    """
    Custom callback for logging OT2-specific metrics during training.
    
    Logs to TensorBoard (visible in ClearML):
    - Episode reward (cumulative reward per episode)
    - Episode length (steps to reach goal or timeout)
    - Success rate (rolling 100-episode average)
    - Final distance to goal (mm)
    """
    
    def __init__(self, threshold=0.005, verbose=0):
        super().__init__(verbose)
        self.threshold = threshold
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_successes = []
        self.episode_final_distances = []
    
    def _on_step(self) -> bool:
        """Called after each step in all environments"""
        # Check if any environment finished an episode
        dones = self.locals.get('dones', [])
        
        for i, done in enumerate(dones):
            if done:
                # Get info for this environment
                infos = self.locals.get('infos', [])
                if i < len(infos):
                    info = infos[i]
                    
                    # Extract metrics from info dict
                    final_dist = info.get('distance', float('inf'))
                    
                    # Get episode info (tracked by SB3)
                    ep_info = info.get('episode')
                    if ep_info is not None:
                        ep_reward = ep_info['r']
                        ep_length = ep_info['l']
                        
                        # Store metrics
                        self.episode_rewards.append(ep_reward)
                        self.episode_lengths.append(ep_length)
                        
                        # Success if final distance < threshold
                        success = float(final_dist < self.threshold)
                        self.episode_successes.append(success)
                        self.episode_final_distances.append(final_dist)
                        
                        # Log individual episode metrics
                        self.logger.record('ot2/episode_reward', ep_reward)
                        self.logger.record('ot2/episode_length', ep_length)
                        self.logger.record('ot2/final_distance_mm', final_dist * 1000)
                        self.logger.record('ot2/success', success)
                        
                        # Log rolling averages (last 100 episodes)
                        if len(self.episode_successes) >= 10:
                            window = min(100, len(self.episode_successes))
                            self.logger.record('ot2/success_rate_100ep', 
                                             np.mean(self.episode_successes[-window:]))
                            self.logger.record('ot2/avg_length_100ep', 
                                             np.mean(self.episode_lengths[-window:]))
                            self.logger.record('ot2/avg_final_dist_mm_100ep', 
                                             np.mean(self.episode_final_distances[-window:]) * 1000)
        
        return True
    
    def _on_training_end(self) -> None:
        """Print summary statistics at end of training"""
        if len(self.episode_successes) > 0:
            print("\n" + "="*60)
            print("TRAINING SUMMARY")
            print("="*60)
            print(f"Total episodes: {len(self.episode_successes)}")
            print(f"Success rate: {100*np.mean(self.episode_successes):.1f}%")
            print(f"Average episode length: {np.mean(self.episode_lengths):.1f} steps")
            print(f"Average final distance: {1000*np.mean(self.episode_final_distances):.3f} mm")
            
            # Stats for successful episodes only
            successful_lengths = [l for l, s in zip(self.episode_lengths, self.episode_successes) if s]
            if successful_lengths:
                print(f"Successful episodes avg length: {np.mean(successful_lengths):.1f} steps")
            
            print("="*60)


# ============================================================================
# Early Stopping Callback
# ============================================================================
class EarlyStoppingCallback(BaseCallback):
    """Stop training when success rate reaches threshold"""
    
    def __init__(self, ot2_callback, success_threshold=0.95, check_freq=10000, min_episodes=100, verbose=1):
        super().__init__(verbose)
        self.ot2_callback = ot2_callback
        self.success_threshold = success_threshold
        self.check_freq = check_freq
        self.min_episodes = min_episodes
        
    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            if len(self.ot2_callback.episode_successes) >= self.min_episodes:
                recent_success = np.mean(self.ot2_callback.episode_successes[-100:])
                if recent_success >= self.success_threshold:
                    if self.verbose > 0:
                        print(f"\n🎉 Early stopping! Success rate {recent_success:.1%} >= {self.success_threshold:.1%}")
                    return False  # Stop training
        return True


# ============================================================================
# Learning Rate Schedule
# ============================================================================
def linear_schedule(initial_value: float):
    """
    Linear learning rate schedule that decays to 0 by end of training.
    Helps with fine-tuning in later stages.
    """
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func


# ============================================================================
# ClearML Setup
# ============================================================================
task_name = f'model_{timestamp}'

task = Task.init(
    project_name='Mentor Group - Uther/Group 2', 
    task_name=task_name,
)

# Set repo
task.set_repo(
    repo='https://github.com/ZofiaPilitowska240272/ot2-rl-training.git',
    branch='main'
)

task.set_base_docker('deanis/2023y2b-rl:latest')
task.set_packages(['tensorboard', 'clearml', 'wandb'])

# ============================================================================
# Command Line Arguments
# ============================================================================
parser = argparse.ArgumentParser()
parser.add_argument("--learning_rate", type=float, default=0.001)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--n_steps", type=int, default=2048)
parser.add_argument("--total_timesteps", type=int, default=3000000)
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--max_steps_truncate", type=int, default=500)
parser.add_argument("--target_threshold", type=float, default=0.001)
parser.add_argument("--max_velocity", type=float, default=0.05)
parser.add_argument("--run_name", type=str, default="default_run")
parser.add_argument("--n_epochs", type=int, default=10)
parser.add_argument("--gae_lambda", type=float, default=0.95)
parser.add_argument("--clip_range", type=float, default=0.2)
parser.add_argument("--ent_coef", type=float, default=0.0)
parser.add_argument("--vf_coef", type=float, default=0.5)
parser.add_argument("--use_lr_schedule", action='store_true', help="Use linear LR decay")
parser.add_argument("--normalize_env", action='store_true', help="Normalize observations and rewards")
parser.add_argument("--checkpoint_freq", type=int, default=100000, help="Save checkpoint every N steps")
parser.add_argument("--early_stop_success", type=float, default=0.95, help="Stop if success rate reaches this")

args = parser.parse_args()

# Execute remotely AFTER capturing arguments
task.execute_remotely(queue_name='default')

# ============================================================================
# Generate Filename
# ============================================================================
def format_lr(lr):
    """Convert learning rate to scientific notation string for filename"""
    return f"{lr:.0e}".replace("+", "").replace("-0", "-")

lr_str = format_lr(args.learning_rate)
filename = f"{timestamp}_lr{lr_str}_b{args.batch_size}_s{args.n_steps}_th{int(args.target_threshold*1000)}mm"

print("="*60)
print(f"Training Configuration:")
print(f"  Learning Rate: {args.learning_rate}")
print(f"  LR Schedule: {'Linear decay' if args.use_lr_schedule else 'Constant'}")
print(f"  Batch Size: {args.batch_size}")
print(f"  N Steps: {args.n_steps}")
print(f"  N Epochs: {args.n_epochs}")
print(f"  Total Timesteps: {args.total_timesteps:,}")
print(f"  Max Episode Steps: {args.max_steps_truncate}")
print(f"  Target Threshold: {args.target_threshold*1000:.1f}mm")
print(f"  Max Velocity: {args.max_velocity}")
print(f"  Gamma: {args.gamma}")
print(f"  GAE Lambda: {args.gae_lambda}")
print(f"  Clip Range: {args.clip_range}")
print(f"  Entropy Coef: {args.ent_coef}")
print(f"  Value Coef: {args.vf_coef}")
print(f"  Normalize Env: {args.normalize_env}")
print(f"  Checkpoint Freq: {args.checkpoint_freq:,}")
print(f"  Early Stop Success: {args.early_stop_success:.1%}")
print(f"  Model Name: {filename}")
print("="*60)

# ============================================================================
# Environment Setup
# ============================================================================
# Create base environment
base_env = OT2GymEnv(
    use_gui=False,
    max_steps=args.max_steps_truncate,
    success_threshold=args.target_threshold,
    max_velocity=args.max_velocity
)

# Wrap in DummyVecEnv (required for VecNormalize)
env = DummyVecEnv([lambda: base_env])

# Optionally normalize observations and rewards
if args.normalize_env:
    print("✅ Using observation and reward normalization")
    env = VecNormalize(
        env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=args.gamma
    )
else:
    print("⚠️ Not using normalization (set --normalize_env to enable)")

# ============================================================================
# Model Setup
# ============================================================================
# Choose learning rate schedule
if args.use_lr_schedule:
    learning_rate = linear_schedule(args.learning_rate)
    print("✅ Using linear LR schedule (decays to 0)")
else:
    learning_rate = args.learning_rate
    print("⚠️ Using constant learning rate")

model = PPO(
    'MlpPolicy',
    env,
    learning_rate=learning_rate,
    batch_size=args.batch_size,
    n_steps=args.n_steps,
    n_epochs=args.n_epochs,
    gamma=args.gamma,
    gae_lambda=args.gae_lambda,
    clip_range=args.clip_range,
    ent_coef=args.ent_coef,
    vf_coef=args.vf_coef,
    tensorboard_log="./tensorboard_logs/",
    verbose=1,
    device='auto'  # Use GPU if available
)

# ============================================================================
# Callbacks Setup
# ============================================================================
# OT2 metrics callback
ot2_callback = OT2Callback(threshold=args.target_threshold, verbose=1)

# Checkpoint callback
os.makedirs('./checkpoints/', exist_ok=True)
checkpoint_callback = CheckpointCallback(
    save_freq=args.checkpoint_freq,
    save_path='./checkpoints/',
    name_prefix=f'ot2_checkpoint_{timestamp}',
    verbose=1
)

# Early stopping callback
early_stop_callback = EarlyStoppingCallback(
    ot2_callback=ot2_callback,
    success_threshold=args.early_stop_success,
    check_freq=10000,
    verbose=1
)

# Combine all callbacks
combined_callback = CallbackList([
    ot2_callback,
    checkpoint_callback,
    early_stop_callback
])

# ============================================================================
# Training
# ============================================================================
print("\n" + "="*60)
print("Starting Training...")
print("="*60 + "\n")

model.learn(
    total_timesteps=args.total_timesteps,
    callback=combined_callback,
    tb_log_name=f"PPO_{args.run_name}_{filename}",
    progress_bar=True
)

# ============================================================================
# Save Final Model and Normalization Stats
# ============================================================================
model_name = f"{filename}.zip"
model.save(model_name)
print(f"\n✅ Model saved: {model_name}")

# Save normalization statistics if used
if args.normalize_env:
    norm_stats_name = f"{filename}_vecnormalize.pkl"
    env.save(norm_stats_name)
    print(f"✅ Normalization stats saved: {norm_stats_name}")
    task.upload_artifact("normalization_stats", artifact_object=norm_stats_name)

# Upload to ClearML
task.upload_artifact("model", artifact_object=model_name)
print(f"✅ Artifact uploaded: {model_name}")

print("\n" + "="*60)
print("🎉 Training Complete!")
print("="*60)

# ============================================================================
# Cleanup
# ============================================================================
try:
    env.close()
except:
    pass