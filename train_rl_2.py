"""
OT-2 RL Training - Simple Version
"""

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from clearml import Task
import argparse
from datetime import datetime
import numpy as np

# Import your wrapper
from ot2_gym_wrapper_2 import OT2GymEnv



# Generate timestamp for unique task name and model filename
timestamp = datetime.now().strftime("%y%m%d.%H%M")
print(f"Timestamp: {timestamp}")

# ============================================================================
# Custom Callback for OT2 Metrics
# ============================================================================
class OT2Callback(BaseCallback):
    """
    Callback for logging OT2-specific metrics during training.
    """
    
    def __init__(self, threshold=0.001, verbose=0):
        super().__init__(verbose)
        self.threshold = threshold
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_successes = []
        self.episode_final_distances = []
    
    def _on_step(self) -> bool:
        """Called after each step in all environments"""
        dones = self.locals.get('dones', [])
        
        for i, done in enumerate(dones):
            if done:
                infos = self.locals.get('infos', [])
                if i < len(infos):
                    info = infos[i]
                    
                    # Extract metrics - YOUR wrapper uses 'distance' not 'distance_to_goal'
                    final_dist = info.get('distance', float('inf'))
                    
                    # Get episode info from SB3
                    ep_info = info.get('episode')
                    if ep_info is not None:
                        ep_reward = ep_info['r']
                        ep_length = ep_info['l']
                        
                        # Store metrics
                        self.episode_rewards.append(ep_reward)
                        self.episode_lengths.append(ep_length)
                        
                        success = float(final_dist < self.threshold)
                        self.episode_successes.append(success)
                        self.episode_final_distances.append(final_dist)
                        
                        # Log to tensorboard
                        self.logger.record('ot2/episode_reward', ep_reward)
                        self.logger.record('ot2/episode_length', ep_length)
                        self.logger.record('ot2/final_distance_mm', final_dist * 1000)
                        self.logger.record('ot2/success', success)
                        
                        # Rolling averages
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
        """Print summary at end of training"""
        if len(self.episode_successes) > 0:
            print("\n" + "="*60)
            print("TRAINING SUMMARY")
            print("="*60)
            print(f"Total episodes: {len(self.episode_successes)}")
            print(f"Success rate: {100*np.mean(self.episode_successes):.1f}%")
            print(f"Average episode length: {np.mean(self.episode_lengths):.1f} steps")
            print(f"Average final distance: {1000*np.mean(self.episode_final_distances):.3f} mm")
            
            successful_lengths = [l for l, s in zip(self.episode_lengths, self.episode_successes) if s]
            if successful_lengths:
                print(f"Successful episodes avg length: {np.mean(successful_lengths):.1f} steps")
            
            print("="*60)


# ============================================================================
# ClearML Setup
# ============================================================================
task_name = f'model_ppo_{timestamp}'

task = Task.init(
    project_name='Mentor Group - Uther/Group 2', 
    task_name=task_name,
)

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
parser.add_argument("--ent_coef", type=float, default=0.0)
parser.add_argument("--gae_lambda", type=float, default=0.95)
parser.add_argument("--clip_range", type=float, default=0.2)
parser.add_argument("--vf_coef", type=float, default=0.5)
parser.add_argument("--n_epochs", type=int, default=10)
parser.add_argument("--run_name", type=str, default="default_run")
args = parser.parse_args()

# Execute remotely AFTER capturing arguments
task.execute_remotely(queue_name='default')

# ============================================================================
# Generate Filename
# ============================================================================
def format_lr(lr):
    """Convert learning rate to scientific notation for filename"""
    return f"{lr:.0e}".replace("+", "").replace("-0", "-")

lr_str = format_lr(args.learning_rate)
filename = f"{timestamp}_lr{lr_str}_b{args.batch_size}_s{args.n_steps}_th{int(args.target_threshold*1000)}mm"

print("="*60)
print(f"Training Configuration:")
print(f"  Learning Rate: {args.learning_rate}")
print(f"  Batch Size: {args.batch_size}")
print(f"  N Steps: {args.n_steps}")
print(f"  N Epochs: {args.n_epochs}")
print(f"  Total Timesteps: {args.total_timesteps:,}")
print(f"  Max Episode Steps: {args.max_steps_truncate}")
print(f"  Target Threshold: {args.target_threshold*1000:.1f}mm")
print(f"  Max Velocity: {args.max_velocity}")
print(f"  Clip Range: {args.clip_range}")
print(f"  GAE Lambda: {args.gae_lambda}")
print(f"  Entropy Coef: {args.ent_coef}")
print(f"  Value Coef: {args.vf_coef}")
print(f"  Model Name: {filename}")
print("="*60)

# ============================================================================
# Environment Setup
# ============================================================================
env = OT2GymEnv(
    use_gui=False,
    max_steps=args.max_steps_truncate,
    success_threshold=args.target_threshold,
    max_velocity=args.max_velocity
)

# ============================================================================
# Model Setup - ALL ARGS NOW USED!
# ============================================================================
model = PPO(
    'MlpPolicy',
    env,
    learning_rate=args.learning_rate,
    batch_size=args.batch_size,
    n_steps=args.n_steps,
    n_epochs=args.n_epochs,
    gamma=args.gamma,
    gae_lambda=args.gae_lambda,
    clip_range=args.clip_range,
    ent_coef=args.ent_coef,
    vf_coef=args.vf_coef,
    verbose=1
)

# ============================================================================
# Training
# ============================================================================
ot2_callback = OT2Callback(threshold=args.target_threshold, verbose=1)

model.learn(
    total_timesteps=args.total_timesteps,
    callback=ot2_callback,
    tb_log_name=f"PPO_{args.run_name}_{filename}"
)

# ============================================================================
# Save and Upload Model
# ============================================================================
model_name = f"{filename}.zip"
model.save(model_name)
print(f"\n✅ Model saved: {model_name}")

task.upload_artifact("model", artifact_object=model_name)
print(f"✅ Artifact uploaded: {model_name}")

print("\n🎉 Training complete!")

# Close environment
try:
    env.close()
except:
    pass