import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from clearml import Task
import argparse
from datetime import datetime
import numpy as np

from ot2_gym_wrapper_4 import OT2Env

timestamp = datetime.now().strftime("%y%m%d.%H%M")
print(f"Timestamp: {timestamp}")

class OT2Callback(BaseCallback):
    def __init__(self, threshold=0.001, verbose=0):
        super().__init__(verbose)
        self.threshold = threshold
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_successes = []
        self.episode_final_distances = []
    
    def _on_step(self) -> bool:
        dones = self.locals.get('dones', [])
        for i, done in enumerate(dones):
            if done:
                infos = self.locals.get('infos', [])
                if i < len(infos):
                    info = infos[i]
                    final_dist = info.get('distance_to_goal', float('inf'))
                    ep_info = info.get('episode')
                    if ep_info is not None:
                        ep_reward = ep_info['r']
                        ep_length = ep_info['l']
                        self.episode_rewards.append(ep_reward)
                        self.episode_lengths.append(ep_length)
                        success = float(final_dist < self.threshold)
                        self.episode_successes.append(success)
                        self.episode_final_distances.append(final_dist)
                        self.logger.record('ot2/episode_reward', ep_reward)
                        self.logger.record('ot2/episode_length', ep_length)
                        self.logger.record('ot2/final_distance_mm', final_dist * 1000)
                        self.logger.record('ot2/success', success)
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

task_name = f'OT2_RL_{timestamp}'

task = Task.init(
    project_name='Mentor Group - Uther/Group 2', 
    task_name=task_name,
)

task.set_repo(
    repo='https://github.com/ZofiaPilitowska240272/ot2-rl-training.git',
    branch='main'
)

task.set_base_docker('deanis/2023y2b-rl:latest')
task.set_packages(['tensorboard', 'clearml'])

parser = argparse.ArgumentParser()
parser.add_argument("--learning_rate", type=float, default=0.0003)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--n_steps", type=int, default=2048)
parser.add_argument("--total_timesteps", type=int, default=1000000)
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--max_steps_truncate", type=int, default=1000)
parser.add_argument("--target_threshold", type=float, default=0.001)
parser.add_argument("--ent_coef", type=float, default=0.0)
parser.add_argument("--gae_lambda", type=float, default=0.95)
parser.add_argument("--clip_range", type=float, default=0.3)
parser.add_argument("--vf_coef", type=float, default=0.5)
parser.add_argument("--n_epochs", type=int, default=10)
args = parser.parse_args()

task.execute_remotely(queue_name='default')

def format_lr(lr):
    return f"{lr:.0e}".replace("+", "").replace("-0", "-")

lr_str = format_lr(args.learning_rate)
filename = f"{timestamp}_lr{lr_str}_b{args.batch_size}_s{args.n_steps}_th{int(args.target_threshold*1000)}mm"

print("="*60)
print(f"Training Configuration:")
print(f"  Learning Rate: {args.learning_rate}")
print(f"  Batch Size: {args.batch_size}")
print(f"  N Steps: {args.n_steps}")
print(f"  Total Timesteps: {args.total_timesteps:,}")
print(f"  Max Episode Steps: {args.max_steps_truncate}")
print(f"  Target Threshold: {args.target_threshold*1000:.1f}mm")
print(f"  Model Name: {filename}")
print("="*60)

env = OT2Env(
    render=False, 
    max_steps=args.max_steps_truncate, 
    target_threshold=args.target_threshold
)

model = PPO(
    'MlpPolicy',
    env,
    learning_rate=args.learning_rate,
    batch_size=args.batch_size,
    n_steps=args.n_steps,
    gamma=args.gamma,
    ent_coef=args.ent_coef,
    gae_lambda=args.gae_lambda,
    clip_range=args.clip_range,
    vf_coef=args.vf_coef,
    n_epochs=args.n_epochs,
    verbose=1,
    tensorboard_log=f"runs/ot2_rl"
)

ot2_callback = OT2Callback(threshold=args.target_threshold, verbose=1)

model.learn(
    total_timesteps=args.total_timesteps,
    callback=ot2_callback,
    tb_log_name=f"PPO_{filename}"
)

model_name = f"{filename}.zip"
model.save(model_name)
print(f"\nModel saved: {model_name}")

task.upload_artifact("model", artifact_object=model_name)
print(f"Artifact uploaded: {model_name}")

print("\nTraining complete!")

try:
    env.close()
except:
    pass