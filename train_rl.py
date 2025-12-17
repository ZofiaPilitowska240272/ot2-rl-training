"""
train_rl.py - Reinforcement Learning Training for OT-2 Pipette Controller

This script trains RL agents (PPO, SAC, TD3) on the OT-2 pipette environment.
Includes:
- Vectorized environments
- Checkpointing
- Evaluation callback
- Model saving
"""

import os
import argparse
import json
import numpy as np
from datetime import datetime
import gymnasium as gym

from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed

from ot2_gym_wrapper import OT2GymEnv  # Make sure your OT2GymEnv wrapper is in PYTHONPATH


# ------------------------ Custom Evaluation Callback ------------------------
class EvalCallback(BaseCallback):
    """
    Custom callback for evaluating RL agent performance on OT-2 environment.
    Logs:
    - Mean reward
    - Success rate
    - Final distance to target
    Saves best model automatically.
    """
    def __init__(self, eval_env, eval_freq=5000, n_eval_episodes=5,
                 best_model_save_path='best_model', verbose=1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.best_model_save_path = best_model_save_path
        self.best_success_rate = 0.0

        os.makedirs(best_model_save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq != 0:
            return True

        success_count = 0
        final_distances = []

        for _ in range(self.n_eval_episodes):
            obs, info = self.eval_env.reset()
            done = False
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.eval_env.step(action)
                done = terminated or truncated

            if info['success']:
                success_count += 1
            final_distances.append(info['distance'])

        success_rate = success_count / self.n_eval_episodes
        mean_final_distance = np.mean(final_distances) * 1000  # mm

        if self.verbose > 0:
            print(f"\nEvaluation at step {self.num_timesteps}")
            print(f"Success rate: {success_rate*100:.1f}% ({success_count}/{self.n_eval_episodes})")
            print(f"Mean final distance: {mean_final_distance:.2f} mm")

        # Save best model
        if success_rate > self.best_success_rate:
            self.best_success_rate = success_rate
            path = os.path.join(self.best_model_save_path, 'best_model')
            self.model.save(path)
            if self.verbose > 0:
                print(f"New best model saved to {path}.zip")

        return True


# ------------------------ Environment Factory ------------------------
def make_env(rank=0, seed=0, **env_kwargs):
    def _init():
        env = OT2GymEnv(**env_kwargs)
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed)
    return _init


# ------------------------ Model Factory ------------------------
def create_model(algorithm, env, args):
    policy_kwargs = {'net_arch': args.net_arch}

    common_params = {
        'policy': 'MlpPolicy',
        'env': env,
        'learning_rate': args.learning_rate,
        'verbose': 1,
        'seed': args.seed,
        'device': 'cuda' if args.use_gpu else 'auto',
        'policy_kwargs': policy_kwargs
    }

    if algorithm == 'PPO':
        return PPO(**common_params,
                   n_steps=args.n_steps,
                   batch_size=args.batch_size,
                   n_epochs=args.n_epochs,
                   gamma=args.gamma,
                   gae_lambda=args.gae_lambda,
                   clip_range=args.clip_range,
                   ent_coef=args.ent_coef)
    elif algorithm == 'SAC':
        return SAC(**common_params,
                   batch_size=args.batch_size,
                   buffer_size=args.buffer_size,
                   tau=args.tau,
                   gamma=args.gamma,
                   learning_starts=args.learning_starts)
    elif algorithm == 'TD3':
        return TD3(**common_params,
                   batch_size=args.batch_size,
                   buffer_size=args.buffer_size,
                   tau=args.tau,
                   gamma=args.gamma,
                   learning_starts=args.learning_starts)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")


# ------------------------ Argument Parser ------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Train RL agent for OT-2 pipette controller")

    # Algorithm
    parser.add_argument('--algorithm', type=str, default='PPO', choices=['PPO', 'SAC', 'TD3'])
    parser.add_argument('--total_timesteps', type=int, default=100000)
    parser.add_argument('--n_envs', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--use_gpu', action='store_true')

    # PPO parameters
    parser.add_argument('--n_steps', type=int, default=2048)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--n_epochs', type=int, default=10)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--gae_lambda', type=float, default=0.95)
    parser.add_argument('--clip_range', type=float, default=0.2)
    parser.add_argument('--ent_coef', type=float, default=0.01)

    # SAC/TD3 parameters
    parser.add_argument('--buffer_size', type=int, default=200000)
    parser.add_argument('--tau', type=float, default=0.005)
    parser.add_argument('--learning_starts', type=int, default=10000)

    # Network architecture
    parser.add_argument('--net_arch', type=int, nargs='+', default=[256, 256])

    # Environment parameters
    parser.add_argument('--max_velocity', type=float, default=0.05)
    parser.add_argument('--success_threshold', type=float, default=0.005)
    parser.add_argument('--max_steps', type=int, default=500)
    parser.add_argument('--reward_scale', type=float, default=1.0)

    # Checkpointing & evaluation
    parser.add_argument('--save_dir', type=str, default='models')
    parser.add_argument('--eval_freq', type=int, default=5000)
    parser.add_argument('--n_eval_episodes', type=int, default=5)
    parser.add_argument('--checkpoint_freq', type=int, default=20000)
    parser.add_argument('--log_interval', type=int, default=10)

    return parser.parse_args()


# ------------------------ Training Function ------------------------
def train(args):
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(f"{args.save_dir}/checkpoints", exist_ok=True)
    os.makedirs(f"{args.save_dir}/best", exist_ok=True)

    env_kwargs = {
        'max_velocity': args.max_velocity,
        'success_threshold': args.success_threshold,
        'max_steps': args.max_steps,
        'reward_scale': args.reward_scale,
        'use_gui': False
    }

    # Create vectorized training environment
    env = DummyVecEnv([make_env(i, args.seed, **env_kwargs) for i in range(args.n_envs)])

    # Create evaluation environment
    eval_env = OT2GymEnv(**env_kwargs)
    eval_env = Monitor(eval_env)

    # Create RL model
    model = create_model(args.algorithm, env, args)

    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=max(args.checkpoint_freq // args.n_envs, 1),
        save_path=f"{args.save_dir}/checkpoints",
        name_prefix=f"{args.algorithm}_checkpoint",
        save_vecnormalize=True,
        save_replay_buffer=True
    )

    eval_callback = EvalCallback(
        eval_env=eval_env,
        eval_freq=max(args.eval_freq // args.n_envs, 1),
        n_eval_episodes=args.n_eval_episodes,
        best_model_save_path=f"{args.save_dir}/best",
        verbose=1
    )

    callbacks = CallbackList([checkpoint_callback, eval_callback])

    # Train model
    model.learn(total_timesteps=args.total_timesteps,
                callback=callbacks,
                log_interval=args.log_interval,
                progress_bar=True)

    # Save final model
    final_path = os.path.join(args.save_dir, f"{args.algorithm}_final")
    model.save(final_path)
    print(f"Final model saved to {final_path}.zip")

    env.close()
    eval_env.close()


# ------------------------ Main ------------------------
def main():
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
