"""
OT-2 RL Training Script with ClearML Integration
-----------------------------------------------
Train RL agents (PPO, SAC, TD3) to control the Opentrons OT-2 pipette system.
Fully integrated with ClearML for experiment tracking, remote execution, and artifact management.
"""

import os
import argparse
import json
import numpy as np
from datetime import datetime

import gymnasium as gym
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback

from clearml import Task
from ot2_gym_wrapper import OT2GymEnv  # Your custom wrapper


# ===================== Custom Evaluation Callback =====================
class EvalCallback(BaseCallback):
    """
    Evaluate agent periodically and save best model based on success rate.
    """
    def __init__(self, eval_env, eval_freq=10000, n_eval_episodes=5, save_path="best_model", verbose=1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.best_success_rate = -np.inf
        self.save_path = save_path

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq != 0:
            return True

        success_count = 0
        for _ in range(self.n_eval_episodes):
            obs, info = self.eval_env.reset()
            done = False
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.eval_env.step(action)
                done = terminated or truncated
            if info['success']:
                success_count += 1

        success_rate = success_count / self.n_eval_episodes

        if success_rate > self.best_success_rate:
            self.best_success_rate = success_rate
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
            self.model.save(self.save_path)
            if self.verbose > 0:
                print(f"New best model saved! Success rate: {success_rate*100:.1f}%")

        return True


# ===================== Gym Environment Factory =====================
def make_env(seed=0, **kwargs):
    def _init():
        env = OT2GymEnv(**kwargs)
        env = Monitor(env)
        env.reset(seed=seed)
        return env
    set_random_seed(seed)
    return _init


# ===================== Argument Parser =====================
def parse_args():
    parser = argparse.ArgumentParser(description="Train RL agent for OT-2 pipette with ClearML")
    parser.add_argument('--algorithm', type=str, default='PPO', choices=['PPO', 'SAC', 'TD3'])
    parser.add_argument('--total_timesteps', type=int, default=500000)
    parser.add_argument('--n_envs', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max_steps', type=int, default=500)
    parser.add_argument('--success_threshold', type=float, default=0.005)
    parser.add_argument('--reward_scale', type=float, default=1.0)
    parser.add_argument('--net_arch', nargs='+', type=int, default=[256, 256])
    parser.add_argument('--run_name', type=str, default=None)
    parser.add_argument('--save_dir', type=str, default='models')
    parser.add_argument('--use_gpu', action='store_true')
    return parser.parse_args()


# ===================== Model Creation =====================
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
        return PPO(**common_params, n_steps=2048, batch_size=128, n_epochs=10, gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01)
    elif algorithm == 'SAC':
        return SAC(**common_params, buffer_size=200000, batch_size=256, tau=0.005, gamma=0.99, learning_starts=10000, train_freq=1, gradient_steps=1, ent_coef='auto')
    elif algorithm == 'TD3':
        return TD3(**common_params, buffer_size=200000, batch_size=256, tau=0.005, gamma=0.99, learning_starts=10000, train_freq=1, gradient_steps=1)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")


# ===================== Training Function =====================
def train(args):
    # ------------------ ClearML Task ------------------
    if args.run_name is None:
        args.run_name = f"{args.algorithm}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    task = Task.init(project_name='OT2-RL-Controller', task_name=args.run_name)
    task.connect(vars(args))
    task.execute_remotely(queue_name="default")
    print(f"ClearML task initialized: {task.id}")

    # ------------------ Environment ------------------
    env_kwargs = {
        'max_steps': args.max_steps,
        'success_threshold': args.success_threshold,
        'reward_scale': args.reward_scale,
        'use_gui': False
    }

    env = DummyVecEnv([make_env(seed=args.seed+i, **env_kwargs) for i in range(args.n_envs)])
    eval_env = Monitor(OT2GymEnv(**env_kwargs))

    # ------------------ Model ------------------
    model = create_model(args.algorithm, env, args)

    # ------------------ Directories ------------------
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    best_model_path = os.path.join(save_dir, 'best_model.zip')
    checkpoint_dir = os.path.join(save_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)

    # ------------------ Callbacks ------------------
    checkpoint_callback = CheckpointCallback(save_freq=50000 // args.n_envs, save_path=checkpoint_dir)
    eval_callback = EvalCallback(eval_env=eval_env, eval_freq=10000 // args.n_envs, save_path=best_model_path)
    callbacks = CallbackList([checkpoint_callback, eval_callback])

    # ------------------ Training ------------------
    print(f"Starting training {args.algorithm} for {args.total_timesteps} timesteps")
    model.learn(total_timesteps=args.total_timesteps, callback=callbacks, log_interval=10, progress_bar=True)

    # ------------------ Save Final Model ------------------
    final_model_path = os.path.join(save_dir, f"{args.algorithm}_final.zip")
    model.save(final_model_path)
    print(f"Final model saved to {final_model_path}")
    task.upload_artifact(name='final_model', artifact_object=final_model_path)
    task.upload_artifact(name='best_model', artifact_object=best_model_path)

    env.close()
    eval_env.close()
    print("Training completed.")


# ===================== Main =====================
if __name__ == "__main__":
    args = parse_args()
    train(args)
