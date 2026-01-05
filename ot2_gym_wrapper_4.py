import gymnasium as gym
import numpy as np
import pybullet as p
from gymnasium import spaces
from sim_class import Simulation

X_MIN, X_MAX = -0.260, 0.134
Y_MIN, Y_MAX = -0.260, 0.130
Z_MIN, Z_MAX = 0.030, 0.200

import os
print(os.getcwd())

class OT2Env(gym.Env):
    def __init__(self, render=False, max_steps=1000, target_threshold=0.001, num_agents=1):
        super(OT2Env, self).__init__()

        self.render = render
        self.max_steps = max_steps
        self.sim = Simulation(num_agents=num_agents, render=render)
        self.target_threshold = target_threshold

        self.action_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(6,), dtype=np.float32)

        self.steps = 0
        self.prev_position = None

    def reset(self, seed=42):
        observation = self.sim.reset()

        robotId = list(observation.keys())[0]

        np.random.seed(seed)
        self.goal_position = np.random.uniform(low=(X_MIN, Y_MIN, Z_MIN), high=(X_MAX, Y_MAX, Z_MAX))

        pipette_position = self.sim.pipette_positions[robotId]

        observation = np.concatenate([
            self._normalize_position(pipette_position),
            self._normalize_position(self.goal_position)
        ], dtype=np.float32)
        self.steps = 0
        self.prev_position = observation[:3]
        return observation, {}

    def get_goal_position(self):
        return self.goal_position

    def step(self, action):
        try:
            observation = self.sim.run(action)
        except IndexError:
            observation = self.sim.run([action])

        robotId = list(observation.keys())[0]
        robot_state = observation.get(robotId, {})
        self.pipette_position = np.array(
            robot_state.get('pipette_position', [0.0, 0.0, 0.0]),
            dtype=np.float32
        )

        observation = np.concatenate([
            self._normalize_position(self.pipette_position),
            self._normalize_position(self.goal_position)
        ], dtype=np.float32)

        distance_to_goal = np.linalg.norm(self.pipette_position - self.goal_position)
        reward = self.calculate_reward(distance_to_goal)

        task_completed = distance_to_goal < self.target_threshold

        truncated = self.steps >= self.max_steps

        info = {
            'distance_to_goal': float(distance_to_goal),
            'current_position': self.pipette_position.tolist(),
            'goal_position': self.goal_position.tolist()
        }

        self.steps += 1
        self.prev_position = observation[:3]
        return observation, reward, task_completed, truncated, info

    def render(self, rendermode="human"):
        if self.render:
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

    def calculate_reward(self, distance_to_goal=None):
        distance_factor = 10

        prev_distance_to_goal = np.linalg.norm(self.prev_position - self.goal_position)
        reward = -distance_to_goal * distance_factor

        if distance_to_goal > prev_distance_to_goal:
            reward_moving_to_goal = -1
        else:
            reward_moving_to_goal = 1

        reward = reward + reward_moving_to_goal
        return reward

    def close(self):
        self.sim.close()

    def _normalize_position(self, position):
        if not isinstance(position, np.ndarray):
            position = np.array(position, dtype=np.float32)
        WORKSPACE_LOW  = np.array([X_MIN, Y_MIN, Z_MIN], dtype=np.float32)
        WORKSPACE_HIGH = np.array([X_MAX, Y_MAX, Z_MAX], dtype=np.float32)
        normalized = 2.0 * (position - WORKSPACE_LOW) / (WORKSPACE_HIGH - WORKSPACE_LOW) - 1.0
        return normalized.astype(np.float32)

if __name__ == "__main__":
    pass