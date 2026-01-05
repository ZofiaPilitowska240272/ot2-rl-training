import gymnasium as gym
import numpy as np
import pybullet as p
from gymnasium import spaces
from sim_class import Simulation

# Working envolope
X_MIN, X_MAX = -0.260, 0.134
Y_MIN, Y_MAX = -0.260, 0.130
Z_MIN, Z_MAX = 0.030, 0.200

import os
print(os.getcwd())
class OT2Env(gym.Env):
    """
    OpenAI Gym environment wrapper for the OT-2 robotic system.

    Properties:
        - Action Space
        - Observation Space
        - Possible actions
        - Possible Observations (current state? all possible states?)
        - Reward function
        - Done conditions

    Methods:
        - Reset
        - Step
        - Render
        - Close
    """

    def __init__(self, render=False, max_steps=1000, target_threshold=0.001, num_agents=1):
        super(OT2Env, self).__init__()

        # Initialize environment parameters
        self.render = render
        self.max_steps = max_steps
        self.sim = Simulation(num_agents=num_agents, render=render)
        self.target_threshold = target_threshold

        # Define action and observation spaces
        self.action_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(6,), dtype=np.float32)

        # Initialize step counter
        self.steps = 0
        self.prev_position = None

    def reset(self, seed=42):
        self.sim.set_start_position(0.0, 0.0, 0.15)
        # Reset the simulation environment
        observation = self.sim.reset()
        print(f"ot2_gym_wrapper:self.sim.reset():{observation}")

        # Extract robot ID from the observation
        robotId = list(observation.keys())[0]
        # print(f"ot2_gym_wrapper:observation.keys:{robotId}")

        # Set a random goal position for the episode
        np.random.seed(seed)
        self.goal_position = np.random.uniform(low=(X_MAX, Y_MIN, Z_MIN), high=(X_MAX, Y_MAX, Z_MAX))
        # print(f"ot2_gym_wrapper:self.goal_position:{self.goal_position}")

        # Get the initial pipette position
        pipette_position = self.sim.pipette_positions[robotId]
        # print(f"ot2_gym_wrapper:self.pipette_position:{self.pipette_position}")

        # Concatenate pipette and goal positions for the observation
        observation = np.concatenate([
            self._normalize_position(pipette_position),
            self._normalize_position(self.goal_position)
        ], dtype=np.float32)
        self.steps = 0
        self.prev_position = observation[:3]
        return observation, {}

    def get_goal_position(self) -> tuple[float, float, float]:
        return self.goal_position

    def step(self, action):
        # Concatenate a placeholder for the "drop" action to the given action
        # action = np.concatenate([action[:3], [0]])
        # print(f"ot2_gym_wrapper:step:action:{action}")
        # threshold = 0.5
        # action[-1] = 1 if action[-1] > threshold else 0

        try:
            # Run the simulation with the given action
            observation = self.sim.run(action)
        except IndexError:
            observation = self.sim.run([action])

        # Extract robot ID from the observation
        robotId = list(observation.keys())[0]
        robot_state = observation.get(robotId, {})
        self.pipette_position = np.array(
            robot_state.get('pipette_position', [0.0, 0.0, 0.0]),
            dtype=np.float32
        )
       

    
        # Concatenate pipette and goal positions for the observation
        observation = np.concatenate([
            self._normalize_position(self.pipette_position),
            self._normalize_position(self.goal_position)
        ], dtype=np.float32)

        
        # Calculate the Euclidean distance between current and goal positions
        distance_to_goal = np.linalg.norm(self.pipette_position - self.goal_position)
        reward = self.calculate_reward(distance_to_goal)

        # if action[3] == 1:
        #     penalty = 0.1 * np.linalg.norm(self.pipette_position - self.goal_position)
        #     reward += penalty


        # Check if the task is completed (distance below a threshold)
        task_completed = distance_to_goal < self.target_threshold

        # Check if the episode should be truncated
        truncated = self.steps >= self.max_steps

        # Additional info
        info = {
            'distance_to_goal': float(distance_to_goal),
            'current_position': self.pipette_position.tolist(),
            'goal_position': self.goal_position.tolist()
        }

        # Increment the number of steps
        self.steps += 1
        self.prev_position = observation[:3]
        return observation, reward, task_completed, truncated, info

    def render(self, rendermode="human"):
        # super(OT2Env, self).render()
        if self.render:
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

    # def get_reward(self):
    #     # Define a step penalty to encourage faster completion
    #     step_penalty_coefficient = 0.1  # Adjust as needed

    #     # Calculate the Euclidean distance between the current position and the goal
    #     distance_to_goal = -np.linalg.norm(self.pipette_position - self.goal_position)

    #     # Calculate the step-based penalty
    #     step_penalty = -step_penalty_coefficient * self.steps

    #     # Calculate the total reward as a combination of distance and step penalties
    #     total_reward = distance_to_goal + step_penalty

    #     return total_reward

    def calculate_reward(self, distance_to_goal=None):
        distance_factor = 10

        # logging.warning(f'ot2_gym_wrapper:calculate_reward:distance_to_goal:{distance_to_goal}')
        prev_distance_to_goal = np.linalg.norm(self.prev_position - self.goal_position)
        # logging.info(f'ot2_gym_wrapper:calculate_reward:prev_distance_to_goal:{prev_distance_to_goal}')
        reward = -distance_to_goal * distance_factor
        # logging.info(f'ot2_gym_wrapper:calculate_reward:reward:1:{reward}')

        if distance_to_goal > prev_distance_to_goal:
            reward_moving_to_goal = -1
        else:
            reward_moving_to_goal = 1     

        # time_penalty = self.step * 0.2
        # logging.info(f'ot2_gym_wrapper:calculate_reward:reward_moving_to_goal:{reward_moving_to_goal}')
        reward = reward + reward_moving_to_goal
        # logging.info(f'ot2_gym_wrapper:calculate_reward:reward:2:{reward}')
        return reward

    def close(self):
        # Close the simulation environment
        self.sim.close()

    def _normalize_position(self, position):
        """Normalize position from workspace bounds to [-1, 1]."""
        if not isinstance(position, np.ndarray):
            position = np.array(position, dtype=np.float32)
        WORKSPACE_LOW  = np.array([X_MIN, Y_MIN, Z_MIN], dtype=np.float32)
        WORKSPACE_HIGH = np.array([X_MAX, Y_MAX, Z_MAX], dtype=np.float32)
        normalized = 2.0 * (position - WORKSPACE_LOW) / (WORKSPACE_HIGH - WORKSPACE_LOW) - 1.0
        return normalized.astype(np.float32)

if __name__ == "__main__":
    pass