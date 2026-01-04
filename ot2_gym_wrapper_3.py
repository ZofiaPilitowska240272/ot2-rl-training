"""
OT2 Gym Environment - Multiple Reward Functions
Properly scaled for 1mm precision threshold
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from sim_class import Simulation


class OT2GymEnv(gym.Env):

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    
    def __init__(
        self, 
        render=False, 
        max_steps=500, 
        success_threshold=0.001, 
        reward_type='normalized_progress', 
        reward_params=None,
        use_gui=False,
        max_velocity=2.0
    ):
        super().__init__()
        
        self.render_mode = "human" if render or use_gui else None
        self.max_steps = max_steps
        self.success_threshold = success_threshold
        self.reward_type = reward_type
        self.max_velocity = max_velocity
        
        # Create simulation
        self.sim = Simulation(num_agents=1, render=(render or use_gui))
        
        # Action space: [vx, vy, vz]
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(3,),
            dtype=np.float32
        )
        
        # Observation space: [current_x, current_y, current_z, goal_x, goal_y, goal_z]
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(6,),
            dtype=np.float32
        )
        
        # Workspace bounds
        self.workspace_low = np.array([-0.1871, -0.1706, 0.1700], dtype=np.float32)
        self.workspace_high = np.array([0.2532, 0.2197, 0.2897], dtype=np.float32)
        
        # Reward parameters
        self.reward_params = self._get_default_reward_params(reward_type)
        if reward_params is not None:
            self.reward_params.update(reward_params)
        
        # Episode tracking
        self.steps = 0
        self.goal_position = None
        self.initial_distance = None
        self.previous_distance = None
        self.previous_action = np.zeros(3, dtype=np.float32)
        self.achieved_stages = set()
        self.cumulative_reward = 0.0
        self.robot_id = None
    
    def reset(self, seed=None, options=None):
        """Reset environment."""
        super().reset(seed=seed)
        
        if seed is not None:
            np.random.seed(seed)
        
        # Generate random target
        if options and 'target_position' in options:
            self.goal_position = np.array(options['target_position'], dtype=np.float32)
        else:
            self.goal_position = np.random.uniform(
                self.workspace_low,
                self.workspace_high
            ).astype(np.float32)
        
        # Reset simulation
        state_dict = self.sim.reset(num_agents=1)
        
        # Extract robot ID and position
        self.robot_id = list(state_dict.keys())[0]
        current_pos = self._extract_position(state_dict)
        
        # Calculate initial distance
        self.initial_distance = float(np.linalg.norm(current_pos - self.goal_position))
        self.previous_distance = self.initial_distance
        
        # Create observation
        observation = np.concatenate([
            self._normalize_position(current_pos),
            self._normalize_position(self.goal_position)
        ], dtype=np.float32)
        
        # Reset episode tracking
        self.steps = 0
        self.previous_action = np.zeros(3, dtype=np.float32)
        self.achieved_stages = set()
        self.cumulative_reward = 0.0
        
        info = {
            'target_position': self.goal_position.copy(),
            'initial_distance': self.initial_distance,
            'distance': self.initial_distance,
            'distance_to_goal': self.initial_distance
        }
        
        return observation, info
    
    def step(self, action):
        """Execute one step."""
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, -1.0, 1.0)
        
        # Scale action to velocity
        velocity = action * self.max_velocity
        
        # Create full action with drop command
        full_action = [
            float(velocity[0]), 
            float(velocity[1]), 
            float(velocity[2]), 
            0.0  # No dropping
        ]
        
        # Run simulation
        state_dict = self.sim.run([full_action])
        
        # Extract current position
        current_pos = self._extract_position(state_dict)
        distance_to_goal = float(np.linalg.norm(current_pos - self.goal_position))
        
        # Calculate reward
        reward = self._calculate_reward(distance_to_goal, action)
        self.cumulative_reward += reward
        
        # Check termination
        terminated = bool(distance_to_goal < self.success_threshold)
        
        self.steps += 1
        truncated = bool(self.steps >= self.max_steps)
        
        # Update tracking
        self.previous_distance = distance_to_goal
        self.previous_action = action.copy()
        
        # Create observation
        observation = np.concatenate([
            self._normalize_position(current_pos),
            self._normalize_position(self.goal_position)
        ], dtype=np.float32)
        
        info = {
            'distance': distance_to_goal,
            'distance_to_goal': distance_to_goal,
            'success': terminated,
            'position': current_pos.copy(),
            'target': self.goal_position.copy(),
            'current_position': current_pos.tolist(),
            'goal_position': self.goal_position.tolist(),
            'step': self.steps,
            'cumulative_reward': self.cumulative_reward,
            'reward_type': self.reward_type
        }
        
        return observation, reward, terminated, truncated, info
    
    def _calculate_reward(self, distance_to_goal, action):
        """Route to appropriate reward function."""
        if self.reward_type == 'normalized_progress':
            return self._reward_normalized_progress(distance_to_goal, action)
        elif self.reward_type == 'exponential':
            return self._reward_exponential(distance_to_goal)
        elif self.reward_type == 'staged':
            return self._reward_staged(distance_to_goal)
        elif self.reward_type == 'dense_shaping':
            return self._reward_dense_shaping(distance_to_goal)
        elif self.reward_type == 'energy_efficient':
            return self._reward_energy_efficient(distance_to_goal, action)
        else:
            raise ValueError(f"Unknown reward type: {self.reward_type}")
    
    def _reward_normalized_progress(self, distance_to_goal, action):
        """Properly scaled for 1mm precision with progress tracking."""
        params = self.reward_params
        
        # Progress reward scaled by initial distance
        progress = (self.previous_distance - distance_to_goal) / self.initial_distance
        progress_reward = params['progress_scale'] * progress
        
        # Small time penalty
        time_penalty = -params['time_penalty']
        
        # Large success bonus with efficiency component
        if distance_to_goal < self.success_threshold:
            remaining_steps = self.max_steps - self.steps
            efficiency_bonus = remaining_steps * params['efficiency_multiplier']
            success_bonus = params['success_bonus'] + efficiency_bonus
        else:
            success_bonus = 0.0
        
        reward = progress_reward + time_penalty + success_bonus
        return float(reward)
    
    def _reward_exponential(self, distance_to_goal):
        """Safe exponential using hyperbolic tangent."""
        params = self.reward_params
        
        # Normalize distance to [0, 1]
        normalized_distance = distance_to_goal / self.initial_distance
        
        # Use tanh to prevent explosion
        distance_reward = -params['scale'] * (1.0 - np.tanh(params['alpha'] * (1.0 - normalized_distance)))
        
        time_penalty = -params['time_penalty']
        success_bonus = params['success_bonus'] if distance_to_goal < self.success_threshold else 0.0
        
        reward = distance_reward + time_penalty + success_bonus
        return float(reward)
    
    def _reward_staged(self, distance_to_goal):
        """Progressive milestones leading to 1mm."""
        params = self.reward_params
        reward = 0.0
        
        # Check for new threshold achievements
        for i, threshold in enumerate(params['thresholds']):
            if distance_to_goal < threshold and i not in self.achieved_stages:
                self.achieved_stages.add(i)
                reward += params['bonuses'][i]
        
        # Small time penalty
        reward -= params['time_penalty']
        
        return float(reward)
    
    def _reward_dense_shaping(self, distance_to_goal):
        """Potential-based reward shaping."""
        params = self.reward_params
        
        # Potential-based shaping
        current_potential = -params['shaping_scale'] * distance_to_goal
        previous_potential = -params['shaping_scale'] * self.previous_distance
        shaping_reward = current_potential - previous_potential
        
        time_penalty = -params['time_penalty']
        success_bonus = params['success_bonus'] if distance_to_goal < self.success_threshold else 0.0
        
        reward = shaping_reward + time_penalty + success_bonus
        return float(reward)
    
    def _reward_energy_efficient(self, distance_to_goal, action):
        """Progress-focused with small action penalties."""
        params = self.reward_params
        
        # Strong progress reward
        progress = (self.previous_distance - distance_to_goal) / self.initial_distance
        progress_reward = params['progress_scale'] * progress
        
        # Small action penalties
        action_magnitude = np.linalg.norm(action)
        action_penalty = -params['action_penalty'] * (action_magnitude ** 2)
        
        action_change = np.linalg.norm(action - self.previous_action)
        smoothness_penalty = -params['smoothness_penalty'] * (action_change ** 2)
        
        time_penalty = -params['time_penalty']
        success_bonus = params['success_bonus'] if distance_to_goal < self.success_threshold else 0.0
        
        reward = progress_reward + action_penalty + smoothness_penalty + time_penalty + success_bonus
        return float(reward)
    
    def _get_default_reward_params(self, reward_type):
        """Default parameters properly scaled for 1mm precision."""
        defaults = {
            'normalized_progress': {
                'progress_scale': 300,
                'time_penalty': 0.05,
                'efficiency_multiplier': 1.0,
                'success_bonus': 200
            },
            'exponential': {
                'alpha': 5.0,
                'scale': 100,
                'time_penalty': 0.05,
                'success_bonus': 250
            },
            'staged': {
                'thresholds': [0.05, 0.02, 0.005, 0.002, 0.001],
                'bonuses': [20, 40, 80, 120, 200],
                'time_penalty': 0.05
            },
            'dense_shaping': {
                'shaping_scale': 300,
                'time_penalty': 0.05,
                'success_bonus': 200
            },
            'energy_efficient': {
                'progress_scale': 300,
                'action_penalty': 0.02,
                'smoothness_penalty': 0.02,
                'time_penalty': 0.05,
                'success_bonus': 200
            }
        }
        return defaults.get(reward_type, {})
    
    def render(self):
        """Render environment."""
        pass
    
    def close(self):
        """Clean up."""
        try:
            self.sim.close()
        except:
            pass
    
    def _extract_position(self, state_dict):
        """Extract pipette position from state dictionary."""
        robotId = list(sorted(state_dict.keys()))[0]
        robot_state = state_dict.get(robotId, {})
        position = np.array(
            robot_state.get('pipette_position', [0.0, 0.0, 0.0]),
            dtype=np.float32
        )
        return position

    def _normalize_position(self, position):
        """Normalize position from workspace bounds to [-1, 1]."""
        normalized = 2.0 * (position - self.workspace_low) / (self.workspace_high - self.workspace_low) - 1.0
        return normalized.astype(np.float32)