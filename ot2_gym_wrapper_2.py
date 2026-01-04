"""
Gymnasium Wrapper for OT-2 RL Controller
"""

import gymnasium as gym
import numpy as np
import pybullet as p
from gymnasium import spaces
from typing import Tuple, Dict, Any, Optional
from sim_class import Simulation

# Workspace bounds
X_MIN, X_MAX = -0.260, 0.134
Y_MIN, Y_MAX = -0.260, 0.130
Z_MIN, Z_MAX = 0.030, 0.200


class OT2GymEnv(gym.Env):
    """
    Gymnasium environment for OT-2 pipette positioning control.
    Uses sim_class.py as simulation backend.
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    
    def __init__(
        self,
        render_mode: Optional[str] = None,
        max_velocity: float = 0.05,
        success_threshold: float = 0.005,
        max_steps: int = 500,
        use_gui: bool = False,
        record_trajectory: bool = False,
        num_agents: int = 1
    ):
        super().__init__()
        
        # Environment parameters
        self.max_velocity = max_velocity
        self.success_threshold = success_threshold
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.use_gui = use_gui or (render_mode == "human")
        self.record_trajectory = record_trajectory
        
        # Create simulation using sim_class
        self.sim = Simulation(num_agents=num_agents, render=self.use_gui)
        
        # Action space: [vx, vy, vz, drop]
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(4,),
            dtype=np.float32
        )
        
        # Observation space: [current_x, current_y, current_z, goal_x, goal_y, goal_z]
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(6,),
            dtype=np.float32
        )
        
        # Episode tracking
        self.current_step = 0
        self.cumulative_reward = 0.0
        self.episode_trajectory = []
        self.prev_position = None
        
        # Robot state
        self.target_position = None
        self.pipette_position = None
        self.robot_id = None
        
    def _normalize_position(self, position) -> np.ndarray:
        """Normalize position from workspace bounds to [-1, 1]."""
        if not isinstance(position, np.ndarray):
            position = np.array(position, dtype=np.float32)
        
        workspace_low = np.array([X_MIN, Y_MIN, Z_MIN], dtype=np.float32)
        workspace_high = np.array([X_MAX, Y_MAX, Z_MAX], dtype=np.float32)
        
        normalized = 2.0 * (position - workspace_low) / (workspace_high - workspace_low) - 1.0
        return normalized.astype(np.float32)
    
    def _compute_distance(self, pos1: np.ndarray, pos2: np.ndarray) -> float:
        """Compute Euclidean distance."""
        return float(np.linalg.norm(pos1 - pos2))
    
    def _compute_reward(self, distance_to_goal: float) -> float:
        distance_factor = 10
        
        # Distance penalty
        reward = -distance_to_goal * distance_factor
        
        # Progress reward
        if self.prev_position is not None:
            prev_distance = np.linalg.norm(self.prev_position - self.target_position)
            
            if distance_to_goal > prev_distance:
                reward_moving_to_goal = -1
            else:
                reward_moving_to_goal = 1
            
            reward += reward_moving_to_goal
        
        return reward

    def reset(
        self, 
        seed: Optional[int] = None, 
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment."""
        super().reset(seed=seed)
        
        self.current_step = 0
        self.cumulative_reward = 0.0
        self.episode_trajectory = []
        self.prev_position = None
        
        # Reset simulation
        observation = self.sim.reset(num_agents=1)
        
        # Extract robot ID
        self.robot_id = list(observation.keys())[0]
        
        # Generate random target position
        if options and 'target_position' in options:
            self.target_position = np.array(options['target_position'], dtype=np.float32)
        else:
            if seed is not None:
                np.random.seed(seed)
            self.target_position = np.random.uniform(
                low=[X_MIN, Y_MIN, Z_MIN],
                high=[X_MAX, Y_MAX, Z_MAX]
            ).astype(np.float32)
        
        # Get initial pipette position
        self.pipette_position = np.array(
            self.sim.pipette_positions[self.robot_id],
            dtype=np.float32
        )
        
        # Set previous position for progress tracking
        self.prev_position = self.pipette_position.copy()
        
        # Calculate initial distance
        distance = self._compute_distance(self.pipette_position, self.target_position)
        
        if self.record_trajectory:
            self.episode_trajectory.append({
                'position': self.pipette_position.copy(),
                'target': self.target_position.copy(),
                'action': np.zeros(4),
                'distance': distance,
                'reward': 0.0
            })
        
        # Create observation
        observation = np.concatenate([
            self._normalize_position(self.pipette_position),
            self._normalize_position(self.target_position)
        ]).astype(np.float32)
        
        info = {
            'target_position': self.target_position.copy(),
            'initial_distance': distance,
            'distance': distance,
            'distance_to_goal': distance  # For compatibility with both callback styles
        }
        
        return observation, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one step."""
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        
        # Scale velocity actions (first 3 dimensions)
        scaled_action = action.copy()
        scaled_action[:3] = action[:3] * self.max_velocity
        
        
        try:
            observation = self.sim.run([scaled_action.tolist()])
        except:
            observation = self.sim.run(scaled_action.tolist())
        
        # Extract robot state
        robot_state = observation.get(self.robot_id, {})
        self.pipette_position = np.array(
            robot_state.get('pipette_position', [0.0, 0.0, 0.0]),
            dtype=np.float32
        )
        
        # Calculate distance to goal
        distance = self._compute_distance(self.pipette_position, self.target_position)
        
        # Compute reward
        reward = self._compute_reward(distance)
        self.cumulative_reward += reward
        
        # Update previous position for next step
        self.prev_position = self.pipette_position.copy()
        
        # Check termination
        terminated = False
        success = False
        
        if distance < self.success_threshold:
            terminated = True
            success = True
        
        self.current_step += 1
        truncated = self.current_step >= self.max_steps
        
        if self.record_trajectory:
            self.episode_trajectory.append({
                'position': self.pipette_position.copy(),
                'target': self.target_position.copy(),
                'action': action.copy(),
                'distance': distance,
                'reward': reward
            })
        
        # Create observation
        observation = np.concatenate([
            self._normalize_position(self.pipette_position),
            self._normalize_position(self.target_position)
        ]).astype(np.float32)
        
        info = {
            'distance': distance,
            'distance_to_goal': distance,  
            'success': success,
            'position': self.pipette_position.copy(),
            'target': self.target_position.copy(),
            'current_position': self.pipette_position.tolist(),
            'goal_position': self.target_position.tolist(),
            'step': self.current_step,
            'cumulative_reward': self.cumulative_reward,
            'trajectory': self.episode_trajectory if self.record_trajectory else []
        }
        
        return observation, reward, terminated, truncated, info
    
    def render(self):
        """Render environment."""
        if self.render_mode == "human":
            if self.use_gui:
                p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
        return None
    
    def get_trajectory(self) -> list:
        """Get recorded trajectory."""
        return self.episode_trajectory
    
    def get_goal_position(self) -> tuple:
        """Get current goal position."""
        return tuple(self.target_position.tolist())
    
    def close(self):
        """Clean up."""
        try:
            self.sim.close()
        except:
            pass