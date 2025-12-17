"""
Gymnasium Wrapper for Opentrons OT-2 Reinforcement Learning Controller
Adapted to use actual OT-2 robot with measured work envelope

This module implements a Gymnasium-compatible environment for training RL agents
to control the OT-2 pipette positioning system using your existing PyBullet simulation.

"""

import gymnasium as gym
import numpy as np
import pybullet as p
from gymnasium import spaces
from typing import Tuple, Dict, Any, Optional
import json
import os


class OT2GymEnv(gym.Env):
    """
    Gymnasium environment for OT-2 pipette positioning control.
    
    Uses actual OT-2 simulation with ot_2_simulation_v6.urdf robot.
    Work envelope measured from actual robot joint limits.
    
    Observation Space (10D):
        - Pipette tip position: [x, y, z] in meters
        - Target position: [x_t, y_t, z_t] in meters
        - Distance to target: scalar in meters
        - Previous action: [vx, vy, vz] normalized
    
    Action Space (3D):
        - Continuous velocity commands: [vx, vy, vz]
        - Range: [-1, 1] (normalized)
        - Maps to joint velocities for the 3 prismatic joints
    
    Reward Function:
        - Dense: -distance_to_target (encourages moving closer)
        - Sparse: +100 for reaching goal (distance < threshold)
        - Penalty: -0.01 * ||action||^2 (energy efficiency)
    
    Termination Conditions:
        - Success: distance < success_threshold (default 5mm)
        - Failure: pipette exits work envelope
        - Truncation: max_steps reached (default 500)
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    
    def __init__(
        self,
        render_mode: Optional[str] = None,
        max_velocity: float = 0.05,
        success_threshold: float = 0.005,
        max_steps: int = 500,
        reward_scale: float = 1.0,
        use_gui: bool = False,
        record_trajectory: bool = False
    ):
        """
        Initialize the OT-2 Gymnasium environment.
        
        Args:
            render_mode: Visualization mode ('human' or 'rgb_array')
            max_velocity: Maximum velocity for pipette movement (m/s)
            success_threshold: Distance threshold for success (m)
            max_steps: Maximum steps per episode
            reward_scale: Scaling factor for rewards
            use_gui: Whether to use PyBullet GUI (slower but visual)
            record_trajectory: Whether to record trajectory for visualization
        """
        super().__init__()
        
        # Environment parameters
        self.max_velocity = max_velocity
        self.success_threshold = success_threshold
        self.max_steps = max_steps
        self.reward_scale = reward_scale
        self.render_mode = render_mode
        self.use_gui = use_gui or (render_mode == "human")
        self.record_trajectory = record_trajectory
        
        # Pipette offset from your simulation
        self.pipette_offset = np.array([0.073, 0.0895, 0.0895], dtype=np.float32)
        
        # Load actual work envelope from measurements
        self.work_envelope = self._load_work_envelope()
        
        # Action space: continuous 3D velocity commands
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(3,),
            dtype=np.float32
        )
        
        # Observation space: [pos(3), target(3), distance(1), prev_action(3)]
        obs_low = np.array([
            self.work_envelope['x_min'],
            self.work_envelope['y_min'],
            self.work_envelope['z_min'],
            self.work_envelope['x_min'],
            self.work_envelope['y_min'],
            self.work_envelope['z_min'],
            0.0,  # distance
            -1.0, -1.0, -1.0  # previous action
        ], dtype=np.float32)
        
        obs_high = np.array([
            self.work_envelope['x_max'],
            self.work_envelope['y_max'],
            self.work_envelope['z_max'],
            self.work_envelope['x_max'],
            self.work_envelope['y_max'],
            self.work_envelope['z_max'],
            1.0,  # max normalized distance
            1.0, 1.0, 1.0
        ], dtype=np.float32)
        
        self.observation_space = spaces.Box(
            low=obs_low,
            high=obs_high,
            dtype=np.float32
        )
        
        # Episode tracking
        self.current_step = 0
        self.previous_action = np.zeros(3, dtype=np.float32)
        self.cumulative_reward = 0.0
        self.episode_trajectory = []
        
        # PyBullet and robot setup
        self.physics_client = None
        self.robot_id = None
        self.specimen_id = None
        self.target_position = None
        self.initial_position = None
        self.robot_base_position = None
        
        # Initialize simulation
        self._setup_simulation()
        
    def _load_work_envelope(self) -> Dict[str, float]:

        
        # Measured values
        envelope = {
            'x_min': -0.1878,
            'x_max': 0.2547,
            'y_min': -0.1722,
            'y_max': 0.2199,
            'z_min': 0.1193,
            'z_max': 0.2908,
            'x_range': 0.4425,
            'y_range': 0.3921,
            'z_range': 0.1715
        }
        
        print(f"Work envelope:")
        print(f"  X: [{envelope['x_min']:.4f}, {envelope['x_max']:.4f}] m")
        print(f"  Y: [{envelope['y_min']:.4f}, {envelope['y_max']:.4f}] m")
        print(f"  Z: [{envelope['z_min']:.4f}, {envelope['z_max']:.4f}] m")
        
        return envelope
        
    def _setup_simulation(self):
        """Initialize PyBullet simulation with your OT-2 robot."""
        # Connect to PyBullet
        if self.use_gui:
            self.physics_client = p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        else:
            self.physics_client = p.connect(p.DIRECT)
        
        # Configure simulation (matching sim_class.py)
        import pybullet_data
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -10)
        p.setTimeStep(1/240)  # 240 Hz simulation
        
        # Set camera for better view
        if self.use_gui:
            p.resetDebugVisualizerCamera(
                cameraDistance=0.8,
                cameraYaw=90,
                cameraPitch=-35,
                cameraTargetPosition=[0, 0, 0.2]
            )
        
        # Load base plane
        self.base_plane_id = p.loadURDF("plane.urdf")
        
        # Load your OT-2 robot
        self._create_ot2_robot()
        


    def _create_ot2_robot(self):
        """
        Actual OT-2 robot from  ot_2_simulation_v6.urdf file.
        """
        # Robot position at origin
        robot_position = [0, 0, 0.03]
        robot_orientation = [0, 0, 0, 1]
        
        # Load the OT-2 URDF
        try:
            self.robot_id = p.loadURDF(
                "ot_2_simulation_v6.urdf",
                robot_position,
                robot_orientation,
                flags=p.URDF_USE_INERTIA_FROM_FILE
            )
        except Exception as e:
            raise FileNotFoundError(
                f"Could not load ot_2_simulation_v6.urdf. "
                f"Make sure it's in the current directory. Error: {e}"
            )
        
        # Fix the robot base in space
        start_position, start_orientation = p.getBasePositionAndOrientation(self.robot_id)
        p.createConstraint(
            parentBodyUniqueId=self.robot_id,
            parentLinkIndex=-1,
            childBodyUniqueId=-1,
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=[0, 0, 0],
            childFramePosition=start_position,
            childFrameOrientation=start_orientation
        )
        
        # Store base position
        self.robot_base_position = np.array(start_position, dtype=np.float32)
        
        # Set initial joint positions
        initial_joints = [0.04, 0.065, 0.11]  # Center of measured ranges
        for i in range(3):
            p.resetJointState(self.robot_id, i, targetValue=initial_joints[i])
        
        # Store initial pipette position
        self.initial_position = self._get_pipette_position()
        
        print(f"Robot loaded. Initial pipette position: {self.initial_position}")
        


    def _get_pipette_position(self) -> np.ndarray:
        """
        Get current pipette tip position using your calculation method.
        

        Returns:
            3D position [x, y, z] of pipette tip
        """
        # Get robot base position
        robot_position = list(p.getBasePositionAndOrientation(self.robot_id)[0])
        
        # Get joint states for the 3 prismatic joints
        joint_states = p.getJointStates(self.robot_id, [0, 1, 2])
        
        
        robot_position[0] -= joint_states[0][0]  # X joint (negative)
        robot_position[1] -= joint_states[1][0]  # Y joint (negative)
        robot_position[2] += joint_states[2][0]  # Z joint (positive)
        
        # Add pipette offset
        pipette_position = np.array([
            robot_position[0] + self.pipette_offset[0],
            robot_position[1] + self.pipette_offset[1],
            robot_position[2] + self.pipette_offset[2]
        ], dtype=np.float32)
        
        return pipette_position
    


    def _set_joint_velocities(self, action: np.ndarray):
        """
        Set joint velocities based on action.

        
        Args:
            action: Normalized velocity commands [vx, vy, vz] in range [-1, 1]
        """
        # Scale action to actual velocity
        velocity = action * self.max_velocity
        
        
        # Joint 0 (X): negative velocity
        # Joint 1 (Y): negative velocity  
        # Joint 2 (Z): positive velocity
        p.setJointMotorControl2(
            self.robot_id, 0, 
            p.VELOCITY_CONTROL, 
            targetVelocity=-velocity[0], 
            force=500
        )
        p.setJointMotorControl2(
            self.robot_id, 1, 
            p.VELOCITY_CONTROL, 
            targetVelocity=-velocity[1], 
            force=500
        )
        p.setJointMotorControl2(
            self.robot_id, 2, 
            p.VELOCITY_CONTROL, 
            targetVelocity=velocity[2], 
            force=800
        )



    
    def _generate_random_target(self) -> np.ndarray:
        """
        Generate a random target position within the work envelope.
        
        Returns:
            3D target position [x, y, z]
        """
        # small margin to avoid boundary issues (5mm margin)
        margin = 0.005
        
        target = np.array([
            np.random.uniform(
                self.work_envelope['x_min'] + margin,
                self.work_envelope['x_max'] - margin
            ),
            np.random.uniform(
                self.work_envelope['y_min'] + margin,
                self.work_envelope['y_max'] - margin
            ),
            np.random.uniform(
                self.work_envelope['z_min'] + margin,
                self.work_envelope['z_max'] - margin
            )
        ], dtype=np.float32)
        
        return target
    


    def _compute_distance(self, pos1: np.ndarray, pos2: np.ndarray) -> float:
        """
        Compute Euclidean distance between two 3D points.
        
        Args:
            pos1: First position
            pos2: Second position
            
        Returns:
            Euclidean distance
        """
        return float(np.linalg.norm(pos1 - pos2))
    



    def _is_within_bounds(self, position: np.ndarray) -> bool:
        """
        Check if position is within work envelope.
        
        Args:
            position: 3D position to check
            
        Returns:
            True if within bounds, False otherwise
        """
        x, y, z = position
        return (self.work_envelope['x_min'] <= x <= self.work_envelope['x_max'] and
                self.work_envelope['y_min'] <= y <= self.work_envelope['y_max'] and
                self.work_envelope['z_min'] <= z <= self.work_envelope['z_max'])
    



    def _compute_reward(self, current_pos, target_pos, action, distance):
        # Dense reward: encourage moving closer, stronger near target
        dense_reward = -distance  # coarse reward
        precision_reward = np.exp(-10.0 * distance**2)  # strong reward near target
        
        # Sparse goal reward
        sparse_reward = 100.0 if distance < self.success_threshold else 0.0
        
        # Action penalty for smoothness
        action_penalty = -0.005 * np.sum(action**2)  # reduce weight slightly
        
        # Combine
        total_reward = (dense_reward + precision_reward + sparse_reward + action_penalty) * self.reward_scale
        
        return total_reward

    


    def reset(
        self, 
        seed: Optional[int] = None, 
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment to initial state.
        
        Args:
            seed: Random seed for reproducibility
            options: Additional options (e.g., specific target position)
            
        Returns:
            observation: Initial observation
            info: Additional information dictionary
        """
        super().reset(seed=seed)
        
        # Reset episode tracking
        self.current_step = 0
        self.previous_action = np.zeros(3, dtype=np.float32)
        self.cumulative_reward = 0.0
        self.episode_trajectory = []
        
        # Reset joints to initial position (centered)
        initial_joints = [0.04, 0.065, 0.11]
        for i in range(3):
            p.resetJointState(self.robot_id, i, targetValue=initial_joints[i])
            # Also reset velocities
            p.setJointMotorControl2(
                self.robot_id, i, 
                p.VELOCITY_CONTROL, 
                targetVelocity=0, 
                force=0
            )
        
        # Generate new target or use provided target
        if options and 'target_position' in options:
            self.target_position = np.array(options['target_position'], dtype=np.float32)
        else:
            self.target_position = self._generate_random_target()
        
        # Create visual marker for target
        self._create_target_marker()
        
        # Get initial observation
        current_pos = self._get_pipette_position()
        distance = self._compute_distance(current_pos, self.target_position)
        
        # Record trajectory if enabled
        if self.record_trajectory:
            self.episode_trajectory.append({
                'position': current_pos.copy(),
                'target': self.target_position.copy(),
                'action': np.zeros(3),
                'distance': distance,
                'reward': 0.0
            })
        
        observation = np.concatenate([
            current_pos,
            self.target_position,
            [distance],
            self.previous_action
        ]).astype(np.float32)
        
        info = {
            'target_position': self.target_position.copy(),
            'initial_distance': distance
        }
        
        return observation, info
    

    def _create_target_marker(self):
        """Create visual marker for target position."""
        # Remove old marker if exists
        if hasattr(self, 'target_marker_id'):
            try:
                p.removeBody(self.target_marker_id)
            except:
                pass
        
        # Create sphere marker at target
        marker_visual = p.createVisualShape(
            p.GEOM_SPHERE,
            radius=0.01,
            rgbaColor=[1, 0, 0, 0.5]
        )
        
        self.target_marker_id = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=marker_visual,
            basePosition=self.target_position.tolist()
        )
    
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.
        
        Args:
            action: Normalized velocity command [vx, vy, vz] in range [-1, 1]
            
        Returns:
            observation: Current observation
            reward: Reward for this step
            terminated: Whether episode ended (success or failure)
            truncated: Whether episode was truncated (max steps)
            info: Additional information dictionary
        """
        # Clip action to valid range
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        
        # Apply velocity control to joints
        self._set_joint_velocities(action)
        
        # Step simulation multiple times for smooth control (matching 240Hz physics)
        # With 30Hz control rate, we step 8 times per control step
        for _ in range(8):
            p.stepSimulation()
        
        # Get new state
        current_pos = self._get_pipette_position()
        distance = self._compute_distance(current_pos, self.target_position)
        
        # Compute reward
        reward = self._compute_reward(current_pos, self.target_position, action, distance)
        self.cumulative_reward += reward
        
        # Check termination conditions
        terminated = False
        success = False
        
        # Success: reached target
        if distance < self.success_threshold:
            terminated = True
            success = True
        
        # Failure: out of bounds
        if not self._is_within_bounds(current_pos):
            terminated = True
            reward -= 50.0  # Large penalty for leaving workspace
        
        # Truncation: max steps reached
        self.current_step += 1
        truncated = self.current_step >= self.max_steps
        
        # Record trajectory if enabled
        if self.record_trajectory:
            self.episode_trajectory.append({
                'position': current_pos.copy(),
                'target': self.target_position.copy(),
                'action': action.copy(),
                'distance': distance,
                'reward': reward
            })
        
        # Build observation
        observation = np.concatenate([
            current_pos,
            self.target_position,
            [distance],
            action
        ]).astype(np.float32)
        
        # Update previous action
        self.previous_action = action.copy()
        
        # Build info dict
        info = {
            'distance': distance,
            'success': success,
            'position': current_pos.copy(),
            'target': self.target_position.copy(),
            'step': self.current_step,
            'cumulative_reward': self.cumulative_reward,
            'trajectory': self.episode_trajectory if self.record_trajectory else []
        }
        
        return observation, reward, terminated, truncated, info
    
    def render(self):
        """
        Render the environment.
        
        For 'human' mode, this is handled automatically by PyBullet GUI.
        For 'rgb_array' mode, this returns an image array.
        """
        if self.render_mode == "rgb_array":
            # Get camera image from PyBullet
            view_matrix = p.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=[0, 0, 0.2],
                distance=0.8,
                yaw=90,
                pitch=-35,
                roll=0,
                upAxisIndex=2
            )
            proj_matrix = p.computeProjectionMatrixFOV(
                fov=60,
                aspect=1.0,
                nearVal=0.1,
                farVal=3.0
            )
            
            (_, _, px, _, _) = p.getCameraImage(
                width=640,
                height=480,
                viewMatrix=view_matrix,
                projectionMatrix=proj_matrix,
                renderer=p.ER_BULLET_HARDWARE_OPENGL
            )
            
            rgb_array = np.array(px, dtype=np.uint8)
            rgb_array = np.reshape(rgb_array, (480, 640, 4))[:, :, :3]
            return rgb_array
        
        return None
    
    def get_trajectory(self) -> list:
        """
        Get the recorded trajectory from the current episode.
        
        Returns:
            List of trajectory points with positions, actions, and rewards
        """
        return self.episode_trajectory
    
    def close(self):
        """Clean up and close the environment."""
        if self.physics_client is not None:
            p.disconnect(self.physics_client)
            self.physics_client = None


# Example usage and testing
if __name__ == "__main__":
    # Create environment
    env = OT2GymEnv(success_threshold=0.002, max_steps=500)
    
    print("\nEnvironment created ")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    
    # Run one episode
    obs, info = env.reset()
    print(f"\nInitial observation shape: {obs.shape}")
    print(f"Initial pipette position: {obs[:3]}")
    print(f"Target position: {info['target_position']}")
    print(f"Initial distance: {info['initial_distance']:.4f}m ({info['initial_distance']*1000:.2f}mm)")
    
    done = False
    step = 0
    while not done:
        # Random action
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        step += 1
        
        if step % 50 == 0:
            print(f"Step {step}: Distance = {info['distance']*1000:.2f}mm, Reward = {reward:.2f}")
    
    print(f"\nEpisode finished after {step} steps")
    print(f"Success: {info['success']}")
    print(f"Final distance: {info['distance']*1000:.2f}mm")
    print(f"Final pipette position: {info['position']}")
    
    env.close()