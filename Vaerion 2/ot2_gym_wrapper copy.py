"""
FIXED Gymnasium Wrapper for OT-2 RL Controller
-----------------------------------------------
This version fixes the reward function for better learning.

Key fixes:
1. Better reward shaping with multiple distance scales
2. Improved precision rewards near target
3. Progress-based rewards
4. Better action penalties

Author: Zofia Pilitowska (240272)
"""

import gymnasium as gym
import numpy as np
import pybullet as p
from gymnasium import spaces
from typing import Tuple, Dict, Any, Optional


class OT2GymEnv(gym.Env):
    """
    FIXED Gymnasium environment for OT-2 pipette positioning control.
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
        super().__init__()
        
        # Environment parameters
        self.max_velocity = max_velocity
        self.success_threshold = success_threshold
        self.max_steps = max_steps
        self.reward_scale = reward_scale
        self.render_mode = render_mode
        self.use_gui = use_gui or (render_mode == "human")
        self.record_trajectory = record_trajectory
        
        # Pipette offset
        self.pipette_offset = np.array([0.073, 0.0895, 0.0895], dtype=np.float32)
        
        # Load work envelope
        self.work_envelope = self._load_work_envelope()
        
        # Action space: continuous 3D velocity commands
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(3,),
            dtype=np.float32
        )
        
        # Observation space: [pos(3), target(3)] - simplified like Aaron's
        obs_low = np.array([
            self.work_envelope['x_min'],
            self.work_envelope['y_min'],
            self.work_envelope['z_min'],
            self.work_envelope['x_min'],
            self.work_envelope['y_min'],
            self.work_envelope['z_min']
        ], dtype=np.float32)
        
        obs_high = np.array([
            self.work_envelope['x_max'],
            self.work_envelope['y_max'],
            self.work_envelope['z_max'],
            self.work_envelope['x_max'],
            self.work_envelope['y_max'],
            self.work_envelope['z_max']
        ], dtype=np.float32)
        
        self.observation_space = spaces.Box(
            low=obs_low,
            high=obs_high,
            dtype=np.float32
        )
        
        # Episode tracking (simplified)
        self.current_step = 0
        self.previous_distance = None  # For progress rewards
        self.cumulative_reward = 0.0
        self.episode_trajectory = []
        
        # PyBullet and robot setup
        self.physics_client = None
        self.robot_id = None
        self.target_position = None
        self.initial_position = None
        self.robot_base_position = None
        
        # Initialize simulation
        self._setup_simulation()
        
    def _load_work_envelope(self) -> Dict[str, float]:
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
        return envelope
        
    def _setup_simulation(self):
        """Initialize PyBullet simulation."""
        if self.use_gui:
            self.physics_client = p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        else:
            self.physics_client = p.connect(p.DIRECT)
        
        import pybullet_data
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -10)
        p.setTimeStep(1/240)
        
        if self.use_gui:
            p.resetDebugVisualizerCamera(
                cameraDistance=0.8,
                cameraYaw=90,
                cameraPitch=-35,
                cameraTargetPosition=[0, 0, 0.2]
            )
        
        self.base_plane_id = p.loadURDF("plane.urdf")
        self._create_ot2_robot()

    def _create_ot2_robot(self):
        """Load OT-2 robot."""
        robot_position = [0, 0, 0.03]
        robot_orientation = [0, 0, 0, 1]
        
        try:
            self.robot_id = p.loadURDF(
                "ot_2_simulation_v6.urdf",
                robot_position,
                robot_orientation,
                flags=p.URDF_USE_INERTIA_FROM_FILE
            )
        except Exception as e:
            raise FileNotFoundError(f"Could not load ot_2_simulation_v6.urdf: {e}")
        
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
        
        self.robot_base_position = np.array(start_position, dtype=np.float32)
        
        initial_joints = [0.04, 0.065, 0.11]
        for i in range(3):
            p.resetJointState(self.robot_id, i, targetValue=initial_joints[i])
        
        self.initial_position = self._get_pipette_position()

    def _get_pipette_position(self) -> np.ndarray:
        """Get current pipette tip position."""
        robot_position = list(p.getBasePositionAndOrientation(self.robot_id)[0])
        joint_states = p.getJointStates(self.robot_id, [0, 1, 2])
        
        robot_position[0] -= joint_states[0][0]
        robot_position[1] -= joint_states[1][0]
        robot_position[2] += joint_states[2][0]
        
        pipette_position = np.array([
            robot_position[0] + self.pipette_offset[0],
            robot_position[1] + self.pipette_offset[1],
            robot_position[2] + self.pipette_offset[2]
        ], dtype=np.float32)
        
        return pipette_position

    def _set_joint_velocities(self, action: np.ndarray):
        """Set joint velocities based on action."""
        velocity = action * self.max_velocity
        
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
        """Generate random target position."""
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
        """Compute Euclidean distance."""
        return float(np.linalg.norm(pos1 - pos2))
    
    def _is_within_bounds(self, position: np.ndarray) -> bool:
        """Check if position is within work envelope."""
        x, y, z = position
        return (self.work_envelope['x_min'] <= x <= self.work_envelope['x_max'] and
                self.work_envelope['y_min'] <= y <= self.work_envelope['y_max'] and
                self.work_envelope['z_min'] <= z <= self.work_envelope['z_max'])
    
    def _compute_reward(self, current_pos, target_pos, action, distance, previous_distance):
        """
        IMPROVED reward function with better shaping.
        
        Key improvements:
        1. Multi-scale distance rewards (coarse + fine)
        2. Progress-based rewards
        3. Stronger precision incentives near target
        4. Better action penalties
        """
        # 1. COARSE DISTANCE REWARD (linear, always active)
        # Scale: -1000 at 1m, -50 at 0.05m, -5 at 0.005m
        coarse_reward = -distance * 1000.0
        
        # 2. FINE DISTANCE REWARD (quadratic, emphasizes precision)
        # Becomes dominant at <10cm
        fine_reward = -10000.0 * distance**2
        
        # 3. PRECISION REWARD (exponential, very strong near target)
        # Kicks in at <5cm, maximum at target
        if distance < 0.05:  # Within 5cm
            precision_reward = 50.0 * np.exp(-100.0 * distance)
        else:
            precision_reward = 0.0
        
        # 4. PROGRESS REWARD (encourage getting closer)
        if previous_distance is not None:
            progress = previous_distance - distance
            # Reward for moving closer, penalty for moving away
            progress_reward = progress * 500.0
        else:
            progress_reward = 0.0
        
        # 5. GOAL REWARD (sparse, big bonus for success)
        if distance < self.success_threshold:
            goal_reward = 1000.0
        elif distance < self.success_threshold * 2:  # Close!
            goal_reward = 500.0
        elif distance < self.success_threshold * 4:  # Getting there
            goal_reward = 250.0
        else:
            goal_reward = 0.0
        
        # 6. ACTION PENALTY (encourage smooth, efficient movements)
        action_magnitude = np.linalg.norm(action)
        action_penalty = -0.5 * action_magnitude 
        
        # 7. EFFICIENCY BONUS (reward quick success)
        if distance < self.success_threshold:
            steps_remaining = self.max_steps - self.current_step
            efficiency_bonus = steps_remaining * 0.1
        else:
            efficiency_bonus = 0.0
        
        # COMBINE A REWARDS
        total_reward = (
            coarse_reward +
            fine_reward +
            precision_reward +
            progress_reward +
            goal_reward +
            action_penalty +
            efficiency_bonus
        ) * self.reward_scale
        
        return total_reward

    def reset(
        self, 
        seed: Optional[int] = None, 
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment."""
        super().reset(seed=seed)
        
        self.current_step = 0
        self.previous_distance = None  # Reset previous distance
        self.cumulative_reward = 0.0
        self.episode_trajectory = []
        
        # Reset joints
        initial_joints = [0.04, 0.065, 0.11]
        for i in range(3):
            p.resetJointState(self.robot_id, i, targetValue=initial_joints[i])
            p.setJointMotorControl2(
                self.robot_id, i, 
                p.VELOCITY_CONTROL, 
                targetVelocity=0, 
                force=0
            )
        
        # Generate target
        if options and 'target_position' in options:
            self.target_position = np.array(options['target_position'], dtype=np.float32)
        else:
            self.target_position = self._generate_random_target()
        
        self._create_target_marker()
        
        current_pos = self._get_pipette_position()
        distance = self._compute_distance(current_pos, self.target_position)
        self.previous_distance = distance  # Initialize
        
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
            self.target_position
        ]).astype(np.float32)
        
        info = {
            'target_position': self.target_position.copy(),
            'initial_distance': distance,
            'distance': distance  # Add for compatibility
        }
        
        return observation, info
    
    def _create_target_marker(self):
        """Create visual marker for target."""
        if hasattr(self, 'target_marker_id'):
            try:
                p.removeBody(self.target_marker_id)
            except:
                pass
        
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
        """Execute one step."""
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        
        # Apply control
        self._set_joint_velocities(action)
        
        # Step simulation
        for _ in range(8):
            p.stepSimulation()
        
        # Get new state
        current_pos = self._get_pipette_position()
        distance = self._compute_distance(current_pos, self.target_position)
        
        # Compute reward (now includes previous_distance)
        reward = self._compute_reward(
            current_pos, 
            self.target_position, 
            action, 
            distance, 
            self.previous_distance
        )
        self.cumulative_reward += reward
        
        # Update previous distance for next step
        self.previous_distance = distance
        
        # Check termination
        terminated = False
        success = False
        
        if distance < self.success_threshold:
            terminated = True
            success = True
        
        if not self._is_within_bounds(current_pos):
            terminated = True
            reward -= 500.0  # Large penalty
        
        self.current_step += 1
        truncated = self.current_step >= self.max_steps
        
        if self.record_trajectory:
            self.episode_trajectory.append({
                'position': current_pos.copy(),
                'target': self.target_position.copy(),
                'action': action.copy(),
                'distance': distance,
                'reward': reward
            })
        
        # Simplified observation (6D like Aaron's)
        observation = np.concatenate([
            current_pos,
            self.target_position
        ]).astype(np.float32)
        
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
        """Render environment."""
        if self.render_mode == "rgb_array":
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
        """Get recorded trajectory."""
        return self.episode_trajectory
    
    def close(self):
        """Clean up."""
        if self.physics_client is not None:
            p.disconnect(self.physics_client)
            self.physics_client = None