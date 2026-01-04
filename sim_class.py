import pybullet as p
import time
import pybullet_data
import math
import logging
import os
import random
import numpy as np

# logging.basicConfig(level=logging.INFO)

class Simulation:
    def __init__(self, num_agents, render=True, rgb_array=False):
        self.render = render
        self.rgb_array = rgb_array

        mode = p.GUI if render else p.DIRECT
        self.physicsClient = p.connect(mode)

        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -10)

        # -------- TEXTURES --------
        texture_list = os.listdir("textures")
        random_texture = random.choice(texture_list[:-1])
        random_texture_index = texture_list.index(random_texture)
        self.plate_image_path = f'textures/_plates/{os.listdir("textures/_plates")[random_texture_index]}'
        self.textureId = p.loadTexture(f'textures/{random_texture}')

        # -------- CAMERA --------
        cameraDistance = 1.1 * (math.ceil(num_agents ** 0.3))
        cameraYaw = 90
        cameraPitch = -35
        cameraTargetPosition = [-0.2, -(math.ceil(num_agents ** 0.5) / 2) + 0.5, 0.1]
        p.resetDebugVisualizerCamera(cameraDistance, cameraYaw, cameraPitch, cameraTargetPosition)

        self.baseplaneId = p.loadURDF("plane.urdf")

        # -------- OT-2 PARAMETERS --------
        self.pipette_offset = np.array([0.073, 0.0895, 0.0895], dtype=np.float32)

        # Workspace (Aaron-style safety clamp)
        self.workspace_low = np.array([-0.19, -0.18, 0.12], dtype=np.float32)
        self.workspace_high = np.array([0.26, 0.22, 0.29], dtype=np.float32)

        # Max Cartesian step per sim step
        self.max_cartesian_step = 0.02  # meters

        self.pipette_positions = {}
        self.sphereIds = []
        self.droplet_positions = {}

        self.create_robots(num_agents)

    # =========================================================================
    # ROBOT CREATION
    # =========================================================================

    def create_robots(self, num_agents):
        spacing = 1
        grid_size = math.ceil(num_agents ** 0.5)

        self.robotIds = []
        self.specimenIds = []

        agent_count = 0
        for i in range(grid_size):
            for j in range(grid_size):
                if agent_count >= num_agents:
                    break

                position = [-spacing * i, -spacing * j, 0.03]
                robotId = p.loadURDF(
                    "ot_2_simulation_v6.urdf",
                    position,
                    [0, 0, 0, 1],
                    flags=p.URDF_USE_INERTIA_FROM_FILE
                )

                start_pos, start_ori = p.getBasePositionAndOrientation(robotId)
                p.createConstraint(
                    robotId, -1, -1, -1,
                    p.JOINT_FIXED,
                    [0, 0, 0],
                    [0, 0, 0],
                    start_pos,
                    start_ori
                )

                offset = [0.1827, 0.137, 0.057]
                spec_pos = [position[0] + offset[0], position[1] + offset[1], position[2] + offset[2]]
                planeId = p.loadURDF("custom.urdf", spec_pos, p.getQuaternionFromEuler([0, 0, -math.pi / 2]))

                p.setCollisionFilterPair(robotId, planeId, -1, -1, enableCollision=0)
                p.changeVisualShape(planeId, -1, textureUniqueId=self.textureId)

                self.robotIds.append(robotId)
                self.specimenIds.append(planeId)

                self.pipette_positions[f'robotId_{robotId}'] = self.get_pipette_position(robotId)
                agent_count += 1

    # =========================================================================
    # STATE HELPERS
    # =========================================================================
    def reset(self, num_agents=None):
        """Reset robot joints to initial positions."""
        # Reset all robots to initial joint positions
        initial_joints = [0.04, 0.065, 0.11]
        
        for robotId in self.robotIds:
            for i in range(3):
                p.resetJointState(robotId, i, targetValue=initial_joints[i])
                p.setJointMotorControl2(
                    robotId, i,
                    p.VELOCITY_CONTROL,
                    targetVelocity=0,
                    force=0
                )
            
            # Update pipette position after reset
            self.pipette_positions[f'robotId_{robotId}'] = self.get_pipette_position(robotId)
        
        # Clear any existing droplets
        for sphereId in self.sphereIds[:]:
            try:
                p.removeBody(sphereId)
            except:
                pass
        self.sphereIds = []
        self.droplet_positions = {}
        
        return self.get_states()

    def get_pipette_position(self, robotId):
        base_pos = np.array(p.getBasePositionAndOrientation(robotId)[0], dtype=np.float32)
        joints = p.getJointStates(robotId, [0, 1, 2])

        base_pos[0] -= joints[0][0]
        base_pos[1] -= joints[1][0]
        base_pos[2] += joints[2][0]

        return (base_pos + self.pipette_offset).tolist()

    # =========================================================================
    # AARON-STYLE CARTESIAN CONTROL (KEY CHANGE)
    # =========================================================================

    def apply_actions(self, actions):
        """
        actions[i] = [vx, vy, vz, drop]
        Cartesian velocity control (Aaron-style)
        """
        for i, robotId in enumerate(self.robotIds):
            action = np.clip(actions[i][:3], -1.0, 1.0)
            drop_cmd = actions[i][3]

            current_pos = np.array(self.get_pipette_position(robotId))
            delta = action * self.max_cartesian_step
            target_pos = current_pos + delta

            # Clamp to workspace
            target_pos = np.clip(target_pos, self.workspace_low, self.workspace_high)

            base_pos = np.array(p.getBasePositionAndOrientation(robotId)[0])

            joint_targets = [
                base_pos[0] + self.pipette_offset[0] - target_pos[0],
                base_pos[1] + self.pipette_offset[1] - target_pos[1],
                target_pos[2] - base_pos[2] - self.pipette_offset[2]
            ]

            for j in range(3):
                p.setJointMotorControl2(
                    robotId,
                    j,
                    p.POSITION_CONTROL,
                    targetPosition=joint_targets[j],
                    force=500
                )

            if drop_cmd == 1:
                self.drop(robotId)

    # =========================================================================
    # SIM LOOP
    # =========================================================================

    def run(self, actions, num_steps=1):
        for _ in range(num_steps):
            self.apply_actions(actions)
            p.stepSimulation()

            for specimenId, robotId in zip(self.specimenIds, self.robotIds):
                self.check_contact(robotId, specimenId)

            if self.render:
                time.sleep(1. / 240.)

        return self.get_states()

    # =========================================================================
    # DROPLETS & CONTACT
    # =========================================================================

    def drop(self, robotId):
        pos = self.get_pipette_position(robotId)
        pos[2] -= 0.0015

        sphereRadius = 0.003
        visual = p.createVisualShape(p.GEOM_SPHERE, radius=sphereRadius, rgbaColor=[1, 0, 0, 0.5])
        collision = p.createCollisionShape(p.GEOM_SPHERE, radius=sphereRadius)
        sphere = p.createMultiBody(0.1, collision, visual)

        p.resetBasePositionAndOrientation(sphere, pos, [0, 0, 0, 1])
        self.sphereIds.append(sphere)

    def check_contact(self, robotId, specimenId):
        for sphereId in self.sphereIds[:]:
            if p.getContactPoints(sphereId, specimenId):
                p.setCollisionFilterPair(sphereId, specimenId, -1, -1, 0)
                pos, ori = p.getBasePositionAndOrientation(sphereId)
                p.createConstraint(sphereId, -1, -1, -1, p.JOINT_FIXED, [0, 0, 0], [0, 0, 0], pos, ori)

            if p.getContactPoints(sphereId, robotId):
                p.removeBody(sphereId)
                self.sphereIds.remove(sphereId)

    # =========================================================================
    # STATE OUTPUT
    # =========================================================================

    def get_states(self):
        states = {}
        for robotId in self.robotIds:
            states[f'robotId_{robotId}'] = {
                "pipette_position": [round(x, 4) for x in self.get_pipette_position(robotId)]
            }
        return states

    def get_plate_image(self):
        return self.plate_image_path

    def close(self):
        p.disconnect()
