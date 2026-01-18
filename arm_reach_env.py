import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data
import time


class ArmReachEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, render=False):
        super().__init__()
        self.render_mode = render
        self.physics_client = p.connect(p.GUI if render else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)
        
        self.time_step = 1.0 / 240.0
        p.setTimeStep(self.time_step)
        
        self.max_steps = 500
        self.step_count = 0
        
        # Load plane and robot
        self.plane = p.loadURDF("plane.urdf")
        self.robot = p.loadURDF("kuka_iiwa/model.urdf", useFixedBase=True)
        self.num_joints = p.getNumJoints(self.robot)
        
        # Get controllable joints (exclude fixed joints)
        self.controllable_joints = []
        for j in range(self.num_joints):
            joint_info = p.getJointInfo(self.robot, j)
            if joint_info[2] != p.JOINT_FIXED:
                self.controllable_joints.append(j)
        
        self.num_controllable = len(self.controllable_joints)
        
        # Action: joint position targets (normalized to -1, 1)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.num_controllable,), dtype=np.float32
        )
        
        # Get joint limits
        self.joint_lower_limits = []
        self.joint_upper_limits = []
        for j in self.controllable_joints:
            joint_info = p.getJointInfo(self.robot, j)
            self.joint_lower_limits.append(joint_info[8])
            self.joint_upper_limits.append(joint_info[9])
        
        self.joint_lower_limits = np.array(self.joint_lower_limits)
        self.joint_upper_limits = np.array(self.joint_upper_limits)
        
        # Observation: joint positions + velocities + target position + ee position + distance
        obs_dim = self.num_controllable * 2 + 3 + 3 + 1
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        
        self.target_position = np.array([0.6, 0.0, 0.5])
        self.target_visual = None
        
        # Track previous distance for reward shaping
        self.previous_distance = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        
        # Reset robot to random initial position
        for j in self.controllable_joints:
            init_pos = np.random.uniform(
                self.joint_lower_limits[self.controllable_joints.index(j)],
                self.joint_upper_limits[self.controllable_joints.index(j)]
            ) * 0.3  # Start near center
            p.resetJointState(self.robot, j, init_pos)
        
        # Random target position
        self.target_position = np.random.uniform(
            low=[0.3, -0.3, 0.2],
            high=[0.7, 0.3, 0.7]
        )
        
        # Create visual target
        if self.target_visual is not None:
            p.removeBody(self.target_visual)
        
        self.target_visual = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_SPHERE, radius=0.05),
            baseVisualShapeIndex=p.createVisualShape(
                p.GEOM_SPHERE, radius=0.05, rgbaColor=[1, 0, 0, 0.8]
            ),
            basePosition=self.target_position
        )
        
        # Initialize previous distance
        obs = self._get_obs()
        ee_position = obs[-4:-1]
        self.previous_distance = np.linalg.norm(ee_position - self.target_position)
        
        return obs, {}

    def _get_obs(self):
        joint_states = p.getJointStates(self.robot, self.controllable_joints)
        joint_positions = np.array([state[0] for state in joint_states])
        joint_velocities = np.array([state[1] for state in joint_states])
        
        # Get end-effector position (last link)
        end_effector_state = p.getLinkState(
            self.robot, 
            self.num_joints - 1,
            computeLinkVelocity=1
        )
        ee_position = np.array(end_effector_state[0])
        
        # Calculate distance to target
        distance = np.linalg.norm(ee_position - self.target_position)
        
        return np.concatenate([
            joint_positions,
            joint_velocities,
            self.target_position,
            ee_position,
            [distance]
        ]).astype(np.float32)

    def step(self, action):
        self.step_count += 1
        
        # Convert normalized action to actual joint positions
        action = np.clip(action, -1.0, 1.0)
        target_positions = self.joint_lower_limits + (action + 1.0) / 2.0 * (
            self.joint_upper_limits - self.joint_lower_limits
        )
        
        # Apply position control
        for idx, j in enumerate(self.controllable_joints):
            p.setJointMotorControl2(
                bodyIndex=self.robot,
                jointIndex=j,
                controlMode=p.POSITION_CONTROL,
                targetPosition=target_positions[idx],
                force=500,
                maxVelocity=2.0
            )
        
        # Step simulation
        p.stepSimulation()
        if self.render_mode:
            time.sleep(self.time_step)
        
        obs = self._get_obs()
        ee_position = obs[-4:-1]
        distance = obs[-1]
        
        # Improved reward shaping
        # 1. Distance reward
        distance_reward = -distance
        
        # 2. Progress reward (getting closer)
        progress_reward = (self.previous_distance - distance) * 10.0
        self.previous_distance = distance
        
        # 3. Success bonus
        success_bonus = 100.0 if distance < 0.05 else 0.0
        
        # 4. Small penalty for time (encourage efficiency)
        time_penalty = -0.01
        
        reward = distance_reward + progress_reward + success_bonus + time_penalty
        
        terminated = distance < 0.05
        truncated = self.step_count >= self.max_steps
        
        return obs, reward, terminated, truncated, {"distance": distance}

    def close(self):
        if self.target_visual is not None:
            p.removeBody(self.target_visual)
        p.disconnect()
