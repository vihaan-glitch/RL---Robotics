"""
arm_reach_env.py
================
Gymnasium environment for a 7-DOF Kuka-style arm reaching task under a
Constrained MDP formulation, backed by MuJoCo.

Observation (21 dims):
  joint_positions (7) | joint_velocities (7) | target_position (3) | ee_position (3) | distance (1)

Action (7 dims):
  Normalised joint position targets in [-1, 1].

Reward:
  Dense shaped: distance_reward + progress_reward + success_bonus + time_penalty

Cost (safety signal):
  Binary per-step: 1 if any joint torque exceeds TORQUE_LIMIT, else 0.
  Returned in info["cost"] — consumed by PPO-Lagrangian to enforce the CMDP constraint.

CMDP constraint:
  Expected cumulative episode cost <= COST_LIMIT (default 25).
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco

try:
    import mujoco.viewer as _mj_viewer
    _VIEWER_AVAILABLE = True
except ImportError:
    _VIEWER_AVAILABLE = False

# ── Safety thresholds ─────────────────────────────────────────────────────────
TORQUE_LIMIT = 100.0  # Nm — any joint exceeding this triggers a cost event
COST_LIMIT   = 25     # Maximum allowed cumulative cost per episode
# ─────────────────────────────────────────────────────────────────────────────

# Policy step = SIM_SUBSTEPS × model timestep (0.002 s) = 0.02 s → 50 Hz
SIM_SUBSTEPS = 10

# ── MuJoCo model (7-DOF Kuka-style arm) ──────────────────────────────────────
_ARM_XML = """
<mujoco model="kuka_reach">
  <compiler angle="radian"/>
  <!-- implicitfast: semi-implicit integration, stable under stiff PD gains -->
  <option gravity="0 0 -9.81" timestep="0.002" integrator="implicitfast"/>

  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
    <rgba haze="0.15 0.25 0.35 1"/>
  </visual>

  <worldbody>
    <light name="main" pos="0 0 3" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>
    <geom name="floor" type="plane" size="3 3 0.1" rgba="0.7 0.7 0.7 1"/>
    <!-- Fixed camera for rgb_array rendering (must live inside worldbody) -->
    <camera name="fixed" pos="1.8 -1.2 1.4" xyaxes="0.55 1 0 -0.35 0 0.94" fovy="55"/>

    <body name="base" pos="0 0 0">
      <geom type="cylinder" size="0.06 0.04" rgba="0.2 0.2 0.2 1"
            contype="0" conaffinity="0"/>

      <body name="link1" pos="0 0 0.16">
        <joint name="joint1" type="hinge" axis="0 0 1"
               range="-2.967 2.967" limited="true" damping="2.0" armature="0.5"/>
        <geom type="capsule" fromto="0 0 -0.06 0 0 0.20"
              size="0.040" rgba="0.55 0.55 0.65 1"/>

        <body name="link2" pos="0 0 0.20">
          <joint name="joint2" type="hinge" axis="0 1 0"
                 range="-2.094 2.094" limited="true" damping="2.0" armature="0.5"/>
          <geom type="capsule" fromto="0 0 0 0 0 0.19"
                size="0.038" rgba="0.55 0.55 0.65 1"/>

          <body name="link3" pos="0 0 0.19">
            <joint name="joint3" type="hinge" axis="0 0 1"
                   range="-2.967 2.967" limited="true" damping="1.0" armature="0.3"/>
            <geom type="capsule" fromto="0 0 0 0 0 0.21"
                  size="0.035" rgba="0.55 0.55 0.65 1"/>

            <body name="link4" pos="0 0 0.21">
              <joint name="joint4" type="hinge" axis="0 -1 0"
                     range="-2.094 2.094" limited="true" damping="1.0" armature="0.3"/>
              <geom type="capsule" fromto="0 0 0 0 0 0.19"
                    size="0.033" rgba="0.55 0.55 0.65 1"/>

              <body name="link5" pos="0 0 0.19">
                <joint name="joint5" type="hinge" axis="0 0 1"
                       range="-2.967 2.967" limited="true" damping="0.5" armature="0.1"/>
                <geom type="capsule" fromto="0 0 0 0 0 0.12"
                      size="0.030" rgba="0.55 0.55 0.65 1"/>

                <body name="link6" pos="0 0 0.12">
                  <joint name="joint6" type="hinge" axis="0 1 0"
                         range="-2.094 2.094" limited="true" damping="0.5" armature="0.1"/>
                  <geom type="capsule" fromto="0 0 0 0 0 0.09"
                        size="0.028" rgba="0.55 0.55 0.65 1"/>

                  <body name="link7" pos="0 0 0.09">
                    <joint name="joint7" type="hinge" axis="0 0 1"
                           range="-3.054 3.054" limited="true" damping="0.2" armature="0.05"/>
                    <geom type="capsule" fromto="0 0 0 0 0 0.06"
                          size="0.025" rgba="0.40 0.40 0.50 1"/>
                    <!-- End-effector site (tracked for FK) -->
                    <site name="ee_site" pos="0 0 0.07" size="0.015"
                          rgba="1 1 0 1"/>
                  </body>
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>

    <!-- Target: mocap body so we can reposition it freely via data.mocap_pos -->
    <body name="target" mocap="true" pos="0.5 0 0.5">
      <geom type="sphere" size="0.05" rgba="1 0 0 0.5"
            contype="0" conaffinity="0"/>
    </body>
  </worldbody>

  <!-- PD position servos. kv adds derivative damping: force = kp*(ctrl-pos) - kv*vel.
       forcerange caps peak torque so random-policy spikes can't destabilise dynamics. -->
  <actuator>
    <position name="act1" joint="joint1" kp="100" kv="20" forcerange="-200 200"/>
    <position name="act2" joint="joint2" kp="100" kv="20" forcerange="-200 200"/>
    <position name="act3" joint="joint3" kp="80"  kv="16" forcerange="-150 150"/>
    <position name="act4" joint="joint4" kp="80"  kv="16" forcerange="-150 150"/>
    <position name="act5" joint="joint5" kp="50"  kv="10" forcerange="-100 100"/>
    <position name="act6" joint="joint6" kp="50"  kv="10" forcerange="-100 100"/>
    <position name="act7" joint="joint7" kp="30"  kv="6"  forcerange="-60  60"/>
  </actuator>
</mujoco>
"""


class ArmReachEnv(gym.Env):
    """
    7-DOF Kuka-style arm reaching task with torque-based safety constraints.

    Parameters
    ----------
    render_mode : str | None
        "human"     → MuJoCo passive viewer (GUI)
        "rgb_array" → headless camera frames via mujoco.Renderer
        None        → fully headless (training)
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, render_mode: str | None = None):
        super().__init__()

        self.render_mode = render_mode

        # ── Load MuJoCo model ─────────────────────────────────────────────────
        self.model = mujoco.MjModel.from_xml_string(_ARM_XML)
        self.data  = mujoco.MjData(self.model)

        self.num_controllable = self.model.nu  # 7

        # Joint limits from model
        self.joint_lower_limits = self.model.jnt_range[:, 0].astype(np.float32)
        self.joint_upper_limits = self.model.jnt_range[:, 1].astype(np.float32)

        # Persistent IDs for named model objects
        self._ee_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "ee_site"
        )
        _target_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "target"
        )
        self._target_mocap_id = int(self.model.body_mocapid[_target_body_id])

        # ── Spaces ────────────────────────────────────────────────────────────
        n = self.num_controllable
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(n,), dtype=np.float32
        )
        obs_dim = n * 2 + 3 + 3 + 1  # 21
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # ── Named obs slices ──────────────────────────────────────────────────
        self._idx_jp  = slice(0, n)          # joint positions
        self._idx_jv  = slice(n, 2 * n)      # joint velocities
        self._idx_tgt = slice(2*n, 2*n+3)    # target position
        self._idx_ee  = slice(2*n+3, 2*n+6)  # end-effector position
        self._idx_d   = -1                    # distance scalar

        self.max_steps  = 500
        self.step_count = 0

        self.target_position   = np.zeros(3, dtype=np.float32)
        self.previous_distance = None

        # ── Renderer / viewer (created lazily) ────────────────────────────────
        self._viewer   = None
        self._renderer = None

        if render_mode == "human":
            if not _VIEWER_AVAILABLE:
                raise RuntimeError(
                    "mujoco.viewer is not available on this system. "
                    "Use render_mode=None or 'rgb_array'."
                )
            self._viewer = _mj_viewer.launch_passive(self.model, self.data)

    # ─────────────────────────────────────────────────────────────────────────
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0

        mujoco.mj_resetData(self.model, self.data)

        # Random near-zero initial joint configuration
        # Use self.np_random (seeded by super().reset) for reproducibility.
        init_pos = (
            self.np_random.uniform(self.joint_lower_limits, self.joint_upper_limits)
            * 0.3
        )
        self.data.qpos[:self.num_controllable] = init_pos

        # Randomise target across reachable workspace
        self.target_position = self.np_random.uniform(
            low=[0.2, -0.4, 0.1],
            high=[0.7,  0.4, 0.8],
        ).astype(np.float32)
        self.data.mocap_pos[self._target_mocap_id] = self.target_position

        # Recompute forward kinematics so site positions are valid immediately
        mujoco.mj_forward(self.model, self.data)

        obs = self._get_obs()
        self.previous_distance = float(obs[self._idx_d])

        if self._viewer is not None:
            self._viewer.sync()

        return obs, {}

    # ─────────────────────────────────────────────────────────────────────────
    def _get_obs(self) -> np.ndarray:
        joint_positions  = self.data.qpos[:self.num_controllable].astype(np.float32)
        joint_velocities = self.data.qvel[:self.num_controllable].astype(np.float32)
        ee_position      = self.data.site_xpos[self._ee_site_id].astype(np.float32)
        distance         = float(np.linalg.norm(ee_position - self.target_position))

        return np.concatenate([
            joint_positions,
            joint_velocities,
            self.target_position,
            ee_position,
            [distance],
        ]).astype(np.float32)

    # ─────────────────────────────────────────────────────────────────────────
    def _compute_cost(self) -> float:
        """
        Binary safety cost: 1 if any joint torque exceeds TORQUE_LIMIT, else 0.
        Uses data.qfrc_actuator — generalized actuator forces at each joint (Nm).
        """
        torques = self.data.qfrc_actuator[:self.num_controllable]
        return float(np.any(np.abs(torques) > TORQUE_LIMIT))

    # ─────────────────────────────────────────────────────────────────────────
    def step(self, action):
        self.step_count += 1

        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        target_positions = (
            self.joint_lower_limits
            + (action + 1.0) / 2.0
            * (self.joint_upper_limits - self.joint_lower_limits)
        )
        self.data.ctrl[:] = target_positions

        # Action repeat — accumulate cost across substeps
        substep_cost = 0.0
        for _ in range(SIM_SUBSTEPS):
            mujoco.mj_step(self.model, self.data)
            substep_cost += self._compute_cost()

        # Binary cost for the whole policy step
        cost = float(substep_cost > 0)

        if self._viewer is not None:
            self._viewer.sync()

        obs      = self._get_obs()
        distance = float(obs[self._idx_d])
        ee_position = obs[self._idx_ee]

        # ── Reward ────────────────────────────────────────────────────────────
        distance_reward = -distance
        progress_reward = (self.previous_distance - distance) * 10.0
        success_bonus   = 100.0 if distance < 0.05 else 0.0
        time_penalty    = -0.01
        reward = distance_reward + progress_reward + success_bonus + time_penalty
        self.previous_distance = distance

        terminated = bool(distance < 0.05)
        truncated  = bool(self.step_count >= self.max_steps)

        info = {
            "distance":    distance,
            "cost":        cost,
            "cost_limit":  COST_LIMIT,
            "ee_position": ee_position.tolist(),
        }

        return obs, reward, terminated, truncated, info

    # ─────────────────────────────────────────────────────────────────────────
    def render(self):
        if self.render_mode == "rgb_array":
            if self._renderer is None:
                self._renderer = mujoco.Renderer(self.model, height=480, width=640)
            self._renderer.update_scene(self.data, camera="fixed")
            return self._renderer.render()

    def close(self):
        if self._viewer is not None:
            try:
                self._viewer.close()
            except Exception:
                pass
            self._viewer = None
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


class LagrangianRewardWrapper(gym.Wrapper):
    """
    Augments the per-step reward with the Lagrangian penalty:

        r_aug = r - λ · c(s, a)

    `lam` is set externally by LagrangianSafetyCallback after each episode,
    implementing the dual variable update of the CMDP primal-dual algorithm.
    Without this wrapper the Lagrange multiplier is computed but never applied,
    so the policy would train on unpenalised rewards regardless of constraint violations.
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.lam: float = 0.0

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        cost = info.get("cost", 0.0)
        reward = float(reward) - self.lam * cost
        info["lagrangian_penalty"] = self.lam * cost
        return obs, reward, terminated, truncated, info
