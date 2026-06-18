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
  Configurable via the `reward_mode` constructor argument (see REWARD_MODES).
  The default "full" mode is the original dense-shaped reward:
      distance_reward + progress_reward + success_bonus + time_penalty

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

# ── Reward modes (for reward-ablation study) ──────────────────────────────────
# Each mode toggles which of the four reward terms are summed into the final
# per-step reward. All other environment logic (termination, randomization,
# observation/action spaces) is identical across modes — only the reward
# computation changes. See ArmReachEnv._compute_reward for the term definitions.
#
#   distance : distance_reward = -distance               (dense distance penalty)
#   progress : progress_reward = 10 * (prev_d - d)       (dense shaping)
#   success  : success_bonus   = +100 if distance < 0.05 (terminal-style bonus)
#   time     : time_penalty    = -0.01 per step          (encourages speed)
REWARD_MODES = {
    "full":            ("distance", "progress", "success", "time"),
    "sparse":          ("success",),
    "no_progress":     ("distance", "success", "time"),
    "no_time_penalty": ("distance", "progress", "success"),
    "distance_only":   ("distance",),
}

# ── Obstacles (for the hard-task ablation) ────────────────────────────────────
# Three fixed static boxes placed in the frontal workspace. Their positions are
# chosen so that a naive straight-line reach from a typical initial joint
# configuration toward many targets would pass through a box — forcing the policy
# to learn a more deliberate, curved trajectory. Targets are resampled away from
# the boxes (see ArmReachEnv._target_clear) so the goal itself is always in free
# space and remains reachable; only the *path* is obstructed.
#
# Each entry: (name, center (x, y, z), half-size (x, y, z))  — MuJoCo box `size`
# is the half-extent along each axis.
SIMPLE_OBSTACLES = [
    ("obstacle0", (0.35,  0.00, 0.42), (0.04, 0.18, 0.16)),  # central vertical slab
    ("obstacle1", (0.30, -0.26, 0.28), (0.09, 0.06, 0.18)),  # low side block
    ("obstacle2", (0.46,  0.22, 0.60), (0.12, 0.10, 0.04)),  # high shelf
]

# Reward penalty applied once per policy step in which the arm contacts any
# obstacle (binary, not per-contact-point — keeps the penalty bounded). Applied
# uniformly across all reward modes, since collision avoidance is a property of
# the harder *task*, not one of the reward-shaping terms under ablation.
COLLISION_PENALTY = -10.0

VALID_OBSTACLE_MODES = (None, "simple")


def _obstacle_xml(obstacle_mode: str | None) -> str:
    """Return the MuJoCo geom XML for the requested obstacle mode (empty if None)."""
    if obstacle_mode is None:
        return ""
    geoms = [
        f'<geom name="{name}" type="box" pos="{px} {py} {pz}" '
        f'size="{sx} {sy} {sz}" rgba="0.40 0.40 0.48 1"/>'
        for name, (px, py, pz), (sx, sy, sz) in SIMPLE_OBSTACLES
    ]
    return "\n      ".join(geoms)


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

    <!-- Static obstacles injected here at build time (empty unless obstacle_mode). -->
      <!-- OBSTACLES -->
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
    reward_mode : str
        Which reward terms are active. One of REWARD_MODES (default "full"):
          "full"            distance + progress + success + time (original)
          "sparse"          success bonus only (+100 at goal, 0 otherwise)
          "no_progress"     distance + success + time (no shaping)
          "no_time_penalty" distance + progress + success
          "distance_only"   negated distance penalty only (no bonus/progress/time)
        Only the reward computation changes between modes; all other dynamics
        (termination, randomization, spaces) are held identical.
    obstacle_mode : str | None
        Controls static obstacles in the workspace (default None):
          None       no obstacles — the original easy task.
          "simple"   three fixed static box obstacles (see SIMPLE_OBSTACLES)
                     placed in the frontal workspace so that naive straight-line
                     reaches collide, forcing more deliberate trajectories.
        With obstacles active:
          * The arm physically collides with the boxes (real MuJoCo contacts).
          * A COLLISION_PENALTY (-10) is added to the reward once per policy step
            in which any arm link touches an obstacle. This penalty is applied
            uniformly across ALL reward modes — collision avoidance is a property
            of the harder task, not a reward-shaping term under ablation, so the
            relative comparison between reward modes is preserved.
          * Targets are resampled away from the boxes so the goal is always in
            free space; only the path to it is obstructed.
        The observation and action spaces are unchanged (obstacles are at fixed
        positions, so a policy can learn to avoid them without observing them).
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, render_mode: str | None = None, reward_mode: str = "full",
                 obstacle_mode: str | None = None):
        super().__init__()

        self.render_mode = render_mode

        if reward_mode not in REWARD_MODES:
            raise ValueError(
                f"Unknown reward_mode={reward_mode!r}. "
                f"Expected one of {sorted(REWARD_MODES)}."
            )
        if obstacle_mode not in VALID_OBSTACLE_MODES:
            raise ValueError(
                f"Unknown obstacle_mode={obstacle_mode!r}. "
                f"Expected one of {VALID_OBSTACLE_MODES}."
            )
        self.reward_mode = reward_mode
        self._active_terms = REWARD_MODES[reward_mode]
        self.obstacle_mode = obstacle_mode

        # ── Load MuJoCo model ─────────────────────────────────────────────────
        xml = _ARM_XML.replace("<!-- OBSTACLES -->", _obstacle_xml(obstacle_mode))
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data  = mujoco.MjData(self.model)

        self.num_controllable = self.model.nu  # 7

        # ── Geom bookkeeping for collision detection ──────────────────────────
        # Obstacle geoms by name; arm geoms = those on the base/link bodies.
        self._obstacle_geom_ids: set[int] = set()
        if obstacle_mode is not None:
            for name, _, _ in SIMPLE_OBSTACLES:
                gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
                if gid >= 0:
                    self._obstacle_geom_ids.add(gid)
        self._arm_geom_ids: set[int] = set()
        for gid in range(self.model.ngeom):
            bname = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_BODY, int(self.model.geom_bodyid[gid])
            )
            if bname and (bname.startswith("link") or bname == "base"):
                self._arm_geom_ids.add(gid)

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

        # Randomise target across reachable workspace (clear of obstacles).
        self.target_position = self._sample_target()
        self.data.mocap_pos[self._target_mocap_id] = self.target_position

        # Recompute forward kinematics so site positions are valid immediately
        mujoco.mj_forward(self.model, self.data)

        obs = self._get_obs()
        self.previous_distance = float(obs[self._idx_d])

        if self._viewer is not None:
            self._viewer.sync()

        return obs, {}

    # ─────────────────────────────────────────────────────────────────────────
    def _target_clear(self, t: np.ndarray, margin: float = 0.08) -> bool:
        """True if target t is outside every obstacle's (expanded) bounding box."""
        for _, (px, py, pz), (sx, sy, sz) in SIMPLE_OBSTACLES:
            if (abs(t[0] - px) < sx + margin
                    and abs(t[1] - py) < sy + margin
                    and abs(t[2] - pz) < sz + margin):
                return False
        return True

    def _sample_target(self) -> np.ndarray:
        """Uniform target in the workspace; resampled to avoid obstacle volumes."""
        low, high = [0.2, -0.4, 0.1], [0.7, 0.4, 0.8]
        t = self.np_random.uniform(low=low, high=high).astype(np.float32)
        if self.obstacle_mode is not None:
            for _ in range(100):
                if self._target_clear(t):
                    break
                t = self.np_random.uniform(low=low, high=high).astype(np.float32)
        return t

    def _count_obstacle_contacts(self) -> int:
        """Number of active contact points between an arm link and an obstacle."""
        if not self._obstacle_geom_ids:
            return 0
        n = 0
        for i in range(self.data.ncon):
            g1 = self.data.contact[i].geom1
            g2 = self.data.contact[i].geom2
            obs = ((g1 in self._obstacle_geom_ids and g2 in self._arm_geom_ids)
                   or (g2 in self._obstacle_geom_ids and g1 in self._arm_geom_ids))
            if obs:
                n += 1
        return n

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
    def _compute_reward(self, distance: float) -> float:
        """
        Reward = sum of the terms enabled by the active reward_mode.

        Term definitions (all use `distance`, the current EE→target distance,
        and `self.previous_distance` from the prior step):

          distance : -distance               dense penalty pulling EE toward goal
          progress : 10 * (prev_d - d)       dense shaping; positive when closing in
          success  : +100 if distance < 0.05 large bonus for reaching the goal
          time     : -0.01                   constant per-step penalty

        Which terms are summed is fixed at construction time by `reward_mode`
        (see REWARD_MODES). Modes never change termination, randomization, or
        the observation/action spaces — only this scalar.
        """
        terms = self._active_terms
        reward = 0.0
        if "distance" in terms:
            reward += -distance
        if "progress" in terms:
            reward += (self.previous_distance - distance) * 10.0
        if "success" in terms:
            reward += 100.0 if distance < 0.05 else 0.0
        if "time" in terms:
            reward += -0.01
        return reward

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

        # Action repeat — accumulate cost and obstacle contacts across substeps
        substep_cost = 0.0
        substep_contacts = 0
        for _ in range(SIM_SUBSTEPS):
            mujoco.mj_step(self.model, self.data)
            substep_cost += self._compute_cost()
            substep_contacts += self._count_obstacle_contacts()

        # Binary cost for the whole policy step
        cost = float(substep_cost > 0)
        # Binary collision flag for the whole policy step (any arm-obstacle contact)
        collision = float(substep_contacts > 0)

        if self._viewer is not None:
            self._viewer.sync()

        obs      = self._get_obs()
        distance = float(obs[self._idx_d])
        ee_position = obs[self._idx_ee]

        # ── Reward ────────────────────────────────────────────────────────────
        # Reward-mode terms (the ablation variable) plus, on the hard task, a
        # uniform collision penalty applied identically across all modes.
        reward = self._compute_reward(distance)
        if self.obstacle_mode is not None:
            reward += COLLISION_PENALTY * collision
        self.previous_distance = distance

        terminated = bool(distance < 0.05)
        truncated  = bool(self.step_count >= self.max_steps)

        info = {
            "distance":    distance,
            "cost":        cost,
            "cost_limit":  COST_LIMIT,
            "ee_position": ee_position.tolist(),
            "collision":   collision,
            "n_contacts":  substep_contacts,
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
