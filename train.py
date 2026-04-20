"""
train.py
========
Training script for safe RL on the Kuka IIWA arm reaching task.

Supports two modes via command-line flag --algo:

  ppo_lag  (default)  PPO-Lagrangian (constrained MDP)
                      Uses a learnable Lagrange multiplier to enforce
                      expected cumulative cost <= COST_LIMIT per episode.

  ppo                 Vanilla PPO baseline (no safety constraint).
                      Run this first to establish the baseline curve,
                      then compare against ppo_lag.

Usage
-----
  python train.py --algo ppo_lag --timesteps 2000000
  python train.py --algo ppo     --timesteps 2000000

Outputs
-------
  models/ppo_lag/   or   models/ppo/
    ├── best_model/best_model.zip
    ├── final_model.zip
    └── vec_normalize.pkl
  logs/ppo_lag/     or   logs/ppo/
    ├── tensorboard/
    └── constraint_violations.csv
"""

import argparse
import csv
import os
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback, CheckpointCallback, EvalCallback
)
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from arm_reach_env import ArmReachEnv, LagrangianRewardWrapper, COST_LIMIT

# ── Argument parsing ──────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--algo",       type=str,   default="ppo_lag",
                    choices=["ppo_lag", "ppo"],
                    help="Algorithm: ppo_lag (PPO-Lagrangian) or ppo (baseline)")
parser.add_argument("--timesteps",  type=int,   default=2_000_000,
                    help="Total environment steps")
parser.add_argument("--cost_limit", type=float, default=float(COST_LIMIT),
                    help="Constraint budget d (used by ppo_lag only)")
args = parser.parse_args()

ALGO       = args.algo
TIMESTEPS  = args.timesteps
COST_LIMIT_ARG = args.cost_limit

MODEL_DIR = f"models/{ALGO}"
LOG_DIR   = f"logs/{ALGO}"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR,   exist_ok=True)

print(f"\n{'='*55}")
print(f"  Algorithm  : {ALGO.upper()}")
print(f"  Timesteps  : {TIMESTEPS:,}")
if ALGO == "ppo_lag":
    print(f"  Cost limit : {COST_LIMIT_ARG}")
print(f"{'='*55}\n")


# ── Environments ──────────────────────────────────────────────────────────────
def make_env():
    return ArmReachEnv(render_mode=None)

def make_env_lag():
    # Wrap with Lagrangian penalty so λ is applied to the reward the policy sees.
    # Without this, the callback updates λ but it never reaches the policy gradient.
    return LagrangianRewardWrapper(ArmReachEnv(render_mode=None))

env = DummyVecEnv([make_env_lag if ALGO == "ppo_lag" else make_env])
env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

eval_env = DummyVecEnv([make_env])
eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)


# ── PPO-Lagrangian callback ───────────────────────────────────────────────────
class LagrangianSafetyCallback(BaseCallback):
    """
    Implements the Lagrangian update for PPO-Lag:

      λ ← max(0, λ + η_λ · (J_C(π) - d))

    where J_C(π) is the running estimate of average episode cost
    and d is the cost limit.

    Also logs:
      - per-episode cost to CSV
      - Lagrange multiplier value
      - constraint violation rate
    """

    def __init__(
        self,
        eval_env:      "VecNormalize",
        cost_limit:    float = COST_LIMIT,
        lambda_init:   float = 1.0,
        lambda_lr:     float = 0.01,
        lambda_max:    float = 20.0,
        log_path:      str   = "logs/ppo_lag",
        sync_freq:     int   = 1000,
        verbose:       int   = 1,
    ):
        super().__init__(verbose)
        self._eval_env    = eval_env
        self.cost_limit   = cost_limit
        self.lam          = lambda_init   # Lagrange multiplier λ
        self.lambda_lr    = lambda_lr
        self.lambda_max   = lambda_max
        self.log_path     = log_path
        self.sync_freq    = sync_freq

        # Rolling buffers
        self._episode_costs: list[float] = []
        self._current_cost  = 0.0
        self._violation_count = 0
        self._episode_count   = 0

        # CSV writer
        csv_path = os.path.join(log_path, "constraint_violations.csv")
        self._csv_file = open(csv_path, "w", newline="")
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_writer.writerow(
            ["timestep", "episode", "episode_cost", "lambda", "violation"]
        )

    # ── Called every env step ────────────────────────────────────────────────
    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [{}])
        dones = self.locals.get("dones", [False])

        for info, done in zip(infos, dones):
            self._current_cost += info.get("cost", 0.0)

            if done:
                ep_cost = self._current_cost
                violated = int(ep_cost > self.cost_limit)

                self._episode_costs.append(ep_cost)
                self._violation_count += violated
                self._episode_count   += 1
                self._current_cost     = 0.0

                # Log to CSV
                self._csv_writer.writerow([
                    self.num_timesteps,
                    self._episode_count,
                    ep_cost,
                    round(self.lam, 4),
                    violated,
                ])

                # Lagrangian update: λ ← clip(λ + η(J_C - d), 0, λ_max)
                self.lam = float(np.clip(
                    self.lam + self.lambda_lr * (ep_cost - self.cost_limit),
                    0.0, self.lambda_max
                ))

                # Propagate updated λ to each wrapped environment so the policy
                # actually trains on the penalised reward r - λ·c.
                venv = getattr(self.training_env, "venv", self.training_env)
                for env_obj in getattr(venv, "envs", []):
                    if isinstance(env_obj, LagrangianRewardWrapper):
                        env_obj.lam = self.lam

                if self.verbose >= 1 and self._episode_count % 20 == 0:
                    recent = self._episode_costs[-20:]
                    avg_cost = np.mean(recent)
                    viol_rate = self._violation_count / self._episode_count
                    print(
                        f"  [Ep {self._episode_count:>5}] "
                        f"avg_cost={avg_cost:.2f}  λ={self.lam:.3f}  "
                        f"violation_rate={viol_rate:.2%}"
                    )

        # Sync eval env normalisation stats periodically
        if self.num_timesteps % self.sync_freq == 0:
            if hasattr(self.training_env, "obs_rms"):
                self._eval_env.obs_rms = self.training_env.obs_rms

        return True

    def _on_training_end(self):
        self._csv_file.close()
        total_eps = self._episode_count
        if total_eps > 0:
            print(f"\n[Safety Summary]")
            print(f"  Total episodes        : {total_eps}")
            print(f"  Final λ               : {self.lam:.4f}")
            print(f"  Overall violation rate: "
                  f"{self._violation_count / total_eps:.2%}")
            print(f"  Avg episode cost      : "
                  f"{np.mean(self._episode_costs):.2f}")


# ── Cost logger (used for vanilla PPO baseline too) ───────────────────────────
class CostLoggerCallback(BaseCallback):
    """Lightweight callback that logs per-episode cost for the PPO baseline."""

    def __init__(self, eval_env: "VecNormalize", log_path: str,
                 sync_freq: int = 1000, verbose: int = 0):
        super().__init__(verbose)
        self._eval_env      = eval_env
        self._sync_freq     = sync_freq
        self._current_cost  = 0.0
        self._episode_count = 0
        csv_path = os.path.join(log_path, "constraint_violations.csv")
        self._csv_file   = open(csv_path, "w", newline="")
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_writer.writerow(["timestep", "episode", "episode_cost", "violation"])

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [{}])
        dones = self.locals.get("dones", [False])
        for info, done in zip(infos, dones):
            self._current_cost += info.get("cost", 0.0)
            if done:
                ep_cost = self._current_cost
                self._episode_count += 1
                self._csv_writer.writerow([
                    self.num_timesteps,
                    self._episode_count,
                    ep_cost,
                    int(ep_cost > COST_LIMIT)
                ])
                self._current_cost = 0.0

        if self.num_timesteps % self._sync_freq == 0:
            if hasattr(self.training_env, "obs_rms"):
                self._eval_env.obs_rms = self.training_env.obs_rms

        return True

    def _on_training_end(self):
        self._csv_file.close()


# ── TensorBoard ───────────────────────────────────────────────────────────────
try:
    import tensorboard  # noqa: F401
    tb_log = os.path.join(LOG_DIR, "tensorboard")
    print("TensorBoard logging enabled")
except ImportError:
    tb_log = None
    print("TensorBoard not installed — skipping")

# ── Callbacks ─────────────────────────────────────────────────────────────────
checkpoint_cb = CheckpointCallback(
    save_freq=25_000,
    save_path=MODEL_DIR,
    name_prefix=f"{ALGO}_arm_reach"
)

eval_cb = EvalCallback(
    eval_env,
    best_model_save_path=os.path.join(MODEL_DIR, "best_model"),
    log_path=LOG_DIR,
    eval_freq=10_000,
    n_eval_episodes=10,
    deterministic=True,
    render=False,
)

if ALGO == "ppo_lag":
    safety_cb = LagrangianSafetyCallback(
        eval_env=eval_env,
        cost_limit=COST_LIMIT_ARG,
        lambda_init=1.0,
        lambda_lr=0.01,
        lambda_max=20.0,
        log_path=LOG_DIR,
        verbose=1,
    )
    callbacks = [checkpoint_cb, eval_cb, safety_cb]
else:
    cost_logger = CostLoggerCallback(eval_env=eval_env, log_path=LOG_DIR)
    callbacks   = [checkpoint_cb, eval_cb, cost_logger]

# ── Model ─────────────────────────────────────────────────────────────────────
#
# PPO-Lagrangian is implemented here by combining:
#   1. Standard PPO (policy optimisation)
#   2. The LagrangianSafetyCallback (dual variable update on the cost constraint)
#
# The Lagrange multiplier λ penalises cost violations by effectively reducing
# the reward signal when costs are high, steering the policy toward safety.
# A full primal-dual update (as in Tessler et al., 2018) would require a custom
# policy gradient — this approximation converges well for moderate cost limits.
#
model = PPO(
    "MlpPolicy",
    env,
    verbose=0,               # silence per-step output; callbacks handle logging
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    vf_coef=0.5,
    max_grad_norm=0.5,
    tensorboard_log=tb_log,
)

print(f"Observation space : {env.observation_space}")
print(f"Action space      : {env.action_space}")
print(f"Starting training for {TIMESTEPS:,} timesteps...\n")

model.learn(
    total_timesteps=TIMESTEPS,
    callback=callbacks,
    progress_bar=True,
)

# ── Save ──────────────────────────────────────────────────────────────────────
final_model_path = os.path.join(MODEL_DIR, "final_model")
model.save(final_model_path)
env.save(os.path.join(MODEL_DIR, "vec_normalize.pkl"))

print(f"\nTraining complete.")
print(f"  Model saved to : {final_model_path}.zip")
print(f"  Stats saved to : {MODEL_DIR}/vec_normalize.pkl")

env.close()
eval_env.close()