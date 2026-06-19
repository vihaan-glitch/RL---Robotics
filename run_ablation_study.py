"""
run_ablation_study.py
======================
Controlled reward-ablation experiment for the Kuka-style arm reaching task.

Trains a *vanilla PPO* agent (no safety/Lagrangian constraint — the constraint
is orthogonal to the reward-shaping question under study) for each of the five
reward modes defined in arm_reach_env.REWARD_MODES, across 2 random seeds:

    5 reward modes  ×  2 seeds  =  10 runs

All runs share the PPO hyperparameters and the per-run timestep budget. Only the
environment's `reward_mode` changes between conditions; every other dynamic
(termination, randomization, observation/action spaces) is held fixed.

Per-run outputs (under results/{reward_mode}_seed{N}/)
-----------------------------------------------------
    final_model.zip        trained PPO policy
    vec_normalize.pkl      VecNormalize statistics (obs normalization)
    tensorboard/           full TensorBoard log
    evaluations.npz        periodic eval rewards (EvalCallback, every EVAL_FREQ)
    best_model/            best-by-eval checkpoint
    first_success.json     timestep of the first successful episode in training
                           (first step where distance < SUCCESS_DISTANCE), or
                           null if the goal was never reached during training
    run_meta.json          mode, seed, timesteps, wall-clock duration

The runner is:
  * Sequential   — one run at a time (CPU-only friendly).
  * Progress-aware — prints "Run k/10: reward_mode=..., seed=..." and an ETA
                     extrapolated from the mean duration of completed runs.
  * Resumable    — a run whose final_model.zip already exists is skipped, so an
                   interrupted study can be restarted without losing work.

Usage
-----
  python run_ablation_study.py            # prints config, then asks to confirm
  python run_ablation_study.py --yes      # skip the confirmation prompt
  python run_ablation_study.py --timesteps 500000 --seeds 0 1
"""

import argparse
import json
import os
import sys
import time

import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import (
    DummyVecEnv, VecNormalize, sync_envs_normalization,
)

from arm_reach_env import ArmReachEnv, REWARD_MODES

# ── Experiment configuration ──────────────────────────────────────────────────
SUCCESS_DISTANCE = 0.05      # metres; distance < this counts as reaching the goal
EVAL_FREQ        = 5_000     # env steps between periodic evaluations
N_EVAL_EPISODES  = 10        # episodes averaged per periodic evaluation
RESULTS_ROOT     = "results" # set per --task-difficulty in main()
OBSTACLE_MODE    = None      # set per --task-difficulty in main()

# Task-difficulty presets: difficulty → (obstacle_mode, results_root).
# Saving hard runs under results_hard/ keeps the easy-task results intact.
DIFFICULTY_PRESETS = {
    "easy": (None,     "results"),
    "hard": ("simple", "results_hard"),
}

# PPO hyperparameters — copied verbatim from train.py so the ablation matches the
# existing pipeline exactly. (verbose=0; callbacks handle all logging.)
PPO_KWARGS = dict(
    policy="MlpPolicy",
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
    verbose=0,
)


# ── First-success tracker ─────────────────────────────────────────────────────
class FirstSuccessCallback(BaseCallback):
    """
    Records the global timestep of the first successful episode during training,
    i.e. the first env step at which info["distance"] < SUCCESS_DISTANCE.

    Also keeps the eval VecNormalize statistics in sync with the training env so
    that periodic evaluations see correctly normalized observations. Syncing here
    (before EvalCallback runs) avoids the stale-stats pitfall with VecNormalize.
    """

    def __init__(self, eval_env, sync_freq: int = 1000, verbose: int = 0):
        super().__init__(verbose)
        self._eval_env = eval_env
        self._sync_freq = sync_freq
        self.first_success_timestep: int | None = None

    def _on_step(self) -> bool:
        if self.first_success_timestep is None:
            for info in self.locals.get("infos", []):
                if info.get("distance", np.inf) < SUCCESS_DISTANCE:
                    self.first_success_timestep = int(self.num_timesteps)
                    if self.verbose:
                        print(f"    first success at timestep "
                              f"{self.first_success_timestep:,}")
                    break

        # Keep eval normalization stats current for the next periodic eval.
        if self.num_timesteps % self._sync_freq == 0:
            try:
                sync_envs_normalization(self.training_env, self._eval_env)
            except Exception:
                # Fall back to the lightweight obs_rms copy used in train.py.
                if hasattr(self.training_env, "obs_rms"):
                    self._eval_env.obs_rms = self.training_env.obs_rms
        return True


# ── Single run ────────────────────────────────────────────────────────────────
def run_dir(reward_mode: str, seed: int) -> str:
    return os.path.join(RESULTS_ROOT, f"{reward_mode}_seed{seed}")


def is_complete(reward_mode: str, seed: int) -> bool:
    """A run is complete once its final model has been written."""
    return os.path.exists(os.path.join(run_dir(reward_mode, seed), "final_model.zip"))


def train_one(reward_mode: str, seed: int, timesteps: int,
              obstacle_mode: str | None = None) -> None:
    """Train a single PPO agent for (reward_mode, seed) and persist all outputs."""
    out_dir = run_dir(reward_mode, seed)
    tb_dir  = os.path.join(out_dir, "tensorboard")
    os.makedirs(tb_dir, exist_ok=True)

    set_random_seed(seed)

    def make_env():
        return ArmReachEnv(render_mode=None, reward_mode=reward_mode,
                           obstacle_mode=obstacle_mode)

    # Training env: normalize obs and reward (matches train.py).
    train_env = DummyVecEnv([make_env])
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    train_env.seed(seed)

    # Eval env: normalize obs only (reward kept raw so eval rewards are comparable
    # across modes), training=False so its stats are not updated during eval.
    eval_env = DummyVecEnv([make_env])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)
    eval_env.seed(seed + 10_000)

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(out_dir, "best_model"),
        log_path=out_dir,                # writes evaluations.npz here
        eval_freq=EVAL_FREQ,
        n_eval_episodes=N_EVAL_EPISODES,
        deterministic=True,
        render=False,
        verbose=0,
    )
    first_success_cb = FirstSuccessCallback(eval_env, sync_freq=1000, verbose=1)

    model = PPO(env=train_env, seed=seed, tensorboard_log=tb_dir, **PPO_KWARGS)

    model.learn(
        total_timesteps=timesteps,
        # Sync callback must precede eval so eval sees fresh normalization stats.
        callback=[first_success_cb, eval_cb],
        progress_bar=True,
    )

    # ── Persist outputs ───────────────────────────────────────────────────────
    model.save(os.path.join(out_dir, "final_model"))
    train_env.save(os.path.join(out_dir, "vec_normalize.pkl"))

    with open(os.path.join(out_dir, "first_success.json"), "w") as f:
        json.dump(
            {
                "reward_mode": reward_mode,
                "seed": seed,
                "success_distance": SUCCESS_DISTANCE,
                "first_success_timestep": first_success_cb.first_success_timestep,
            },
            f,
            indent=2,
        )

    train_env.close()
    eval_env.close()


# ── Orchestration ─────────────────────────────────────────────────────────────
def fmt_duration(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--timesteps", type=int, default=500_000,
                        help="Per-run training budget (default: 500000)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1],
                        help="Random seeds (default: 0 1)")
    parser.add_argument("--modes", type=str, nargs="+", default=list(REWARD_MODES),
                        choices=list(REWARD_MODES),
                        help="Reward modes to run (default: all five)")
    parser.add_argument("--task-difficulty", type=str, default="easy",
                        choices=list(DIFFICULTY_PRESETS),
                        help="easy = no obstacles (results/); "
                             "hard = simple obstacles (results_hard/)")
    parser.add_argument("--yes", action="store_true",
                        help="Skip the interactive confirmation prompt")
    args = parser.parse_args()

    # Apply the difficulty preset: obstacle mode + output root.
    global OBSTACLE_MODE, RESULTS_ROOT
    OBSTACLE_MODE, RESULTS_ROOT = DIFFICULTY_PRESETS[args.task_difficulty]

    # Build the full run plan (mode-major ordering for readable progress).
    plan = [(mode, seed) for mode in args.modes for seed in args.seeds]
    total = len(plan)
    done_already = [(m, s) for (m, s) in plan if is_complete(m, s)]
    todo = [(m, s) for (m, s) in plan if not is_complete(m, s)]

    # ── Configuration banner ──────────────────────────────────────────────────
    line = "=" * 64
    print(f"\n{line}")
    print("  REWARD-ABLATION STUDY — configuration")
    print(line)
    print(f"  Algorithm       : vanilla PPO (no safety constraint)")
    print(f"  Task difficulty : {args.task_difficulty}  "
          f"(obstacle_mode={OBSTACLE_MODE})")
    print(f"  Reward modes    : {', '.join(args.modes)}")
    print(f"  Seeds           : {', '.join(map(str, args.seeds))}")
    print(f"  Timesteps/run   : {args.timesteps:,}")
    print(f"  Total runs      : {total}  ({len(done_already)} already complete, "
          f"{len(todo)} to run)")
    print(f"  Eval frequency  : every {EVAL_FREQ:,} steps "
          f"({N_EVAL_EPISODES} episodes each)")
    print(f"  Output root     : {RESULTS_ROOT}/{{reward_mode}}_seed{{N}}/")
    print(line)
    if done_already:
        print("  Resuming — these completed runs will be skipped:")
        for m, s in done_already:
            print(f"    • {m}_seed{s}")
        print(line)

    if not todo:
        print("  Nothing to do — all requested runs are already complete.\n")
        return

    if not args.yes:
        try:
            ans = input(f"\n  Proceed with {len(todo)} training run(s)? "
                        f"This may take many hours. [y/N] ").strip().lower()
        except EOFError:
            ans = ""
        if ans not in ("y", "yes"):
            print("  Aborted — no training started.\n")
            return

    # ── Sequential training loop ────────────────────────────────────────────────
    durations: list[float] = []
    started_at = time.time()

    for run_idx, (mode, seed) in enumerate(plan, start=1):
        if is_complete(mode, seed):
            print(f"\nRun {run_idx}/{total}: reward_mode={mode}, seed={seed} "
                  f"— already complete, skipping.")
            continue

        # ETA from the mean duration of runs completed *this session*.
        if durations:
            mean_dur = sum(durations) / len(durations)
            remaining = sum(
                1 for j, (m, s) in enumerate(plan, start=1)
                if j >= run_idx and not is_complete(m, s)
            )
            eta = mean_dur * remaining
            eta_str = f"  | est. time remaining: {fmt_duration(eta)}"
        else:
            eta_str = "  | est. time remaining: (computing after first run)"

        print(f"\n{'-'*64}")
        print(f"Run {run_idx}/{total}: reward_mode={mode}, seed={seed}{eta_str}")
        print(f"{'-'*64}")

        t0 = time.time()
        train_one(mode, seed, args.timesteps, obstacle_mode=OBSTACLE_MODE)
        dur = time.time() - t0
        durations.append(dur)

        with open(os.path.join(run_dir(mode, seed), "run_meta.json"), "w") as f:
            json.dump(
                {"reward_mode": mode, "seed": seed,
                 "task_difficulty": args.task_difficulty,
                 "obstacle_mode": OBSTACLE_MODE,
                 "timesteps": args.timesteps, "duration_seconds": round(dur, 1)},
                f, indent=2,
            )

        print(f"  ✓ {mode}_seed{seed} finished in {fmt_duration(dur)}")

    print(f"\n{line}")
    print(f"  All runs complete. Total wall-clock: "
          f"{fmt_duration(time.time() - started_at)}")
    print(f"  Results under {RESULTS_ROOT}/")
    print(line + "\n")


if __name__ == "__main__":
    main()
