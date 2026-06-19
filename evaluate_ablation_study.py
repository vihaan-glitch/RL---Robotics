"""
evaluate_ablation_study.py
===========================
Evaluate every trained model from the reward-ablation study and write a single
seed-level results table.

For each of the 10 runs under results/{reward_mode}_seed{N}/ this script:
  1. Loads final_model.zip together with its VecNormalize statistics.
  2. Runs N_EVAL_EPISODES (default 50) deterministic episodes — matching the
     protocol in evaluate.py (deterministic actions, norm_reward disabled so the
     reported rewards are the environment's true reward under that reward_mode).
  3. Records per run: success rate, mean/std episode reward, mean/std episode
     length, and mean/std final distance to the target.

Outputs
-------
  results/ablation_results.csv
      One row per individual run (seed-level granularity retained):
        reward_mode, seed, n_episodes, success_rate,
        mean_reward, std_reward, mean_length, std_length,
        mean_final_distance, std_final_distance

  Also prints a per-mode aggregate table (mean ± std across the 2 seeds).

Usage
-----
  python evaluate_ablation_study.py
  python evaluate_ablation_study.py --episodes 50
"""

import argparse
import os

import numpy as np
import pandas as pd

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from arm_reach_env import ArmReachEnv, REWARD_MODES

SUCCESS_DISTANCE = 0.05

# Task-difficulty presets: difficulty → (obstacle_mode, results_root).
# Must match run_ablation_study.DIFFICULTY_PRESETS so eval uses the same physics
# and reads from the matching results directory.
DIFFICULTY_PRESETS = {
    "easy": (None,     "results"),
    "hard": ("simple", "results_hard"),
}

# Set per --task-difficulty in main().
RESULTS_ROOT  = "results"
OBSTACLE_MODE = None


def run_dir(reward_mode: str, seed: int) -> str:
    return os.path.join(RESULTS_ROOT, f"{reward_mode}_seed{seed}")


# ── Single-run evaluation ─────────────────────────────────────────────────────
def evaluate_run(reward_mode: str, seed: int, n_episodes: int) -> dict | None:
    """Run n_episodes deterministic episodes for one trained model."""
    out_dir    = run_dir(reward_mode, seed)
    model_path = os.path.join(out_dir, "final_model")
    norm_path  = os.path.join(out_dir, "vec_normalize.pkl")

    if not os.path.exists(model_path + ".zip"):
        print(f"  [skip] {reward_mode}_seed{seed}: no final_model.zip")
        return None

    def make_env():
        # Same obstacle_mode as training so eval physics/penalty are consistent.
        return ArmReachEnv(render_mode=None, reward_mode=reward_mode,
                           obstacle_mode=OBSTACLE_MODE)

    env = DummyVecEnv([make_env])
    if os.path.exists(norm_path):
        env = VecNormalize.load(norm_path, env)
        env.training    = False
        env.norm_reward = False
    env.seed(seed + 50_000)  # fixed eval seed offset for reproducibility

    model = PPO.load(model_path, env=env)

    rewards, lengths, final_dists, successes, coll_rates = [], [], [], [], []
    obs = env.reset()
    for _ in range(n_episodes):
        ep_reward, ep_len, done = 0.0, 0, False
        ep_collisions, last_distance = 0, np.inf
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done_arr, info_arr = env.step(action)
            ep_reward    += float(reward[0])
            ep_len       += 1
            ep_collisions += int(info_arr[0].get("collision", 0.0) > 0)
            last_distance = float(info_arr[0].get("distance", np.inf))
            if done_arr[0]:
                done = True
                rewards.append(ep_reward)
                lengths.append(ep_len)
                final_dists.append(last_distance)
                successes.append(last_distance < SUCCESS_DISTANCE)
                coll_rates.append(ep_collisions / max(ep_len, 1))
                obs = env.reset()
    env.close()

    return {
        "reward_mode":         reward_mode,
        "seed":                seed,
        "n_episodes":          n_episodes,
        "success_rate":        float(np.mean(successes)),
        "mean_reward":         float(np.mean(rewards)),
        "std_reward":          float(np.std(rewards)),
        "mean_length":         float(np.mean(lengths)),
        "std_length":          float(np.std(lengths)),
        "mean_final_distance": float(np.mean(final_dists)),
        "std_final_distance":  float(np.std(final_dists)),
        "mean_collision_rate": float(np.mean(coll_rates)),  # frac of steps w/ contact
        # Raw per-episode rewards (kept for the box-plot figure; not written to CSV).
        "_episode_rewards":    rewards,
    }


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=50,
                        help="Deterministic eval episodes per run (default: 50)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--modes", type=str, nargs="+", default=list(REWARD_MODES),
                        choices=list(REWARD_MODES))
    parser.add_argument("--task-difficulty", type=str, default="easy",
                        choices=list(DIFFICULTY_PRESETS),
                        help="easy = results/, hard = results_hard/ (with obstacles)")
    args = parser.parse_args()

    global OBSTACLE_MODE, RESULTS_ROOT
    OBSTACLE_MODE, RESULTS_ROOT = DIFFICULTY_PRESETS[args.task_difficulty]
    csv_path = os.path.join(RESULTS_ROOT, "ablation_results.csv")

    rows, episode_reward_records = [], []
    for mode in args.modes:
        for seed in args.seeds:
            print(f"Evaluating {mode}_seed{seed} for {args.episodes} episodes...")
            res = evaluate_run(mode, seed, args.episodes)
            if res is None:
                continue
            # Persist per-episode rewards separately for the box-plot figure.
            for ep_idx, r in enumerate(res.pop("_episode_rewards")):
                episode_reward_records.append(
                    {"reward_mode": mode, "seed": seed,
                     "episode": ep_idx, "episode_reward": r}
                )
            rows.append(res)
            print(f"    success_rate={res['success_rate']:.1%}  "
                  f"mean_reward={res['mean_reward']:.2f}  "
                  f"mean_final_dist={res['mean_final_distance']:.3f}")

    if not rows:
        print("\nNo trained runs found — run run_ablation_study.py first.")
        return

    df = pd.DataFrame(rows)
    os.makedirs(RESULTS_ROOT, exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"\nWrote seed-level results → {csv_path}  ({len(df)} rows)")

    # Per-episode rewards (long format) for the box-plot figure.
    ep_csv = os.path.join(RESULTS_ROOT, "ablation_episode_rewards.csv")
    pd.DataFrame(episode_reward_records).to_csv(ep_csv, index=False)
    print(f"Wrote per-episode rewards → {ep_csv}")

    # ── Aggregate across seeds (mean ± std per reward_mode) ────────────────────
    agg = (
        df.groupby("reward_mode")
          .agg(success_rate_mean=("success_rate", "mean"),
               success_rate_std=("success_rate", "std"),
               reward_mean=("mean_reward", "mean"),
               reward_std=("mean_reward", "std"),
               final_dist_mean=("mean_final_distance", "mean"))
          .reindex([m for m in REWARD_MODES if m in df["reward_mode"].values])
    )
    print(f"\n{'='*72}")
    print("  PER-MODE AGGREGATE (mean ± std across seeds)")
    print('='*72)
    print(f"  {'reward_mode':<16}{'success':<18}{'mean_reward':<22}{'final_dist':<10}")
    for mode, r in agg.iterrows():
        sr_std = 0.0 if np.isnan(r.success_rate_std) else r.success_rate_std
        rw_std = 0.0 if np.isnan(r.reward_std) else r.reward_std
        success_col = f"{r.success_rate_mean:.1%} ± {sr_std:.1%}"
        reward_col  = f"{r.reward_mean:.2f} ± {rw_std:.2f}"
        print(f"  {mode:<16}{success_col:<18}{reward_col:<22}{r.final_dist_mean:.3f}")
    print('='*72 + "\n")


if __name__ == "__main__":
    main()
