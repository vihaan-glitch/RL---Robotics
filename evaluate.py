"""
evaluate.py
===========
Evaluate a trained PPO or PPO-Lagrangian model on the Kuka IIWA arm reaching task.

Usage
-----
  # Evaluate PPO-Lagrangian (default)
  python evaluate.py --algo ppo_lag

  # Evaluate vanilla PPO baseline
  python evaluate.py --algo ppo

  # Compare both side by side
  python evaluate.py --compare

  # Visual playback (works with --compare too)
  python evaluate.py --algo ppo_lag --render
  python evaluate.py --compare --render

Outputs
-------
  Prints a statistics table with:
    - Success rate
    - Average reward
    - Average episode cost
    - Constraint violation rate  (episodes where cost > COST_LIMIT)
"""

import argparse
import os
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from arm_reach_env import ArmReachEnv, COST_LIMIT

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--algo",     type=str,  default="ppo_lag",
                    choices=["ppo_lag", "ppo"])
parser.add_argument("--episodes", type=int,  default=50)
parser.add_argument("--render",   action="store_true",
                    help="Show PyBullet GUI during evaluation")
parser.add_argument("--compare",  action="store_true",
                    help="Evaluate both ppo and ppo_lag and print side-by-side table")
args = parser.parse_args()


# ── Core evaluation function ──────────────────────────────────────────────────
def evaluate_model(algo: str, n_episodes: int, render: bool) -> dict:
    """
    Load a saved model and run n_episodes episodes.

    Returns a dict with keys:
      success_rate, mean_reward, std_reward, mean_cost,
      std_cost, violation_rate, mean_length
    """
    model_dir = f"models/{algo}"
    render_mode = "human" if render else None

    def make_env():
        return ArmReachEnv(render_mode=render_mode)

    env = DummyVecEnv([make_env])

    # Load normalisation stats if available
    norm_path = os.path.join(model_dir, "vec_normalize.pkl")
    if os.path.exists(norm_path):
        env = VecNormalize.load(norm_path, env)
        env.training   = False
        env.norm_reward = False
        print(f"  [{algo}] Loaded normalisation stats from {norm_path}")
    else:
        print(f"  [{algo}] Warning: no vec_normalize.pkl found — proceeding without normalisation")

    # Load model (only under models/<algo>/ — never fall back to another algorithm)
    candidates = [
        os.path.join(model_dir, "final_model"),
        os.path.join(model_dir, "best_model", "best_model"),
        os.path.join(model_dir, f"{algo}_arm_reach_final"),
    ]
    if algo == "ppo":
        candidates.append(os.path.join(model_dir, "ppo_arm_reach"))

    def _exists(p: str) -> bool:
        return os.path.exists(p + ".zip") or os.path.exists(p)

    model = None
    for path in candidates:
        if _exists(path):
            model = PPO.load(path, env=env)
            print(f"  [{algo}] Loaded model from {path}")
            break
    if model is None:
        raise FileNotFoundError(
            f"No model found for algo='{algo}' under {model_dir}. "
            f"Tried: {candidates}"
        )

    # ── Run episodes ──────────────────────────────────────────────────────────
    rewards     = []
    costs       = []
    lengths     = []
    successes   = []

    obs = env.reset()

    for ep in range(n_episodes):
        ep_reward = 0.0
        ep_cost   = 0.0
        ep_length = 0
        done      = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done_arr, info_arr = env.step(action)

            ep_reward += float(reward[0])
            ep_cost   += float(info_arr[0].get("cost", 0.0))
            ep_length += 1

            if done_arr[0]:
                done = True
                success = info_arr[0].get("distance", 1.0) < 0.05
                successes.append(success)

                status = "SUCCESS" if success else "failed "
                print(
                    f"  [{algo}] Ep {ep+1:>3}/{n_episodes}  "
                    f"{status}  reward={ep_reward:>8.2f}  "
                    f"cost={ep_cost:>5.1f}  steps={ep_length}"
                )

                rewards.append(ep_reward)
                costs.append(ep_cost)
                lengths.append(ep_length)

                obs = env.reset()

    env.close()

    violation_rate = np.mean([c > COST_LIMIT for c in costs])

    return {
        "algo":           algo,
        "n_episodes":     n_episodes,
        "success_rate":   np.mean(successes),
        "mean_reward":    np.mean(rewards),
        "std_reward":     np.std(rewards),
        "mean_cost":      np.mean(costs),
        "std_cost":       np.std(costs),
        "violation_rate": violation_rate,
        "mean_length":    np.mean(lengths),
    }


# ── Print results table ───────────────────────────────────────────────────────
def print_results(results: list[dict]):
    line = "=" * 60
    print(f"\n{line}")
    print("  EVALUATION RESULTS")
    print(line)

    headers = [
        ("Algorithm",             lambda r: r["algo"].upper()),
        ("Episodes",              lambda r: str(r["n_episodes"])),
        ("Success Rate",          lambda r: f"{r['success_rate']:.1%}"),
        ("Avg Reward",            lambda r: f"{r['mean_reward']:.2f} ± {r['std_reward']:.2f}"),
        ("Avg Episode Cost",      lambda r: f"{r['mean_cost']:.2f} ± {r['std_cost']:.2f}"),
        (f"Violation Rate (>{COST_LIMIT})", lambda r: f"{r['violation_rate']:.1%}"),
        ("Avg Episode Length",    lambda r: f"{r['mean_length']:.1f}"),
    ]

    col_w = 32
    for label, fn in headers:
        row = f"  {label:<{col_w}}"
        for r in results:
            row += f"  {fn(r):<18}"
        print(row)

    print(line)

    # Highlight safety improvement if comparing
    if len(results) == 2:
        baseline = next((r for r in results if r["algo"] == "ppo"), None)
        safe     = next((r for r in results if r["algo"] == "ppo_lag"), None)
        if baseline and safe:
            delta_viol = baseline["violation_rate"] - safe["violation_rate"]
            delta_cost = baseline["mean_cost"]       - safe["mean_cost"]
            print(f"\n  Safety improvement (PPO → PPO-Lag):")
            print(f"    Violation rate reduction : {delta_viol:+.1%}")
            print(f"    Episode cost reduction   : {delta_cost:+.2f}")
            print(line)


# ── Main ──────────────────────────────────────────────────────────────────────
if args.compare:
    algos = ["ppo", "ppo_lag"]
    all_results = []
    for algo in algos:
        print(f"\nEvaluating {algo.upper()} for {args.episodes} episodes...")
        try:
            r = evaluate_model(algo, args.episodes, render=args.render)
            all_results.append(r)
        except FileNotFoundError as e:
            print(f"  Skipped: {e}")
    print_results(all_results)
else:
    print(f"\nEvaluating {args.algo.upper()} for {args.episodes} episodes...")
    result = evaluate_model(args.algo, args.episodes, render=args.render)
    print_results([result])