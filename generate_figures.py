"""
generate_figures.py
====================
Publication-quality figures for the reward-ablation study.

Reads:
  results/ablation_results.csv            (seed-level summary metrics)
  results/ablation_episode_rewards.csv    (per-episode eval rewards; box plot)
  results/{mode}_seed{N}/evaluations.npz  (periodic eval log from EvalCallback —
                                           the same data mirrored to TensorBoard)
  results/{mode}_seed{N}/first_success.json

Writes 300-DPI PNGs to figures/ plus a caption file figures/captions.md:
  fig1_learning_curves.png   eval reward vs timestep, all modes, ±1 SD shading
  fig2_success_rate.png      final success rate per mode (sorted desc), error bars
  fig3_first_success.png     timestep of first success per mode (sorted asc)
  fig4_efficiency_vs_success.png   sample-efficiency vs final success scatter
  fig5_reward_boxplot.png    distribution of final eval-episode rewards per mode

A single fixed color palette maps each reward_mode to one color across all
figures.

Usage
-----
  python generate_figures.py
"""

import glob
import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from arm_reach_env import REWARD_MODES

RESULTS_ROOT      = "results"       # easy task
RESULTS_ROOT_HARD = "results_hard"  # hard task (obstacles)
FIG_DIR           = "figures"
BUDGET            = 500_000         # training budget (for never-succeeded fill)

# ── Fixed, colorblind-friendly palette: reward_mode → color (consistent everywhere) ──
PALETTE = {
    "full":            "#0072B2",  # blue
    "sparse":          "#D55E00",  # vermillion
    "no_progress":     "#009E73",  # green
    "no_time_penalty": "#CC79A7",  # purple
    "distance_only":   "#E69F00",  # orange
}
# Human-readable labels for axes/legends.
LABELS = {
    "full":            "full",
    "sparse":          "sparse",
    "no_progress":     "no_progress",
    "no_time_penalty": "no_time_penalty",
    "distance_only":   "distance_only",
}


# ── Global publication styling (overrides default matplotlib look) ────────────
def set_style() -> None:
    plt.rcParams.update({
        "figure.dpi":          300,
        "savefig.dpi":         300,
        "savefig.bbox":        "tight",
        "font.family":         "DejaVu Sans",
        "font.size":           11,
        "axes.titlesize":      13,
        "axes.titleweight":    "bold",
        "axes.labelsize":      12,
        "axes.spines.top":     False,
        "axes.spines.right":   False,
        "axes.linewidth":      0.9,
        "axes.grid":           True,
        "grid.color":          "#cccccc",
        "grid.linewidth":      0.6,
        "grid.alpha":          0.6,
        "legend.frameon":      False,
        "legend.fontsize":     10,
        "xtick.labelsize":     10,
        "ytick.labelsize":     10,
        "figure.facecolor":    "white",
        "axes.facecolor":      "white",
    })


def present_modes(df: pd.DataFrame) -> list[str]:
    """Reward modes present in the data, in canonical REWARD_MODES order."""
    return [m for m in REWARD_MODES if m in df["reward_mode"].values]


# ── Data loaders ──────────────────────────────────────────────────────────────
def discover_seeds(results_root: str, mode: str) -> list[int]:
    """All seeds present for a mode under results_root (auto-detected, sorted).

    The easy and hard studies may have different seed counts (e.g. 2 vs 5), so
    seeds are discovered from the run directories rather than assumed.
    """
    seeds = []
    for d in glob.glob(os.path.join(results_root, f"{mode}_seed*")):
        suffix = os.path.basename(d).rsplit("_seed", 1)[-1]
        if suffix.isdigit():
            seeds.append(int(suffix))
    return sorted(seeds)


def load_learning_curves(results_root: str = RESULTS_ROOT) -> dict[str, dict]:
    """
    For each mode return aligned eval timesteps and per-seed mean eval rewards.
    Returns {mode: {"timesteps": (T,), "rewards": (n_seeds, T)}}.
    """
    curves = {}
    for mode in REWARD_MODES:
        per_seed_ts, per_seed_rew = [], []
        for seed in discover_seeds(results_root, mode):
            npz = os.path.join(results_root, f"{mode}_seed{seed}", "evaluations.npz")
            if not os.path.exists(npz):
                continue
            data = np.load(npz)
            per_seed_ts.append(data["timesteps"])
            per_seed_rew.append(data["results"].mean(axis=1))  # mean over eval eps
        if not per_seed_rew:
            continue
        # Align on the shortest curve (eval grids are identical in practice).
        n = min(len(r) for r in per_seed_rew)
        curves[mode] = {
            "timesteps": per_seed_ts[0][:n],
            "rewards":   np.vstack([r[:n] for r in per_seed_rew]),
        }
    return curves


def load_first_success(results_root: str = RESULTS_ROOT) -> pd.DataFrame:
    rows = []
    for mode in REWARD_MODES:
        for seed in discover_seeds(results_root, mode):
            p = os.path.join(results_root, f"{mode}_seed{seed}", "first_success.json")
            if not os.path.exists(p):
                continue
            d = json.load(open(p))
            rows.append({"reward_mode": mode, "seed": seed,
                         "first_success_timestep": d.get("first_success_timestep")})
    return pd.DataFrame(rows)


# ── Figures ───────────────────────────────────────────────────────────────────
def fig_learning_curves(curves: dict) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for mode in REWARD_MODES:
        if mode not in curves:
            continue
        ts   = curves[mode]["timesteps"]
        rew  = curves[mode]["rewards"]
        mean = rew.mean(axis=0)
        std  = rew.std(axis=0)
        ax.plot(ts, mean, color=PALETTE[mode], label=LABELS[mode], linewidth=2)
        ax.fill_between(ts, mean - std, mean + std, color=PALETTE[mode], alpha=0.18,
                        linewidth=0)
    ax.set_xlabel("Training timestep")
    ax.set_ylabel("Mean evaluation reward")
    ax.set_title("Learning curves by reward configuration")
    ax.legend(title="reward_mode", loc="best")
    ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    fig.savefig(os.path.join(FIG_DIR, "fig1_learning_curves.png"))
    plt.close(fig)


def fig_success_rate(df: pd.DataFrame) -> None:
    g = (df.groupby("reward_mode")["success_rate"]
           .agg(["mean", "std"]).fillna(0.0))
    g = g.reindex(present_modes(df)).sort_values("mean", ascending=False)
    modes = list(g.index)
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.bar(range(len(modes)), g["mean"] * 100, yerr=g["std"] * 100,
           color=[PALETTE[m] for m in modes], capsize=5, edgecolor="black",
           linewidth=0.7, error_kw={"elinewidth": 1.2})
    ax.set_xticks(range(len(modes)))
    ax.set_xticklabels([LABELS[m] for m in modes], rotation=20, ha="right")
    ax.set_ylabel("Final success rate (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Final success rate by reward configuration")
    fig.savefig(os.path.join(FIG_DIR, "fig2_success_rate.png"))
    plt.close(fig)


def fig_first_success(fs: pd.DataFrame) -> None:
    # Treat "never succeeded" (None) as the full budget for plotting honesty.
    BUDGET = 500_000
    fs = fs.copy()
    fs["ts"] = fs["first_success_timestep"].fillna(BUDGET)
    g = fs.groupby("reward_mode")["ts"].agg(["mean", "std"]).fillna(0.0)
    g = g.reindex([m for m in REWARD_MODES if m in fs["reward_mode"].values])
    g = g.sort_values("mean", ascending=True)
    modes = list(g.index)
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.bar(range(len(modes)), g["mean"], yerr=g["std"],
           color=[PALETTE[m] for m in modes], capsize=5, edgecolor="black",
           linewidth=0.7, error_kw={"elinewidth": 1.2})
    ax.set_xticks(range(len(modes)))
    ax.set_xticklabels([LABELS[m] for m in modes], rotation=20, ha="right")
    ax.set_ylabel("Timestep of first success (lower = more efficient)")
    ax.set_title("Sample efficiency by reward configuration")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    fig.savefig(os.path.join(FIG_DIR, "fig3_first_success.png"))
    plt.close(fig)


def fig_efficiency_vs_success(df: pd.DataFrame, fs: pd.DataFrame) -> None:
    """
    Paired (twin-axis) bar chart comparing the two metrics' rankings.

    A scatter is unreadable here because the dense modes have near-identical
    first-success timesteps (their points coincide). Paired bars sorted by final
    success rate make rank (dis)agreement obvious: if the first-success bars are
    NOT monotonically increasing left→right, the two metrics rank modes
    differently — the question under study.
    """
    BUDGET = 500_000
    fs = fs.copy()
    fs["ts"] = fs["first_success_timestep"].fillna(BUDGET)
    eff = fs.groupby("reward_mode")["ts"].agg(["mean", "std"]).fillna(0.0)
    suc = (df.groupby("reward_mode")["success_rate"]
             .agg(["mean", "std"]).fillna(0.0)) * 100
    # Order by final success rate (descending) — the reference ranking.
    order = [m for m in suc.sort_values("mean", ascending=False).index
             if m in eff.index]

    x = np.arange(len(order))
    w = 0.4
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()

    b1 = ax1.bar(x - w/2, [suc.loc[m, "mean"] for m in order], width=w,
                 yerr=[suc.loc[m, "std"] for m in order],
                 color=[PALETTE[m] for m in order], edgecolor="black",
                 linewidth=0.7, capsize=4, error_kw={"elinewidth": 1.0})
    b2 = ax2.bar(x + w/2, [eff.loc[m, "mean"] for m in order], width=w,
                 yerr=[eff.loc[m, "std"] for m in order],
                 color=[PALETTE[m] for m in order], edgecolor="black",
                 linewidth=0.7, hatch="///", alpha=0.55, capsize=4,
                 error_kw={"elinewidth": 1.0})

    ax1.set_ylabel("Final success rate (%)  — solid bars")
    ax2.set_ylabel("Timestep of first success — hatched bars")
    ax1.set_ylim(0, 109)
    ax2.set_ylim(0, max(eff["mean"] + eff["std"]) * 1.15)
    ax2.grid(False)
    ax1.set_xticks(x)
    ax1.set_xticklabels([LABELS[m] for m in order], rotation=20, ha="right")
    ax1.set_title("Sample efficiency vs. final success rate\n"
                  "(modes ordered by success rate; non-monotone hatched bars = "
                  "ranking disagreement)", fontsize=11)
    # Legend mapping the two encodings.
    from matplotlib.patches import Patch
    handles = [Patch(facecolor="0.4", edgecolor="black", label="Final success rate (left axis)"),
               Patch(facecolor="0.4", edgecolor="black", hatch="///", alpha=0.55,
                     label="First-success timestep (right axis)")]
    ax1.legend(handles=handles, loc="center right")
    fig.savefig(os.path.join(FIG_DIR, "fig4_efficiency_vs_success.png"))
    plt.close(fig)


def fig_reward_boxplot(ep_df: pd.DataFrame) -> None:
    modes = [m for m in REWARD_MODES if m in ep_df["reward_mode"].values]
    data  = [ep_df.loc[ep_df["reward_mode"] == m, "episode_reward"].values
             for m in modes]
    fig, ax = plt.subplots(figsize=(7.5, 5))
    bp = ax.boxplot(data, patch_artist=True, widths=0.6,
                    medianprops={"color": "black", "linewidth": 1.4},
                    flierprops={"marker": "o", "markersize": 3,
                                "markerfacecolor": "none", "alpha": 0.5})
    for patch, m in zip(bp["boxes"], modes):
        patch.set_facecolor(PALETTE[m])
        patch.set_alpha(0.75)
        patch.set_edgecolor("black")
    ax.set_xticklabels([LABELS[m] for m in modes], rotation=20, ha="right")
    ax.set_ylabel("Final eval-episode reward")
    ax.set_title("Distribution of eval-episode rewards by reward configuration")
    fig.savefig(os.path.join(FIG_DIR, "fig5_reward_boxplot.png"))
    plt.close(fig)


# ── Easy-vs-Hard comparison figures ───────────────────────────────────────────
# Convention across all comparison figures: color encodes reward_mode (shared
# palette); EASY is drawn solid/lighter, HARD is drawn dashed/hatched/darker.

def _agg_success(df):
    return (df.groupby("reward_mode")["success_rate"]
              .agg(["mean", "std"]).fillna(0.0))


def _agg_first_success(fs):
    fs = fs.copy()
    fs["ts"] = fs["first_success_timestep"].fillna(BUDGET)
    return fs.groupby("reward_mode")["ts"].agg(["mean", "std"]).fillna(0.0)


def fig_learning_curves_comparison(curves_easy: dict, curves_hard: dict) -> None:
    """Two panels (easy | hard), shared axes, same per-mode colors."""
    fig, (axe, axh) = plt.subplots(1, 2, figsize=(13, 5), sharex=True, sharey=True)
    for ax, curves, title in [(axe, curves_easy, "Easy (no obstacles)"),
                              (axh, curves_hard, "Hard (obstacles)")]:
        for mode in REWARD_MODES:
            if mode not in curves:
                continue
            ts, rew = curves[mode]["timesteps"], curves[mode]["rewards"]
            mean, std = rew.mean(axis=0), rew.std(axis=0)
            ax.plot(ts, mean, color=PALETTE[mode], label=LABELS[mode], linewidth=2)
            ax.fill_between(ts, mean - std, mean + std, color=PALETTE[mode],
                            alpha=0.18, linewidth=0)
        ax.set_title(title)
        ax.set_xlabel("Training timestep")
        ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    axe.set_ylabel("Mean evaluation reward")
    axh.legend(title="reward_mode", loc="best")
    fig.suptitle("Learning curves: easy vs. hard task", fontweight="bold",
                 fontsize=14)
    fig.savefig(os.path.join(FIG_DIR, "fig1_learning_curves_comparison.png"))
    plt.close(fig)


def fig_success_rate_comparison(df_easy, df_hard) -> None:
    """Grouped bars: easy vs hard final success rate per mode."""
    ge, gh = _agg_success(df_easy), _agg_success(df_hard)
    modes = [m for m in REWARD_MODES if m in ge.index or m in gh.index]
    x, w = np.arange(len(modes)), 0.4
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.bar(x - w/2, [ge.loc[m, "mean"]*100 if m in ge.index else 0 for m in modes],
           width=w, yerr=[ge.loc[m, "std"]*100 if m in ge.index else 0 for m in modes],
           color=[PALETTE[m] for m in modes], edgecolor="black", linewidth=0.7,
           capsize=4, label="Easy")
    ax.bar(x + w/2, [gh.loc[m, "mean"]*100 if m in gh.index else 0 for m in modes],
           width=w, yerr=[gh.loc[m, "std"]*100 if m in gh.index else 0 for m in modes],
           color=[PALETTE[m] for m in modes], edgecolor="black", linewidth=0.7,
           hatch="///", alpha=0.6, capsize=4, label="Hard")
    ax.set_xticks(x); ax.set_xticklabels([LABELS[m] for m in modes],
                                         rotation=20, ha="right")
    ax.set_ylabel("Final success rate (%)"); ax.set_ylim(0, 109)
    ax.set_title("Final success rate: easy (solid) vs. hard (hatched)")
    ax.legend(loc="best")
    fig.savefig(os.path.join(FIG_DIR, "fig2_success_rate_comparison.png"))
    plt.close(fig)


def fig_first_success_comparison(fs_easy, fs_hard) -> None:
    """Grouped bars: easy vs hard timestep of first success per mode."""
    ge, gh = _agg_first_success(fs_easy), _agg_first_success(fs_hard)
    modes = [m for m in REWARD_MODES if m in ge.index or m in gh.index]
    x, w = np.arange(len(modes)), 0.4
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.bar(x - w/2, [ge.loc[m, "mean"] if m in ge.index else 0 for m in modes],
           width=w, yerr=[ge.loc[m, "std"] if m in ge.index else 0 for m in modes],
           color=[PALETTE[m] for m in modes], edgecolor="black", linewidth=0.7,
           capsize=4, label="Easy")
    ax.bar(x + w/2, [gh.loc[m, "mean"] if m in gh.index else 0 for m in modes],
           width=w, yerr=[gh.loc[m, "std"] if m in gh.index else 0 for m in modes],
           color=[PALETTE[m] for m in modes], edgecolor="black", linewidth=0.7,
           hatch="///", alpha=0.6, capsize=4, label="Hard")
    ax.set_xticks(x); ax.set_xticklabels([LABELS[m] for m in modes],
                                         rotation=20, ha="right")
    ax.set_ylabel("Timestep of first success (lower = more efficient)")
    ax.set_title("Sample efficiency: easy (solid) vs. hard (hatched)")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.legend(loc="best")
    fig.savefig(os.path.join(FIG_DIR, "fig3_first_success_comparison.png"))
    plt.close(fig)


def fig_efficiency_vs_success_comparison(df_easy, fs_easy, df_hard, fs_hard) -> None:
    """
    Scatter: each mode has an easy point (circle) and a hard point (square)
    joined by a line, so the shift induced by task difficulty is visible per mode.
    """
    se, sh = _agg_success(df_easy)["mean"]*100, _agg_success(df_hard)["mean"]*100
    ee, eh = _agg_first_success(fs_easy)["mean"], _agg_first_success(fs_hard)["mean"]
    modes = [m for m in REWARD_MODES if m in se.index and m in sh.index]

    # Clip x so the dense-mode structure is visible; points at/over the budget
    # (i.e. "never succeeded") are pinned to the right edge and flagged.
    XMAX = 60_000
    def cx(v):  # clipped x
        return min(v, XMAX)

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for m in modes:
        xe, xh = cx(ee[m]), cx(eh[m])
        ax.plot([xe, xh], [se[m], sh[m]], color=PALETTE[m], linewidth=1.2,
                alpha=0.7, zorder=2)
        ax.scatter(xe, se[m], s=150, color=PALETTE[m], edgecolor="black",
                   linewidth=0.8, marker="o", zorder=3)
        ax.scatter(xh, sh[m], s=160, color=PALETTE[m], edgecolor="black",
                   linewidth=0.8, marker="s", zorder=3, label=LABELS[m])
        # Flag modes whose hard-task first-success is off-scale (never reached).
        if eh[m] >= XMAX:
            ax.annotate(f"{LABELS[m]} (never)", (xh, sh[m]),
                        textcoords="offset points", xytext=(-10, 8),
                        fontsize=8, ha="right")
        else:
            ax.annotate(LABELS[m], (xh, sh[m]), textcoords="offset points",
                        xytext=(8, -2), fontsize=8)
    ax.axvline(XMAX, color="0.6", linestyle=":", linewidth=1)
    ax.set_xlim(0, XMAX * 1.02)
    ax.set_ylim(-5, 109)
    ax.set_xlabel("Timestep of first success  (→ less sample-efficient; "
                  f"≥{XMAX//1000}k pinned to right edge)")
    ax.set_ylabel("Final success rate (%)")
    ax.set_title("Efficiency vs. success — easy (○) → hard (□) shift per mode")
    ax.legend(title="reward_mode", loc="center right", ncol=1)
    fig.savefig(os.path.join(FIG_DIR, "fig4_efficiency_vs_success_comparison.png"))
    plt.close(fig)


def fig_reward_boxplot_comparison(ep_easy, ep_hard) -> None:
    """Two panels (easy | hard) of per-episode reward distributions per mode."""
    fig, (axe, axh) = plt.subplots(1, 2, figsize=(13, 5))
    for ax, ep_df, title in [(axe, ep_easy, "Easy (no obstacles)"),
                             (axh, ep_hard, "Hard (obstacles)")]:
        if ep_df is None or ep_df.empty:
            ax.set_visible(False)
            continue
        modes = [m for m in REWARD_MODES if m in ep_df["reward_mode"].values]
        data = [ep_df.loc[ep_df["reward_mode"] == m, "episode_reward"].values
                for m in modes]
        bp = ax.boxplot(data, patch_artist=True, widths=0.6,
                        medianprops={"color": "black", "linewidth": 1.4},
                        flierprops={"marker": "o", "markersize": 3,
                                    "markerfacecolor": "none", "alpha": 0.5})
        for patch, m in zip(bp["boxes"], modes):
            patch.set_facecolor(PALETTE[m]); patch.set_alpha(0.75)
            patch.set_edgecolor("black")
        ax.set_xticklabels([LABELS[m] for m in modes], rotation=20, ha="right")
        ax.set_title(title)
    axe.set_ylabel("Final eval-episode reward")
    fig.suptitle("Eval-episode reward distributions: easy vs. hard task",
                 fontweight="bold", fontsize=14)
    fig.savefig(os.path.join(FIG_DIR, "fig5_reward_boxplot_comparison.png"))
    plt.close(fig)


# ── Captions ──────────────────────────────────────────────────────────────────
CAPTIONS = """\
# Figure captions — reward-ablation study

**Figure 1. Learning curves by reward configuration.** Mean evaluation reward
versus training timestep for each reward mode. Curves show the mean across two
random seeds; shaded bands denote ±1 SD across seeds. Evaluations were run every
5,000 timesteps (10 deterministic episodes each).

**Figure 2. Final success rate by reward configuration.** Fraction of 50
deterministic evaluation episodes reaching the target (end-effector within
0.05 m), averaged over two seeds and sorted in descending order. Error bars
denote ±1 SD across seeds.

**Figure 3. Sample efficiency by reward configuration.** Training timestep at
which the agent first reaches the target during learning (lower is more sample
efficient), averaged over two seeds and sorted in ascending order. Error bars
denote ±1 SD across seeds; configurations that never succeeded within the
500,000-step budget are plotted at the full budget.

**Figure 4. Sample efficiency versus final success rate.** Each point is one
reward mode, positioned by its mean timestep-to-first-success (x) and mean final
success rate (y). Divergence between the horizontal and vertical orderings
indicates that the two metrics rank the reward configurations differently — the
central question of this study.

**Figure 5. Distribution of evaluation-episode rewards by reward configuration.**
Box plots over all individual evaluation episodes (both seeds pooled) showing the
spread, not just the mean, of the learned policy's per-episode reward. Boxes span
the interquartile range, whiskers extend to 1.5×IQR, and points beyond are drawn
as outliers.

## Easy-vs-Hard comparison figures

Across all comparison figures, color encodes reward mode (shared palette); the
easy task (no obstacles) is drawn solid/circles and the hard task (three static
box obstacles) is drawn hatched/squares.

**Figure 1c. Learning curves, easy vs. hard.** Mean evaluation reward versus
training timestep, with the easy task (left panel) and hard task (right panel) on
shared axes. Shaded bands denote ±1 SD across two seeds.

**Figure 2c. Final success rate, easy vs. hard.** Grouped bars give each reward
mode's final success rate on the easy (solid) and hard (hatched) tasks. Error
bars denote ±1 SD across seeds.

**Figure 3c. Sample efficiency, easy vs. hard.** Grouped bars give each mode's
timestep of first success on the easy (solid) and hard (hatched) tasks; modes
that never succeeded within the 500,000-step budget are plotted at the full
budget. Error bars denote ±1 SD across seeds.

**Figure 4c. Efficiency vs. success shift under difficulty.** Each mode is shown
as an easy point (circle) and a hard point (square) joined by a line; the length
and direction of the line show how added task complexity moves that mode in the
sample-efficiency/final-success plane. Divergence in how modes move is the
signature of reward-shaping trade-offs emerging under complexity.

**Figure 5c. Reward distributions, easy vs. hard.** Per-episode evaluation reward
distributions for each mode on the easy (left) and hard (right) tasks, showing
how the spread and consistency of the learned policy change with difficulty.
"""


def main() -> None:
    set_style()
    os.makedirs(FIG_DIR, exist_ok=True)

    csv_path = os.path.join(RESULTS_ROOT, "ablation_results.csv")
    if not os.path.exists(csv_path):
        print(f"Missing {csv_path} — run evaluate_ablation_study.py first.")
        return
    df = pd.read_csv(csv_path)

    ep_path = os.path.join(RESULTS_ROOT, "ablation_episode_rewards.csv")
    ep_df = pd.read_csv(ep_path) if os.path.exists(ep_path) else None

    curves = load_learning_curves()
    fs     = load_first_success()

    if curves:
        fig_learning_curves(curves)
        print("  ✓ fig1_learning_curves.png")
    fig_success_rate(df)
    print("  ✓ fig2_success_rate.png")
    if not fs.empty:
        fig_first_success(fs)
        print("  ✓ fig3_first_success.png")
        fig_efficiency_vs_success(df, fs)
        print("  ✓ fig4_efficiency_vs_success.png")
    if ep_df is not None and not ep_df.empty:
        fig_reward_boxplot(ep_df)
        print("  ✓ fig5_reward_boxplot.png")

    # ── Easy-vs-hard comparison figures (only if hard results are present) ──────
    hard_csv = os.path.join(RESULTS_ROOT_HARD, "ablation_results.csv")
    if os.path.exists(hard_csv):
        print("\nHard-task results found — generating comparison figures:")
        df_hard = pd.read_csv(hard_csv)
        ep_hard_path = os.path.join(RESULTS_ROOT_HARD, "ablation_episode_rewards.csv")
        ep_hard = pd.read_csv(ep_hard_path) if os.path.exists(ep_hard_path) else None
        curves_hard = load_learning_curves(RESULTS_ROOT_HARD)
        fs_hard     = load_first_success(RESULTS_ROOT_HARD)

        if curves and curves_hard:
            fig_learning_curves_comparison(curves, curves_hard)
            print("  ✓ fig1_learning_curves_comparison.png")
        fig_success_rate_comparison(df, df_hard)
        print("  ✓ fig2_success_rate_comparison.png")
        if not fs.empty and not fs_hard.empty:
            fig_first_success_comparison(fs, fs_hard)
            print("  ✓ fig3_first_success_comparison.png")
            fig_efficiency_vs_success_comparison(df, fs, df_hard, fs_hard)
            print("  ✓ fig4_efficiency_vs_success_comparison.png")
        fig_reward_boxplot_comparison(ep_df, ep_hard)
        print("  ✓ fig5_reward_boxplot_comparison.png")
    else:
        print(f"\n(No hard-task results at {hard_csv} — skipping comparison figures.)")

    with open(os.path.join(FIG_DIR, "captions.md"), "w") as f:
        f.write(CAPTIONS)
    print(f"  ✓ captions.md\n\nAll figures written to {FIG_DIR}/")


if __name__ == "__main__":
    main()
