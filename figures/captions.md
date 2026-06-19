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
