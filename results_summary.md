# Reward-Ablation Study — Results Summary

**Task.** 7-DOF Kuka-style arm reaching 3D targets (MuJoCo), trained with vanilla
PPO (no safety constraint). Five reward configurations, 2 random seeds each
(10 runs), 500,000 timesteps per run, identical PPO hyperparameters. Each model
was evaluated over 50 deterministic episodes.

**Central question.** *Do sample efficiency (timestep of first success) and final
success rate rank the reward configurations differently?*

## Key numbers

Aggregated across the two seeds (mean ± SD). Success = end-effector within
0.05 m of the target.

| reward_mode | Final success rate | Mean eval reward | First success (timestep) | Mean ep. length | Mean final dist (m) |
|---|---|---|---|---|---|
| full | 100.0% ± 0.0% | 104.09 ± 0.12 | 10,280 ± 1,803 | 10.3 | 0.033 |
| no_progress | 100.0% ± 0.0% | 95.11 ± 0.24 | 10,280 ± 1,804 | 10.3 | 0.031 |
| no_time_penalty | 99.0% ± 1.4% | 102.45 ± 2.62 | 10,280 ± 1,802 | 19.4 | 0.035 |
| distance_only | 100.0% ± 0.0% | −4.68 ± 0.21 | 10,279 ± 1,803 | 10.3 | 0.035 |
| sparse | 19.0% ± 24.0% | 19.00 ± 24.04 | 45,170 ± 37,074 | 414.1 | 0.430 |

*(Reward magnitudes are not comparable across modes — each mode optimizes a
different reward function. Only success rate, episode length, and final distance
are cross-comparable; reward is reported for completeness.)*

## Answer to the central question

**No — in this experiment the two metrics rank the conditions concordantly, not
differently.** Both sample efficiency and final success rate produce the same
ordering: the four dense-reward variants (`full`, `no_progress`,
`no_time_penalty`, `distance_only`) are statistically indistinguishable —
~100% success and first success at ~10,300 timesteps for all of them — while
`sparse` is the clear loser on **both** axes (19% success, and first success at
~45,000 timesteps with very high seed variance). There is no configuration that
is sample-efficient but ultimately poor, or slow to first success but ultimately
strong. The rankings agree (see Figure 4: the hatched first-success bars are
monotone with the solid success-rate bars).

## What actually drives the result

The discriminating factor is not *which* shaping terms are present but whether a
**dense distance signal** exists at all:

- **`distance_only` matches `full` on both metrics** (100% success, identical
  first-success timestep) despite stripping out the progress reward, success
  bonus, and time penalty. For this task, the dense `−distance` term alone is
  sufficient for both fast learning and high final performance; the other three
  terms are not necessary.
- **`sparse` is the only configuration that degrades performance**, because it
  removes the dense distance signal entirely. With reward only at the goal, the
  agent must stumble onto success by exploration, which is slow and unreliable —
  reflected in its enormous seed variance (one seed first succeeded at 8,096
  steps, the other at 82,243; final success 2% vs 36%).
- **The time penalty affects *speed*, not *success*.** `no_time_penalty` reaches
  the goal just as reliably (99–100%) and just as quickly to first success, but
  its episodes are ~2× longer (19.4 vs 10.3 steps) — without the per-step
  penalty the policy has no incentive to reach the target quickly.

## Takeaway

For this reaching task, reward shaping is effectively binary: any dense distance
feedback yields fast, reliable learning; removing it (the `sparse` case) is the
only manipulation that hurts. Because the single weak configuration is weak on
*both* sample efficiency and final success, the two metrics do not separate the
conditions here. A study where these metrics diverge would likely require a
harder task (e.g., obstacles, longer horizons, or sparser goals) where shaping
terms trade off against one another rather than all saturating at ceiling.

---

# Comparison: Easy vs. Hard Task

The prediction at the end of the easy-task study — that a harder task would make
the reward-shaping terms trade off — was tested directly. The **hard task** adds
three fixed static box obstacles to the workspace (`obstacle_mode="simple"`),
with a uniform −10 collision penalty applied identically across all reward modes.
Targets are resampled to remain in free space, so only the *path* is obstructed.
The hard-task ablation was run at **two seeds first, then extended to five seeds
per mode (25 runs)** for statistical robustness. Easy-task results are unchanged
(2 seeds) and stored under `results/`; hard under `results_hard/`.

> **Headline correction.** A compelling apparent result at n=2 — that the
> reward-mode rankings *changed* under difficulty and that sample efficiency and
> final success *diverged* — **did not survive the extension to n=5.** With five
> seeds the four dense modes are statistically indistinguishable on the hard task,
> just as they were on the easy task. The n=2 "divergence" was seed noise. This is
> reported in full below because it is itself the most important methodological
> finding of the study.

## Side-by-side numbers (hard = 5 seeds, easy = 2 seeds)

50 deterministic eval episodes per run. "First success" = mean training timestep
of first goal reach (≈500k ⇒ rarely/never reached within budget). ± is SD across
seeds.

| reward_mode | Success (easy) | Success (hard, n=5) | First success (hard, n=5) | Collision rate (hard) |
|---|---|---|---|---|
| full | 100% | 14.4% ± 6.2% | 24,828 ± 6,667 | 2.2% |
| no_progress | 100% | 14.4% ± 4.8% | 31,123 ± 16,033 | 0.2% |
| no_time_penalty | 99% | 13.6% ± 5.7% | 20,934 ± 2,905 | 1.0% |
| distance_only | 100% | 14.0% ± 6.9% | 25,666 ± 11,473 | 0.0% |
| sparse | 19% | 1.2% ± 2.7% | 455,792 ± 88,416 | 0.0% |

## Did the reward-mode rankings change? **No — not robustly.**

**Ranking by final success rate:**
- Easy (n=2):  `full` = `no_progress` = `distance_only` (100%) > `no_time_penalty` (99%) ≫ `sparse` (19%)
- Hard (n=5):  `full` = `no_progress` (14.4%) ≈ `distance_only` (14.0%) ≈ `no_time_penalty` (13.6%) ≫ `sparse` (1.2%)

On the hard task the four dense modes span just **13.6–14.4%**, while their
per-seed SDs are **±5–7 percentage points** — i.e. the between-mode differences
are far smaller than the within-mode seed scatter. They are not distinguishable.
The **only robust effect, present at both difficulties, is that any dense distance
signal massively beats `sparse`** (which collapses from 19% on easy to ~1% on
hard, essentially never finding the goal behind the obstacles).

So the harder task did **not** separate the dense shaping variants. It lowered the
ceiling for all of them roughly equally (~100% → ~14%) without changing their
relative ordering in any statistically supportable way.

## What the n=2 run had claimed, and why it was wrong

At two seeds the hard task appeared to show `no_progress` as the clear winner
(16% vs `full` 13%) and a divergence between sample efficiency and final success.
Extending to five seeds erased it: `full` recovered to tie for the top (its n=2
value of 13% sat at the low end of a wide 5-seed spread, ~6–24%), and the
`no_progress` "lead" shrank to a 0.0-percentage-point tie. With only two seeds, a
single lucky/unlucky run moved a mode several ranks — exactly the failure mode
that motivated adding seeds.

The one mild signal that *does* persist at n=5: `no_time_penalty` reaches its
first success both fastest and most consistently on the hard task
(20,934 ± 2,905 steps), yet this earlier first contact does **not** translate into
higher final success. Sample efficiency and final success are therefore weakly
*decoupled* on the hard task — efficiency separates the modes slightly while final
success does not — but this is a minor effect, not the strong ranking inversion
the n=2 data falsely suggested.

## Takeaway

Two lessons, one about the task and one about method:

1. **Task.** Across both difficulties the reward-shaping ablation has a single
   robust conclusion: dense distance feedback (in any of its four tested forms) is
   what matters; the presence or absence of the progress, time, or
   success-bonus terms does not measurably change final success on this reaching
   task, even with obstacles. The hoped-for trade-off between shaping terms did
   not materialise at adequate sample size.

2. **Method (the real headline).** The exciting n=2 result — changed rankings and
   diverging metrics — was a statistical artifact. It took five seeds to see that
   the four dense modes are equivalent on the hard task too. Reward-ablation
   conclusions on noisy, low-success regimes require enough seeds to separate
   signal from seed variance; two is not enough. Had the study stopped at n=2, it
   would have reported a finding that does not replicate.

If a genuine reward-shaping trade-off is the goal, the evidence here suggests the
obstacle task as configured is still not the right probe (it depresses all dense
modes uniformly). A task where the shaping terms are made to actively conflict —
e.g. a tight time budget pitted against a detour-forcing obstacle, or a much
larger collision penalty — would be a better next step, again run at ≥5 seeds.

---

*Easy figures `figures/figN_*.png`; easy-vs-hard comparisons
`figures/figN_*_comparison.png` (captions in `figures/captions.md`). Seed-level
metrics in `results/ablation_results.csv` (easy, 2 seeds) and
`results_hard/ablation_results.csv` (hard, 5 seeds).*
