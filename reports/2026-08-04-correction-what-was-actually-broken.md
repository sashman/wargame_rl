# Correction: the policy was never trained, and most earlier conclusions do not hold

**Date:** 2026-08-04
**Supersedes:** large parts of
[the reward-phase curriculum report](2026-08-04-reward-phase-curriculum.md)
**Config:** `examples/env_config/25v25_scripted_opponent_random_objectives_3_reward_phases_terrain.yaml`

## The measurement that reframes everything

Seven runs were spent tuning reward weights and curriculum thresholds. None of them asked
what a trivial hand-written policy scores on the same board.

| | final fraction on objectives | win rate |
|---|---|---|
| untrained network | 0.013 | 0% |
| **trained, 945 epochs** | **0.173** | **17%** |
| `greedy_nearest` heuristic | 1.000 | 28% |
| `split_evenly` heuristic | 1.000 | 68% |
| **`squad_march` heuristic** | **1.000** | **80%** |

`squad_march` is twelve lines: send squad *k* to objective *k mod 3*, steer on the squad
centroid, stop on the disc. It does not shoot. It beats a 6.5M-parameter network trained
for 945 epochs by nearly 5×.

The reason, measured directly on the checkpoint:

| | movement-head entropy (ceiling `ln 97` = 4.575) |
|---|---|
| freshly initialised network | 4.525 (98.9% of maximum) |
| **after 945 epochs** | **4.512 (98.6% of maximum)** |

**The movement policy never left its initialisation.** ~91 of 97 movement actions remained
live, which under 16 evenly-spaced angle bins is a near-zero-drift random walk. Every
threshold, rung and `success_rate` in the earlier reports was measured against that.

This also removes the need for the objective-drift report's potential-function
explanation: a uniform movement policy produces the observed "reaches 3/4 objectives, then
0/4 for eight steps, then back" trace directly. The analysis of `closest_objective` as a
potential is still correct about the calculator; it is not established as the cause of the
trace.

## Why it could not learn

The reward is a **mean over 25 models**, so one model's action explains ~4% of the number
it is credited with. That already-thin signal was then attributed through:

- **one joint importance ratio** — log-probs summed over all 25 models, so PPO's
  `eps_clip=0.2` was breached at `ln(1.2)/25 = 0.0073` nats of change per model;
- **one scalar critic** reading only the game token, so its residual was the advantage
  noise floor for all 25 policy factors at once;
- **an entropy bonus summed over models**, making the effective coefficient
  `ent_coef × 25` — 0.75 where 0.03 was configured, ~25× a conventional setting.

The per-model decomposition needed to fix the first two already existed inside
`calculate_reward` and was averaged away one line later.

## Four mechanism bugs, all confirmed in source

| | Bug | Consequence |
|---|---|---|
| **D6** | Rollout envs were rebuilt every `training_step`, each with a fresh `RewardPhaseManager` at index 0; phase advancement mutated only the eval env | **Training reward always came from phase 0.** `vp_gain`, `objective_flip_bonus`, `killing`, `closest_objective_v2` and `group_cohesion` were never trained on, in any run |
| **D7** | Each rebuilt env was re-seeded to its own index | Training replayed **8 fixed layouts** every epoch for 945 epochs while eval drew random ones |
| **D8** | The opponent policy was invoked once per battle phase, and `ActionHandler.apply` ignored its `phase` argument | The opponent moved **4× per round** to the player's 1×, seizing every objective before VP scoring opens at round 2 |
| **D9** | `KillingReward` read `opponent_models_killed` — player models the opponent killed | Paid **+5 per friendly casualty**. Inert only because the scripted opponent cannot shoot |

D6 has a visible fingerprint in the old runs: `reward_phase` climbed to 3 while
`reward/components/*` never contained `vp_gain` or `objective_flip_bonus`, which exist
only in the later phases. The metric was there; the check was not.

## The diagnostic

Two 30-epoch runs, identical except for `ent_coef`, on code with all four bugs fixed and
per-model credit assignment in place. Predictions were registered before the runs.

| | prediction | **A: `ent_coef` 0.75** | **B: `ent_coef` 0.03** |
|---|---|---|---|
| movement entropy (ceiling 4.575) | falls below 4.3 | 4.520 → **4.519** ✗ | 4.524 → **3.952** ✓ |
| `clip_fraction` | 0.1–0.3 (was 0.45–0.55) | 0.23 → **0.05** ✓ | 0.27 → **0.14** ✓ |
| `explained_variance` | rising | −0.05 → **0.44** ✓ | −0.03 → 0.17 ✓ |
| `success_rate` (phase criteria) | — | noisy 0–90, ends ~30 | 13 → **~80, sustained** |
| `eval/win_rate`, 10-epoch means | — | **0 / 0 / 0** | **21 → 62 → 47** |
| `eval/vp_margin`, 10-epoch means | — | −163 → −91 | −62 → **+10** → −4 |

Arm A holds the *pre-change* entropy pressure constant (0.03 summed over 25 models ≈ 0.75
on the mean), so it isolates credit assignment alone. Arm B additionally uses the
scale-free conventional coefficient. Nothing else differs between them.

### What this establishes

1. **The joint-ratio pathology was real and is fixed.** `clip_fraction` fell from a
   measured 0.45–0.55 to 0.05–0.19 in both arms.
2. **Per-model credit assignment alone is not sufficient.** Arm A's movement entropy is
   flat at 4.52 across all 30 epochs (slope +0.0004/epoch) and it wins **0%** of episodes.
   By the pre-registered criterion — *"refuted if entropy is still above 4.45 with clip
   fraction in the healthy band"* — **H-A is refuted as a sufficient explanation.**
3. **The entropy coefficient was the binding constraint.** At 0.03 the movement head
   concentrates monotonically (4.52 → 3.95, flattening near 3.92), the phase criteria go
   from 13% to a sustained ~80%, and VP margin crosses from −107 to positive.

So the original H5 was directionally right for a reason nobody had measured: entropy
pressure *was* pinning the policy — but not at the nominal coefficient. It was pinned by
the loss summing entropy over 25 models, which no change to `ent_coef` in the tested range
could have overcome.

**The B1 fix is verified in a live run.** `train/rollout_phase_index` now tracks
`reward_phase` with a one-epoch lag (the rollout logs before the epoch-end evaluation
advances the phase). Previously it would have stayed at 0 forever.

### The win-rate decline is informative, not noise

Arm B's win rate peaks at 62% (epochs 10–19) then falls to 47% (epochs 20–29) while
`success_rate` *holds at ~80%*. The two diverge at exactly the epoch the curriculum
advances to `mass_on_objectives`.

That phase rewards `models_at_objectives` (1.5) and `objective_coverage` (0.8) and
contains **no VP term at all**. The agent did what it was paid to do: it optimised
occupancy, and occupancy is not winning — control requires *strictly more* models than the
opponent on a point, and VP accrue per round.

This is the first time any phase past the first has actually influenced a gradient, so it
is also the first evidence about whether the ladder's shape is right. On this evidence it
is not: an intermediate rung that drops the VP signal entirely trains away from the goal.
That is now a testable question rather than a speculative one.

## Limitations

**Not a like-for-like comparison with the 17%.** These runs also fix the 4×-speed
opponent, so the opponent is weaker than in every earlier run. The honest framing is
against the baseline measured on the *same* code: `squad_march` scores 85% there, so the
agent went from **17% against an 80% bar** to a peak of **62% against an 85% bar**, ending
the run at 47%. Arm A vs arm B is the only clean single-variable comparison, and it
isolates `ent_coef`, not the bug fixes.

**Necessity of per-model credit assignment is untested.** Both arms have it. Establishing
whether it is needed *alongside* the entropy fix requires a third arm with the old joint
ratio and `ent_coef` 0.03.

**30 epochs, one seed per arm.** Arm B's win rate peaked mid-run and fell back; with 30
eval episodes an epoch's win rate has a binomial standard deviation of ~9 points, so
single-epoch readings mean little and even the 10-epoch means carry real uncertainty.
These runs show a direction, not a converged result, and the phase-1 explanation for the
decline is inferred from the reward composition rather than tested by ablation.

**The curriculum remains untested.** It has now trained for the first time, but no run has
compared it against a single-phase reward. Every threshold in the config is uncalibrated;
the commentary recording the old measurements has been replaced with a warning.

## Method lessons

1. **Measure a trivial baseline before tuning anything.** Twelve lines of numpy would have
   shown, on day one, that the policy was 12% of the way from random to a heuristic. Seven
   runs of threshold tuning were spent instead.
2. **Verify the thing you are tuning is the thing being trained.** The earlier lesson was
   *"before tuning a policy against a gate, verify the gate can open."* It needs the
   clause: *and that the reward you are tuning reaches the gradient.*
3. **Log raw quantities, not loss terms.** `loss/entropy_loss` is `-ent_coef × entropy`.
   Reading it as entropy produced a confident, wrong conclusion that stood for four runs.
4. **Instrument the optimiser.** `clip_fraction` and `approx_kl` are two lines each and
   would have shown the objective was degenerate at any point in those 945 epochs.
5. **A coefficient applied to a sum over N is not a coefficient.** `ent_coef` and the PPO
   clip both silently scaled with army size. Any hyperparameter whose effective magnitude
   depends on `n_models` will be mis-tuned at one scale and untunable at another.

## Next

- **Re-derive the ladder.** Every threshold in the config is a placeholder. Now that the
  phases actually train, size them from live rates against the baselines.
- **Test the ladder against a single-phase reward.** Never done. The phase-1 decline is
  direct evidence that at least one rung trains away from the goal.
- **Third arm:** old joint ratio with `ent_coef` 0.03, to establish whether per-model
  credit assignment is *necessary* alongside the entropy fix or merely sufficient.
- **Longer runs, ≥3 seeds** before any hyperparameter claim.
