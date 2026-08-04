# Reward-phase curriculum at 25v25: why it stalled, and what moved it

> ## ⚠️ Substantially retracted, 2026-08-04
>
> Later measurement invalidated most of this report's cross-run conclusions. Read
> [the correction](2026-08-04-correction-what-was-actually-broken.md) first.
>
> In short: **training reward always came from phase 0**, no matter what `reward_phase`
> reported, because the rollout environments built their own phase manager. `vp_gain`,
> `objective_flip_bonus`, `killing`, `closest_objective_v2` and `group_cohesion` were
> never trained on in any run below. Training also replayed **8 fixed objective layouts**
> every epoch while evaluation drew random ones, and the scripted opponent moved **four
> times per round** to the player's one.
>
> Most decisively: a **12-line scripted heuristic scores an 80% win rate** on this config
> where the 945-epoch policy scores 17%. The policy's movement head sat at 98.6% of
> maximum entropy — barely moved from initialisation. Every rung, threshold and
> `success_rate` below was measured against a near-random policy.
>
> What survives: **D1, D2 and D4's method lesson** (arithmetic, independent of policy
> quality) and the fact that the drift and the gate mechanics were real. What does not:
> every hyperparameter verdict, the ladder sizing, and the claim that reaching
> `win_at_the_end` meant anything about play.

**Date:** 2026-08-04
**Config:** `examples/env_config/25v25_scripted_opponent_random_objectives_3_reward_phases_terrain.yaml`
**Algorithm:** PPO + TransformerNetwork (6.51M params), 25v25, 3 random objectives, 20 rounds
(`max_turns = 40`), 7 terrain pieces, scripted advancing opponent
**Branch / PR:** `feature/25v25-terrain-config` — [#127](https://github.com/sashman/wargame_rl/pull/127)

## Question

An earlier 25v25 run gated on `all_at_objectives` held `success_rate` at exactly 0 for 330
epochs while other metrics improved — that criteria needs `p**25` at this army size and is
unreachable. Replacing it with a `fraction_at_objectives` ladder produced non-zero rates
but the successor runs below still plateaued under their phase-0 gate. **Why does the
curriculum never advance, and can it be made to reach the final `win_at_the_end` phase?**

## Summary of outcome

The curriculum reached `win_at_the_end`. The cause of the stall was **four defects in the
advancement mechanism**, not the quality of the learned policy — three of the four make
advancement impossible independently of how well the agent plays.

The hypotheses about *learning dynamics* (discount factor, entropy coefficient) were
mostly wrong and cost three runs. The defects were found by measuring the mechanism.

## Runs

All runs: PPO + transformer, `n_steps=2048`. All were stopped manually; none ran to their
epoch cap. `success_rate` figures are the mean of the last 40 logged epochs.

| # | Run | Changed vs previous | γ | ent | n_eval | Epochs | Final phase | `success_rate` |
|---|---|---|---|---|---|---|---|---|
| 1 | `wd71v79p` | baseline (fraction ladder) | 0.90 | 0.03 | 10 | 511 | 0 | 48.2 |
| 2 | `km1equ2u` | terminal-bonus fix; γ 0.99; gate 3x30 | 0.99 | 0.03 | 30 | 301 | 0 | 33.2 |
| 3 | `5ibr1cls` | + `models_at_objectives`; bonus 5.0→3.0 | 0.99 | 0.03 | 30 | 163 | 0 | 15.1 |
| 4 | `zdo84kkc` | + `ent_coef` 0.01 | 0.99 | **0.01** | 30 | 148 | 0 | 3.7 |
| 5 | `cad3hbz7` | γ 0.95; ent reverted; thresholds lowered | 0.95 | 0.03 | 30 | 301 | 0 | 32.7 |
| 6 | `zf3ebdvs` | ladder re-sized from checkpoint sweep; warm start | 0.95 | 0.03 | 30 | 373 | **1** | 17.3 |
| 7 | `g5zuqts1` | ladder re-sized from **live** rates; warm start | 0.95 | 0.03 | 30 | 945 | **3** | 18.5 |

Run 7 advanced `approach_objectives → mass_on_objectives` at epoch 5,
`→ win_only_by_vp` at epoch 30, and `→ win_at_the_end` at epoch 50, confirmed by phase
name in the training log.

Note that `success_rate` is **not comparable across rows**: the active phase's criteria
and `min_fraction` change between runs, so the same number measures different things.

## Hypotheses and verdicts

| # | Hypothesis | Verdict | Evidence |
|---|---|---|---|
| H1 | `terminal_success_bonus` is under-delivered | **Confirmed** | Arithmetic + direct measurement: nominal 5.0 delivered 0.125 |
| H2 | The advancement gate demands more than its nominal threshold | **Confirmed** | Binomial: 10 consecutive epochs ≥0.7 at n=10 needs a true rate ≈0.85 |
| H3 | The criteria has no dense reward behind it | **Confirmed** (gap real) | No calculator paid for the gated quantity; effect on `success_rate` **not** demonstrated |
| H4 | γ=0.9 is too myopic for 40-step episodes | **Refuted** | γ 0.9→0.99 lowered `success_rate` 48.2 → 33.2 |
| H5 | `ent_coef=0.03` pins the policy and prevents commitment | **Refuted** | ent 0.03→0.01 collapsed eval `success_rate` to 3.7 |
| H6 | The ladder is mis-shaped, not the policy | **Confirmed** | Rung measured harder than the final goal; re-sizing produced advancement |

Full evidence for H1, H2, H3 and H6 is in
[mechanism defects](2026-08-04-mechanism-defects.md).

### H4 — raising the discount factor (refuted)

γ was raised 0.9 → 0.99 so the terminal bonus, which lands on the last step, would be
visible to early actions (`0.9^40 = 0.015` vs `0.99^40 = 0.669`).

Measured: `success_rate` fell from 48.2 (run 1) to 33.2 (run 2). Reverting to an
intermediate γ=0.95 recovered to 32.7 in run 5 under lower thresholds.

Interpretation — **inferred, not measured.** Raising γ multiplied the discounted dense
shaping at t=0 by ~3.3x but the terminal bonus by ~41x, shifting the split at t=0 from
roughly 11% terminal / 89% dense to 59% / 41%. A terminal bonus is a sparse binary
signal: it reports *whether* an episode succeeded, not which direction to move. The
plausible reading is that it displaced the dense shaping that performs the navigation.
This was not isolated experimentally — run 2 also changed the terminal-bonus magnitude
and the gate sampling.

### H5 — lowering the entropy coefficient (RETRACTED — the mechanism was misread)

> **Correction, 2026-08-04.** This section's central claim is wrong.
> `loss/entropy_loss` is **`-ent_coef × entropy`**, not entropy. Dividing by the
> respective coefficients gives **59.7 nats at `ent_coef` 0.03 and 59.0 nats at 0.01** —
> a 3× change in the coefficient moved total policy entropy by about **1%**. The stated
> reading ("0.59 nats ⇒ ~1.8 actions live") is off by roughly 100×: it is 59 nats across
> 25 models, ~2.4 nats each, ~11 actions each.
>
> H5 is therefore **untested**, not refuted. The measurements below stand; the
> explanation attached to them does not.
>
> The correct inference is the one nobody drew: if entropy is nearly invariant to a 3×
> change in its own regulariser, something else is setting it. That something was the
> loss shape — entropy was summed over 25 models, making the effective coefficient
> 25× the nominal one. See [mechanism defects, D6](2026-08-04-mechanism-defects.md).

`loss/entropy_loss_epoch` sat in `[-1.81, -1.72]` — a 0.09-nat band — across 975 epochs
and three configurations differing in γ, terminal-bonus magnitude (40x), and reward
composition. Invariance under that much change suggested the coefficient was pinning it.

Measured, over epochs 100–149:

| Metric (epochs 100–149) | ent 0.03 | ent 0.01 |
|---|---|---|
| `closest_objective` | 0.898 | **1.063** |
| `models_at_objectives` | 0.670 | **0.833** |
| `objective_coverage` | 1.785 | **2.041** |
| `mean_episode_reward` (eval) | **5.03** | 1.27 |
| `min_episode_reward` (eval) | −0.06 | **−2.25** |

Every *training-rollout* component improved while *evaluation* reward fell 4x.

That divergence is real and is the durable finding here — a clean instance of the
training-rollout vs evaluation distinction in `docs/metrics.md`, with the two quantities
disagreeing in **sign of change**. But the original explanation (a policy sharpened by
the lower coefficient overfitting to a rigid routine) cannot be right, because the
coefficient barely moved entropy.

The likelier cause, found later: training rollouts replayed **8 fixed objective layouts**
every epoch while evaluation drew random ones
([D7](2026-08-04-mechanism-defects.md)). Training-rollout numbers improving while eval
falls is exactly what fitting a handful of maps looks like, and it would occur under any
`ent_coef`. This has not been isolated experimentally.

## What produced the advancement

Runs 6 and 7 differ **only** in the ladder's rung sizes; run 7 warm-started from run 6's
checkpoint and shares its code, γ, `ent_coef`, and eval-episode count.

| Rung | Run 6 | Run 7 |
|---|---|---|
| `approach_objectives` | `min_fraction` 0.15 @ 0.45 | unchanged |
| `mass_on_objectives` | `min_fraction` 0.25 @ 0.35 | **0.20 @ 0.25** |
| `win_only_by_vp` | `fraction_of_max` 0.3 @ 0.35 | **@ 0.25** |

Run 6 advanced once (epoch 38) then held `success_rate` at 17.3 for 330 epochs against a
0.35 gate. Run 7 cleared three gates in 50 epochs.

**This comparison is well-controlled and supports attributing the advancement to the rung
sizes.** It does not show the policy improved — see the limitation below.

## Limitations

**The bar was lowered.** Reaching `win_at_the_end` was achieved partly by reducing
thresholds and `min_fraction` values to measured capability. This was the correct response
to gates that were provably unopenable (D2, D4 in the defects report), but the agent
cleared easier rungs than originally specified. Any claim of "the curriculum works now"
must carry this caveat.

**The agent did not improve at the final phase.** Over 895 epochs in `win_at_the_end`,
`success_rate` (fraction of episodes ending ahead on VP) was:

| Epochs | 50–199 | 200–349 | 350–499 | 500–649 | 650–799 | 800–949 |
|---|---|---|---|---|---|---|
| mean | 8.9 | 17.7 | 15.4 | 17.3 | 12.7 | 17.2 |

An initial rise to ~17% then flat. **Reaching the phase is established; winning it is
not.** The agent loses the VP race roughly 83% of the time.

**Single seed, one variable-set per run.** Runs 2, 5 and 6 each changed more than one
thing. These runs support "this mechanism was broken" and do not support ranking
hyperparameters. H4 and H5 are refuted as *directions that helped*, not isolated
as sole causes of the observed drops.

**Runs were stopped early** (148–945 of 500–1200 epochs) based on visible trends. Runs 3
and 4 were stopped at 163 and 148 epochs; both were tracking or below their predecessors
at matched epochs, but a later reversal cannot be excluded.

**H3 is a confirmed gap with unconfirmed benefit.** `models_at_objectives` closes a real
hole — the criteria measured a quantity nothing rewarded — but run 3 tracked run 2 rather
than beating it, and no run has isolated its effect at the raised weights.

**The opponent cannot shoot.** `ScriptedAdvanceToObjectivePolicy` emits only movement and
stay actions. Terrain blocks line-of-sight, and line-of-sight only affects shooting, so
terrain protects the opponent from the player and never the reverse. The player takes no
casualties, making `killing` risk-free in the VP phases. Every VP result here is against
a defenceless opponent.

## Conclusions

1. **The curriculum stall was a mechanism failure, not a learning failure.** Three of the
   four defects make advancement impossible regardless of policy quality.
2. **Measure the mechanism before tuning the policy.** Three runs (~2.5 hours) were spent
   on discount and entropy hypotheses. The defects were found by arithmetic and by
   evaluating criteria directly against checkpoints.
3. **Gate thresholds must be calibrated against the distribution the gate reads** — the
   live per-epoch `success_rate`, not a best-checkpoint sweep.
4. **A curriculum rung must be verified easier than the rungs after it.** One rung here
   was measurably harder than the final goal.

## Follow-up

- **Make the opponent shoot.** Add a shooting branch to
  `ScriptedAdvanceToObjectivePolicy.select_action` and mask its shots with
  `compute_shooting_masks`. Without masking the opponent would shoot through terrain at
  unlimited range: `_resolve_shooting_action` validates alive/slice/weapons but neither
  range nor line-of-sight. Until then no VP result is a fair test.
- **Isolate `models_at_objectives`** at its raised weights (1.0 / 1.5) against a matched
  control, with the `at_objective` trace as the primary signal rather than `success_rate`.
- **Raise the phase-3 win rate**, currently ~17% and flat over 895 epochs.
