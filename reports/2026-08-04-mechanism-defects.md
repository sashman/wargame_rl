# Mechanism defects blocking curriculum advancement

**Date:** 2026-08-04
**Context:** [reward-phase curriculum experiment](2026-08-04-reward-phase-curriculum.md)

Four defects in the reward-phase advancement mechanism. Three of them (D1, D2, D4) make
advancement impossible or near-impossible **independently of how well the agent plays**,
which is why several hundred epochs of visible learning produced no phase transition.

Each entry states the symptom, the measurement, the root cause, and how it was verified.

---

## D1 — `terminal_success_bonus` delivered at 1/40th of its configured value

**Symptom.** Phase 0 plateaued at ~45% against a 0.7 gate for ~400 epochs (run
`wd71v79p`) while `objective_coverage` and `closest_objective/progress` continued to rise.
Suspicion fell on the terminal bonus after decomposing episode reward by component.

**Root cause.** `RewardPhaseManager` scaled the bonus by the fraction of turns remaining —
a *speed* incentive that presumes success ends the episode:

```python
remaining = max(0.0, float(ctx.max_turns - ctx.current_turn + 1))
remaining_frac = remaining / float(ctx.max_turns)
bonus = phase.terminal_success_bonus * remaining_frac
```

With `terminate_on_success: false` every episode runs to `max_turns`, so
`remaining_frac = 1/max_turns`. Note `max_turns = number_of_battle_rounds x (5 - len(skip_phases))`,
so a 20-round config with three skipped phases has `max_turns = 40`, not 20.

**Measurement.** Evaluated against the live config before and after the fix:

| | delivered bonus | value at t=0 |
|---|---|---|
| before | **0.125** | 0.082 (γ=0.9) |
| after | **5.0** | 3.379 (γ=0.99) |

Logged reward components, converted to per-episode contributions
(per-step mean x `mean_episode_steps`):

| Component | per-episode | share |
|---|---|---|
| `objective_coverage` (weight 0.3) | 3.12 | **61%** |
| `closest_objective` (weight 1.0) | 1.54 | 30% |
| `terminal_success_bonus` (nominal 5.0) | **0.06** | **1.2%** |

The criterion the curriculum gates on contributed ~1% of episode reward.

**Fix.** Apply the remaining-turn scale only when `terminate_on_success` is true.

**Verification.** `reward/components/terminal_success_bonus` now logs exactly
`0.00244140625` per successful episode, which is `5.0 / 2048` (the bonus over the rollout
step count). Before the fix the same quantity would be `0.125 / 2048 = 0.000061`.

---

## D2 — the advancement gate demanded far more than its nominal threshold

**Symptom.** A nominal `success_threshold: 0.7` never fired despite epochs frequently
reading above 0.7.

**Root cause.** `min_epochs_above_threshold` requires that many *consecutive* epochs above
the bar, and each epoch's `success_rate` is an `n_episodes`-sample binomial — a noisy
estimate, not the policy's true rate. Requiring a run of them compounds the noise.

**Measurement.** At `n_episodes=10`, threshold 0.7:

| True rate | P(one epoch ≥0.7) | P(10 consecutive) |
|---|---|---|
| 0.6 | 0.38 | 0.0001 |
| 0.7 | 0.65 | 0.013 |
| 0.8 | 0.88 | 0.28 |
| 0.9 | 0.99 | 0.88 |

A nominal 0.7 gate was effectively an 0.85 gate.

**Fix.** `--n-eval-episodes 30` with `min_epochs_above_threshold: 3`, which restores the
effective bar to near the nominal value at comparable evaluation cost.

**Verification.** Predictive, on a later run: `approach_objectives` held a live mean of
0.365 against a 0.45 gate, giving P(one epoch) ≈ 0.20 and P(three consecutive) ≈ 0.008 —
about 1 attempt in 125, so roughly 40 epochs. It advanced at **epoch 38**.

---

## D3 — the gated quantity had no dense reward behind it

**Symptom.** Phases gated on `fraction_at_objectives` improved on every reward component
without improving on the criteria.

**Root cause.** No calculator paid for the fraction of models on objectives:

| Calculator | Pays for | Why it does not serve the criteria |
|---|---|---|
| `objective_coverage` | fraction of objectives *controlled* | saturates once the player out-counts the opponent (~2 models a point) |
| `closest_objective` | distance *closed* | rewards approaching, not staying |

**Fix.** Added `models_at_objectives`: dense reward = alive models within some objective
radius / alive models. Non-saturating as more models arrive; dead models leave both
numerator and denominator.

**Status.** The gap is confirmed real. **The benefit is not yet demonstrated** — the run
introducing it tracked its predecessor rather than beating it, and no run has isolated it
at the raised weights (1.0 / 1.5). See also
[objective drift](2026-08-04-objective-drift.md), where the same term is the mechanism
that opposes drift.

---

## D4 — rungs calibrated against the wrong distribution

**Symptom.** A rung sized from a measured curve still never advanced, over 330 epochs.

**Root cause.** The rungs were sized by sweeping criteria against a **best checkpoint**,
but the gate reads the **live per-epoch `success_rate`**. Best checkpoints are selected on
reward, so they are systematically optimistic — by 10–15 points here.

**Measurement.** `mass_on_objectives` at `min_fraction 0.25`:

- Best-checkpoint sweep: **0.33** → looks marginal against a 0.35 gate
- Live per-epoch mean over 330 epochs: **0.174** → 16.9, 13.6, 17.6, 19.9, 10.3, 18.5, 18.9, 17.4, 17.4 (50-epoch buckets), no trend

At p=0.174 with n=30, P(one epoch ≥0.35) ≈ 0.003 and P(three consecutive) ≈ **2x10⁻⁸**.
The gate could not open at any epoch budget. Meanwhile `objective_coverage` rose
0.055 → 0.060 and `progress` 0.028 → 0.032 across the same window: the policy was
improving the whole time.

**Fix.** Size rungs from live rates. Measured live: `min_fraction` 0.15 → ~36%,
0.20 → ~25%, 0.25 → ~17%. Rungs set to 0.15 @ 0.45, 0.20 @ 0.25, VP @ 0.25.

**Verification.** Three gates cleared in 50 epochs, against one gate in 373 epochs for the
checkpoint-calibrated ladder. The two runs differ only in rung sizes.

---

## D5 — the ladder was not monotonic (found during D4 investigation)

**Symptom.** An intermediate phase blocked while later phases measured passable.

**Measurement.** All four criteria evaluated against one checkpoint, 30–40 episodes each,
through the same path training uses (run to episode end, check against
`last_step_context`):

| Phase | Criteria | Threshold | Measured |
|---|---|---|---|
| `approach_objectives` | fraction ≥ 0.2 | 0.45 | 0.37 |
| `mass_on_objectives` | fraction ≥ 0.35 | 0.35 | **0.20** |
| `win_only_by_vp` | `player_vp_min` 0.3 | 0.45 | ~0.40 |
| `win_at_the_end` | `player_ahead_on_vp` | 0.80 | ~0.30 |

`mass_on_objectives` — massing 35% of 25 models (≈9) on objectives at the final step — was
**harder than winning the VP race and harder than the final goal itself**. A rung stricter
than its own destination stalls everything behind it, and no reward tuning can fix it.

**Fix.** `min_fraction` 0.35 → 0.20, so difficulty increases monotonically along the
ladder.

---

## Latent issue, fixed but deliberately not enabled

`terminate_on_success` was hardcoded to `all_models_at_objectives` and never consulted the
phase's configured criteria, so fraction- and VP-gated phases could not end on their own
success. This is now fixed, but **left disabled in config**.

Enabling it redefines the metric:

| `terminate_on_success` | `success_rate` measures |
|---|---|
| `false` | criteria **holds at the final step** |
| `true` | criteria was **ever met** — the episode stops there, so it holds at the last step by construction |

Given peak occupancy runs ~3x final occupancy, flipping this raises `success_rate` by
roughly that factor with no policy improvement, invalidating every calibrated threshold.
For VP phases it is also semantically wrong: `win_at_the_end` must mean ahead *at the end*.

---

## Cross-cutting lesson

D1 and D2 are arithmetic. D4 and D5 are measurement discipline. None required a training
run to find — but all four were found only after several runs had been spent on
hyperparameter hypotheses that turned out to be wrong.

**Before tuning a policy against a gate, verify the gate can open.**
