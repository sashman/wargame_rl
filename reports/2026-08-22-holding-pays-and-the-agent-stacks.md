# Holding pays. The agent is not hiding — it is stacking.

**2026-08-22.** All measurement, no GPU. Everything on
`configs/experiments/24v24_maps_spare_squads.yaml` (the arm config, unrefereed
— it is the reward the agent optimised, and the referee's attrition would move
control), held-out nine, seeds 700000+, decode as named per row.

## The hypothesis, and why it is refuted

The agent finishes with **52.9% of its army alive against the scripts'
27.4–30.9%** while holding 0.63 fewer objectives. Six explanations for that
offence deficit had already died. The seventh: standing on an objective means
standing in the open, so if holding does not pay for the casualties, **hiding is
correct play** and the agent found it.

**Its premise is false.** `docs/rules/implementation-status.md` records that
*every one of the 270 objective markers in `configs/evaluation/maps/` sits inside
a ruin*. Standing on an objective is standing in a building. The models in the
open are the ones walking between them.

And the trade measures the same way. `just measure-hold-hazard` records, per
alive model per step, whether it is inside an objective radius (`norms_offset <=
radius` — the test VP scoring uses), what `last_per_model_reward` paid it, and
whether it is dead on the next step:

| policy | steps on obj | income differential | excess death hazard | break-even |
|---|---|---|---|---|
| agent s1 (K=3) | 53.9% | **+0.4123** | **−0.29%** | +5.98% |
| agent s2 (K=3) | 55.4% | **+0.4237** | **−0.28%** | +5.33% |
| agent s3 (K=3) | 53.9% | **+0.3973** | **−0.13%** | +5.05% |
| `squad_march_take` (K=1) | 76.2% | +0.4419 | −1.04% | +3.52% |
| `squad_march_deny` (K=1) | 74.7% | +0.3748 | −1.43% | +3.38% |

**Unanimous, 5 of 5, and not marginal.** Standing on an objective pays +0.37 to
+0.44 more per model-step *and* the excess hazard is **negative in every row** —
it is *safer* than being off one. Breaking even would need +3.4% to +6.0% of
excess hazard; the measured value has the opposite sign.

⚠ The comparison is conditional, not causal: models on objectives differ from
models off them in more than their footing. But the margin is not close and the
sign is wrong for the hypothesis, so no plausible confound rescues it.

**Verdict: holding pays. The agent is leaving return on the table.**

## Where the misjudgement actually is

Not risk. Two things, both measured.

**It spends less time on the points.** 54.4% of model-steps against the scripts'
**75.5%**. Its models are walking for nearly half the game.

**And when it arrives, it arrives where it already is.** `just
measure-objective-split`, n=60:

| | agent s3 (K=3) | `squad_march_take` (K=1) |
|---|---|---|
| models alive at end | **12.5** | 6.6 |
| objectives held | 2.45 | **2.65** |
| models on its top objective | **4.90** | 2.73 |
| surplus models on held points | **6.15** | 3.73 |
| objectives abandoned | 0.553 | 0.498 |
| redistribution ceiling | **4.65 (+2.20)** | 3.52 (+0.87) |

It has **8.6 of its 12.5 survivors standing on objectives** — proportionally more
than the script — piled **4.90 deep on its top point where the script puts
2.73**, while 55.3% of objectives hold nobody. The **+2.20 redistribution
ceiling is the largest recorded here**; per the standing rule a large ceiling
does not prove re-allocation would work, but it does not rule it out.

**The reward already prices this, and the agent pays it.** `objective_hold` runs
at `crowding_exponent: 1.0` — a fixed pot split between occupants, so spreading
strictly raises income. `just measure-income-share`, n=30:

| calculator | kind | agent s3 | `squad_march_take` |
|---|---|---|---|
| `objective_hold` | per-model | 6.76 (0.307) | **13.48 (0.608)** |
| `vp_gain` | **global** | 6.91 (0.314) | 0.29 (0.013) |
| `objective_coverage` | **global** | 4.92 (0.223) | 5.43 (0.245) |
| `model_kills` | per-model | 2.50 (0.113) | 2.61 (0.118) |
| **global share of income** | | **0.537** | **0.258** |

**Exactly half the script's `objective_hold` income**, from a pot it splits over
half as many points. The shaping term that was supposed to prevent this is
already switched on and already correct; the agent is simply not following it.

## The desk check passes

The standing rule is to confirm the agent can *observe* whatever a lever keys on
— two mechanically opposite levers once both halved objective occupancy because
neither could be seen. `observe_objective_control: true` is set, and
`_objectives_to_obs` supplies **per-objective counts of alive models for both
sides**, normalised by a shared establishment. Occupancy is observable. That is
not the defect this time.

## What a fix would take, and what it would cost

Not reward retuning. `crowding_exponent` is the measured-good lever, it is on,
and it is being ignored — so the case for shaping is weaker here than it has ever
been, not stronger.

The live candidates, unpriced and **not run**:

1. **The travel term's fallback.** `closest_objective_v2` has
   `fallback_to_nearest: true` at `progress_scale: 6.0`, and a model in a group
   assigned no objective is *paid* to close on whatever is nearest — usually a
   point its own side already holds. That is a documented defect and it produces
   precisely this signature. Screening it costs one config and no GPU.
2. **Squad-level dispersion.** Squads of three under a 2" chain move as a body,
   and `objective_hold` requires coherence, so the allocation quantum is the
   squad. Whether the agent's squads bunch, or spread but arrive thick, is
   measurable and has not been measured.

Do (2) first: it is free, and it decides whether (1) is even the right target.

## Do not re-run

- **"Objectives are exposed."** They are ruins. On-objective death hazard is
  *lower* than off-objective for all five policies measured.
- **Anti-stacking shaping, until squad dispersion is measured.** The
  pot-splitting lever is already correct and already ignored.
