# Pre-registering the melee arm, before any number exists

**Date:** 2026-08-25 · **Status:** committed before the arm was launched · **Code:** `feature/melee-stage-0`

This exists because a vp gate for melee is **unpowered by construction**, and writing the
criterion afterwards is how this project has twice turned a screen into a retraction.

## Why the obvious gate cannot work

The shipped melee profile (`A1 / MS6+ / S1 / AP2`) is **lethality-negligible** by design:
the blade returns roughly a tenth of what an engaged model's forfeited shot costs. So the
mechanic is *designed* to move vp by less than the estimator can see.

| n seeds | one-sided half-width at 50% power | **MDE at 80% power** |
|---|---|---|
| 3 | 19.05 | **25.97** |
| 6 | 9.32 | **13.54** |

⚠ **The first version of this table quoted the 50%-power column as the detectable effect.**
It is not: a 50%-power test misses a real effect of that size half the time. The right-hand
column is the number a gate must clear.

And the standing rule this replaces — *"≥ −8 vp on 3 of 3 seeds"* — **passes a do-nothing
feature 44% of the time** and *fails a costless one 56% of the time*, because the per-seed
paired sd is 11.3. A bound tighter than the estimator's own noise is not a bound.

## The primary readouts are mechanism counts

All of them move from a **hard floor of zero**, so they are detectable at n=3 where vp is
not. Every one is measured on `configs/experiments/25v25_maps_melee.yaml` against
`..._melee_dark.yaml`, which differ in exactly one scalar and therefore **pair**.

| readout | floor | why it is the right question |
|---|---|---|
| charges **declared** per episode | 0 | does the agent use the rule at all |
| charges **standing** per episode | 0 | does it use it *competently* |
| standing fraction | n/a | below ~0.5 it is proposing charges the referee reverts |
| model-steps **locked** in melee | 0 | the shooting shield, which is the charge's whole measured value |
| `coherent` | — | a rigid charge should preserve formation; a stretched one reverts |

⚠ **`charged_this_turn` is NOT a valid readout.** With `fight` in `skip_phases` the fight
resolves on the boundary *inside the same step* and clears the flag, so it always reads
zero. This has now cost two measurements. A charge **stood** iff the referee did not put its
models back where they started — displacement is durable, the flag is not.

## The secondary readout, and its trichotomy

`vp_margin`, paired, at **n ≥ 6**, reported as **PASS / FAIL / UNDERPOWERED** — and
UNDERPOWERED must be a reportable outcome, not a euphemism. If the confidence interval
contains both zero and the effect that would matter, that is the finding.

## What must accompany any vp number

Both of these are requirements, not suggestions. Each was paid for.

1. **The ablation.** The same arm with `exclude_engaged_targets` off. A charging script
   measured **+62.50 ± 14.74** with it and **−4.00 ± 17.39** without — so the charge's value
   is the *shooting shield*, not the blade. A number quoted without the ablation does not
   say which of those it measured.
2. **The 2×2.** Both sides walking, both charging, and each alone. Measuring a symmetric
   change with both sides changed at once published **"+15.5 to the bar"** for Advance when
   the truth was two self-inflicted wounds cancelling.

⚠ **Six independently hand-rolled charging scripts produced +6.5, +48.0, +52.0, +59.2,
+82.9 and +88.8 vp for nominally the same measurement — a 14× spread.** Nobody has measured
"the value of melee"; each measured their own heuristic. `squad_march_take_charge` is one
more heuristic and its number must be reported as such.

## The bar, and its own gates

Until 2026-08-25 `BaselinePolicy.select_action` returned STAY for every phase outside
command, movement and shooting, so **no scripted baseline and no scripted opponent could
charge**. An arm launched before that fix would have measured `baseline/policy.py`, exactly
as the Advance arm measured a bar that could not advance.

`squad_march_take_charge` was gated on four mechanism criteria written before it was run.
Measured on the shipped melee config, n=8, seeds 700000+:

| gate | criterion | measured | verdict |
|---|---|---|---|
| 1 | declared > 0 and standing > 0 | 9.75 and 6.12 per episode | **pass** |
| 2 | standing fraction > 0.5 | **0.628** | **pass** |
| 3 | `coherent` no lower than `squad_march_take` | 0.8636 v 0.8026 | **pass** |
| 4 | melee off ⇒ byte-identical to `squad_march_take` | identical on the golden **and on the dark control** | **pass** |

⚠ **Gate 2 failed on the first two versions of the rule, and both failures were mine.**
Aiming to stop "just inside" contact left **44.8%** of moved charges touching nobody,
because `best_action_toward` rounds *down* to a rung and spent the margin on rounding.
Reading the charge cap as `Move + roll` — the *advance's* rule — made it **55.9%**: a charge
is capped by `min(Move, roll)`, since the ladder's longest rung is Move
(`DEFERRED: charge.beyond_move_ladder`) and the roll exceeds Move on 59.1% of declarations.
The gate is what caught both. **A bar that has not been gated on its own mechanism counts is
an unvalidated instrument, and its vp number means nothing.**

**Still open at 28.2% of moved charges: ending incoherent.** Rigid translation preserves
formation exactly, so this is the *resolver* deflecting members off friendly and enemy
bases. It is the same wall three movement-side fixes have already been measured away
against — **do not attempt a fourth.** It is a known, quantified property of the instrument.

## What is NOT claimed here

A 2×2 has been *run* — n=6, seeds 700000+, on the melee config — purely to show the
instrument works on both seats. Its cells are internally inconsistent (both sides appear to
gain from charging), which at n=6 against a per-episode vp sd of 51–83 is what noise looks
like. **Those numbers are not reported as a finding and must not be quoted.** The opponent
seat charges: models engaged after the opponent's charge phase go from 0.0 to 6.2–8.0 per
episode.
