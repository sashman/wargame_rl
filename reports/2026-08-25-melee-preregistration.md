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
| 2 | standing fraction > 0.5 | **0.628**, and **0.887** after the audit fix below | **pass** |
| 3 | `coherent` no lower than `squad_march_take` | 0.8636 v 0.8026 | **pass, but see below** |
| 4 | melee off ⇒ byte-identical to `squad_march_take` | identical on the golden **and on the dark control** | **pass** |

⚠ **Gate 2 failed on the first two versions of the rule, and both failures were mine.**
Aiming to stop "just inside" contact left **44.8%** of moved charges touching nobody,
because `best_action_toward` rounds *down* to a rung and spent the margin on rounding.
Reading the charge cap as `Move + roll` — the *advance's* rule — made it **55.9%**: a charge
is capped by `min(Move, roll)`, since the ladder's longest rung is Move
(`DEFERRED: charge.beyond_move_ladder`) and the roll exceeds Move on 59.1% of declarations.
The gate is what caught both. **A bar that has not been gated on its own mechanism counts is
an unvalidated instrument, and its vp number means nothing.**

⚠ **RETRACTED, same day, by an audit panel — "still open at 28.2%: ending incoherent.
Rigid translation preserves formation exactly, so this is the RESOLVER; do not attempt a
fourth movement-side fix."** Wrong on all three clauses. Four panel probes at four
different n (76.2%, 80.8%, 80.9%, 85.0%) and my own re-derivation (**82.2%**, n=12) agree
that the incoherent failures were on units **already out of coherency before the charge
began** — only 8 of 135 moved charges were broken *by* the move. The inference was
backwards: rigid translation preserving formation is exactly *why* a squad that was broken
when it declared is still broken when it lands. And the fix is policy-side, so the
movement-side prohibition never applied to it. There is also a selection effect pulling the
wrong way — a stretched squad's nearest member is nearer the enemy, so a broken unit looks
*more* chargeable.

**The fix is one clause: decline to charge with an incoherent unit.** Standing fraction
**0.628 → 0.887**; incoherent failures 45 → 7 of 151 moved charges.

⚠ **This is the third time on this project's record that a published explanation was built
against a missing within-policy control** — after the angle-collapse statistic and the
travel-reward gates. The standing rule *"run the within-policy control before building a
diagnosis on a behavioural statistic"* applies to **mechanism counts exactly as it does to
vp**, and the first version of this document did not carry it.

## ⚠ The blocker was closed on the BAR and left open on the OPPONENT

Found by an audit panel, inside the commit whose message says the blocker is closed.
**Both melee configs seated `squad_march_take`**, whose `charge_when_it_lands` is False —
so the arm would have trained in the **unilateral cell** of a mechanic whose entire measured
value is the *asymmetry between the seats*. This is the Advance failure moved one seat over.

The panel's 2×2, n=100 per cell, paired on seeds 700000+, argmax, vp_margin to the player:

| | opponent walks | opponent charges |
|---|---|---|
| **player walks** | +14.05 | −51.55 |
| **player charges** | **+38.00** | +0.95 |

Charging against a walker is worth **+23.95 ± 9.94** (t = +2.41); both charging against both
walking is **−13.10 ± 10.88**. Every same-row comparison in the top row says melee is worth
+24, and **the mechanic is worth about zero.** Both configs now seat
`squad_march_take_charge`, pinned by a test. With melee off it is byte-identical to
`squad_march_take`, so the dark control's digest is unchanged and the pair still differs in
exactly one scalar.

⚠ **The standing rule was on file and I did not apply it.** *"Never measure a symmetric
change with both sides changed at once — run the 2×2."* I *did* run a 2×2 — at n=6, where
the SE on a cell is ~36 vp — then recorded it as noise and reasoned about the mechanic from
a single cell anyway. **The missing half of the rule: a 2×2 run below its resolvable n is
not a 2×2, and the resolvable n is computable from the per-episode sd before you run it.**
"Record it as noise" was the wrong remedy when the right one cost four CPU-minutes and would
have caught the opponent-seat defect before this document was committed.

## The shield ablation, run at last — and the standing prior does NOT reproduce

n=60 per cell, paired on layouts, seeds 700000+, argmax, on the shipped melee config.
The first run of this measurement ever possible, because `shield_engaged_targets` did not
exist until the audit found requirement 1 unsatisfiable.

| player \ opponent | walks | charges | | walks (no shield) | charges (no shield) |
|---|---|---|---|---|---|
| **walks** | −2.58 | −55.67 | | −2.58 | −26.58 |
| **charges** | +19.58 | −3.42 | | +10.08 | −6.92 |

| what charging is worth | shield ON | shield OFF |
|---|---|---|
| against a walking opponent | **+22.17 ± 11.50** (t=1.93, **31/60**) | +12.67 ± 11.79 (t=1.07, 33/60) |
| against a charging opponent | **+52.25 ± 12.02** (t=4.35, 39/60) | +19.67 ± 12.94 (t=1.52, 33/60) |

⚠ **RETRACTED: "the charge's value is ENTIRELY the shooting shield (+62.50 → −4.00 when
ablated)."** That does not reproduce. Ablated, charging is still worth **+12.67 and +19.67**,
not −4.00. The shield is roughly **half** the effect, not all of it. The prior was measured
with a different script on pre-per-unit-gate code, and this project's own caution applies to
it — six hand-rolled charging scripts spanned 14×.

⚠ **My reasoning that the per-unit gate makes a charge MORE valuable was unjustified in
direction, and remains unresolved.** The gate silences the *charger's* whole unit too, not
only the target's. The two measurements are not comparable (different script, different
code), so nothing here settles it.

**The mechanic is worth about zero, confirmed at an independent n.** Both-walk **−2.58**
against both-charge **−3.42**. All of the apparent value is the *unilateral* cell.

⚠ **Read the sign counts, not the means.** `+22.17` sits on **31 of 60** — a coin flip — so
that cell is **tail-driven and is not a robust effect**, exactly what the standing rule
"quote a t AND a sign count" exists to catch. Only the +52.25 cell (39/60, t=4.35) has both.

## Two more defects the audit found in this document

⚠ **Requirement 1 was unsatisfiable.** `exclude_engaged_targets` was hardwired to
`config.melee.enabled` at both mask call sites and `MeleeConfig` had no such field, so *"the
same arm with the shield ablated"* could not be expressed at all. Now
`melee.shield_engaged_targets`, default True.

⚠ **`declared` and `standing` are not decode-independent, and this document never states K.**
A panel measured the same untrained weights declaring 25–35 charges per episode at K=1 and
2.25–3.62 at K=3 — a 10× swing in the headline mechanism count from the decode setting
alone. **Every readout here must be quoted with its K**, and the pre-registration's own
central insight — a gate tighter than its estimator's noise is not a gate — applies to three
of its four mechanism gates.

## The control nobody ran: the untrained-network floor

⚠ A randomly-initialised network passes **gate 1 and gate 3** having learned nothing
(declared/ep 2.25–3.62, stood/ep 0.25–0.88, coherency 0.916–0.994 against gate 3's 0.803
bar), because the decoder enforces coherency for it. **New standing rule: any behavioural
readout that gates an arm must be floored on a random-init network through the arm's own
selector path, before the arm launches.** It costs one inference run.

## What is NOT claimed here

The 2×2 above is the panel's, at n=100. My own n=6 version was internally inconsistent and
is superseded rather than retained. The opponent seat does charge: models engaged after the
opponent's charge phase go from 0.0 to 6.2–8.0 per episode.
