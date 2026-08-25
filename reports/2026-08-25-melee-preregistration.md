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

## ⚠ THE ARM MAY MEASURE THE COORDINATION CEILING, NOT THE MECHANIC

Nominated by a second, uncoordinated audit panel as the reason not to fund the arm.
**Verified by me**, n=20, referee's own verdict (not displacement), on the shipped config:
take the bar's own charge order and perturb *j* of a unit's five members to a **different
legal charge rung** — which is what a factored per-model policy emits the moment it is not
unanimous.

| j perturbed | standing fraction | vp_margin |
|---|---|---|
| **0** (unanimous, the script) | **0.842** | **+5.75** |
| **1** | **0.472** | **−21.50** |
| 2 | 0.522 | −18.75 |
| 5 | 0.423 | −18.75 |

**One dissenting model of five halves the standing fraction and costs 27 vp.** And the
collapse is **not monotone** — j=2 and j=5 are no worse than j=1. The penalty is paid **in
full at the first dissenter**.

⚠ **"Unanimity or nothing" is RETRACTED, by my own follow-up the same hour.** That
perturbation moved the dissenter to a **uniformly random** legal rung, which is a large
change — it measures *"one model does something quite different"*, not *"one model is
slightly off"*, and the difference decides whether the problem is learnable at all. Perturbing
one member by a **single bin** instead, n=20, referee's verdict:

| one member perturbed by | standing fraction | vp |
|---|---|---|
| nothing (the script) | 0.842 | +5.75 |
| **± one SPEED bin** | **0.757** | −10.75 |
| **± one ANGLE bin** | **0.756** | +1.50 |
| a random legal rung | 0.472 | −21.50 |

**Near-agreement is worth ~90% of exact agreement.** The charge requires the unit to end in
a *region*, not to emit identical actions, so a policy that is approximately unanimous keeps
most of the value and only a *large* disagreement collapses it. That is a far weaker
requirement than the first measurement implied, and it is the difference between "a factored
policy cannot express this" and "a factored policy has to be roughly consistent".

⚠ Read the standing fraction here, not vp: the two single-bin rows land at the same 0.757
with vp −10.75 and +1.50, which at n=20 is the noise.

That matters because the referee is an **all-or-nothing joint constraint over five
independently sampled actions**, and this project has already measured what a factored
network does with one: `clone_squad_march_take.ckpt` reproduces its rigid teacher's shared
heading on **42.2% of unit-turns against the teacher's 91.8%**, and consensus decoding to
force agreement lost **−4.8 / −4.1 / −9.1 vp on 3 of 3 seeds**.

⚠ **So the bar's +31.5 may be architecturally unreachable, and no seed count or epoch budget
would fix it.** Training decodes at **K=1**, where the untrained standing fraction is
0.000–0.066 (measured above) — the joint decoder that rescues legality is a *play-time* tool
and is explicitly excluded from training.

**The decisive next experiment is therefore the CLONE CONTROL, not the arm** — behaviour-clone
`squad_march_take_charge` and measure the clone's standing fraction at K=1. It costs about
**1 GPU-hour against the arm's 49–74**, and its two outcomes point to different spends:
a clone at K=1 ≥ 0.45 says the mechanic is expressible and the arm is funded; ≤ 0.25 says
melee's measured value is a joint-action artefact of a rigid-body script, the arm would
measure the coordination ceiling, and the finding generalises to **every move type the rules
add**. CLAUDE.md already carries running the clone control as a standing rule; skipping it
would be the third published explanation here checked against the wrong control.

## THE CLONE CONTROL, RUN — verdict INCONCLUSIVE, and the mechanism is not the one predicted

Three behaviour clones of `squad_march_take_charge` (200 demonstration episodes, 8 epochs,
seeds 0/1/2), ~2 minutes of GPU each. Pre-registered read: **accept ≥ 0.45**, **reject
≤ 0.25**, between = inconclusive, on the clone's standing fraction at K=1.

### As played, it rejects — but for the wrong reason

| | declared/ep | stood/ep | standing fraction (K=1) | vp |
|---|---|---|---|---|
| teacher (the script) | 5.90 | 4.60 | **0.780** | +10.2 |
| clone s0 / s1 / s2 | 0.53 / 0.50 / 1.17 | 0.00 / 0.03 / 0.00 | **0.000 / 0.067 / 0.000** | −71 / −49 / −36 |

⚠ **The clone does not fail to coordinate — it fails to DECLARE.** Action match against the
teacher on teacher-driven states, split by phase: shooting **0.988–0.990**, movement
**0.628–0.638**, charge phase 0.940–0.945 — but that last figure is an artefact of class
imbalance (most models are told STAY, so predicting STAY scores 94%). **Of the teacher's
actual charge ORDERS, the clone echoes 0.8–2.4%.** A charge order is ~3.7% of charge-phase
model-decisions and a plain imitation loss predicts STAY everywhere.

### Forcing the declaration isolates the coordination, and it is HALFWAY THERE

In every charge phase where the teacher would charge a unit, each of that unit's models takes
its own argmax over **charge actions only**, and the referee judges the result. The decision
to charge is taken out of the clone's hands; the coordination is entirely its own.

| | standing fraction | unit picks one shared action |
|---|---|---|
| teacher (ceiling) | **0.849** | 1.000 (rigid by construction) |
| clone s0 / s1 / s2 | **0.367 / 0.303 / 0.391** | 0.603 / 0.520 / 0.742 |

Against an untrained network's K=1 floor of **0.000–0.066**, a clone with two minutes of naive
imitation and no class weighting lands **5–6× the floor** and produces a unanimous unit on
**52–74%** of charges.

### Verdict, against the criterion committed before the numbers existed

**INCONCLUSIVE** — 0.303–0.391 is inside the 0.25–0.45 band on 3 of 3. The pre-registration's
own instruction for this band is *"run three more demonstration seed bases before any GPU is
spent on the arm. Do not resolve it by training."*

⚠ **But the panel's headline claim is NOT supported.** "The mechanic may be architecturally
unreachable; melee's value is a joint-action artefact of a rigid-body script" predicts a clone
that tries to charge and cannot coordinate. The measurement shows the opposite: it coordinates
about half the time and barely tries. **The binding constraint is proposing a rare
all-or-nothing action, not executing a coordinated one** — and that is an exploration and
class-balance problem, which has different remedies from a coordination problem.

This also joins the retraction above: near-agreement is worth ~90% of exact agreement, so a
policy that is *approximately* unanimous keeps most of the value. Both measurements point the
same way.

## Two attempted fixes, and what they found: the CHARGE HAS NO DECLARATION

### 1. Balancing the clone loss — moved the probability, not the argmax

The unweighted clone learned "always STAY" in the charge phase because a charge order is
~3.7% of that phase's deciding rows. `phase_balanced_weights` balances STAY against
everything else **within each phase** (globally would cancel: STAY is rare in movement and
dominant in the charge, and the mask cannot tell them apart because a charge reuses the
movement slice).

Measured in states where the teacher charges this model:

| clone | P(stay) | P(any legal charge) | teacher's rung ranks |
|---|---|---|---|
| unweighted s0 / s1 | 0.637 / 0.613 | 0.363 / 0.387 | 16th / 16th |
| **balanced** s0 / s1 / s2 | 0.413 / 0.396 / 0.464 | **0.587 / 0.605 / 0.536** | 14th / 13th / 18th |

**The weighting flipped the mass onto charging — and the argmax still says STAY.** Charge
declarations per episode rose 2–3× (0.50–1.17 → 1.75–3.20) and the standing fraction did
not move off ~0. ⚠ **The class imbalance was real and was not the binding constraint.**

### 2. The mechanism, and it is the ACTION SPACE

`P(any charge)` of 0.59 is spread across ~48 individual rungs at ~0.012 each, against STAY
as **one** action at 0.41. An argmax over a spread loses to a concentrated alternative even
when the spread carries more total mass. And it compounds per model:

| | declarations | mean share of the unit declaring | **whole unit declares** |
|---|---|---|---|
| teacher | 82 | **1.000** | **1.000** |
| balanced clone s0 / s1 / s2 | 26 / 23 / 48 | 0.539 / 0.616 / 0.597 | **0.231 / 0.348 / 0.354** |

**A charge fails because only half the unit charges at all.** The rest stay put, the unit
stretches, and the referee reverts it however good the rung choice was. At a per-model
declare rate of ~0.6 over ~3.3 living members, `0.6^3.3 ≈ 0.20` — against the measured
0.23–0.35.

⚠ **This is the defect CLAUDE.md already names for the advance, unfixed for the charge.**
*"Leader-binds inside the movement slice would SHATTER formation — move type and displacement
were the same action... That is why the declaration had to be split out into its own phase."*
The advance got a 2-action `move_type` slice declared by the unit's leader, which **binds the
whole unit**. The charge never did: its declaration *is* the choice of a rung, so "charge or
not" is decided independently by every model.

**The proposal that falls out, and it needs no new mechanism** — one more value in the
existing `move_type` slice, declared by the leader, binding the unit, exactly as
`MOVE_TYPE_ADVANCE` already is. It turns a 1-against-48 argmax repeated per model into a
single binary unit choice. ⚠ It costs one action (102 → 103) so it is **not pairable against
the current control**, and a `dark_action_slices` control of identical shape is the bridge —
the pattern this project already used for the advance.

✅ **IMPLEMENTED 2026-08-25.** `MOVE_TYPE_CHARGE` in the existing slice, declared by the
leader in the command phase, with both halves: a rung is legal only for a declared unit, and
a declared unit may not stand still. Sized so advance configs keep their 152 actions, and
registered via `dark_action_slices` on the control so the pair stays at 104 each.

⚠ **It voids every melee number in this document.** The melee configs now step `command`, so
`max_turns` is 60 → **80** — the gate table, the 2×2 and the shield ablation were all taken
at 60. The bar still clears gate 2 under the declaration (4.40 declared/ep, standing fraction
**0.750**), which is the only figure re-measured so far.

⚠ **And it does not settle whether melee is worth training.** The 2×2 still says the mechanic
is worth about zero, so a declaration that makes charging easy could teach a behaviour that
does not pay. What it changes is that the arm would now measure the *mechanic* rather than
the agent's inability to express it.

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

Three randomly-initialised `TransformerNetwork`s through the arm's **own** selector path
(argmax on masked logits, then the joint decode), melee config, n=6 episodes, seeds 700000+.
Measured by me after an audit panel named it as the missing control.

| decode | declared/ep | stood/ep | standing fraction | coherent |
|---|---|---|---|---|
| **K=1** | 25.17 / 36.33 / 32.00 | 1.67 / 0.00 / 0.33 | 0.066 / 0.000 / 0.010 | 0.532 / 0.918 / 0.749 |
| **K=3** | 4.50 / 5.17 / 2.33 | 3.67 / 1.17 / 1.50 | **0.815** / 0.226 / 0.643 | 0.809 / 0.991 / 0.937 |

⚠ **`declared` swings 7–15× on identical weights from the decode setting alone**, and this
document never stated K. A readout that moves an order of magnitude on a setting that is not
the feature is not a readout.

⚠ **All three thresholded gates are cleared by a network that has learned nothing.** Gate 1
("declared > 0 and standing > 0") passes 3 of 3 at K=3. Gate 2 (standing fraction > 0.5)
passes on the first seed at **0.815**. Gate 3 (coherency ≥ 0.803) passes on 2 of 3 and is
marginal on the third — because at K=3 **the decoder enforces coherency for the network**.
The pre-registration's own central insight — a gate tighter than its estimator's noise is not
a gate — applies to its own gates, and one class harder: these are not merely noisy, they are
*passed by the machinery*.

**The conclusion the floor forces, and it is the useful one: at K=3 the mechanism counts
measure the DECODER, not the policy.** An untrained network stands 1.17–3.67 charges an
episode purely because the decoder picks legal combinations for it. At K=1 the same weights
stand 0.00–1.67 at a standing fraction of 0.000–0.066 — *there* the policy's own competence
shows.

### The thresholds, written against that floor

Every readout is quoted at **both** K, and the K=1 column is the one that decides.

| readout | floor (K=1, untrained) | ACCEPT | notes |
|---|---|---|---|
| standing fraction, K=1 | 0.000–0.066 | **> 0.35** | ~5× the worst floor seed; the policy is choosing charges that land |
| standing/ep, K=1 | 0.00–1.67 | **> 3.0** | above every floor seed |
| declared/ep, K=1 | 25.2–36.3 | **< 15** | ⚠ **an upper bound.** The floor DECLARES constantly and lands nothing; learning shows as declaring *less* |
| coherency | 0.532–0.918 | ≥ the walker's on the same layouts | K=3 is uninformative here — the decoder supplies it |
| standing fraction, K=3 | 0.226–0.815 | **not a gate** | the floor spans it; report it, do not gate on it |

⚠ **`declared` is an upper bound, not a lower one, and that inverts the obvious reading.**
An untrained network declares 25–36 charges an episode and stands almost none. "Uses the
lever a lot" is the *floor's* signature, not competence — the same shape as the advance
lever, where two of three seeds drifted into *using* a move that does not pay and 700 extra
epochs made it worse.

**New standing rule, earned here: any behavioural readout that gates an arm must be floored
on a random-init network through the arm's own selector path, before the arm launches.** It
costs one inference run, and it would have caught all three of these.

## What is NOT claimed here

The 2×2 above is the panel's, at n=100. My own n=6 version was internally inconsistent and
is superseded rather than retained. The opponent seat does charge: models engaged after the
opponent's charge phase go from 0.0 to 6.2–8.0 per episode.
