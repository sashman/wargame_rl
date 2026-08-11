# Spreading objectives lowers the bar, not the abandonment

**2026-08-11** · `configs/experiments/25v25_spread_{objectives,hold}.yaml` ·
wandb group `spread-symmetric` · two arms x two seeds, 600 epochs, `--no-tf32`

## The question

The agent lost to the scripted bar by 17 vp on the clustered scenario, entirely
by abandoning 38% of objectives. Two hypotheses, in order of what the evidence
favoured at the time:

1. **Reward weighting.** The income breakdown showed the approach term earning
   ~zero and a *global* term outpaying the per-model one.
2. **Scenario geometry.** All three objectives sat inside a ~16" circle with 47%
   of pairs inside one weapon range, so there was no travel trade-off to price.

## The answer

**Neither fixes the abandonment. One of them is still worth keeping.**

n=100, seeds 700000-700099, epoch 600:

| arm | s1 | s2 | mean | vs bar |
|---|---|---|---|---|
| `spread_objectives` (control weights) | +42.0 | +36.3 | **+39.2** | -19.7 |
| **`spread_hold`** (`objective_hold` 2.5, `objective_coverage` 0.1) | +48.2 | +49.7 | **+49.0** | **-9.9** |
| `squad_march_shoot` (the bar) | | | **+58.9** | — |

### What replicated

`objective_hold` 1.25 -> 2.5 with `objective_coverage` 0.3 -> 0.1 is worth
**+9.8 vp**, both seeds, 1.5 apart. On the clustered scenario the same change
measured +5.2 at ~1.5 sigma — weak. Reproducing it at nearly twice the size on a
*different scenario* makes it a real result. The mechanism is visible in
`on_obj`: 0.77-0.82 against the control's 0.64-0.77.

Reading: moving income from a **global** term (broadcast whole, absorbed by the
value baseline, cannot differentiate one model's choice from another's) to a
**per-model** one sharpens the gradient. That is the same lesson as
[the crowding result](2026-08-08-paying-the-pot-beats-the-bar.md), arrived at
from the opposite direction.

### What was refuted

**`overstack_penalty_per_extra` is not the constraint.** Zeroing it (verified
logged at exactly 0.0000) scored +49.4 against the control's +50.6 — null. The
income breakdown that motivated it was read correctly and the conclusion drawn
from it was wrong: the approach term nets ~zero because the penalty cancels the
progress reward, but removing the penalty does not make the agent approach.

**Spreading objectives does not fix abandonment.** It was the better-supported
hypothesis, measured at 3.6 sigma as a natural experiment, and it did not
transfer to training.

### The gap narrowed, but not the way it was supposed to

| scenario | agent | bar | deficit |
|---|---|---|---|
| clustered | +50.6 | +67.7 | -17.1 |
| spread, symmetric | +49.0 | +58.9 | **-9.9** |

**The bar fell 8.8; the agent did not move.** The natural experiment predicted a
clustered-trained agent gains ~+37 paired margin on spread layouts, and it does
— but an agent *trained* on spread layouts gains nothing absolute. Two candidate
explanations, not separated: the symmetry fix left the contrast smaller than the
one that produced the effect (10.7" min separation against the spread group's
11.7"), or transferring-in and training-on are different things here.

## The invariant

| | control | no_overstack | hold_over_coverage | spread_hold |
|---|---|---|---|---|
| **abandoned** | 0.380 | — | 0.370 | **0.357** |
| surplus models on held | 8.58 | — | 9.66 | 9.40 |

Against the bar's **0.147 abandoned** and **3.88 surplus**.

Five weight configurations and a scenario change have moved abandonment by 2.3
points. The agent is pushed off an objective **1.7%** of the time against the
bar's **24.7%** — it is not losing fights over objectives, it is not turning up.
Meanwhile it wins the firefight (firepower 1.59-1.74 against 0.85) and keeps
0.60-0.63 alive against 0.50.

**That invariance is the most robust finding here**, and it says the cause is
neither the reward weighting nor the objective spacing.

## What this does not support

- **Not "spreading objectives was wrong".** It makes the scenario match the real
  layouts, it lowered a bar that was inflated by objectives packed close enough
  for one squad to cover all of them, and it is where the honest comparison now
  lives. It just is not the fix for this failure.
- **Not "the agent cannot learn the scenario".** It beats every movement-only
  baseline and wins the shooting exchange roughly two to one.
- **Not a verdict on `crowding_exponent`.** Still never retuned for objectives
  holding ~18 models. It was the prime suspect and the income breakdown
  eliminated it before a run was spent on it — at `a=1.0` the pot is conserved,
  so the ninth model on a point already earns a ninth of a lone model's pay.

## Next, and what to stop

**Stop** tuning reward weights and scenario geometry against the abandonment.
Three rounds, seven arms, two scenarios, invariant behaviour.

**The untested hypothesis is representational.** An objective reaches the network
as a location, a control count and a size. Nothing encodes "an objective nobody
is contesting is worth walking 20 inches for", and the value function must infer
a multi-round investment from a per-step reward that pays **zero during the
walk**. That is credit assignment and observability, not weighting — and it is
consistent with a hand-written heuristic beating a 12.9M-parameter network at
exactly this one thing while losing every firefight.

Cheapest probes, in order:

1. **Does a model that leaves a crowded objective ever get paid?** Instrument
   per-model reward along a transit and check the integral against staying. If
   staying dominates for the whole walk, no weighting fixes it and the answer is
   a potential-based term on *assigned* objective, not nearest.
2. **`closest_objective_v2.fallback_to_nearest: true`** sends surplus models to
   their *nearest* objective, which is the one they are already standing on.
   Strict de-stacking (`false`) is a one-line arm and directly targets the 9.4
   surplus models.


---

## Follow-up: filling the trough with `progress_scale` makes it worse

Two probes on the `spread_hold` agent aimed the next arm, and the arm was
refuted. Recording both, because the probes were right and the intervention
drawn from them was wrong.

### Probe 1 — the walk is a trough, and the reward is not mis-weighted

Per-model income by what the model is doing (`phase_manager.last_per_model_reward`,
n=15 episodes):

| activity | reward/step |
|---|---|
| alone on an objective | **1.30** |
| on a crowded one (8 others) | 0.62 |
| loitering outside | 0.50 |
| **in transit** | **0.26** |

Moving toward an objective pays **half of standing still doing nothing**. But the
crowding gradient itself works — pay falls monotonically 1.30 → 0.36 from one
occupant to twelve — so arriving at an empty objective roughly *doubles* a
model's income. Leaving costs 0.383/step over a ~3.3 round walk (~1.26 of certain
loss), then pays +0.66/step, breaking even ~1.9 rounds after arrival. Over a
20-round episode, leaving is clearly profitable.

**So the reward is not mis-weighted.** The landscape has a trough between two
peaks and the policy sits on the smaller one. Crossing needs a *sustained*
multi-round walk that single-step exploration is punished for starting. That is
why five weight configurations moved abandonment by 2 points.

### Probe 2 — the assignment confirms the pile

**98.4%** of models standing on an objective have that same objective as their
assigned target. With `fallback_to_nearest: true` a model on an objective is
always nearest to it, so `closest_objective_v2` tells it to stay and pays it
progress for a journey of zero length.

`fallback_to_nearest: false` was the planned fix and was **dropped before
running**: an unassigned model returns `overstack_penalty` and *zero* progress
reward, and group assignment gives each objective to one group, so with 5 squads
and 3 objectives two squads would earn nothing but a penalty. It deepens the
trough.

### The arm: refuted, monotonically

Transit earned 0.207/step of progress at `progress_scale` 6.0, so covering the
0.383/step gap calculates to ~17. Two arms, two seeds, scored at epoch ~270,
n=100:

| arm | `progress_scale` | s1 | s2 | mean |
|---|---|---|---|---|
| `spread_hold` (@300) | 6.0 | +46.4 | +47.0 | **+46.7** |
| `progress12` | 12.0 | +37.7 | +33.1 | **+35.4** |
| `progress17` | 17.0 | +37.1 | +27.0 | **+32.1** |

**Monotonically worse in the scale**, all four runs below the arm they were built
on. Killed at ~epoch 280 rather than spend two more hours; last round gained only
+2.3 between epoch 300 and 600, which does not change the ranking.

### The reasoning error, which is the part worth keeping

The arm was justified as safe to push hard because progress is potential-based
and "telescopes to `scale × (initial − final distance)`, so oscillation cannot
farm it". **That holds only if the potential function is fixed, and it is not.**
`target_switched` measured **0.273/step in transit** — models change assigned
objective constantly, and each switch re-bases the distance the term measures
against. The shaping is farmable, and raising the scale amplifies the farm rather
than the intended pull.

The lesson generalises past this arm: *potential-based shaping is safe by
construction only when the potential is a fixed function of state.* A shaping
term keyed on an **assigned** target is not potential-based at all if the
assignment can change mid-episode.

The untried repair is therefore to fix the *switching*, not the scale: pin a
model's target once assigned, which would make the telescoping argument true and
would also stop 98.4% of models being re-assigned to the ground under their feet.

---

## Correction (2026-08-11, same day): the 0.273 switch rate is an artefact

**The repair proposed immediately above was measured before it was built, and it
is not worth building.** Two probes on the same checkpoint:

`target_switched` in transit is **0.292** — and **0.063 once each model's first
step of the episode is excluded**. Of 335 transit switches, **280 are each
model's single initial assignment**, which lands in the transit bucket because
every model is walking on step 1. There is exactly one per model per episode and
no pin can remove it.

| bucket | `target_switched` | after step 1 |
|---|---|---|
| on an objective | 0.0230 | 0.0189 |
| **in transit** | **0.2921** | **0.0634** |
| loitering | 0.0182 | 0.0182 |

So *"models re-base their target constantly"* is false. Real mid-walk churn is
one switch per ~16 steps of walking against a walk of ~3.3 rounds — **most walks
are never interrupted at all**. A switch costs that step's progress (~0.155 at
`progress_scale` 6.0), so the whole effect is worth **~0.01/step against a
0.383/step trough: about 2.5%**. Classifying the switches makes it worse for the
proposal — only **8 of 203** are candidate-to-candidate churn; **96% involve the
`fallback_to_nearest` selection** on one side or the other, which pinning genuine
targets would not touch.

**What this means for the arm above.** The refutation stands — `progress_scale`
12 and 17 really are monotonically worse — but *the farming explanation for it is
withdrawn*. At 0.063/step the potential is close enough to fixed that telescoping
approximately holds, so amplifying a farm cannot be what happened. Two candidates
remain, unseparated: at scale 17 the progress term is worth ~0.44/step against a
per-term budget of ~10 an episode, so it simply crowds out `objective_hold` and
`vp_gain`; and it is symmetric, so every repositioning, retreat or lateral
manoeuvre is penalised at the same amplified rate.

**The general lesson survives intact and is worth more than the arm**: a shaping
term keyed on an assigned target is only potential-based while the assignment
holds. The error here was the *second* one — quoting a rate that bundled an
unavoidable initialisation into a churn measurement, and then designing against
it. **Bucket a rate by whether the event could have been otherwise before
building on it.**
