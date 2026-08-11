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
