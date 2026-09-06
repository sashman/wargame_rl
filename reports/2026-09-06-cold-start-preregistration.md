# Pre-registration — is the bar clone buying anything self-play and 1000 epochs would not?

Written 2026-09-06 while arm 4 seeds 4–6 were at epoch 830/1000, **before the arm
was launched and before any of its numbers existed**.

## The question

Every winning lineage in this goal warm-starts from `barclone-s{n}.ckpt`, a
behaviour clone of `squad_march_take_charge` — the very policy the goal is to
beat — and is then held near it by a KL anchor. That recipe rests on an
inherited premise: *gradient descent cannot reach the coordinated policy,
because a unilateral squad advance is punished per-model, so every step toward
it is locally downhill.* That was measured on an **older, non-melee scenario**
(`reports/2026-08-16-the-cap-makes-it-a-denial-game.md`). It has never been
tested on this config, with self-play and the modern decode stack.

## ⚠ This is NOT a one-factor ablation, and it cannot be made into one

**The KL anchor's reference IS the clone's weights.** Remove the clone and there
is nothing to anchor to, so the anchor goes with it. The arm therefore differs
from its control in **two** factors at once, and the second is the single
largest lever measured in this goal. No design fixes this: an anchor to a random
init would hold the policy at random, and an anchor to the run's own past is a
different mechanism, not this one.

Anything that survives the clone's removal: the decode stack (play-time, so
weight-independent), the self-play pool and its scripted floor, the epoch
budget, `ent_coef`.

## Design

Arm: `25v25_maps_melee_approach.yaml`, **no `--warm-start-ckpt-path`, no KL
anchor**, `--self-play --pool-anchor squad_march_take_charge,squad_march_shoot`
(the same 1:1 floor as arm 4), 1000 epochs, `ent_coef` 0.003, **3 seeds (1–3)**.
Control: arm 4 at the same seeds, epochs, floor and config.

Scored exactly as the ladder is: n=45, seeds 700000+, K=3, charge decode on,
**both** `reallocate` arms, all four cells, CPU.

⚠ **UNPAIRED BY CONSTRUCTION.** The arm's init is random and the control's is a
clone, so no weights are shared and the per-seed difference is not a paired
estimator. Seeds still fix the layout and dice streams.

## Power check, before the bound

Per-seed spread on this config is 12–16 vp. An unpaired 3-v-3 comparison at
sd 15 has SE ≈ 12, so **only differences larger than ~24 vp are resolvable**.
This is stated first because this project has twice written a bound tighter than
its own estimator and had to record it as a defect in the test.

The comparison is therefore powered for the headline question — cold self-play
at 300 epochs already sits ~50 vp behind arm 4 on the mirror — and **not**
powered for anything finer. No claim about a small difference may be made from
this arm.

## Readouts, in order

1. **Does the cold arm beat the bar** on each of the four cells (the goal's own
   criterion: gap > 2 SE)?
2. **Does it match arm 4?** A cold arm that merely gets *close* does not
   establish the clone is unnecessary — the claim requires matching, and at this
   power "matching" can only mean "within ~24 vp", which is a weak statement and
   is labelled as one.

## Predictions, recorded before the run

- **The cold arm loses to arm 4 on `refereed` by the largest margin of the four
  cells.** The anchor is what won the mirror; nothing else in the recipe does.
- **It charges at least as much as arm 4.** Already observed at 300 epochs
  cold + self-play: declares 14–16/ep, stands 7.6–9.4/ep, against arm 4's
  5.9–8.3 and the bar's 5.8–6.3. Self-play teaches charging from scratch.
- **It does not beat the bar on all four cells.**
- ⚠ **It may BEAT arm 4 on an offensive cell**, and this is predicted now so it
  is not discovered later as a surprise. The record's standing diagnosis is that
  this lineage's defence is excellent and its offence is capped; an untethered
  policy has no such cap. `vs_shoot` and `vs_take` are where to look.

## What a result here does NOT license

A cold arm that loses does **not** re-confirm the inherited premise, because the
two factors are confounded — it would be equally consistent with "the anchor is
load-bearing and the clone is incidental". Separating those needs a third arm
(clone warm start, anchor OFF), which is cheap to specify and is **not** part of
this registration.
