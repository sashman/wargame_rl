# The critic already knows the stack is wrong

**2026-08-23. No GPU. Frozen weights, three seeds, 634 forked games.**

An eleven-expert panel review converged, across two independently-run panels, on
one diagnosis: the per-model return is gated on the model's own survival
(`reward/phase_manager.py:272-274` pays the global stream only to `alive_models`),
so the agent has learned a survival premium the game does not contain. Both
panels' top-ranked intervention followed from it — pay the dead, or redistribute
the global pot by pivotality.

**That diagnosis is refuted.** It predicts the critic prefers a surplus model
standing on a held point to the same model taking an empty one. The critic
prefers the opposite, on every seed, at t = +8.3.

## What was measured

`just measure-critic-probe` forks a live game at a chosen battle round, finds an
over-stacked objective and an empty one, and **rigidly translates** one surplus
squad from the first to the second. Rigid because a translation preserves the
unit's internal geometry and so cannot break the 2" chain, which would otherwise
confound the value read with a coherency penalty. The donor must leave the
stacked objective still controlled, so the question is what a *surplus* model is
worth, not what abandoning a point is worth.

Two numbers per fork:

- **dV** — the critic's per-model values summed over the living army,
  counterfactual minus factual. What the agent *believes* redistribution is worth.
- **dVP** — the realised `vp_margin`, counterfactual minus factual, from playing
  both branches to the end. What it *is* worth.

Both branches continue from a `deepcopy` taken at the fork, so they share the dice
stream to the first divergence — common random numbers, not a perfect pairing,
because the two states consume the RNG at different rates once they differ.

Three seeds of `24v24_maps_spare_squads` at 300 epochs, scored on
`24v24_maps_spare_squads_refereed.yaml`, held-out nine, 10 episodes per table,
K=3, forks at rounds 3, 6 and 10.

## The control that makes it mean anything

The counterfactual is a state the policy never produces, and critics are
optimistic off-distribution. So the probe was run **backwards** as well: take a
squad already holding a point of its own and stack it onto the army's biggest
pile. A critic that merely extrapolates upward scores *both* counterfactuals
above the factual.

| direction | n | dV (critic) | dVP (realised) |
|---|---|---|---|
| **forward** — spread a surplus squad onto an empty point | 397 | **+2.63 +/- 0.32** (t = +8.32) | **+3.85 +/- 1.81** (t = +2.13) |
| **reverse** — stack another squad onto the pile | 237 | **-7.18 +/- 0.58** (t = -12.40) | **-11.52 +/- 2.51** (t = -4.59) |

Per seed and round, `dV` is positive in 6 of 6 forward cells and negative in 6 of
6 reverse cells. The critic is directionally correct **both ways**, and it gets
the asymmetry approximately right: it rates stacking 2.7x worse than spreading is
good, against a realised 3.0x.

## What this kills

**The survival-premium account of the agent's stacking.** The critic is trained on
the very reward the account says over-prices survival. If that were the mechanism,
the critic would value the surplus model staying put. It does the reverse, at
t = +8.3 forward and t = -12.4 backward.

That does not make the broadcast at `phase_manager.py:272-274` a good design — it
is still 53.7% of income paid for being alive rather than for anything a model
did, and it is still the largest untouched variable in the reward. It means the
broadcast is **not the reason the agent stacks**, so `dead_share_fraction` and
pivotality redistribution should not be funded *on that rationale*. Either may
still be worth running for its own reasons; neither is the fix for offence.

## What is left

**The reward and the critic both correctly value spreading. The policy does not do
it.** That is a search and optimisation failure, not an attribution one — and it
is the branch the probe's own pre-registered decision table assigns:

    dV > 0, dVP > 0   the critic already wants the spread. The failure is SEARCH;
                      do not spend a run on reward shaping.

Two qualifications, both against the finding's strength:

- **`dVP` forward is weak.** +3.85 +/- 1.81 pooled; no single seed-round cell is
  significant. "Redistribution pays" is supported, not established. The reverse
  direction is far stronger (-11.52 +/- 2.51), which is the asymmetry below.
- **`corr(dV, dVP)` is ~0** (mean +0.07 forward, +0.08 reverse). The critic has the
  level and the direction, and no grip on *which* particular redistribution pays.
  A search method that trusts the critic to rank candidate reallocations will not
  work; one that only needs the direction will.

## The asymmetry is the useful part

Marginal spreading gains +3.85. Marginal stacking loses -11.52. The agent sits
**slightly past** the optimum on the over-stacked side, with a shallow gradient
out and a steep one further in. That reframes "the agent hoards": it is not
parked in a basin the reward dug for it, it is a little way past a broad optimum
and the return for climbing out is small.

⚠ It also predicts that any lever measured by how far it moves top-stack occupancy
will read as a large behavioural change for a small score change. Read `dVP`, not
occupancy.

## The allocation ceiling, tested and inconclusive

`assignment_optimal` is `squad_march_take` with one thing changed: the greedy
squad -> objective matching (cheapest ground first, nearest unassigned squad to
each) is replaced by an exact minimum-cost assignment over the same preference
order, by subset DP, verified against brute force on 300 random instances.

| policy | vp_margin | held | alive | coherent |
|---|---|---|---|---|
| `squad_march_take` (greedy) | **+7.6 +/- 3.8** | 2.80 | 0.311 | 0.947 |
| `assignment_optimal` (exact) | **-26.1 +/- 9.4** | 2.21 | 0.336 | 0.943 |

Globally optimal matching is **33.7 vp worse** than the greedy heuristic.

⚠ **Do not read this as "allocation is at its ceiling."** The pre-registered
criterion was "within +/-5 vp ⇒ near the ceiling", and -26.1 satisfies it by the
letter, but the honest inference is narrower: **one plausible untuned cost model
loses badly to greedy.** It is weak evidence that easy allocation gains are not
lying around, not proof that none exist. Tuning would have to happen on the 36
non-held-out tables, never on the nine.

It does bear on one proposal directly: an **allocation-aware decode** (a Hungarian
squad -> objective assignment outside the network, predicted at +4 to +8 vp) would
be replacing the greedy rule that just beat its exact counterpart by 33.7. Re-cost
before funding.

## Reproduce

    just measure-critic-probe <ckpt> configs/experiments/24v24_maps_spare_squads_refereed.yaml 10 6,10 3
    just measure-critic-probe <ckpt> configs/experiments/24v24_maps_spare_squads_refereed.yaml 10 6,10 3 reverse
    just measure-maps assignment_optimal configs/experiments/24v24_maps_spare_squads_refereed.yaml 30 configs/evaluation/maps_heldout 1
