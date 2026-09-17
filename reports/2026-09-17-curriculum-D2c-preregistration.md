# Pre-registration: curriculum rung D2c — PPO from the escort clone with the KL anchor, cold critic

Written 2026-09-17 16:55, **before any anchored PPO round from the clone
at this budget exists** (a 2,048-round smoke at these settings, seed 8,
was run to size the anchor and is not a result), on branch
`feature/curriculum-d2` (PR #377). Issue #378, parent question #340,
rung **D2** — third arm, one change from #376. Carries #332.

## The one change

`--kl-ref-coef 10 --kl-ref-target 0.03` on D2's recipe: the full
masked `KL(policy || clone)` over the selector and the step's head,
weighted by a coefficient the phase facade's rule adapts toward 0.03
nats per decision. Same clone (`escort-c3b-1200-s0.pt`, cold critic),
same seeds, same everything else. Wandb `curriculum-d2` tag `d2c`.

## Why

D2 collapsed from the first update. The whole-army anchor held a clone
where unanchored self-play destroyed it (`reports/2026-09-04-the-anchor-holds.md`),
and the per-model anchor's smoke from this clone held in-run success at
100% (n=10) through 2,048 rounds where D2 read 6 / 0 / 46% at 2,560. The
measured drift sat at ~0.09 nats per decision per update whatever the
coefficient — Adam normalises the penalty's gradient, so the coefficient
climbs to its cap without the drift falling — and the number that
matters is the success it protects.

## Comparator

As D2b's: the clone paired per episode (0.960 / +59.6 / ordering 1.00),
the escort (0.990 / +63.5), D2's arm beside every row.

## Criteria

D2's, unchanged (IMPROVES / HOLDS / DESTROYS at n=180 on 700000+; NULL
only if the escort fails). Readouts: the in-run curve; `train/kl_ref`
and `train/kl_ref_coef` over the run; the final policy's held-out match
against the escort beside the clone's; `alive`, `held`, coherency; the
panel; the census by phase.

## What I expect (a guess, written so it can be wrong)

HOLDS 3/3 — success 0.94–0.97 at the end with the order intact — and
IMPROVES on none: the anchor keeps the plan and the cold critic's noise
has nowhere to go, so the policy sits where it started. The fourth arm
(both changes) is where an improvement would first show.
