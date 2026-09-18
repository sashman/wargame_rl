# Pre-registration: the A5 half-steps — A5-bodies and A5-points

Written 2026-09-18 13:40, **before any training round on either exists**,
on branch `feature/per-model-actor-credit` (PR #383). Parent question
#340. Asked by Sash: is the A4 → A5 jump too big? A4 is four squads over
three points (twelve bodies); A5 doubled both axes in one rung, against
the ladder's one-axis rule, and fails every per-model arm on file
(A5b 0.26 / 0.18 / 0.16; T1 from A4x 0.45 / 0.29 / 0.20; A5d and A5e
under the actor credit 1–14% in-run). These two rungs separate the axes.

## The rungs

| | A5-bodies | A5-points |
|---|---|---|
| config | `configs/experiments/curriculum/a5_bodies.yaml` | `configs/experiments/curriculum/a5_points.yaml` |
| army | 24 bodies, eight squads of three (A5's) | 18 bodies, six squads of three |
| points | THREE, A4's column at x=35 | FIVE: A5's near column of three, a far column of two |
| everything else | A5's band, ten rounds, radius 4, A5's `arrive` reward, `all_objectives_occupied`, `terminate_on_success` | the same |
| bar (`squad_march_take`, n=100, 700000+) | **1.000**, held 3.00, 4.43 turns, coherent 0.759 | **1.000**, held 5.00, 6.71 turns, coherent 0.832 |
| bridge | identical on every shared field (n=3) | identical |

## The arms

Per-model from scratch, the ladder's standard recipe (128 rounds per
update, `--ent-coef 0.003`, the default `mean` credit so the rows pair
with A4x and A5b), seeds 1 / 2 / 3, **122,880 rounds** (the cap; a rung
that fails at the cap is a real fail), in-run eval every 512 rounds at
n=30 on 500000+, Wandb `curriculum-a5`, tags `a5bodies` / `a5points`.
A5-bodies launches now beside the 2,000-game clone fit; A5-points when
the fit releases its memory. No whole-army control on these — they are
diagnostic half-steps, not ladder rungs, and the budget is the cap.

## Criteria

Read greedy at n=100 on 700000+ at 122,880.

- **PASS:** success ≥ 0.95 on all three seeds, no in-run dip below 0.80
  after the first rolling pass.
- **FAIL:** any seed below 0.95; then the census (bodies on points,
  held, max stack, turns).
- **NULL:** only if the bar fails its own criterion; both read 1.000.

## What the pair says

- bodies PASS, points FAIL: the wall is the number of points to cover at
  once — the conjunction the critic first struggled to value on A3.
- bodies FAIL, points PASS: the wall is the army — twenty-four bodies
  dissolve their squads (A5b's coherency 0.10) before they arrive.
- both PASS: A5's difficulty is the interaction, and A5 warm-started
  from A5-points is the next arm.
- both FAIL: the jump was not the problem; A5's wall is already present
  one half-step up from A4 on either axis.

## Prediction

Bodies passes (0.95–1.00 by 60k rounds: three points with eight squads
is A2's one-point problem three times over); points fails on at least
one seed (0.85–0.95), the far column's points the ones left short.
