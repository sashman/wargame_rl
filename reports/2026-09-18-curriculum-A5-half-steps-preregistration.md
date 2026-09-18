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

## Amendment 1 — written 2026-09-18 18:10, A5-bodies at ~106k rounds after a reboot, A5-points at ~29k

**The machine rebooted at about 15:20** (a crash, cause not recorded)
and killed every process: the three A5-bodies runs at about 84k rounds
and the 2,000-game clone fit four and a half hours in. At 17:01–17:05
A5-bodies was **resumed seed-for-seed from each run's `last.pt`**
(`--resume-from`; the first evaluation after the resume at 84,480 /
81,920 / 84,992 rounds, so under three thousand rounds of each run are
replayed), the clone fit was relaunched from the start, and
**A5-points launched from scratch beside them** rather than waiting for
the fit to release its memory as the plan above said — the box carries
the six trainers and the fit together at 128 rounds per update with
about four gigabytes to spare, and the points half-step is the one this
pair cannot be read without. Same recipe as written: 122,880 rounds,
`--ent-coef 0.003`, the `mean` credit, in-run eval every 512 rounds on
500000+, Wandb `curriculum-a5` (A5-bodies resumed `j1e0mpam` /
`0ioij69h` / `nazcv56j`, the pre-reboot runs `5qbyx99r` / `w15c7xih` /
`3vb272ip`; A5-points `uagmul66` / `ici5py46` / `wedjjozy`).

A resume replays a few thousand rounds under a fresh rollout seed and
a fresh Adam state, so the resumed curve is not the same trajectory the
run would have followed; on the criteria it changes nothing — the read
is at 122,880 greedy on 700000+, and the dip rule counts in-run
evaluations from the first rolling pass, which no seed had reached
before the reboot (in-run at 80k: 91 / 62 / 71%, held 2.6–2.9 of 3).
Written before either half-step's end read: A5-bodies' rolling in-run
success over the last eight evaluations at ~106k is 94 / 96 / 98%
(held 2.90–3.00), climbing from the 80k plateau; A5-points at ~29k
reads 0% in-run with 0.7–1.5 of five points held. Neither number is a
result.
