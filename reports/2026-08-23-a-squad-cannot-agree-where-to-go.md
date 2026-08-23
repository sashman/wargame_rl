# A squad cannot agree where to go, so it never gets anywhere

**2026-08-23. No GPU. Three seeds, held-out nine, n=10 per table.**

The question was whether the models in a unit all aim at the same spot and so
lock each other in, instead of moving as a group. The answer is that they lock
each other in, and it is **not** because they aim at the same spot. It is because
they cannot agree on a direction at all — and under a 2" chain, disagreement is
the same thing as standing still.

## The scripts move as a body. The agent does not.

`just measure-angle-collapse` reads each model's own argmax heading at
`decode_topk=1`, so the play-time joint decoder cannot launder the answer.

| policy | within-squad circular variance | squads all on ONE heading | distinct bins (of 16) |
|---|---|---|---|
| `squad_march_take` | **0.0007** | **96.8%** | 1.03 |
| agent s1 | 0.1215 | 57.3% | 1.51 |
| agent s2 | 0.1475 | 52.9% | 1.56 |
| agent s3 | 0.1081 | 58.5% | 1.48 |

The script computes **one vector for the whole unit** and every model follows it
— `model_observation.py` says so outright, and calls it the reason their
formation "holds by construction". The agent's models disagree with each other on
**roughly half** of all squad-moves, on every seed.

⚠ **The decoder does not fix this.** Re-run at `decode_topk=3`, the setting the
agent actually plays at: within-squad variance **0.1312**, all-on-one-heading
**57.0%** — indistinguishable from K=1. The joint decoder filters combinations
for *legality*; it never makes a squad agree where to go. Every previous reading
of "the decoder solved coherency" is about legality only, and this is the part it
does not touch.

## And the army ends up in a heap

`just measure-squad-dispersion`, same tables and seeds, agent at K=3.

| policy | squads alive | distinct objectives held | squads per occupied point | sharing a point | gap between squad centres |
|---|---|---|---|---|---|
| `squad_march_take` | 4.79 | **3.28** | **1.21** | **31.9%** | **19.54"** |
| agent s1 | 6.24 | 2.08 | 1.94 | 77.2% | 11.71" |
| agent s2 | 6.49 | 2.30 | 1.96 | 79.1% | 12.16" |
| agent s3 | 6.27 | 2.20 | 1.89 | 75.6% | 12.50" |

**More squads alive, on fewer places.** Nearly two squads piled on every point the
agent holds against the script's 1.21, three quarters of them sharing, and the
whole army inside 60% of the script's footprint.

## The two tables are one mechanism

A squad under a 2" chain whose members pull in different directions **cannot
travel**. The constraint converts internal disagreement directly into
immobility. So:

- the script's squads translate cleanly across the table and land on separate
  objectives — 3.28 of them;
- the agent's squads mill in place, and a squad that never goes anywhere stays
  sitting on top of the one beside it — 2.08 objectives, 77% sharing.

This is what the freezing measurement was seeing from the other end: **91.8% of
frozen model-steps have a friendly base in contact against 27.7% of moving ones**,
and **75.5% of frozen model-steps have no legal shorter move along that heading at
all.** Those models are not blocked by the enemy or by terrain. They are blocked
by squadmates walking a different way.

It also resolves the standing puzzle that the agent stands still on far fewer
unit-moves than the scripts (stay share 33.5% against 65.5% here) while *moving*
much less far. It is not choosing to stand still. It is failing to go anywhere.

## Why the advance move did not pay

**Extra speed is worth nothing to a unit that cannot commit to a heading.** Three
models pulling three ways at speed 6 diverge twice as far as at speed 3, so a
longer move amplifies the disagreement instead of covering ground, and then the
chain binds or the referee reverts the whole unit-move.

⚠ This does **not** reinstate the freezing explanation that was retracted. That
one failed because the arm and the control froze at the same rate (26.3% v
18–28%), and it still does — because *both* have the same internal disagreement.
The claim here is not that the advance arm froze more. It is that neither arm
could use speed, so adding a faster move bought nothing while enlarging the
action space by 47%.

## What this does and does not license

- ⚠ **`observe_unit_centroid` is still a measured null (−62.1 vp, the worst arm on
  record).** Handing a model the direction to its unit's centroid does not make a
  squad agree; it was tried.
- ⚠ **A rigid unit-level action space is still a measured null (coherency 0.444).**
  But read its verdict precisely: rigid translation "preserves formation and
  cannot *restore* it". That is a judgement about recovering a broken formation,
  which is **not** what this data says the problem is. The problem is committing
  to a heading in the first place.
- **Untried, and different from both:** a hierarchical move where the squad picks
  one heading and each model picks an offset around it. It is not the centroid
  observation (which added an input and changed nothing about the action), and it
  is not rigid translation (which removed per-model freedom entirely).
- ⚠ **It is an action-space change, so it is UNPAIRABLE by construction** — the
  same trap the advance move fell into. Budget for unpaired variance, and build
  the cross-config bridge first: verify a non-committing policy scores identically
  on both configs.

## The one number to watch

**Within-squad circular variance.** The script sits at 0.0007 and the agent at
0.11–0.15 on every seed. Any intervention aimed at this must move that number,
and it can be read for free on frozen weights before a single epoch is scored.

## Reproduce

    just measure-angle-collapse <policy|ckpt> configs/experiments/24v24_maps_spare_squads_refereed.yaml 10 1
    just measure-squad-dispersion <policy|ckpt> configs/experiments/24v24_maps_spare_squads_refereed.yaml 10 3
