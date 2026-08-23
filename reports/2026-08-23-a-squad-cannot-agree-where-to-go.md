# A squad cannot agree where to go

> ⚠ **The causal claim in this report was RETRACTED the same day, after two expert
> panels attacked it. Read the CORRECTION at the end FIRST.** The descriptive finding
> (the agent covers a third fewer objectives with the same squads) survives and is
> larger than published; the explanation does not.

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

---

# CORRECTION, 2026-08-23 (same day)

**The causal claim in this report is RETRACTED. The descriptive finding stands and is
larger than published. The instrument had a bug.**

Two independently-run expert panels were given this report and asked to attack it before
building on it. Both refuted the causal half, by different routes, and one found a bug in
the measurement. Everything below has been reproduced here.

## 1. The instrument had no movement-phase filter

`measure_angle_collapse` decoded **every** non-STAY action as a heading, including the
shooting slice (indices 97-104), via `(action - 1) // n_speed_bins` -- which lands on angle
bin 16 of a **16**-bin wheel. Squadmates shoot the same target, so those rows read as
unanimous and diluted whichever policy shoots more. Now fixed, with a phase guard and a
hard assert on the bin index. Corrected, movement phase only:

| policy | variance | all on ONE heading | stay share |
|---|---|---|---|
| `squad_march_take` | 0.0006 | 97.9% | 56.9% |
| agent s1/s2/s3 | **0.142-0.190** (published: 0.108-0.148) | **41.6-47.7%** (published: 52.9-58.5%) | **0.0-4.0%** |

⚠ **The published "stay share 33.5% v 65.5%" is RETRACTED as a phase artefact.** Movement
phase only it is ~0% against 56.9%, which *confirms* CLAUDE.md's standing 0.4%-v-38-57%
figure rather than superseding it.

## 2. The headline statistic mostly measures ARCHITECTURE, not skill

The control this report should have run, and did not: `clone_squad_march_take.ckpt` is a
**factored per-model transformer behaviour-cloned from the winning script**. Same
architecture as the agent, same policy as the script. On
`configs/evaluation/25v25_maps_vs_squad_march_deny.yaml`, n=10 per table, corrected
instrument:

| | within-squad variance | all on ONE heading | per-model modal agreement |
|---|---|---|---|
| `squad_march_take` (teacher) | 0.0033 | **91.8%** | 0.979 |
| **a clone of that teacher** | 0.0781 | **42.2%** | 0.806 |
| agent | 0.2217 | 35.1% | 0.770 |

Normalising for squad size (p = all-on-one^(1/(k-1)), k=5): **83% of the script-to-agent
gap is bought by the ARCHITECTURE alone** -- a factored per-model policy cannot reproduce a
shared vector -- and only **17%** is the agent being worse than a clone of the winner.

⚠ **"Squads all on one heading" is not fit to carry a diagnosis.** It ranks a clone of the
winning script (42.2%) barely above the agent (35.1%) and nowhere near its own teacher
(91.8%). Report per-model modal agreement, and always beside a clone control.

## 3. The agent is not failing to move

Executed squad-centroid travel per squad-step: **agent 2.82" against the script's 2.05"**.
It covers ~40% more ground while holding a third fewer objectives. The report's central
sentence -- "the agent's squads mill in place, and a squad that never goes anywhere stays
sitting on top of the one beside it" -- is **wrong**. It is not failing to go anywhere; it
is going somewhere useless, constantly.

## 4. Forcing agreement was tested directly, and LOSES

Consensus decoding on frozen weights -- the play-time realisation of "make the squad agree"
-- drives within-squad variance to **exactly 0.0000** and buys 7.8% more travel, and costs
**-4.8 / -4.1 / -9.1 vp, 3/3 seeds negative** (two independent implementations, second
reading -9.7 / -9.5). `held` 2.03 -> 2.02.

⚠ **So "make the squad commit to a heading" is a measured null.** Do not fund it. The
"one number to watch" this report nominated should be retired as a target: driving it to
zero is worth about -6 vp.

## What SURVIVES

**The dispersion finding**, and it survives its confound. Both panels reconstructed the
numerator independently: the script puts **3.97** squads on objectives and the agent
**4.03-4.51** -- essentially identical -- and the agent crams them onto **2.08-2.30**
distinct points against the script's **3.28**. Correcting for squad count makes the gap
*larger*: distinct objectives per alive squad, script 0.685 against agent 0.35. Against a
random-allocation null the agent allocates **worse than chance**.

So the agent really does cover a third fewer objectives with the same number of squads.
What is refuted is the *explanation*, not the phenomenon.

## The better candidate, which CLAUDE.md already told us to check

`closest_objective_v2` with `fallback_to_nearest: true` (the shipped training config) pays
each model `progress_scale * (delta_inches / board_diagonal)` = **+0.081 per inch closed on
the CENTRE POINT** of its own nearest objective. The target is chosen **per model**
(`_choose_target_objective`), and `norms_offset` saturates only within ~0.63" of the
marker centre, **not** at the control radius. Three consequences, each matching a standing
measurement:

- **STAY is strictly dominated** -- standing still earns 0.000 from this term, any inward
  step earns +0.081/inch, and `measure-hold-hazard` established the excess death hazard is
  *negative*. The agent's ~0% stay rate is the correct policy for the MDP it was trained in.
- **Every model is pulled to a point of measure zero** that one or two bases can occupy --
  a direct stacking generator, matching 4.90 models on the top point and 91.8% friendly-base
  contact on frozen model-steps.
- **Members of one squad can hold DIFFERENT targets.** `_compute_group_assignment` assigns
  objectives to *groups*, but 8 groups over 5-6 markers leaves 2-3 unassigned every step, and
  `fallback_to_nearest` then drops each of their models onto its own nearest marker. **Two
  members of one squad are paid to walk apart.** Compounding it, a target switch returns
  progress 0.0 and re-anchors the baseline, so **abandoning a target is free**.

That makes heading disagreement a **symptom** of a per-model reward pulling squadmates to
different points, not a cause of immobility.

⚠ **CLAUDE.md's holding-pays report already said to check `fallback_to_nearest`, "which
pays an unassigned group to close on the nearest point, usually one already held". That
check was never done, and this report went to angle variance instead.** Do it before
anything else.

## The methodological lesson

**A statistic that separates two policies does not thereby explain the difference.** The
clone control costs one inference run and would have caught this before publication: if a
clone of the *winning* policy scores near the *losing* one on your statistic, the statistic
is measuring your architecture. **Run the clone control on any behavioural statistic before
building a diagnosis on it.**
