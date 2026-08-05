# 2026-08-05 — Does the agent break line of sight when the opponent shoots back?

**Question.** When the enemy returns fire and the ruins move every episode, does the
policy do anything beyond rushing forward and out-shooting it? "Cover" here means one
thing only: deliberately breaking line of sight. There is no cover save in this game —
terrain blocks LOS and nothing else.

**Outcome.** Return fire changes positioning by ~4.7x, and the effect is not survivorship.
The agent ends up **less exposed and more alive than any policy that contests objectives**,
including the shooting baseline — but it does *not* out-score that baseline, and there is
**no positive evidence yet that terrain is the mechanism**. A no-terrain control (batch 2,
arm E) is running to settle that. The reward curriculum was refuted.

## What had to be built first

The question was unanswerable as the repo stood:

- **Nothing shot back.** `ScriptedAdvanceAndShootPolicy` shipped in PR #132 but no config
  used it, so terrain constrained only the *player's* shooting and cover had no value.
- **Nothing measured it.** `analyze_match` has no LOS, casualty or damage metric, and
  `GameStateSnapshot` carries no terrain, so it could not be recovered from traces.
- **Terrain was static**, built once at `__init__`. A policy that looked like it used cover
  might only have memorised seven rectangles; the two are indistinguishable in the numbers.

PR #135 added per-episode random terrain (fixed piece count — observation batching stacks
terrain, so the count cannot vary), `exposure_rate` / `terrain_proximity` /
`fraction_alive`, and added `squad_march_shoot` to the auto-logged baselines.

## Setup

Three arms, PPO + transformer, 1000 epochs each, one seed per arm, Wandb group
`train-multi-2026-08-05-00-36-25`. 25v25 on 60x44, 3 random objectives, 7 mirrored random
ruins regenerated per episode, weapon range 12, 20 battle rounds, movement + shooting
phases. All three share one reward (six calculators, `vp_gain` weight 2.0); arms A and C
are single-phase, arm B is the two-rung ladder.

| arm | run | opponent | reward |
|---|---|---|---|
| A | `ttmerrnr` | `scripted_advance_and_shoot` | single phase |
| B | `rhizvkte` | `scripted_advance_and_shoot` | 2-rung curriculum |
| C | `2fgj1zny` | `scripted_advance_to_objective` | single phase |

## Measured — training curves

Rolling means over 1040 eval points; last 100 epochs in bold.

| arm | exposure | terrain_d | alive | vp_margin | win |
|---|---|---|---|---|---|
| A, first 174 epochs | 0.220 | 5.98 | 0.455 | −45.1 | 28.7 |
| **A, last 100** | **0.099** | **5.85** | **0.706** | **+13.6** | **53.9** |
| **B, last 100** | **0.089** | **5.76** | **0.727** | **+8.6** | **52.7** |
| C, first 174 epochs | 0.544 | 6.12 | 1.000 | +57.0 | 81.1 |
| **C, last 100** | **0.463** | **5.99** | **1.000** | **+115.7** | **100.0** |

## Measured — held-out evaluation

Arm A's best checkpoint (`ppo-927`) and every scripted baseline on **identical layouts**,
seeds 700000–700029, scored through the same `evaluate_selector` path.

| policy | on_obj | win | player VP | opp VP | alive | exposure | terrain_d |
|---|---|---|---|---|---|---|---|
| random | 0.015 | 0.00 | 10.8 | 167.8 | 0.811 | 0.016 | 6.9 |
| greedy_nearest | 0.700 | 0.23 | 90.5 | 157.2 | 0.436 | 0.299 | 4.6 |
| split_evenly | 0.700 | 0.23 | 86.8 | 157.7 | 0.225 | 0.298 | 4.7 |
| squad_march | 0.833 | 0.27 | 88.2 | 147.5 | 0.271 | 0.368 | 5.1 |
| **squad_march_shoot** (the bar) | 1.000 | **0.63** | 130.8 | 109.8 | 0.388 | 0.220 | 5.0 |
| **agent (arm A)** | 0.882 | 0.53 | 111.0 | 117.5 | **0.660** | **0.127** | 4.6 |

## What this supports

**1. Return fire changes positioning, by a lot.** Arms A and C differ in exactly one
thing — whether the enemy can shoot. Same reward, same terrain generator, same board. With
return fire the policy converges to 0.099 exposure; without it, 0.463. That is not a
general "stay hidden" prior; it is a response to being shot at. *Measured.*

**2. The survivorship confound is ruled out, in the right direction.** `exposure_rate`
averages over alive models, so casualties depress it on their own. In arm A exposure fell
0.220 → 0.099 while survival **rose** 0.455 → 0.706. The confound predicts exposure falling
when mortality *rises*; the opposite happened. *Measured.*

**3. It is not standing off.** `on_obj` 0.882 against the bar's 1.000 — the agent is on the
objectives, not hiding at the board edge. The `random` failure mode (exposure 0.016 for
10.8 VP) is decisively not what this is. *Measured.*

**4. The agent dominates every movement-only baseline and trades against the shooting
one.** Win 0.53 vs 0.23–0.27 for the movement-only baselines. Against `squad_march_shoot`
it takes 42% less exposure and 70% more survivors for 15% less VP and 0.10 less win rate.
*Measured.* That it is a deliberate trade rather than a partial failure is *inferred*.

**5. The reward curriculum is refuted for this scenario.** Arm B reached rung 1 and
converged to the same place as arm A (win 52.7 vs 53.9, VP +8.6 vs +13.6, exposure 0.089
vs 0.099). It is not better and it is not worse; it is the same policy reached more slowly.
*Measured, one seed.*

## What this does NOT support

**Terrain is not established as the mechanism.** `terrain_proximity` is flat at 5.76–6.03
across *all three arms* and across the whole of training, and sits inside the baselines'
range (4.6–6.9). Nothing in the data shows the policy moving toward ruins. Its held-out
5.4→4.6 edge over the bar is within the spread of the movement-only baselines.

Proximity is a weak proxy — a ruin can break a sightline from 15 cells away, so this does
not *refute* cover. But there is currently no positive evidence for it. Two hypotheses
remain live and the data cannot separate them:

- **Cover**: the policy positions so ruins fall on enemy sightlines.
- **Range management**: the policy fights at the edge of the 12-cell band, where most
  enemies simply cannot reach it, and terrain is incidental.

There is a third contributor either way: **killing the enemy also lowers exposure**, since a
dead opponent is one fewer model with line of sight. The agent's opponent VP of 117.5 says
it is not thinning the enemy as hard as the bar does (109.8), so this is unlikely to be the
whole story, but it is not zero.

## Defect found and fixed during the experiment

Objective placement drew each objective independently, with no separation constraint at all
(`placement.py:145-148`). Measured over 400 episodes on this config:

| | frequency |
|---|---|
| two objective discs overlap | **25% of episodes** |
| an objective centre inside another's disc | 7.8% |
| an objective inside a ruin | **11% of objectives** |

A quarter of episodes were silently running as a two-objective mission. An objective inside
a ruin is also not covered ground — the see-out / see-into rule means its occupant is still
visible to everything outside, so the ruin protects nobody while still blocking that lane.

`objective_min_separation` and `objective_terrain_clearance` now constrain the draw. **Both
default to the old behaviour, and every number above was measured without them** — so
batch-2 results are not comparable to this report's.

## Open — batch 2

Four arms, all with `objective_min_separation: 6`, each varying one thing from a matched
control (`25v25_terrain_range12`):

| arm | varies | decides |
|---|---|---|
| E `noterrain_range12` | no terrain at all | **the open question.** With zero ruins the only ways to lower exposure are range and mortality. If E still reaches ~0.10, terrain was doing nothing |
| F `terrain_range24` | weapon range 12 → 24 | at range 12 most of the board is out of range, so range is what keeps a model safe. Doubling it makes range nearly free and leaves LOS as the only lever |
| G `terrain_range12_clear_objectives` | objectives ≥5 from any ruin | whether un-coverable objectives were suppressing cover use |

## Confounds and limits

- **One seed per arm.** Adequate to establish the 4.7x A-vs-C gap, which is far outside
  epoch-to-epoch noise; inadequate to rank A against B, whose difference is within it.
- **`exposure_rate` has four documented traps** — see [docs/metrics.md](../docs/metrics.md)
  § Cover metrics. Absolute levels are not comparable between arms with different mortality;
  arm C loses nobody, so its 0.463 and arm A's 0.099 are not a like-for-like ratio. The
  *direction of change within* arm A is the load-bearing evidence, not the cross-arm ratio.
- **Terrain is stochastic, so memorisation is excluded** — this is the one confound the
  experiment was designed to remove, and it is removed.
- Arm A's win rate was still drifting upward in the last bucket (51.8 → 56.5). It has not
  provably converged.

## Reproducing

```bash
just measure-baselines examples/env_config/25v25_stochastic_terrain_shooting.yaml 30 "" 700000
just measure-checkpoint <ckpt> examples/env_config/25v25_stochastic_terrain_shooting.yaml 30 record
just run-summary ttmerrnr 50
```
