# 2026-08-05 — Does the agent break line of sight when the opponent shoots back?

**Question.** When the enemy returns fire and the ruins move every episode, does the policy
do anything beyond rushing forward and out-shooting it? "Cover" here means one thing only:
deliberately breaking line of sight. There is no cover save in this game — terrain blocks
LOS and nothing else.

**Answer: no.** The agent does not use terrain. It learns **range management** — it fights
at the edge of the weapon band, where most enemies cannot reach it. Deleting every ruin
from the board changes its exposure by 10%, against a 4.7x effect from return fire alone.
The reward curriculum was also refuted.

This is a negative result on the headline question and a positive one on the mechanism: the
policy *does* learn a real, non-obvious defensive behaviour in response to being shot at.
It is just not the behaviour we went looking for.

## What had to be built first

The question was unanswerable as the repo stood:

- **Nothing shot back.** `ScriptedAdvanceAndShootPolicy` shipped in PR #132 but no config
  used it, so terrain constrained only the *player's* shooting and cover had no value.
- **Nothing measured it.** `analyze_match` has no LOS, casualty or damage metric, and
  `GameStateSnapshot` carries no terrain, so it could not be recovered from traces.
- **Terrain was static**, built once at `__init__`. A policy that looked like it used cover
  might only have memorised seven rectangles; the two are indistinguishable in the numbers.

PR #135 added per-episode random terrain (fixed piece count — observation batching stacks
terrain, so the count cannot vary), the `exposure_rate` / `terrain_proximity` /
`fraction_alive` metrics, and `squad_march_shoot` to the auto-logged baselines.

## Setup

PPO + transformer, 1000 epochs per arm, one seed per arm. 25v25 on 60x44, 3 random
objectives, 7 mirrored random ruins regenerated per episode, 20 battle rounds, movement +
shooting phases. Every arm shares one reward (six calculators, `vp_gain` weight 2.0) except
the curriculum arm, whose final rung is identical to the others'.

**Batch 1** (`train-multi-2026-08-05-00-36-25`), weapon range 12, objectives drawn
independently:

| arm | run | opponent | reward |
|---|---|---|---|
| A | `ttmerrnr` | `scripted_advance_and_shoot` | single phase |
| B | `rhizvkte` | `scripted_advance_and_shoot` | 2-rung curriculum |
| C | `2fgj1zny` | `scripted_advance_to_objective` | single phase |

**Batch 2** (`train-multi-2026-08-05-05-52-10`), all with `objective_min_separation: 6`:

| arm | run | varies from D |
|---|---|---|
| D | `59sqxio7` | — (matched control) |
| E | `pwg97xnv` | **no terrain at all** |
| F | `2ooomh5i` | weapon range 12 → 24 |
| G | `2r5mkd4d` | objectives >= 5 cells from any ruin |

Batch 2 fixed an objective-placement defect (below), which changes the scenario
distribution — **batch 1 and batch 2 numbers are not comparable to each other.** Each batch
carries its own control.

## Measured — batch 1, converged (last 100 of 1000 epochs)

| arm | exposure | terrain_d | alive | vp_margin | win |
|---|---|---|---|---|---|
| A shooting / single | 0.099 | 5.85 | 0.706 | +13.6 | 53.9 |
| B shooting / curriculum | 0.089 | 5.76 | 0.727 | +8.6 | 52.7 |
| C control, no return fire | 0.463 | 5.99 | 1.000 | +115.7 | 100.0 |

Arm A over training: exposure 0.220 → 0.099 while survival **rose** 0.455 → 0.706.

## Measured — batch 1 held-out evaluation

Arm A's best checkpoint and every scripted baseline on **identical layouts**, seeds
700000-700029, scored through the same `evaluate_selector` path.

| policy | on_obj | win | player VP | opp VP | alive | exposure | terrain_d |
|---|---|---|---|---|---|---|---|
| random | 0.015 | 0.00 | 10.8 | 167.8 | 0.811 | 0.016 | 6.9 |
| greedy_nearest | 0.700 | 0.23 | 90.5 | 157.2 | 0.436 | 0.299 | 4.6 |
| split_evenly | 0.700 | 0.23 | 86.8 | 157.7 | 0.225 | 0.298 | 4.7 |
| squad_march | 0.833 | 0.27 | 88.2 | 147.5 | 0.271 | 0.368 | 5.1 |
| **squad_march_shoot** (the bar) | 1.000 | **0.63** | 130.8 | 109.8 | 0.388 | 0.220 | 5.0 |
| **agent (arm A)** | 0.882 | 0.53 | 111.0 | 117.5 | **0.660** | **0.127** | 4.6 |

## Measured — batch 2 (mean of last 150 epochs; F completed 1000, D/E/G at 926-950)

| arm | exposure | terrain_d | alive | vp_margin | win |
|---|---|---|---|---|---|
| D terrain, range 12 | 0.116 | 5.86 | 0.651 | −1.0 | 45.5 |
| **E no terrain, range 12** | **0.120** | n/a | 0.653 | −11.5 | 43.8 |
| F terrain, range 24 | 0.429 | 5.62 | **0.156** | −75.5 | **6.8** |
| G terrain, clear objectives | 0.179 | 6.58 | 0.531 | −34.6 | 28.3 |

## Conclusions

### 1. Terrain is not the mechanism. Range is.

**D vs E is the decisive comparison**: identical configs, differing only in whether ruins
exist. Deleting all seven changed exposure 0.116 → 0.120, survival 0.651 → 0.653, win
45.5 → 43.8. Everything is within noise. *Measured.*

Set that against the 4.7x exposure gap between arms A and C, which differ only in whether
the enemy can shoot. Return fire drives essentially the whole effect; terrain contributes
at most marginally. *Measured.*

This retroactively resolves batch 1's open question. Every batch-1 signature — exposure
collapsing, survival rising, `on_obj` staying at 0.882, `terrain_proximity` never
moving — is explained by a policy that manages **distance**, not line of sight.

### 2. Arm F is the corroboration, and the most informative single arm

Doubling weapon range to 24 should be where cover pays off: distance stops working across
a 60x44 board, LOS becomes the only remaining lever, and there are seven ruins available.
Instead the agent collapses — exposure 0.429, survival 0.156, win 6.8%. It has no answer
once distance is removed, which is exactly what a distance-only strategy predicts.
*Measured; the causal reading is inferred.*

> **Correction (2026-08-05, later).** "There are seven ruins available" overstates what arm F
> offered. `just measure-terrain` (added afterwards) reports that this profile — 7 pieces of
> 5-7 — leaves only **5.8% of the board hidden** from a squad in weapon range, against 19.8%
> for a 29-piece profile at similar total coverage. Cover was therefore not a working
> alternative in arm F either, so the arm shows that **range was the agent's only lever**, not
> that it ignored a working one. The distinction matters: the original wording reads as
> evidence the policy declined to use cover, which arm F cannot support.
>
> This also weakens conclusion 1 slightly in the same direction. D-vs-E remains a clean
> measurement — deleting seven ruins changed nothing — but it establishes that *this* terrain
> did nothing, not that terrain cannot. Batch 3 re-runs the question on a profile where cover
> is physically available.

### 3. What the policy actually learned is still worth having

It is not a null result about learning. Against the movement-only baselines on held-out
seeds the agent wins 0.53 to their 0.23-0.27. Against `squad_march_shoot` it takes **42%
less exposure and 70% more survivors for 15% less VP**. That is a coherent
survivability-for-offence trade, discovered without being rewarded for it — no calculator
pays for staying alive, and none charges for losing a model. The incentive is entirely
indirect (dead models stop earning `objective_hold`, `closest_objective_v2` and `vp_gain`).

### 4. The reward curriculum is refuted for this scenario

Arm B reached rung 1 and converged to the same place as arm A: win 52.7 vs 53.9, VP +8.6
vs +13.6, exposure 0.089 vs 0.099. Not better, not worse — the same policy reached more
slowly. *Measured, one seed.*

An **opponent** curriculum is also unnecessary: arm A clears the `squad_march` bar
comfortably without one. The pre-registered fallback (train against the movement-only
opponent, then `--warm-start-ckpt-path` into the shooting one) was never needed and
remains untested.

### 5. Objectives clear of terrain make the task harder

Arm G scores worse than its control on every axis (win 28.3 vs 45.5, VP −34.6 vs −1.0,
survival 0.531 vs 0.651) and is *more* exposed, at 0.179 vs 0.116. Given conclusion 1, the
likely cause is not lost cover but geometry: forcing objectives >= 5 cells from any of
seven ruins pushes them into the open middle of the board, further from the deployment
zone and from each other. *Measured; the explanation is inferred and untested.*

## Defect found and fixed during the experiment

Objective placement drew each objective independently, with no separation constraint at all
(`placement.py:145-148`). Measured over 400 episodes on this config:

| | frequency |
|---|---|
| two objective discs overlap | **25% of episodes** |
| an objective centre inside another's disc | 7.8% |
| an objective inside a ruin | **11% of objectives** |

A quarter of episodes were silently running as a two-objective mission. An objective inside
a ruin is also not covered ground — the see-out / see-into rule means its occupant stays
visible to everything outside, so the ruin protects nobody while still blocking that lane.

`objective_min_separation` and `objective_terrain_clearance` now constrain the draw, both
defaulting to the old behaviour. The cost of the fix is visible: batch-2 arm D wins 45.5
where batch-1 arm A won 53.9 on the same scenario with overlapping objectives. Three
genuinely separate objectives is a harder mission.

## Confounds and limits

- **One seed per arm.** Adequate for the A-vs-C gap (4.7x) and the F collapse (win 6.8 vs
  45.5), both far outside epoch-to-epoch noise. **Inadequate for D vs E**: that comparison
  concludes *no difference*, and a single seed can only bound the effect as small, not show
  it is zero. The honest claim is "terrain contributes far less than range", not "terrain
  contributes nothing".
- **`exposure_rate` has four documented traps** — see [docs/metrics.md](../docs/metrics.md)
  § Cover metrics. It averages over alive models, so absolute levels are not comparable
  between arms with different mortality. Arm C loses nobody, so its 0.463 against arm A's
  0.099 is not a like-for-like ratio; the load-bearing evidence is the *direction of change
  within* arm A, where exposure fell as survival rose.
- **`terrain_proximity` is a weak proxy for cover.** A ruin can break a sightline from far
  away. It is arm E, not the flat proximity, that carries conclusion 1.
- **D, E and G stopped at 926-950 of 1000 epochs**, read as means of their last 150. All
  three were flat across their final three buckets.
- **Terrain is stochastic, so memorisation is excluded** — the confound the experiment was
  designed to remove is removed.

## What would actually test cover

The experiment shows the policy has no reason and no signal to use terrain — and, per the
correction above, nowhere much to use it either. **Batch 3 addresses all three**, as a 2x2
over (signal x reason) on a terrain profile where cover is physically available:

1. **Give it somewhere to hide.** 29 pieces of 3-7 instead of 7 of 5-7 raises cells hidden
   from a squad from 5.8% to 19.8%. Count dominates size: hiding means breaking *every*
   sightline at once, which needs ruins in many directions rather than one big one.
2. **Give it the signal.** The observation carries terrain rectangles but never says whether
   an enemy can see you: the shooting mask is computed only during the shooting phase and
   only masks logits — it never enters the encoder (`net.py:491`), so at *movement* time the
   policy has zero LOS information. `observe_threat_count` adds a per-model "fraction of the
   enemy with LOS and range to me" scalar.
3. **Give it a reason.** No calculator charged for losing a model, so exposing one was free.
   The `models_lost` global calculator prices losses to match `model_kills`, making an even
   exchange net to zero and only favourable ones pay.

**Correcting the framing this report used.** It treats cover as an alternative to shooting.
It is better understood as choosing the *exchange ratio*: line of sight is exactly symmetric
in this engine, but symmetry is pairwise and does not equalise the counts — ten models behind
a wall while twelve fire on three is twelve shots out for three back, and Lanchester's square
law compounds that. Hence `firepower_advantage`, which measures the difference directly;
`exposure_rate` counts only our side and cannot tell manoeuvre from hiding.

**Also newly measurable: the noise floor.** `reset(options={"combat_seed": ...})` now varies
the dice independently of the layout. On the batch-3 control, `squad_march_shoot` has a
vp_margin sd of 50.6 *within* a fixed layout against 45.0 between layouts — the dice
contribute more spread than the scenario. That does not invalidate anything above (all the
numbers here are means over the last 100-150 epochs, which averages it down), but it does
mean the D-vs-E gap of 1.7pp was never readable at one seed, exactly as the confounds section
already suspected. Batch 3 runs two seeds per arm.

## Reproducing

```bash
just measure-baselines examples/env_config/25v25_stochastic_terrain_shooting.yaml 30 "" 700000
just measure-checkpoint <ckpt> examples/env_config/25v25_stochastic_terrain_shooting.yaml 30 record
just run-summary ttmerrnr 50
```
