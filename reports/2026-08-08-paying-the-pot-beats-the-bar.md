# 2026-08-08 — Paying the pot: the agent beats the shooting bar

**Question.** Beat `scripted_advance_and_shoot` in the current environment.

**Answer. Yes, on both seeds.** Pricing objective *crowding* — paying a point a fixed pot
split between its occupants instead of paying every occupant the same wage — takes the agent
from **+2.5 to +28.4 vp_margin** and past the `squad_march_shoot` bar at **+17.0**.

| policy | on_obj | win | player VP | opp VP | **vp margin** | held | alive | firepower |
|---|---|---|---|---|---|---|---|---|
| `random` (floor) | 0.005 | 0.00 | 10.2 | 174.8 | −164.6 | 0.09 | 0.745 | 0.21 |
| `squad_march` | 0.870 | 0.22 | 91.9 | 150.8 | −58.9 | 0.99 | 0.269 | 0.54 |
| control (`observe`, 1000ep) | 0.912 | 0.535 | 120.2 | 116.9 | +3.25 | 1.43 | 0.62 | 1.38 |
| `contest_and_spread` | 0.963 | 0.62 | 127.0 | 110.1 | +16.9 | 1.61 | 0.516 | 0.78 |
| **`squad_march_shoot` (the bar)** | **0.960** | **0.65** | **129.9** | **112.9** | **+17.0** | **1.64** | 0.393 | 0.51 |
| `share_soft` (a = 0.5) | 0.941 | 0.625 | 129.8 | 113.1 | +16.7 | 1.545 | 0.574 | 1.22 |
| **`share` (a = 1.0)** | **0.935** | **0.685** | **137.9** | **109.5** | **+28.4** | 1.59 | 0.594 | 1.64 |

All rows: **n = 100 episodes, seeds 700000–700099, identical layouts**, scored through the same
`evaluate_selector` path. Baselines re-measured on the arm's own config to close the loop.

`share` clears the bar **on both seeds independently** — +30.1 (s1) and +26.6 (s2) against
+17.0 — so the margin is not a lucky seed. Within-arm seed spread is 3.5 vp against an 11.4 vp
margin over the bar.

---

## 1. The diagnosis came first, and it is the reason this worked

Seven levers had already been tried and had all measured null or negative. What broke the
deadlock was not a better idea but a better instrument.

`objectives_held` says a policy controls 1.42 of 3 objectives. It does not say *why* the other
1.58 were not held, and abandoned / narrowly-lost / lost-by-a-mile call for different fixes.
`scripts/measure_objective_split.py` (`just measure-objective-split`) reports the per-objective
`(player, opponent)` counts at episode end, ranked by player occupancy within each episode, plus
a **redistribution ceiling**: what the same survivors would hold if every model surplus to
`opponent_count + 1` on an already-held point moved to the cheapest point the policy lost.

At n = 100 on the 1000-epoch control:

| | agent | `squad_march_shoot` |
|---|---|---|
| models alive at end | 15.8 | 9.8 |
| busiest objective, player v opponent | **12.89 v 0.25** | 6.95 v 0.25 |
| second objective, player v opponent | 2.72 v 4.22 | 2.68 v 3.04 |
| second objective held rate | 0.48 | 0.64 |
| surplus models on held points | **14.13** | 7.93 |
| `objectives_held` | 1.42 | 1.64 |
| **redistribution ceiling** | **2.06** | 1.88 |

The agent parked **12.9 models on a point defended by a quarter of a model**, where one would
do, and lost the second by a model and a half. It was not short of material — it survived 60%
better than the bar and out-gunned it 2.7×. The ceiling said allocation alone would clear the
bar, and the ceiling is deliberately optimistic (it ignores travel time and return fire), so it
can only ever *rule a re-allocation lever out*. Here it ruled it in.

**Also visible for the first time: this is a two-objective mission.** Both policies concede the
third objective in essentially every episode — the opponent stacks ~13 models there and
flipping it costs 14. `held` is therefore bounded near 2, and the whole contest is over the
second point.

## 2. The lever

`objective_hold` gained `crowding_exponent` (`a`, default 0.0 = the flat historical behaviour,
bit-identical): a point's value is divided by `occupants ** a`. At `a = 1` the **pot is
conserved** — a point pays its value once, split among whoever stands on it.

Two arms, both built on the `observe` config so the occupant count is actually in the
observation, with weights calibrated so the occupancy that holds the contested point (5 models
against the measured 4.22 defenders) pays exactly the flat term's 0.25/step. The arms differ
only in how sharply crowding is punished, not in the price of correct play:

| arm | `a` | weight | pay at k=5 | at k=13 | at k=1 |
|---|---|---|---|---|---|
| `share` | 1.0 | 1.25 | 0.250 | 0.096 | 1.250 |
| `share_soft` | 0.5 | 0.56 | 0.250 | 0.155 | 0.560 |

### Why this worked where `surplus_value` and the overstack penalty did not

Three levers have now been aimed at the same defect. The two that failed share a property the
one that worked does not:

* **They lower total objective income.** A penalty and a discount both make the objective term
  pay less in aggregate, so the policy experiences either as "objectives pay less" and does
  fewer of them — which is exactly what both rounds measured (occupancy 0.925 → 0.520 and
  0.784 → 0.284). Pot conservation instead makes spreading onto a second point *raise* total
  income: `k` models on one point earn the pot once, `k/2` on each of two earn it twice.
* **They key on something the model cannot observe about itself.** `surplus_value` is a cliff
  keyed on a distance-to-centre rank. `crowding_exponent` keys only on the occupant count,
  which `observe_objective_control` puts directly on the objective token.

This is the generalisation worth carrying forward: **a lever against over-concentration must
redistribute reward, not destroy it.** Ask of any shaping term whether the behaviour it wants
pays *more* in total than the behaviour it is replacing. If it does not, the policy will read
it as a tax on the whole activity.

## 3. The exponent barely matters; the mechanism does

At 300 epochs the two arms were indistinguishable (+12.3 and +10.4 over the control, against a
5–7 vp seed spread). At 1000 they separate: `share` +28.4, `share_soft` +16.7. Full pot
conservation is worth roughly 12 vp over half-strength sharing, but half-strength sharing
already reaches the bar. **The finding is that crowding must be priced with income conserved,
not that `a` = 1 is optimal.** `a` was not swept beyond these two points.

## 4. `held` is a good ranking metric and an incomplete mechanism

Earlier in this line of work `objectives_held` was promoted as "the metric that ranks policies"
because vp_margin was monotonic in it across every policy then measured. That still holds as a
*ranking* — but it is not the mechanism, and this result shows the seam:

* `share` beats the bar by 11.4 vp while holding **fewer** objectives at episode end
  (1.59 against 1.64).
* At 300 epochs `share_soft` gained **+19.3 vp with `held` unchanged**.

`held` is an end-state snapshot; VP accrues every round. A policy can hold more *during* the
episode and end level. Read `held` to rank, but do not treat a flat `held` as a null result —
that mistake was made in this session and corrected within the hour.

## 5. A measurement bug found while verifying this batch

`get_checkpoint_callback` set `save_top_k=3, monitor="reward/mean_episode_reward",
save_last=True` on a single `ModelCheckpoint`. With `monitor` set, `last.ckpt` is only rewritten
on epochs that enter the top-k — so it holds **the last epoch that improved**, not the last
epoch trained. This batch's four `last.ckpt` files held epochs **970, 692, 948 and 998**, and
the epoch-1000 weights of the 692 run were never written at all.

Consequences:

* Every score in this repo labelled "at N epochs" was really "at whatever epoch that run last
  improved by its own training reward". On 25v25 a 300-epoch difference is worth ~13 vp_margin.
* Across arms it is not even a common selection rule, because each arm's
  `reward/mean_episode_reward` is a *different function* — `share` weights `objective_hold` at
  1.25, the control at 0.25.

Fixed by splitting the roles: a monitored callback keeps the top-3 by reward, and a second,
unmonitored one owns `last.ckpt` so it means the last epoch. `tests/test_checkpoint_callback.py`
pins it.

**This does not undermine the headline.** The affected run is `share` s2, scored at epoch 692 —
the *lowest*-epoch checkpoint of the four — and it still returned +26.6 against the bar's +17.0.
The bug is conservative here: s2 at a true epoch 1000 would if anything score higher.

## 6. What this does not establish

* **Only two exponents were tried**, at one calibration point each. Nothing here says `a = 1`
  and weight 1.25 are near optimal.
* **One scenario.** 25v25, 3 objectives, `objective_min_separation: 6`, the batch-3 terrain
  profile, against `scripted_advance_and_shoot`. The third objective is uncontestable in this
  setup, which is precisely the condition that makes crowding so costly. On a map where all
  three points are winnable the effect size is unknown.
* **The shooting rules are going to change.** The current mechanic is not how the tabletop
  rules work. This finding is about *objective allocation* and does not depend on shot
  resolution, so it should survive — unlike the shot-waste and decode findings in the
  [2026-08-06 report](2026-08-06-beat-the-shooting-opponent.md), which do.
* **`share` still holds fewer objectives at episode end than the bar** (1.59 v 1.64) and still
  leaves ~12.6 surplus models on its busiest point against the bar's 7.9. Its redistribution
  ceiling is 2.02. The defect this lever was built for is *reduced, not solved* — there is
  headroom left in exactly the same place.

## 7. Reproducing

```
just measure-baselines examples/env_config/25v25_beat_share.yaml 100 "" 700000
just measure-checkpoint <ckpt> examples/env_config/25v25_beat_share.yaml 100
just measure-objective-split <ckpt> examples/env_config/25v25_beat_share.yaml 100
just train-seed 1000 1 <group> examples/env_config/25v25_beat_share.yaml
```

Arms: `examples/env_config/25v25_beat_{share,share_soft}.yaml`, both one line different from
`25v25_beat_observe.yaml`. Control: the `observe` runs, not retrained (training is
deterministic given seed + config + code).
