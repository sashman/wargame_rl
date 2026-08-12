# 2026-08-12 — The agent beats the opponent on the real tables

**Question.** Train on the 45 real table layouts using the golden reward config, and beat
the opponent there.

**Answer.** Achieved, on both seeds, on tables the agent never trained on — **0.99 and 0.93
win rate, +84.5 and +79.3 vp_margin**, against a floor of 0.28 (`random`) and 0.00
(`hold_deployment`). It does **not** reach the scripted bar at +105.7 there — but on tables it
*did* train on it is level with the bar (+89.5 / +90.3 against +91.2) and holds more
objectives. **The deficit is a generalisation gap, not a capability ceiling.**

Getting there needed two pieces of machinery that did not exist and one bug that would have
invalidated every number.

---

## The real tables are a different mission, not a harder board

The generator can only place objectives in the contested middle: `eligible_objective_pieces`
filters candidates to pieces whose **centre** lies between the two deployment edges. The real
layouts do not obey that. Across 246 objectives on 45 tables the split is exactly

| player's zone | middle | opponent's zone |
|---|---|---|
| **82** | **82** | **82** |

— every table mirror-symmetric, 37 of 45 with a (2, ·, 2) shape. Measured before a single
model moves, on identical seeds:

| | at deployment |
|---|---|
| player holds | **1.98** |
| opponent holds | **1.91** |
| empty | 1.58 |

The generated mission starts 0–0 with everything to fight for. This one deals both sides
roughly two objectives and leaves ~1.6 open. It is the first scenario here where *holding
ground you already have* can score, so **no reward lesson from the generated scenario is
known to carry over**.

## Standing still loses every episode

The obvious worry about a mission that deals you ground is that it can be won by deploying.
It cannot. `hold_deployment` — `STAY` for every model, every step — was written to answer
exactly this and is now a registered baseline so the answer stays quotable.

Nine held-out tables, n=10 each, seeds 700000–700009:

| policy | win | vp_margin | held (of 5.6) | alive |
|---|---|---|---|---|
| `hold_deployment` (STAY) | **0.00** | **−70.2** | 1.63 | 0.998 |
| `random` | 0.28 | −39.2 | 1.98 | 0.830 |
| `squad_march` | 0.94 | +77.4 | 3.36 | 0.639 |
| `squad_march_shoot` (the bar) | 1.00 | +105.7 | 3.78 | 0.693 |
| **PPO, 1000 epochs, seed 1** | **0.99** | **+84.5** | 3.37 | 0.917 |
| **PPO, 1000 epochs, seed 2** | **0.93** | **+79.3** | 3.37 | 0.921 |

STAY ends *below* what it started with — 1.63 against 1.98 — with 99.8% of its force alive.
The opponent takes the middle uncontested and then comes for the home points. The range from
doing nothing to the bar is **176 vp**, and none of it is free.

**An earlier figure is superseded.** A first sweep at n=1 per map over all 45 tables put
`random` at 0.67 win / +12.6. That is 45 single episodes and far too noisy to read a win rate
off; the n=10 held-out figures above replace it. The error is instructive — it made the
scenario look easier than it is, which is the direction that would have made a weak result
look like a win.

## Win rate cannot express the result

The bar sits at 1.00 and cannot go higher. A policy at +10 and one at +105 both "beat the
opponent". This is the saturation this project already knows about from the other direction —
win rate could not resolve TF32's 8.5 vp either. **Read `vp_margin`.** The stated goal was met
comfortably; the interesting question is the ~24 vp that remains on unfamiliar ground.

## On unfamiliar ground the failure is allocation, not combat

The agent wins the firefight outright and then fails to convert it into ground. Everything in
this section is measured on **held-out** tables — on training tables it allocates fine, and
out-holds the bar (see below).

| | agent | bar |
|---|---|---|
| alive at end | **0.92** | 0.69 |
| firepower ratio | up to **8.5** | 0.55 |
| objectives held | 3.37 | **3.78** |

Models on each objective at episode end, mean of 10, agent (seed 2, epoch ~150) against the
bar:

**table_35 — a table it plays well**

| objective x | 9.7 | 10.8 | 26.3 | 33.6 | 49.2 | 50.3 |
|---|---|---|---|---|---|---|
| agent | 4.8 | 5.9 | **8.0** | 3.4 | 0.0 | 0.0 |
| bar | 4.6 | 5.0 | 4.7 | 2.7 | 0.5 | 0.0 |

**table_05 — the one it loses**

| objective x | 11.7 | 17.2 | 29.2 | 30.8 | 42.8 | 48.1 |
|---|---|---|---|---|---|---|
| agent | **5.4** | 0.9 | 0.1 | 0.0 | 0.0 | 0.0 |
| bar | 4.8 | 4.8 | 4.9 | 1.5 | 0.9 | 0.0 |

On a table it handles, it spreads across four objectives and out-occupies the bar on every
one. On table_05 it collapses onto a single point: **6.3 of ~22 survivors are on an objective
at all**, while conceding three that the opponent has **zero** models on. At epoch 1000 seed 1
scores +35.5 there and seed 2 **loses it outright** (0.40 win, −18.5) — against +86 to +106 on
every other table, from both seeds.

table_05's objective geometry is unremarkable: 6 objectives, a 2/2/2 zone split, 72.1 mean
area — indistinguishable from table_15, table_30 and table_35, which both seeds score +87 to
+97 on. Whatever makes it hard is in the terrain, and is not yet identified.

**This is the failure `objective_hold.crowding_exponent` exists to price, and this config
already sets it at a = 1.0.** The lever that took the generated scenario from +2.5 to +28.4
does not prevent abandonment here — but note it does not have to be the culprit, since the
same reward produces correct allocation on the 36 tables the agent trained on.

## The plateau is at epoch ~140

Held-out `vp_margin`, both seeds:

| epoch | 150 | 300 | 1000 |
|---|---|---|---|
| seed 1 | +81.4 | +82.7 | +84.5 |
| seed 2 | +82.8 | +79.9 | +79.3 |

**850 epochs bought nothing** — ±3 vp, in both directions, inside the seed spread. Training
reward peaked at 26.50 (seed 2, epoch 240) and 26.44 (seed 1, epoch 767), both effectively
where they were at epoch 140. Do not spend compute on this gap.

## What had to be built

**Objective count was a hard input dimension.** The per-model block is `2 + n_objectives * 2`
wide, so a model token is 49 columns at three objectives, 53 at five, 55 at six. The real
layouts carry five or six objectives and 15 or 16 terrain pieces, so no single network spanned
them and a checkpoint trained on the generated scenario failed `load_state_dict` outright.
`objective_budget` / `terrain_budget` pad both to a fixed width, with padding explicitly
marked — a padding slot's `(0, 0)` distance delta otherwise reads as *"this model is standing
on that objective"*, the most emphatic thing the feature can say.

**Nothing drew a layout.** `map_pool` is a third terrain mode beside fixed `terrain` and
generated `random_terrain`: a whole layout, terrain *and* objectives, drawn per episode off
the layout RNG. Its `names` field is the train/holdout split, and it exists because training
on all 45 consumes the evaluation set.

## The bug that nearly produced a result

`config_for_map` cleared `random_terrain` but not `map_pool`, so scoring a pool-trained config
redrew a layout at every reset and never used the map. It does not look like a bug. It looks
like this:

```
table_05   0.982  1.00  104.5  3.60  0.672
table_10   0.982  1.00  104.5  3.60  0.672
...        (all nine rows byte-identical)
```

Every row scored the same drawn sequence. The first held-out band published in this session
came from that measurement and was wrong; it was caught only because nine independent tables
agreeing to three decimal places is not something layouts do. **The tell for this class of bug
is identical rows, and it is worth checking for deliberately** — the same failure mode was
already guarded for `random_terrain`, and the guard was not generalised when the third mode
was added.

## What this does and does not support

**Supported.** A PPO agent trained on 36 real tables with the golden reward config beats
`scripted_advance_and_shoot` on 9 held-out tables, ~178 of 180 episodes, reproduced across two
seeds that agree within 5 vp. The scenario is not won by standing still. The agent generalises
— it never saw the tables it was scored on.

**Not supported.** That it beats the scripted bar on unfamiliar ground; it does not, by ~24 vp. That more
training would close that; it plateaued at epoch 140 and 850 further epochs moved nothing.
That table_05's difficulty is understood; it is not.

**And the deficit is mostly a generalisation gap.** See below — this was written up as a
capability ceiling first, and the measurement that closed it says otherwise.

---

## The gap is generalisation, not capability

The obvious follow-up to "it does not reach the bar" is *where* it does not reach it. Scored
on 9 **training** tables (01, 06, 11 … 41), same seeds, same n:

| set | seed 1 | seed 2 | bar | agent − bar |
|---|---|---|---|---|
| 9 **training** tables | +89.5 | +90.3 | +91.2 | **−1.4** |
| 9 **held-out** tables | +84.5 | +79.3 | +105.7 | **−23.8** |

`held`, same order: **3.64 / 3.64** against the bar's 3.57 on training tables; 3.37 / 3.37
against 3.78 on held-out.

On tables it trained on the agent is level with the bar — within 1.4 vp on both seeds, and
**ahead on `held`** while keeping 92–94% of its force alive against the bar's 69%. On tables it
has never seen it falls ~24 vp behind. The difference in differences is ~22 vp, and that is the
whole story of the deficit. Both seeds agree to within 0.8 vp on the training set and both
report `held` to two decimal places identically, so this is not a one-seed artefact.

Note the two sets are not equally hard, and the direction is informative: the **bar scores
higher on the held-out set** (+105.7) than on the training set (+91.2), while the agent scores
*lower* (+84.5 against +89.5). Marching and shooting works better on those nine tables; the
agent's policy works slightly worse. So the gap is not an artefact of an unluckily hard
holdout — the holdout is easier for a scripted policy.

This changes what to do about it. The allocation failure is real and visible on table_05, but
it is a failure to transfer an allocation policy to an unfamiliar layout rather than an
inability to allocate: on familiar ground the agent out-holds the bar. **36 layouts is a small
training distribution**, and the first thing to try is more of it — more tables, or
augmentation over the ones there are — not a reward change.
