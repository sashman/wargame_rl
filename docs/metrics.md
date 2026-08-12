# Metrics Reference

Semantics of every metric logged to Wandb, written to be usable as an **evaluation framework
for an LLM agent** reading runs through the W&B MCP server or API.

The catalogue below states, for each metric, where it is emitted, how often, what its units
are, and — critically — what it does *not* mean. Several metrics in this project are means of
means, or change definition silently; an agent that reads them naively will draw confident
wrong conclusions. Those traps are called out inline and collected in
[Reading rules](#reading-rules).

- Related: [reward-phases.md](reward-phases.md) (curriculum config) · [missions-and-vp.md](missions-and-vp.md) (VP scoring) · [game-state-io.md](game-state-io.md) (per-match event logs, a finer-grained alternative to these aggregates)

---

## 1. Where metrics come from

Two recurring emission paths, on two different step counters. **No single W&B step carries
both**, which is why history rows look ragged. A third path runs once, at `on_train_start`.

```
                   on_train_epoch_end (lightning_base.py:372)
                              │
      ┌───────────────────────┴───────────────────────┐
      │                                               │
  run_episodes(n_episodes)                    _advance_reward_phase(sr)
  → _evaluate_episodes                        → reward_phase
  → success_rate, mean_episode_steps,           phase_advanced_at_epoch
    reward/{mean,max,min}_episode_reward,
    eval/{vp_player,vp_opponent,vp_margin,win_rate}

                   PPO training step (ppo/lightning.py:450)
                              │
              loss/*, train/*, reward/components/*

                   on_train_start (lightning_base.py:59), once per run
                              │
                        eval/baseline_*
```

Evaluation episodes are **separate rollouts** run under `torch.no_grad()` with the policy in
eval mode (`lightning_base.py:233–270`). They are not the training rollouts. So
`reward/mean_episode_reward` (eval) and `reward/components/*` (training rollout) describe
different episodes and need not agree.

### Episode count

`n_episodes` defaults to **10** (`ppo/config.py:59`). This sets the resolution of `success_rate`: with 10 episodes it can only
ever be a multiple of 10. **A change from 80% to 90% is one episode.** Do not read
single-epoch movements as signal; require a trend across epochs.

### Episode cadence

Evaluation runs every epoch by default. `--eval-every-n-epochs N` runs it every Nth
instead, which is worth ~16% of wall-clock at N=4 because evaluation is ~22% of a real
epoch and sits outside `perf/epoch_s`. Two consequences for reading the curves: every
`eval/*` series becomes N times sparser, and the final epoch always evaluates
regardless of cadence, so the last point is never stale. It is rejected on curriculum
configs — see [reward-phases.md](reward-phases.md) § Advancement.

Eval seeds are **fixed across epochs** (`EVAL_SEED_BASE`, `lightning_base.py:27`), and held out
from both the training and baseline seed ranges. So an epoch-to-epoch move is attributable to
the policy rather than to which maps were drawn — but the absolute level belongs to those
`n_episodes` layouts, not to the scenario. Two runs are comparable on it; a run and a
differently-seeded measurement are not.

---

## 2. Metric catalogue

### Episode-level (per epoch, from eval rollouts)

| Key | Source | Units / range | Meaning |
|---|---|---|---|
| `success_rate` | `lightning_base.py:331` | **0–100** | Percent of eval episodes where the current phase's `success_criteria` held on the final step. See trap Ⓐ. |
| `mean_episode_steps` | `lightning_base.py:281` | steps | Mean eval episode length. Compare against `max_turns`; equality means episodes never terminate early. |
| `reward/mean_episode_reward` | `lightning_base.py:276` | reward units | Mean total (undiscounted, summed) reward per eval episode. This is the checkpoint selection metric (`checkpoint_callback.py:28`). |
| `reward/max_episode_reward` | `lightning_base.py:286` | reward units | Best single eval episode. Gap vs. mean indicates variance across seeds/placements. |
| `reward/min_episode_reward` | `lightning_base.py:291` | reward units | Worst single eval episode. Persistently negative alongside a healthy mean signals a failure mode not captured by `success_rate`. |

### Curriculum

| Key | Source | Units / range | Meaning |
|---|---|---|---|
| `reward_phase` | `lightning_base.py:339` and `:353` | index, 0-based | Index into `reward_phases`. Emitted twice on two step counters — see trap Ⓒ. |
| `phase_advanced_at_epoch` | `lightning_base.py:357` | epoch | Logged **only on the epoch an advance happens.** Absent for the whole run means the agent never cleared phase 0. |

Advancement requires all three of (`phase_manager.py:284–326`): `success_rate/100 >=
success_threshold`, at least `min_epochs` spent in the phase, and the threshold held for
`min_epochs_above_threshold` **consecutive** epochs (any miss resets the counter to 0).

The index lives in a `CurriculumPosition` (`phase_manager.py:40`) shared by the training env,
every rollout env and every eval env, so one advance moves all of them. Before that sharing
existed the rollout envs stayed on phase 0 for every run to date while `reward_phase` reported
otherwise — which is what `train/rollout_phase_index` below exists to make visible.

### Losses (per epoch, mean over the epoch's optimizer updates)

| Key | Source | Sign convention |
|---|---|---|
| `loss/train_loss` | `ppo/lightning.py:451` | Total: policy + `vf_coef`·value + `ent_coef`·entropy. Routinely negative — the entropy term is negative. **Negative is not an error.** |
| `loss/policy_loss` | `ppo/lightning.py:458` | Clipped surrogate. Near zero at convergence; magnitude says how much the update moved. |
| `loss/value_loss` | `ppo/lightning.py:465` | Critic MSE. The one loss that should fall monotonically. The clearest single convergence signal. |
| `loss/entropy_loss` | `ppo/lightning.py:472` | **`-ent_coef × entropy`, not entropy.** Divide by `ent_coef` before reading it as nats. Rising *toward zero* = policy sharpening. |

Lightning suffixes these `_step` and `_epoch`. With `on_epoch=True` and one log call per epoch
the two are equal; prefer `_epoch`.

> ⚠️ **`loss/entropy_loss` is not entropy.** It is the coefficient-multiplied
> loss term. A past experiment read `-1.79 → -0.59` as an entropy drop and
> concluded `ent_coef` had sharpened the policy; dividing by the respective
> coefficients (0.03, 0.01) gives 59.7 and 59.0 nats — a 3× coefficient change
> moved entropy by ~1%. Read `train/entropy/movement` and
> `train/entropy/shooting` instead: raw nats, already per model, split by
> phase. The aggregate mixes a heavily-masked shooting phase with a wide-open
> movement phase and is uninterpretable either way.

### Update health (per epoch)

| Key | Meaning | Healthy range |
|---|---|---|
| `train/entropy/movement` | Mean per-model entropy in the movement phase, raw nats. One key per battle phase actually stepped, named from `BattlePhase` | Well below `ln(n_movement_actions)`. At 97 actions the ceiling is 4.575; sitting at ~4.5 means the policy is still uniform |
| `train/entropy/shooting` | Same, shooting phase. Naturally low — masking often leaves 1–2 valid targets | — |
| `train/clip_fraction` | Fraction of samples outside the PPO trust region | 0.1–0.3. Above ~0.4 the objective is saturating and much of each minibatch contributes no gradient |
| `train/approx_kl` | Policy movement per update | ~0.01–0.03 |
| `train/explained_variance` | How much of the return the critic explains | Rising toward 1. Near 0 means it is no better than predicting the mean |
| `train/grad_norm` | Mean joint gradient norm across both networks, **before** clipping | Read beside the next row, never alone |
| `train/grad_clipped_fraction` | Fraction of minibatches where that norm exceeded `max_grad_norm` | Near 1.0 means clipping binds on essentially every update, so `max_grad_norm` — not `lr` — is setting the step size, and raising `lr` alone will change little. Measured at **1.0 for every epoch of an 86-epoch run** on `25v25_shooting_opponent.yaml`. The norm decays 6.19 → 2.61 as training settles but never approaches the 0.5 default, so this is not a fresh-init artefact — clipping is the step-size control for the whole run. At `--max-grad-norm 5.0` the fraction drops to 0.14, i.e. a safety net rather than a constraint |
| `train/rollout_phase_index` | The **rollout** env's reward phase | Must track `reward_phase`. If they diverge, training and evaluation are on different reward functions |
| `train/distinct_layouts_seen` | Distinct objective layouts seen **cumulatively across the run**, not within one rollout | Must keep climbing. Flat means training is replaying the same maps — the within-rollout count it replaced read a constant env count and hid exactly that |
| `train/num_rollout_envs_resolved` | Rollout envs actually used | `num_rollout_envs` defaults to 0 = auto-detect, so the config never showed which path ran |

### Baselines and the scoreboard

| Key | Meaning |
|---|---|
| `eval/vp_player`, `eval/vp_opponent`, `eval/vp_margin` | **The phase-invariant scoreboard.** `success_rate` changes definition at every phase boundary and reward changes with the calculator set, so neither is comparable across a run or between runs. VP is the game's own measure and never changes meaning. |
| `eval/win_rate` | **0–100**, like `success_rate` — percent of eval episodes ending ahead on VP, not a fraction |
| `eval/baseline_squad_march_*` | Whole squads marched onto objectives by a scripted heuristic. Movement-only — **not** the bar, see below |
| `eval/baseline_squad_march_shoot_*` | The bar: the only baseline that fires |
| `eval/baseline_random_*` | The floor |

Each baseline emits `_win_rate` (0–100), `_vp_margin`, `_at_objectives` (a
fraction), `_fraction_alive` (a fraction), and `_exposure` when the config sets
`track_exposure`. `random`, `squad_march` and `squad_march_shoot` are logged
during training (`lightning_base.py:BASELINE_POLICIES`); the remaining rungs
live in `scripts/measure_baselines.py`.

**Always read a number against the baselines.** Measured once per run at
`on_train_start` over 20 held-out seeds, and constant thereafter. Every
`success_rate` in `reports/` predating them was quoted with no floor and no
ceiling — which is how a policy scoring 17% against an 80% heuristic was read as
progress. Reproduce or extend with `just measure-baselines <env_config> [n] record`.

**The bar is `squad_march_shoot`, not `squad_march`.** Every other baseline is
movement-only, so their ~0.78 win rate is the ceiling of a policy class the
agent is not in — it gets a shooting decision every other step. Clearing
`eval/baseline_squad_march_win_rate` is clearing 0.78, not 1.00. Measured on
25v25 over 40 held-out episodes:

| baseline | on obj | win | player VP | opp VP |
|---|---|---|---|---|
| random | 0.018 | 0.00 | 16.8 | 168.6 |
| greedy_nearest | 1.000 | 0.53 | 125.9 | 117.9 |
| split_evenly | 1.000 | 0.78 | 148.4 | 104.8 |
| squad_march | 1.000 | 0.78 | 147.9 | 102.9 |
| **squad_march_shoot** | **1.000** | **1.00** | **176.0** | **46.5** |

### `objectives_held` — the metric that ranks policies

**Mean count of objectives the player controls at episode end**, under the same strict count
comparison VP scores on (`player_count > opponent_count` among alive models inside the disc).
Reported by `measure-baselines` and `measure-checkpoint` as `held`.

**Prefer this to `on_obj` for any occupancy question.** `final_fraction_at_objectives` is the
fraction of *alive* models standing on *any* objective, so it cannot distinguish 15 models on
one point from 5 each on three — the first scores 5 VP a round, the second 15, and both read
~0.95. Measured on 25v25 at n=100, seeds 700000+:

| policy | on_obj | **held** | vp margin |
|---|---|---|---|
| `random` | 0.003 | 0.05 | −164.3 |
| `greedy_nearest` | 0.790 | 0.84 | −66.4 |
| `split_evenly` | 0.760 | 0.83 | −64.8 |
| `squad_march` | 0.870 | 0.99 | −58.9 |
| trained control (1000ep) | 0.945 | 1.42 | +1.7 |
| `contest_and_spread` | 0.963 | 1.61 | +16.9 |
| `squad_march_shoot` | 0.960 | 1.64 | +17.0 |

**Ordered by `held` the VP column is perfectly monotonic; ordered by `on_obj` it is not** —
`contest_and_spread` outranks the bar on `on_obj` while scoring below it, and `on_obj` cannot
separate the trained agent from the bar at all (0.945 against 0.960) though they differ by
15 VP.

The arithmetic is direct: an objective held pays 5 VP across ~19 scoring rounds, so a gap of
0.22 objectives is worth ~21 VP. That is the whole measured deficit between the trained
control and the bar — while the agent keeps **50% more models alive** and has *more* models
on objectives in absolute terms. More survivors, more firepower, fewer objectives.

**Historical note.** Three experimental rounds were designed against an apparent occupancy
deficit read off `on_obj` at n=30 (0.925 against 1.000). At n=100 that gap is 0.945 against
0.960 — mostly measurement noise — while the real deficit was always in `held`, which was not
being measured. Diagnose with `held`.

### Splitting `held` by *why* — `just measure-objective-split`

`held` says a policy controls 1.42 of 3 objectives. It does not say whether the missing ones
were abandoned, contested and narrowly lost, or lost by a mile — and those call for different
fixes. `scripts/measure_objective_split.py` reports, at episode end, the per-objective
`(player, opponent)` counts ranked by player occupancy within each episode, the outcome class,
and the **redistribution ceiling**: how many objectives the same survivors would hold if every
model surplus to `opponent_count + 1` on an already-held point moved to the cheapest point the
policy lost.

The ceiling is deliberately optimistic — it ignores travel time and return fire, both of which
only lower it. So a ceiling near the current `held` **rules re-allocation out**, while a large
one does not rule it in. It costs minutes and can retire a reward-shaping idea before it is
trained.

Measured at n=100, seeds 700000+, on the batch-3 scenario:

| | trained agent (1000ep) | `squad_march_shoot` |
|---|---|---|
| models alive at end | 15.8 | 9.8 |
| busiest objective, player v opponent | **12.89 v 0.25** | 6.95 v 0.25 |
| second objective, player v opponent | 2.72 v 4.22 | 2.68 v 3.04 |
| second objective held rate | 0.48 | 0.64 |
| surplus models on held points | **14.13** | 7.93 |
| `objectives_held` | 1.42 | 1.64 |
| **redistribution ceiling** | **2.06** | 1.88 |

The agent parks 12.9 models on a point defended by a quarter of a model and loses the second by
a model and a half. Nothing is missing but allocation: it survives 60% better than the bar and
out-guns it 2.7×, and moving the surplus would take it past the bar without gaining a single
model or kill. Note also that neither policy contests the third objective — the opponent stacks
~12 models there and flipping it costs 13 — so this scenario is effectively a two-objective
mission, and `held` is bounded near 2.

## Cover metrics

Emitted only when the env config sets `track_exposure: true`. Terrain blocks line
of sight and nothing else, so "using cover" means exactly one thing: positioning
where no enemy can see you.

| Key | Meaning |
|---|---|
| `eval/firepower_ratio` | **Prefer this one.** Over the episode, (alive enemies at least one of our models can see and reach) ÷ (our alive models at least one of theirs can). 1.0 is an even exchange; above 1.0 we bring more guns to bear than we expose |
| `eval/exposure_rate` | Fraction of alive model-shooting-phases where at least one alive enemy had **line of sight and weapon range** to that model |
| `eval/terrain_proximity` | Mean distance from an alive model to the nearest terrain footprint (0 inside) |
| `eval/fraction_alive` | Fraction of player models still alive at episode end. Logged always, not just when tracking exposure |

**Prefer `firepower_ratio` to `exposure_rate` for any question about cover.**
Exposure counts only our side of the exchange, so it falls both when a policy
manoeuvres into a good fight and when it merely hides from every fight — it
cannot rank the two, and all four traps below are consequences of that. Line of
sight is exactly symmetric in this engine (`wargame.py` sorts the endpoints to
guarantee it), but symmetry is *pairwise*: it does not equalise the counts. Ten
models behind a wall while twelve fire on three is twelve shots out for three
back, and that is what the ratio reports.

⚠️ **It was a count difference (`firepower_advantage`) until 2026-08-06, and in
that form it did not work.** Batch 3 falsified it: `random` — a policy that wins
**zero** games — scored **1.78**, above every trained arm (1.03–1.29). A
difference is dominated by how much engagement happens at all rather than by who
wins it, so a policy that scatters into contact everywhere scores well while
losing. **`eval/firepower_advantage` values logged before that date are not
comparable to `eval/firepower_ratio` and should not be read.** The key was
renamed rather than redefined in place, precisely so old history stays
identifiable. See [the batch-3
report](../reports/2026-08-06-cover-signal-reason-geometry.md).

The ratio divides **totals over the episode**, not a mean of per-phase ratios: a
phase with 20 models engaged says more about the exchange than one with 2, and
averaging ratios would weight them equally. It is `None` when our side was never
exposed in any sampled phase, since the ratio is unbounded there — a degenerate
case, not a perfect one.

Same numbers appear as `exposure` / `terrain_d` / `alive` columns in
`just measure-baselines` and `just measure-checkpoint`, produced by the same
`evaluate_selector` path, so an agent row is directly comparable to a baseline row.

**Reading rules — this metric has four traps.**

1. **It is a mean over *alive* models, so casualties push it down on their own.**
   Survivors are disproportionately the models that were out of sight or out of
   range, so a policy that loses half its army scores lower exposure without ever
   choosing a covered position. Never compare `exposure_rate` across configs with
   different mortality. Measured on the 25v25 stochastic-terrain configs, 10 seeds:

   | baseline | vs shooting opponent | vs movement-only opponent |
   |---|---|---|
   | | exposure / alive | exposure / alive |
   | greedy_nearest | 0.234 / 0.496 | 0.452 / 1.000 |
   | squad_march | 0.276 / 0.232 | 0.640 / 1.000 |
   | squad_march_shoot | 0.211 / 0.404 | 0.433 / 1.000 |

   The left column is not "these policies took more cover". It is mostly that the
   exposed models died. **Compare policies within one config**, where every arm
   faces the same fire.

   (Numbers above are 10 seeds; the 25-seed table under trap 3 is the reference.)

2. **Low exposure is not good play.** `random` scores 0.018 exposure against the
   shooting opponent — the best number in the table — by wandering off and never
   closing to weapon range, for 13.6 VP against 164.4. Exposure only means
   something read beside `eval/vp_margin`: the claim worth making is *lower
   exposure at equal or better VP*.

3. **Killing the enemy lowers it too.** A dead opponent is one fewer model with
   line of sight to you, so exposure falls for shooting well as much as for
   hiding well. On `25v25_stochastic_terrain_shooting` (a batch-1 config, since
   deleted; restore with `git checkout batch-1-2-configs -- configs/`),
   25 seeds:

   | baseline | exposure | alive | opp VP |
   |---|---|---|---|
   | squad_march | 0.305 | 0.272 | 145.4 |
   | **squad_march_shoot** | **0.201** | **0.442** | **110.4** |

   `squad_march_shoot` has *both* the lower exposure and the higher survival,
   which mortality alone cannot explain — it is thinning the opposing firing
   line. So a policy showing lower exposure than this bar has to be doing
   something the bar is not, and `terrain_proximity` plus kill-driven VP are what
   separate "broke line of sight" from "shot first".

4. **It is not the same quantity as the shooting mask.** `exposure_rate`
   deliberately ignores the engagement-range and advanced gating that
   `compute_shooting_masks` applies for real shots. A shooter within
   `engagement_range` of any enemy cannot fire at all, so including that gate
   would score a headlong charge as cover.

`terrain_proximity` is the check against reading 2 backwards: a policy that is
merely out of range keeps proximity high, one that is actually using ruins pulls
it down. All three are `None` — printed as `-`, key omitted in Wandb — when
unmeasured, never `0.0`, which would read as "never exposed" (or, for firepower,
as a genuinely even exchange).

**`firepower_ratio` avoids traps 1–3** because both terms are measured in the
same phase: casualties, disengagement and kills all move both. It counts
**shooters, not targets** — models with at least one reachable enemy, on each
side — which is what actually sets how many shots are fired.

Getting that right took two attempts, and the failure is instructive. The
original `firepower_advantage` was `(enemies we can see) − (our models they can
see)`. Both halves were wrong. The arithmetic was wrong because a difference is
dominated by how much engagement happens rather than by who wins it. The
*direction* was wrong because line of sight is symmetric: a model that is
exposed is exactly a model that can fire, so "enemies we can see" is **their**
shooter count, not ours. One of our models walking into view of twenty scored
twenty, which is how `random` topped the table.

Reference values on `25v25_cover_control`, 25 seeds:

| baseline | win | firepower_ratio | exposure | alive |
|---|---|---|---|---|
| random | 0.00 | 0.23 | 0.019 | 0.822 |
| greedy_nearest | 0.20 | 1.51 | 0.398 | 0.389 |
| split_evenly | 0.08 | 0.70 | 0.360 | 0.205 |
| squad_march | 0.20 | 0.68 | 0.391 | 0.262 |
| **squad_march_shoot** (the bar) | **0.56** | 0.49 | 0.193 | 0.406 |

**It does not rank policies, and is not supposed to.** It measures the
*firefight*, and a policy can win on objectives while being outgunned — which is
exactly what the bar does: `squad_march_shoot` wins 0.56 at a ratio of 0.49,
holding ground with roughly half the guns bearing. Read it beside `vp_margin`;
the claim worth making is **higher firepower ratio at equal or better VP**.

What it *can* do, which the old form could not, is tell a competent policy from
a flailing one: `random` sits at 0.23, outgunned four to one, last on the table
rather than first.

### The noise floor

`just measure-noise-floor <config> [n_layouts] [n_combat_seeds]` holds the
layouts fixed and varies only the dice (`reset(options={"combat_seed": ...})`),
separating the two sources of variance. On `25v25_cover_control` with
`squad_march_shoot`, 8 layouts x 8 combat seeds:

```
vp_margin sd within a layout     50.6   <- the dice
vp_margin sd between layouts     45.0   <- the scenario
```

**The dice contribute more spread than the scenario does.** The same scripted
policy on the same map wins 0.00 on one layout and 1.00 on another. Two
consequences:

- A single-epoch `success_rate` is worthless, as already documented — but so is a
  point comparison between two arms. Only converged rolling means over hundreds
  of epochs average this down.
- It bounds nothing about **run-to-run policy variance**, which this measurement
  does not touch at all. Batch 2's D-vs-E gap of 1.7pp on rolling means is well
  inside what two seeds of the same config could produce, which is why that
  comparison could only bound the effect as small rather than show it was zero.

Run at least two seeds per arm before reading a difference smaller than ~10pp.

### Pairing beats sample size — `just measure-paired`

The noise floor above is what makes two aggregate rows nearly useless for small
effects. Per-episode `vp_margin` sd on 25v25 is ~45–90, so at n=100 each row
carries a standard error of ~5–9 and their difference ~7–13 — larger than most
effects this project has ever tried to measure.

`just measure-paired <policy_a> <policy_b> <config> [n] [seed_base]` runs both
policies over the **same seed list** and differences them per episode. The
layout variance that dominates those rows cancels, and what is left is the
standard error of the difference.

It is not a refinement, it is the line between a result and an artefact. The
comparison it was written for — nearest-target versus weakest-target under unit
shooting — read **+8.0** as two aggregate means over 60 episodes and
**+1.7 ± 5.7 (t = 0.30)** once paired over 100.

Two rules it encodes:

- **Append unconditionally.** Every episode contributes one entry to each arm in
  seed order. A probe that skipped an episode in one arm and not the other
  reported +21.1 where the paired truth was +10.6.
- **Read the win count beside the mean.** A positive mean with a losing win
  count is a heavy tail, not an improvement — and only one of those two numbers
  says so. The weakest-target arm above was ahead in 24 of 100 episodes.

Either argument may be a baseline name **or a checkpoint path**, so the
comparison that decides a result — a trained policy against the bar — can be
paired directly rather than read off two aggregate rows.

**It also pairs two code versions.** Run the same policy from a worktree at an
earlier commit and difference per episode; that is how the baseline arrival fix
was resolved from "worth +10.2" (unpaired, one seed set) to +16.7 ± 6.5 on one
seed set, −1.6 ± 6.7 on another and **+7.5 ± 4.7 pooled** — helping occupancy
consistently while its effect on `vp_margin` stayed layout-dependent.

## 3. Trace metrics (`just analyze`, `just analyze-compare`)

Per-step metrics from recorded event logs. Aggregates hid the objective drift
that drove the reward redesign; traces showed it immediately. **But most of
these metrics do not rank policy quality**, which was established by running
them over the scripted baselines above — a check worth repeating whenever one
is added.

| metric | discriminates policy quality? |
|---|---|
| **`vp_per_step`** | **Yes — the one that does.** 0.00 / 2.13 / 3.75 / 5.00 across the baselines, matching win rate exactly |
| `final_fraction_at_objectives`, `peak_*`, `objective_drift_ratio` | Only against random. Every competent policy saturates at 1.00 with a drift ratio of 1.00 — the same saturation that makes `models_at_objectives` unable to rank policies |
| `mean_group_distance` | Measures coherency, not quality. Piling the whole army on one point scores *best*, so read it as a legality check against the phase's `group_max_distance`, never as a score |
| `target_selection_optimality` | Only defined when shooting happens — `N/A` for every movement-only policy |
| `idle_rate` | **Ignore.** Structurally ~50%+: it counts every `Stay` regardless of phase, and half of all steps are the shooting phase where `Stay` is often the only legal action |
| `objective_approach_rate` | **Inverted.** Falls once models *arrive* and stop approaching, so competent policies score 13–15% and random scores 24% |
| `tactical_score` | **Ignore.** A 50-point start with ±5–15 adjustments, two of which are `idle_rate` and `objective_approach_rate` — enough to penalise competent play: it scored random 50/100 and every scripted policy 30/100 |

Two of these were fixed rather than documented: `mean_group_distance` used to be
an all-pairs army-wide mean (so concentration read as cohesion), and
`oscillation_rate` counted a stationary model as oscillating every step, scoring
holding policies at ~70% against random's 5%.

## 4. Reward components (per epoch, from training rollouts)

`reward/components/<calculator>` and `reward/components/<calculator>/<sub_component>`. A phase
that lists the same calculator type twice yields `<type>` and `<type>_2`
(`phase_manager.py:96–100`), so a missing key may be a second instance rather than a disabled
term.

Every value is a **per-step mean over the rollout**:

1. `phase_manager.py:176–278` — per-model calculators summed, divided by `n_alive` (`:210`); sub-breakdowns keyed `<name>/<component>`, also divided by `n_alive` (`:212`); global calculators added flat (`:218`). Terminal bonuses enter the same dict once per episode — trap Ⓑ.
2. `wargame.py:688–693` — `step()` publishes the step's breakdown and accumulates it across the episode.
3. The last division depends on which rollout path ran:
   - `num_rollout_envs > 1` (the default; 0 means auto-detect): `ppo/lightning.py:664–691` sums each key over every env-step and divides once by `total_steps`.
   - `num_rollout_envs == 1`: `agent_base.py:116–120` divides by `episode_reward_steps`, then `ppo/lightning.py:754–769` re-weights by episode length, rescales if the rollout is truncated to `n_steps`, and divides by `total_steps`.

Both paths land on the same units. So `reward/components/vp_gain = -0.10` means **the player
loses 0.10 normalized net VP on every step**, not once per episode. Read all these keys as
per-step rates.

### Calculator semantics

| Component | Kind | Formula / meaning | Healthy direction |
|---|---|---|---|
| `vp_gain` | global | `(player_vp_delta - opponent_vp_delta) / cap_per_turn` (`vp_gain.py:34`). Signed: **negative means the opponent is out-scoring you.** | → positive |
| `objective_coverage` | global | Fraction of objectives the player controls, paid every step. Rewards spreading across *distinct* objectives. | ↑ toward 1 |
| `models_at_objectives` | global | Fraction of alive models inside some objective radius. Saturates at 1.00 for every competent policy, so it ranks nothing — see the trace-metric note above. | ↑ toward 1 |
| `closest_objective` | per-model | Distance-shaping toward the nearest objective. Sub-keys: `progress`, `distance_delta`, `base_penalty`, `best_distance_bonus`. | `progress` ↑ |
| `closest_objective_v2` | per-model | As above plus de-stacking. Extra sub-keys: `target_obj_idx`, `target_switched`, `overstack_penalty`. High `target_switched` (≳0.3) means the policy is dithering between targets. Every sub-key is weight-scaled and averaged over alive models like the rest, so `target_obj_idx` is a mean of indices and means nothing as an index. | `target_switched` ↓ |
| `objective_hold` | per-model | Value of the objective a model occupies, keyed on control state (`player`/`contested`/`opponent`; 0 off-objective and 0 on a neutral one). The only term paying a model that is standing still, so it is the per-model signal that survives after arrival. | ↑ |
| `model_kills` | per-model | `bonus_per_kill` per opponent killed **by that model**. Credits the shooter; `killing` does not. | ↑ |
| `group_cohesion` | per-model | Penalty proportional to distance beyond `group_max_distance` from the nearest same-group model. 0 when in range or alone in group. **Exactly 0 for a whole run means disabled or never violated — not "good cohesion".** | 0 or ↑ |
| `killing` | global | `bonus_killing_opponent` per newly killed opponent, paid identically to every model. | context-dependent |
| `objective_flip_bonus` | global | Change in an objective-control potential, summed over objectives. At `loss_penalty_scale == 1.0` it is a pure potential (farming-proof). | ↑ |
| `terminal_success_bonus` | terminal | Awarded on one terminating step when criteria hold (`phase_manager.py:243–262`), scaled by remaining-turns fraction **only when the phase sets `terminate_on_success: true`**; delivered at full value otherwise. See trap Ⓑ. | ↑ |
| `terminal_vp_bonus` | terminal | Awarded at episode end when player VP clears the phase threshold. Same dilution caveat as above. | ↑ |

### Success criteria

`success_rate` means something different per phase, depending on the configured criteria:

| Criteria | Succeeds when |
|---|---|
| `all_at_objectives` | Every alive model is within an objective radius |
| `fraction_at_objectives` | At least `min_fraction` of **alive** models are within an objective radius |
| `all_models_grouped` | Every model is within `max_distance` of a same-group member (sole members count as grouped) |
| `player_vp_min` | Player VP ≥ a threshold derived from mission, objective count and round count |
| `player_ahead_on_vp` | `player_vp > opponent_vp` |

**Always resolve the phase's criteria from the run config before interpreting `success_rate`.**
A 90% success rate under `all_at_objectives` says nothing about whether the player is winning.

---

## 5. Reading rules

Traps that produce confidently wrong conclusions.

Ⓐ, Ⓑ and Ⓒ are defects rather than inherent properties, and are marked in the source as
`TODO(metrics-trap-A|B|C)` at the site where the fix belongs — grep for `metrics-trap` to find
them. Ⓓ, Ⓔ and Ⓕ are correct by design; this document is their only fix.

**Ⓐ `success_rate` silently changes definition.** `lightning_base.py:324–327`: when
`episode_successes` is empty it falls back to `(steps < max_turns).mean()`, i.e. "episode ended
early" rather than "criteria met". Same key, different meaning, no warning. A success is
recorded per episode only when `env.last_step_context` is set, so the fallback now needs *every*
eval episode to have run zero steps — latent rather than active, but a consumer still cannot
tell from the value which definition produced it.

**Ⓑ `terminal_success_bonus` is not comparable to the other components.** It is paid once per
episode but divided by every step (stage 3 above), so its magnitude scales inversely with
episode length. It moves when episode length moves, with no change in policy quality. Never
compare it across runs with different `mean_episode_steps`, and never read its value as the
configured bonus.

**Ⓒ `reward_phase` is double-logged** — through Lightning at `:339` and again through raw
`wandb.log(step=self.global_step)` at `:353`, on different step counters. Prefer the value on
the epoch-aligned row.

**Ⓓ Percent vs. fraction.** The logged `success_rate` is `sr * 100` (`:331`) but
`success_threshold` in config is a fraction in `[0, 1]`. Divide the metric by 100 before
comparing. `eval/win_rate` and `eval/baseline_*_win_rate` are percentages too, while every win
rate quoted in `reports/` and in the baseline table above is a fraction.

**Ⓔ Eval and training rollouts are different episodes.** `reward/mean_episode_reward` (eval,
greedy) and `reward/components/*` (training, exploratory) will not reconcile. Do not compute
one from the other.

**Ⓕ Components are means over alive models.** Per-model calculators divide by `n_alive`
(`phase_manager.py:210`) — note this is the *logged* component only; the reward the policy is
trained on keeps a per-model vector (`last_per_model_reward`) that is never averaged. As models
die, the same per-survivor behaviour yields a different component value. Cross-check against
casualties before attributing a shift to policy change.

---

## 6. Evaluation procedure

A recommended order of operations for an agent asked to assess a run.

### Step 1 — Establish context before reading any number

Fetch the run config and resolve: `reward_phases` (names, calculators, weights,
`success_criteria`, `success_threshold`), `n_episodes`, `max_turns`, `number_of_opponent_models`
and `opponent_policy`. Without the active criteria, `success_rate` is uninterpretable.

### Step 2 — Triage

| Check | Verdict |
|---|---|
| `state == "crashed"` and `_step == 0` | Startup failure. Report and stop — no metrics to read. |
| `state == "crashed"` with steps logged | Partial run. Metrics valid up to the last step; do not treat final values as converged. |
| `phase_advanced_at_epoch` absent and `reward_phase == 0` | Never cleared phase 0. Investigate the phase-0 criteria before anything else. |
| `mean_episode_steps == max_turns` throughout | Episodes never end early. Either `terminate_on_success: false`, or success is never reached. |

### Step 3 — Convergence

- `loss/value_loss_epoch` should fall and flatten. Still falling at run end ⇒ undertrained.
- `loss/entropy_loss_epoch` should rise toward 0 gradually. Reaching near-0 in the first ~10% of epochs ⇒ premature collapse; suspect `ent_coef` too low.
- `loss/policy_loss_epoch` oscillating around 0 is normal.

### Step 4 — Task performance

- `success_rate` trend across epochs, **not** single points (Ⓐ, and ±1/`n_episodes` resolution).
- `reward/mean_episode_reward` trend; check `min_episode_reward` for a failure mode the mean hides.
- Per-component: is the component the current phase actually weights the one improving?

### Step 5 — Plateau detection

Find the epoch after which the primary components change by less than ~5% of their range.
Everything past that is wasted compute. Report it as an epoch budget recommendation.

### Step 6 — Red flags

| Flag | Condition | Meaning |
|---|---|---|
| **Losing the VP race** | `vp_gain < 0` and flat | Opponent out-scores the player every step; nothing in the reward opposes it. Especially likely when criteria are objective-based rather than VP-based. |
| **Dithering** | `closest_objective_v2/target_switched ≳ 0.3` | Policy re-targets ~1 step in 3. |
| **Dead component** | any component exactly `0` for the whole run | Disabled, unreachable, or zero-weighted. Not evidence of good behaviour. |
| **Reward/criteria mismatch** | components improve, `success_rate` flat | Optimizing something the criteria do not measure. |
| **Success without winning** | `success_rate` high, `vp_gain` negative | The criteria do not select for winning. Check whether that is intended. |

### Step 7 — Score on the real layouts

Everything above reads a run against `random_terrain`, which is the training
distribution. `just measure-maps <ckpt> <config>` scores the same scenario on
the fixed layouts in `configs/evaluation/maps/`. Read it per map, not as the
mean: the case it exists to find is strong on most tables and broken on one.
Quote it against a baseline on the same maps, as with every other number here.

### Step 8 — Report

State the verdict, the evidence with metric keys and epochs, and — for every claim — whether
it rests on a full history or a sampled one. Distinguish "the 5 runs I sampled" from "the
project". If a red flag depends on a trap above, say which.

---

## 7. Programmatic access

Via the W&B MCP server (see the `wandb` entry in `~/.claude.json`):

| Need | Tool |
|---|---|
| Project shape, available keys | `probe_project_tool(entity, project)` |
| Run list with final summaries | `query_wandb_tool` (GraphQL, `order: "-createdAt"`) |
| Training curves for one run | `get_run_history_tool(entity, project, run_id, samples=N)` |
| Two runs side by side | `compare_runs_tool` |

`get_run_history_tool` with an explicit `keys` list can return zero rows even when the keys
exist; call it without `keys` and filter client-side.

Because eval and training metrics land on different steps, history rows are sparse — each row
carries one family or the other. Group by family before computing trends rather than assuming
every row has every key.

For per-step ground truth rather than these aggregates, record an event log and use
`analyze_match` instead — see [game-state-io.md](game-state-io.md).
