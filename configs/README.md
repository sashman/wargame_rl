# Configs

Environment configs, tiered by **what breaks if you change the file**.

| Directory | Changing a file here… | Lifetime |
|---|---|---|
| `golden/` | invalidates a published number | permanent |
| `experiments/` | affects one open question | delete when answered |
| `evaluation/maps/` | changes what final evaluation measures | permanent |
| `dev/` | breaks a test or a demo, never a result | permanent |

## `golden/` — results-bearing

**`25v25_maps_two_mode.yaml` is the coherency baseline of record (2026-08-19).** ⚠ Train it with `just train-coherency-baseline`, never plain `just train` — `ent_coef` is not an env-config field, and the PPO default (0.03) is the worse arm by **+5.9 ± 2.5 vp** read paired on seed. Scored on the nine held-out tables against four opponents with the verified top-3 decode: **ahead of the best script on three of four**, behind `contest_and_spread` by 13.7, and **0.94–0.97 intended unit coherency on all four**. The matching evaluation configs are `evaluation/25v25_maps_take_opponent_refereed.yaml` and `evaluation/25v25_maps_vs_*.yaml` — swapping the opponent voids every baseline, so re-measure the scripts on each.


Each of these backs a number quoted in `CLAUDE.md` or a report. Do not edit one
to try something; copy it into `experiments/` instead.

| Config | What it established | Comparable to |
|---|---|---|
| `25v25_shooting_opponent.yaml` | **Beats the shooting bar**: +28.4 vp_margin against `squad_march_shoot`'s +17.0, both seeds, n=100. The lever is `objective_hold.crowding_exponent`. [Report](../reports/2026-08-08-paying-the-pot-beats-the-bar.md) | `25v25_cover_control.yaml` |
| `25v25_cover_control.yaml` | Batch-3 control. Cover is not used even with a working geometry and a priced trade. [Report](../reports/2026-08-06-cover-signal-reason-geometry.md) | `25v25_shooting_opponent.yaml` |
| `25v25_single_phase.yaml` | The single-phase control for the curriculum question | `25v25_curriculum.yaml` |
| `25v25_curriculum.yaml` | Two-rung ladder; final phase identical to the control's, so the comparison isolates the curriculum | `25v25_single_phase.yaml` |
| `25v25_maps_coherency.yaml` | **Unit coherency on the real tables.** `objective_hold.require_coherent` is the only coherency lever measured to work: formation 0.51 → 0.756–0.886 with the referee off, 81.5 vp_margin on nine held-out tables, six seeds. **Never add `enforce_move` to it** — enforcement is a play-time referee, and training under it drops formation to 0.569 and held-out vp to 70.3. [Report](../reports/2026-08-16-enforcement-is-a-referee.md) | the enforced arms in that report |

**The two scenarios are not comparable to each other.** The shooting pair faces
`scripted_advance_and_shoot` on terrain regenerated every episode with
`objective_min_separation`; the curriculum pair faces the non-shooting
`scripted_advance_to_objective` on fixed terrain. Switching a config's opponent
or terrain profile invalidates every baseline and agent score measured on it —
re-measure both sides, on identical seeds, or the comparison is meaningless.

Two of these are **frozen bit-identically** by `tests/test_reward_golden.py` and
`tests/test_observation_golden.py` (`25v25_single_phase`, `25v25_shooting_opponent`,
plus `dev/4v4_two_phases`). Any edit that changes a reward or an observation
fails those tests loudly. That is the intended behaviour, not an obstacle:
regenerate the fixtures only when you mean to change the numbers.

## `experiments/` — arms

Where a screen's arms live while its question is open. Copy the control, change
**one thing**, and give it a distinct `config_name` — run names are built from
it, and two arms with the same name write checkpoints into one directory and
score whichever process saved last.

Delete an arm once its question is answered; `git log -- configs/` restores any
of them. Every batch so far was disposed of this way (batch 1/2's arms, batch
3's `cover_reason`, batch 4's eight `25v25_beat_*`).

| Config | The open question |
|---|---|
| `25v25_real_maps.yaml` | Does training **on the real tables** produce a policy the generated scenario cannot? Same opponent, forces and reward as `golden/25v25_shooting_opponent.yaml`; only the board changes. Draws from 36 tables and holds out the 9 whose number is divisible by 5 |

## `evaluation/maps/` — the real table layouts

Final evaluation, and — through `map_pool` — training. Generated terrain is what
makes a positioning result falsifiable, but it never asks how the policy does on
the boards the game is actually played on, and it cannot: the generator places
objectives only in the contested middle, while the real layouts put a third of
them inside each player's own deployment zone. Measured across all 45 tables,
the split is exactly 75 player-zone / 75 middle / 75 opponent-zone, every table
mirror-symmetric. **That is a different mission, not a harder board** — win rate
saturates at 1.00 for both scripted rungs, and `random` wins 0.67 by deploying
onto its home objectives and never leaving. ⚠ Those win rates, and the
deployment-time holding figures that used to sit here, were measured on the
hand-traced tables and need re-running; the 75/75/75 split is current.

**The tables are generated, not authored — `just fetch-maps` regenerates all 45
from the public layout API.** They were traced by hand from that same source
originally, and the tracing lost detail: outlines became quads where the source
carries 167–348 vertices, one piece per table went missing, and the objectives
were picked by eye for board symmetry rather than read off the layout. Each is
now the layout's own geometry, simplified to the 8-vertex observation budget by
Douglas-Peucker — chosen by measuring all 720 pieces, where it holds area to a
worst 1.078 of the true silhouette against the footprint rectangle's 1.592.

**An objective is a ruin — a group of terrain pieces sharing at least 1" of
boundary.** A layout's marker only says *which ground* is fought over. The pieces
are kit components, not buildings: a rectangle split along a diagonal seam, two
bars butted into an L, drawn as one blob by the source's own render. A marker
takes the **largest unclaimed ruin within control range** — not the nearest,
because markers often sit in the gap beside a scrap of scatter terrain, and
nearest-wins put objectives on 12.9 sq in slivers while 82.5 sq in ruins stood
two inches away. **There are no disc objectives**; a marker beyond range of every
ruin takes the nearest anyway. Every table carries five.

**Training on these maps consumes them.** `map_pool.names` is the split: name a
subset for training and score the complement, or a transfer number means
nothing.

A map is terrain, and optionally the objectives that go with it:

```yaml
name: table_01
terrain:
  - { footprint: [12, 8, 18, 14] }
  - { footprint: [27, 20, 33, 26] }
objectives:
  - area: [[12, 8], [19, 8], [19, 15], [12, 15]]
```

Every objective must be *determined* — an `area` outline or x/y — because a
fixed map exists so a row means the same thing each run, and one undetermined
entry would silently randomise the lot (`place_for_episode` honours fixed
positions only when all of them have coordinates).

**On the real layouts the objective is the ground.** Every one of the 270
markers sits inside exactly one ruin, so each objective is that ruin's outline:
an area objective, held by standing on it, which is the rules' terrain
objective rather than an abstract disc floating over the same footprint. The
layouts print six markers — one home per side, two centre, two in no man's land
— but on 24 of the 45 the two centre markers share the board's largest ruin,
and one piece of ground is held once. Those maps carry five objectives; the
other 21 carry six.

`just measure-maps <ckpt> configs/golden/25v25_shooting_opponent.yaml` runs the
golden scenario unchanged and swaps only the layout — `terrain`, plus
`objectives` and `number_of_objectives` where the map has them — once per map,
reporting a row each plus the spread. **This makes the maps a six-objective
mission** where the golden config trains on three, so a map score is not
comparable to a `measure-checkpoint` score; it is comparable across policies on
the same maps, which is what it is for.

**But the mission only pays for three of them.** VP is
`min(cap_per_turn, controlled * vp_per_objective)` = `min(15, held * 5)`, so a
fourth objective you control scores nothing extra, and on these maps own VP is
saturated for anything competent (272–277 of a 285 ceiling). Above the cap
`vp_margin` is decided by the *opponent's* score — by denial — which is why
`measure-maps` prints `plr VP` and `opp VP` beside the margin, and why `held`
stops ranking policies here: `squad_march_deny` holds **3.00** and
`squad_march_shoot` **4.00**, and they score level. See
[docs/metrics.md](../docs/metrics.md) and
[the report](../reports/2026-08-16-the-cap-makes-it-a-denial-game.md).

**⚠ Re-measure: the tables were regenerated from the layout API on 2026-08-20
and now carry five objectives, not six.** The finding below was measured on the
hand-traced six-objective tables and its numbers are history until re-run.

**The bar saturates on six objectives, so the per-map spread no longer finds
hard layouts.** On `25v25_shooting_opponent.yaml`, `squad_march_shoot` wins
**every one of the 45 maps** (`held` 3.67, `alive` 0.69); the same maps
with terrain only and three random objectives win 0.70 with a −70..+100
vp_margin spread and `alive` 0.40. Dropping the two home objectives does not
recover it (still win 1.00 on all ten sampled), so this is not the deployment
zones — `scripted_advance_and_shoot` concentrates and can contest about two
points, so every objective past that is uncontested. Exposure falls ~5x: the
armies barely meet. Rank two policies against each other here; do not read a
row as "this layout is hard". Measured at n=1 per map, so read the win column
and the `held`/`alive` gap, not a single per-map vp_margin.

Quote it against the bar on the same maps:

```bash
just measure-maps squad_march_shoot configs/golden/25v25_shooting_opponent.yaml
```

Previews are regenerated with `just render-maps`, which draws each map file
directly rather than resetting an episode — fifty deployed models cover the
layout the picture exists to show.

There is deliberately **no config per map**. A 25v25 scenario is ~10 KB, so
copying it per map means every future reward change must be applied N times —
and the first one missed makes evaluation measure a different game from
training, without failing anything.

## `dev/` — fixtures and demos

Fast, small, and no result is ever quoted from them.

| Config | Used by |
|---|---|
| `ci_smoke.yaml` | `tests/test_z_e2e_training.py`, the README resume examples |
| `4v4_two_phases.yaml` | the `just train` default; frozen by both golden tests |
| `terrain_los_demo.yaml` | `tests/test_terrain_render.py`, `docs/terrain.md` |
| `tiny.yaml` | `train.py`'s default, the `just record` default |

## Conventions

- **Set `config_name`.** It leads the run name, and everything after it
  describes the *scenario* — which arms of one experiment deliberately share.
- **Repeated model entries use YAML anchors.** Five squads of five are one
  profile; the anchor single-sources it so a weapon change is one edit. YAML
  resolves anchors at parse time, so the config the env sees, and the copy
  dumped beside each checkpoint, is fully expanded either way.
- **Comment the *why*, with the measurement.** The golden configs carry the
  numbers that justify each term. That is why they are long, and it is the part
  worth keeping.
