# Pre-registration: curriculum rung E1 — two armed armies on a flat board

Written 2026-09-17 23:20, **before any training number on this rung
exists**, on branch `feature/curriculum-e1` (stacked on PR #377). Issue #381,
parent question #340, rung **E1** — the join of the bodies branch (A5's army)
and the guns branch (C2's rifles), the first rung of the road to the
real game.

## The scenario

`configs/experiments/curriculum/e1.yaml` is
`configs/experiments/24v24_maps_spare_squads.yaml` — the golden family's
twenty-round game, random turn order, the win reward phase
(`closest_objective_v2`, `objective_hold` at `crowding_exponent` 1.0,
`model_kills`, `group_cohesion`, `vp_gain`, `objective_coverage`),
`player_ahead_on_vp` as success, `squad_march_take` on the opponent seat
— with the map pool removed: a flat 60×44 board, six objectives of
radius 3 fixed symmetrically at (18, 12), (18, 32), (30, 8), (30, 36),
(42, 12), (42, 32), the two deployment bands 20" deep at either end,
every one of the twenty-four models on each side in eight squads of
three carrying the family's rifle (range 12, one attack). The observation
budgets stay at the family's values (six objectives, sixteen terrain
slots, all padding here) so a checkpoint trained on E1 loads on E2
(generated terrain) and E3 (the real tables), each of which changes one
thing on the way to the refereed evaluation family.

E1 is not "A5 with guns": A5's reward was the curriculum's occupancy
reward and its success `all_objectives_occupied`; E1's reward and
success are the golden family's, because that is what E3 is scored on
and the E rungs change the board one step at a time toward it. The join
is on the army — the same twenty-four bodies in eight squads.

## The bridge diverges, under a recorded rule — not a design fault

`just measure-bridge` on E1 reads BRIDGE DIVERGES with every scenario
field E1 adds bisected away (the padding budgets, exposure tracking,
turn order, three rounds instead of twenty), and the shooting golden
config `25v25_shooting_opponent.yaml` diverges the same way. The cause is
the per-model facade's own recorded rule: with several shooting units a
side, it judges a unit's targets and cover after the casualties an
earlier unit inflicted, which the whole-army step cannot
(`FacadeDivergence`, `shooting.targets_judged_after_casualties`) — first
in **every** episode, by round 3, on eight of eight. The C and D rungs
never met it because our side had at most one shooting unit. So the
#317 rule ("a divergence there is a design fault") is bounded, with this
rung: **a divergence under a recorded rule is the per-model facade
playing the more rules-faithful game; the rung stands, and each trainer
is read against the bar on its own facade.** `measure-bridge` now names
the recorded rules and exits 2 only when nothing was recorded (a build
this rung is first to need, shipped in its PR with a test on the
verdict).

## The bar, measured first

`squad_march_take` against `squad_march_take`, seeds 700000+ at n=100,
turn order random:

| facade | vp | ±SE | win (`player_ahead_on_vp`) | held | on_obj | alive | coherent | stat / hold |
|---|---|---|---|---|---|---|---|---|
| phase (the control's) | **+14.2** | 9.5 | 0.550 | 2.86 | 0.936 | 0.273 | 0.878 | — |
| per-model (the arm's) | **+20.3** | 9.5 | 0.620 | 2.88 | 0.942 | 0.283 | 0.871 | 0.56 / 0.05 |

A mirror script against itself is not zero: the seat carries an
advantage on this layout (the player deploys on the left band; on
`25v25_maps_two_mode` the same gate read +6.5 ± 6.1). Both trainers
train the player seat, so the bar they are read against is the player
seat's, on their own facade. Per-episode sd is about 95, so at the n=180
the E criterion names the SE of a single row is about 7.1 and the paired
SE smaller. `alive` 0.27 says the rifles decide this game: three quarters
of both armies die in twenty rounds.

The script fails no criterion of its own (NULL needs it to), and the
two facades' rows differ by 6 vp on identical seeds — inside one SE,
the recorded rule's whole effect.

## The arms

| | whole-army control | per-model arm | per-model companion |
|---|---|---|---|
| trainer | `train.py` via `just train-curriculum-control` | `train_per_model.py` | `train_per_model.py` |
| start | from scratch | **from scratch** | `--warm-start-from` A5b's `last.pt`, seed for seed (`per-model-a5-2026-09-15-10-07-*-s{1,2,3}a5b`, the rung below on the bodies branch, 0.260 / 0.180 / 0.160 on its own rung) |
| budget | 60 epochs of 2048 steps, extended once to 120 if still rising | **122,880 rounds** at 128 per update (the cap; the E criterion is a vp bound, so the control's rounds-to-pass is not a signal to scale from); 2× the cap, read once, if the control is extended | the same |
| seeds | 1, 2, 3 | 1, 2, 3 | 1, 2, 3 |
| flags | defaults (`ent_coef` 0.03) | `--num-rollout-envs 4 --rollout-rounds 32 --eval-every-rounds 512 --checkpoint-every-rounds 512 --n-eval-episodes 30 --ent-coef 0.003` | the same plus the warm start |
| logging | Wandb `curriculum-e1`, tag `e1-ctl` | `curriculum-e1`, tag `e1` | `curriculum-e1`, tag `e1w` |

Nine trainers on the box at once, nothing else. A5b's runs took 37
hours for 122,880 rounds without a shooting phase; E1 steps two phases a
round with twenty-four shooters, so the per-model runs are expected to
take two to three days.

## Criteria

Read greedy, no decode, paired per episode against `squad_march_take`
on the arm's own facade, seeds 700000+ at **n=180**.

- **PASS:** on all three seeds, the paired vp difference (arm − script,
  same seeds) has mean ≥ **−1 paired SE**; `held` ≥ bar − 0.2 (phase
  2.66, per-model 2.68); greedy coherency ≥ bar − 0.03 (0.85 / 0.84).
- **FAIL:** any seed misses any clause. The next step is the census:
  where the bodies are when the clock runs out, `alive` against the
  bar's 0.27 (hoarding or under-arrival), the objective split.
- **NULL (scenario):** only if the script fails its own criterion — a
  script is its own bar here, so never on the letter; recorded as the
  clause it is.
- **Pass with a defect:** the bound holds but the health panel is red
  over the last quarter.
- **Readouts:** win rate against 0.55 / 0.62; `alive`; `on_obj`; turns
  (every episode runs the clock, 40 phase-clock); coherency greedy and
  sampled; the passive fingerprint; the panel over the last quarter;
  the control and the companion beside every row; in-run vp and win
  rate (n=30, seeds 500000+).

Power: per-episode sd ≈ 95 unpaired; paired, a script and a policy that
plays a similar game share most of the layout noise, so the paired SE at
n=180 should sit under 5. The bound is one paired SE, so a policy exactly
level with the script fails a seed about one time in six and 3/3 about
four times in ten — the clause is written to #340's letter and this
rung will be read with that in mind, as C2 was.

## What I expect (a guess, written so it can be wrong)

The per-model arm from scratch repeats A5's under-arrival and sits well
under the bar on all three seeds (paired −20 or worse), with `alive`
above the bar's 0.27 — it hoards, as the whole-army lineage does on the
real tables. The companion from A5b starts ahead in-run and finishes in
the same place. The whole-army control, the golden lineage's own trainer
on its own reward, reaches the bar's neighbourhood on two of three seeds
by 120 epochs and fails the letter on the third. If the per-model arm
instead passes, the E rungs have inverted the C rungs and the ladder's
next question is why the toy rungs were harder than the real one.

## Amendment 1 — written 2026-09-17 23:20, minutes after launch, before any evaluation row

**A name collision on the record.** `configs/experiments/curriculum/e1.yaml`
replaced, in the same path, the Stage 0 rung E1 of PR #339 (issue #333,
parent #328: one unit walking onto one objective — the rung that became
A1 when #340 superseded #328). That older rung's pre-registration is
`reports/2026-09-14-curriculum-E1-preregistration.md`, and the older
`e2.yaml` / `e3.yaml` beside this file are #334 / #335's, likewise
superseded; they are left untouched until E2 and E3 replace them in
their own PRs. Two consequences for reading this rung: the Wandb group
`curriculum-e1` and the run suffixes `s{1,2,3}e1` are shared with the
2026-09-14 runs of that older rung, which are void
(`reports/2026-09-14-one-episode-per-update.md`); this rung's runs are
the ones dated 2026-09-17 23:13 (checkpoint directories
`per-model-e1-2026-09-17-23-13-*-s{1,2,3}e1{,w}`, run ids on #381), and
every read of E1 names them by directory.

**Wall-clock, corrected.** The per-model runs update every 25–30 s at
128 rounds (rollout 20–26 s, update 4–10 s), so the 122,880-round budget
is about half a day, not the two to three days written above; A5b's 37
hours were measured with twelve trainers on the box.

## Amendment 2 — written 2026-09-19 01:20, before the relaunch, no number from any E1 run on the record

**The 2026-09-17 23:13 runs were stopped by decision** within the hour
of launch (Sash, 2026-09-17: not to run E1 while A5's per-model arm was
failing, since E1 is A5's army with guns) and none was read; their
directories are stale and every read of this rung names the runs by the
2026-09-19 timestamps below. #381 held since.

**A5 has since passed on the start axis** (A5i 0.960 / 0.980 / 0.960 at
122,880 — the bar cloned from 2,000 games and held by anchored PPO;
`reports/2026-09-19-curriculum-a5-passes-on-the-start-axis.md`), so the
rung below on the bodies branch now has a policy at the bar's level, and
**the companion's warm start changes from A5b's `last.pt` to A5i's**,
seed for seed (`per-model-a5-2026-09-18-20-46-09-s{1,2,3}a5i/last.pt`).
Loading check, three episodes on e1.yaml, not a read: the checkpoint
loads and plays (held 1.33, −101.7 vp) — a policy that has never seen an
enemy or a gun, as expected. The companion runs **without the anchor**
(the question is whether reward can teach shooting on top of a held
walk; anchoring to a policy that cannot shoot would forbid the answer);
the scratch arm and the whole-army control are unchanged. Criteria,
n, seeds and budget are unchanged. Set as a goal by Sash 2026-09-19
01:05: "carry the A5i policy up to E1 and read whether the allocation
survives guns" — pass on the pre-registered clauses, and on a fail the
census says whether the walk broke (`held`, bodies on points by turn)
or the shooting was never learned (`alive`, kills).

Tags: control `e1-ctl`, scratch `e1`, companion `e1w`; Wandb
`curriculum-e1`; run ids on #381 when they exist. Nine trainers and the
C3b clone fit (D3) on the box.

## Amendment 3 — written 2026-09-19 02:25, the whole-army control read at 120 epochs; no per-model number yet

The control finished its 60 epochs in about twenty minutes (the phase
facade steps a 24-v-24 shooting round in ~10 ms; the "two to three
days" of § The arms was the per-model estimate) with the in-run vp
still rising on every seed (by quarter −194 → −19, −196 → −21, −208 →
−69), so the pre-registered once-only extension to 120 ran, resumed
from each seed's 60-epoch `last.ckpt` (Wandb `pp3ep94b` / `rkdln3xc` /
`njr0mg5a`; directories `…-2026-09-19-01-41-2*-s{1,2,3}e1-ctl-x`).
Read greedy, K=1, on the phase facade, n=180 on 700000+, paired per
episode against `squad_march_take` on the same facade:

| seed | vp | script | arm − script, paired | t | held (bar 2.66) | coherent (bar 0.85) | alive | on obj | win |
|---|---|---|---|---|---|---|---|---|---|
| s1 | 4.7 | 20.2 | **−15.6 ± 9.2** | −1.68 | 2.47 | 0.860 | 0.577 | 0.877 | 0.43 |
| s2 | **93.6** | 20.2 | **+73.4 ± 10.3** | +7.10 | 3.05 | **0.780** | 0.502 | 0.781 | 0.79 |
| s3 | **65.4** | 20.2 | **+45.2 ± 10.5** | +4.31 | **1.98** | **0.773** | 0.466 | 0.808 | 0.76 |

**The control FAILS on the letter, on every seed by a different
clause**: s1 on vp (−15.6 against a −9.2 bound), s2 on coherency (0.780
against 0.82), s3 on held (1.98 against 2.46) and coherency. And two of
three seeds beat the script by 45–73 vp, t 4.3–7.1 — the largest
margins over a script on any rung of this ladder — by killing: opponent
VP 121–136 against the script's own ~200 (s1, which does not, concedes
227). What the control learned on E1 is the guns, at the cost of
formation (coherency 0.77–0.78 against the script's 0.85) and, on s3,
of the points. The E clause was written as a conjunction of three
readouts at the script's levels; a policy that wins the fight and
loses the formation fails it, and that is what the clause was for.
Reported as: the control beats the bar on vp on two seeds and fails
the rung. The per-model arm and the companion are read against the
same three clauses on their own facade at 122,880.

## Amendment 4 — written 2026-09-19 10:30, the per-model arm and the companion read at 122,880

Greedy, no decode, on the per-model facade, n=180 on 700000+, paired
per episode against `squad_march_take` on the same facade (the script:
vp +20.2 ± 7.6, win 0.567, held 2.93, on_obj 0.946, coherent 0.873):

| row | vp | win | held (bar 2.93; clause ≥ 2.68) | on obj | coherent (clause ≥ 0.84) | stat / hold |
|---|---|---|---|---|---|---|
| scratch s1 | **−158.1 ± 3.3** | 0.000 | 0.76 | 0.096 | 0.584 | 0.00 / 0.21 |
| scratch s2 | **−151.9 ± 4.8** | 0.028 | 0.71 | 0.169 | 0.383 | 0.49 / 0.09 |
| scratch s3 | **−159.8 ± 3.3** | 0.000 | 1.24 | 0.201 | 0.328 | 0.14 / 0.13 |
| from A5i s1 | **−67.9 ± 5.3** | 0.128 | 1.26 | 0.261 | 0.374 | 0.42 / 0.11 |
| from A5i s2 | **−179.7 ± 5.1** | 0.006 | 0.43 | 0.128 | 0.735 | 0.78 / 0.05 |
| from A5i s3 | **−159.1 ± 5.8** | 0.061 | 0.58 | 0.178 | 0.467 | 0.66 / 0.06 |

**FAIL as pre-registered on every seed of both arms, on every clause,
by a margin no clause was written for**: 170–200 vp behind the script
on five seeds (the sixth 88 behind), a fifth of the bar's points held,
a tenth to a quarter of the bodies on any point, coherency 0.33–0.73
against 0.87. The in-run curve never reached the script's vp on any
seed at any evaluation; the companion's best row (s1, −68) was its best
from 60k on.

The by-turn census (n=60 on 700000+; bodies on points of 24 and points
held of 6 after turns 3 → 5 → 7 → end of 20):

| policy | bodies on points | held | max stack |
|---|---|---|---|
| `squad_march_take` | 8.8 → 10.7 → 10.1 → 6.8 | 2.6 → 3.1 → 3.2 → 2.9 | 2.8 |
| scratch s1 / s2 / s3 | 1.5 → 1.6 → 0.8 → 1.3 / 2.0 → 2.4 → 2.1 → 1.0 / 2.3 → 2.0 → 1.2 → 2.4 | 0.9 → 0.8 / 1.2 → 0.7 / 1.2 → 1.2 | 0.9–1.9 |
| from A5i s1 / s2 / s3 | 2.7 → 3.6 → 3.7 → 2.5 / 2.9 → 3.8 → 3.3 → 0.9 / 2.7 → 2.5 → 2.1 → 0.9 | 1.3 → 1.2 / 1.3 → 0.4 / 1.5 → 0.6 | 0.8–1.9 |

**The walk broke and the shooting was not learned.** The script has
nine bodies on points by turn 3 and holds three through the game while
losing three quarters of both armies. The scratch arm never puts more
than two and a half bodies on a point at any turn. The companion
started from a policy that puts twenty of twenty-four bodies on six
points by turn 7 on A5; on E1 it has three to four on points by turn
5 — a fifth of A5i's walk — and by the end one to two and a half, the
points it reached lost (held 1.3–1.6 at turn 5, 0.4–1.2 at the end).
The A5i walk was overwritten within 10k rounds (stationary share
0.5–0.9 in-run from the first evaluations, 0.42–0.78 at the end), and
what replaced it neither walks nor shoots: vp 88–180 behind a script
that walks and shoots on the way. Sampled play and `alive` are read
with the report.

**Read against the whole-army control** (amendment 3): the control
learned the guns (+45 / +73 vp on two seeds by killing) and lost the
formation; the per-model arm learned neither. This is the first rung
where the per-model arm is not merely behind the control but plays a
different game from every comparator — the ladder's E branch stops
here, and the report says what the E1 census says about why.
