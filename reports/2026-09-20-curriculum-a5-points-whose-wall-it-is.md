# Curriculum A5-points, three arms in one night: the wall is the per-model trainer's, a warm start carries four objectives of five, and a start with four objectives given is walked off

**Verdict first.** Three arms on the half-step A5-points (six squads of
three over five objectives; the bar `squad_march_take` 1.000 in 6.71
turns), read greedy at n=100 on 700000+ from deployment:

| arm | what | verdict | success | held of 5 | report of the read |
|---|---|---|---|---|---|
| **CTL** | the whole-army trainer (`train.py`) from scratch, 60 epochs then the once-only extension to 120 (≈245k rounds) | **FAIL on the letter by two seeds — the best read this half-step has had from any trainer** | 0.980 / 0.620 / 0.870 at 60; **0.910 / 0.810 / 0.950** at 120 | 4.98 / 4.61 / 4.86; 4.91 / 4.80 / 4.95 | 93–97% of bodies on objectives, half a round to a round behind the bar |
| **W** | the per-model trainer warm-started from A4x seed-for-seed (the three-objective pass), 122,880 rounds then the same extension to 245,760 | **FAIL on the letter; ahead of the original on 3/3 at 2× the cap; s3 the best per-model read on the half-step** | 0.460 / 0.260 / 0.350 at 122,880; **0.340 / 0.170 / 0.600** at 245,760 | 3.96 / 3.53 / 3.62; 3.59 / 3.52 / 4.29 | no walk-off; s3 8 of 18 bodies on objectives by turn 7, 8.1 turns |
| **BS** | the per-model trainer from scratch under the backward start curriculum (episodes begin with four squads on four objectives; the level walks down as the level's episodes succeed), 122,880 rounds | **FAIL — the level never left four** | 0.000 / 0.030 / 0.030 | 2.08 / 3.02 / 2.07 | with four objectives given the policy keeps 2.0–3.1 of them and reaches the fifth 63–100% of the time |

The original per-model run from scratch reads 0.030 / 0.060 / 0.330 at
122,880; the four optimiser and reward arms before this night read
0.00–0.18. **The whole-army trainer solves the five-objective spread from
scratch to within a seed of the letter; the per-model trainer does not
approach it from scratch under any setting tried, and its best — 0.60 on
one seed — comes from carrying the three-objective policy in.** The
backward start isolated what the per-model policy does with an
objective it is standing on: it leaves. On level-four episodes the bar
keeps 3.9 of its 4 placed squads on their objectives and succeeds every
time; the arm keeps 2.0–3.1 and succeeds a third to a half of the time,
so the schedule never stepped down. That walk-off is the same event every
census on this half-step has recorded, in its smallest form, and it is a
property of the policy under this reward rather than of the start.

## Provenance

| field | value |
|---|---|
| date | 2026-09-19 22:03 – 2026-09-20 04:35: CTL 22:03–22:14 (60) and 22:17–22:30 (120); W 22:03–02:10 and resumed 02:13–04:15; BS 22:17–02:05 (from a worktree); reads as each exited |
| GPU / no-GPU | GPU (RTX 4090): three whole-army and up to six per-model trainers side by side |
| seeds | 1 / 2 / 3 per arm; W from `per-model-a4-2026-09-15-07-08-53-s{1,2,3}a4/pm-00245760.pt` seed-for-seed (weights only) |
| n | 100 at seed base 700000 (every read, the census, greedy against sampled); in-run 30 at 500000 every 512 rounds (per-model) and per epoch (whole-army); the level-4 census n=30 |
| config | `configs/experiments/curriculum/a5_points.yaml`, unchanged for all three; unrefereed by design; bridge identical 2026-09-18 |
| decode | none on the per-model facade; the whole-army control scored as every A-rung control (argmax, `measure-rung`) |
| paired | per episode against the bar on identical seeds (turns); per seed against the original A5-points on the same seeds |
| comparator | the original A5-points (`uagmul66` / `wedjjozy` / `ici5py46`): 0.030 / 0.060 / 0.330 at 122,880, 0.040 / 0.140 / 0.090 at 40,960, 0.220 / 0.220 / 0.110 at 81,920; the bar 1.000, 6.71 turns, held 5.00 |
| opponent | none |
| budget | CTL 60 epochs (`--n-eval-episodes 30 --record-during-training`, defaults otherwise, `ent_coef` 0.03) + once to 120 via `--resume-ckpt-path`; W and BS 122,880 rounds at 128 per update, `--ent-coef 0.003`, `gamma` 0.9, `mean` credit, eval and checkpoint every 512, recording and video on; W once to 245,760 via `--resume-from` under the cap rule; BS `--backward-start 4 --backward-start-share 0.75 --backward-start-advance 0.8 --backward-start-window 8` |
| code revision | pre-registrations `c9f23d4` (CTL, W) and `8572189` (BS, with the build); runs on `c9f23d4` (CTL, W) and `8572189` (BS); merged at `2cebc65`; amendments `a4c61cb` / `81f56a7` / `852e649` (CTL, W) and `1b3596c` (BS); branch `feature/per-model-actor-credit`, PR #383 |
| checkpoint | CTL `checkpoints/ppo-transformer-curriculum_a5_points-m18-opp0-obj5-b60x44-ph1-2026-09-19-22-03-41-s*a5pts-ctl{,-x}/last.ckpt`; W `checkpoints/per_model/per-model-a5_points-2026-09-19-22-03-42-s*a5pw/last.pt` (resumed in place; `pm-00122880.pt` for the cap read); BS `…-2026-09-19-22-17-{29,30}-s*a5bs/last.pt`; every per-model checkpoint with a recorded greedy episode and one MP4 per 10,240 rounds on the run |
| coherency | greedy at 700000+: CTL 0.57–0.66; W 0.24–0.29 (245,760); BS 0.08–0.47; the bar 0.832 |
| Wandb | `curriculum-a5`: CTL `pic9zjka` / `ccheoveo` / `oct4luxd` and `ryk2bfuc` / `0wlaqa3l` / `xwsqlrtn`; W `8clgcy59` / `klv24tu2` / `8h4eq4lj` and `xw154bst` / `8dcfq1xa` / `kwdxlh12`; BS `inhpvik7` / `fhbs37ln` / `feg4n4uf` |
| pre-registration | `reports/2026-09-19-curriculum-A5-points-control-and-warm-start-preregistration.md` (+3 amendments) · `reports/2026-09-19-curriculum-A5-points-backward-start-preregistration.md` (+1) — every table is in them |

## The read

### The whole-army trainer is two to five times ahead at equal rounds

At 60 epochs (≈123k rounds, the per-model cap) the control reads 0.98 /
0.62 / 0.87 with 4.6–5.0 objectives held and 94–95% of its bodies on
objectives; the per-model trainer from scratch at the same rounds reads
0.03–0.33 with a third of its bodies on objectives, under five settings.
At 120 epochs the control reads 0.91 / 0.81 / 0.95 — one seed drifted
down over the extension, as the whole-army A2 and A5 controls did — and
the per-model arm's best at the same rounds is W's 0.60 on one seed. It
is a FAIL on the rung's letter (two seeds under 0.95) and a finding on
#283's question: on the five-objective spread the step-of-one-model
trainer is the one with the wall.

### The warm start carries four objectives and reward adds the fifth slowly

W is where the original ends by 20,480 rounds (0.03 / 0.12 / 0.18, held
2.9–3.4, six or seven bodies on objectives from turn 3 with no walk-off),
climbs to 0.46 / 0.26 / 0.35 at the cap and, extended, to 0.34 / 0.17 /
0.60 — ahead of the original's final row on every seed, past two SE, and
s3 at 0.60 with held 4.29 the highest per-model read on the half-step.
Two seeds fell back over the extension while s3 kept climbing (in-run
40 → 58% by quarter, a rolling 63%). The census at the end is the first
on this half-step with no walk-off on any seed and misses spread across
all five objectives; greedy and sampled play agree within ±6 vp. T1's
shape (ahead at 20k, flat after) did not repeat; the difference between
T1 and W is one objective (six against five), and the reading is that
the A4x policy's three-objective spread carries to four here and reward
finds the fifth on one seed of three.

### The backward start: given four, it keeps two or three

The level never stepped down on any seed: the level's own success
plateaued at 0.25–0.35 by quarter against the 0.8 the schedule needed.
Under the final checkpoints on level-four starts (n=30): success 0.53 /
0.33 / 0.50; the placed squads still on their objectives at turn 3 2.93 /
3.38 / 3.93 and at the end **2.70 / 1.97 / 3.13** of 4; the free squads
reaching the empty objective 0.83 / 1.00 / 0.63; the bar on the same
starts 1.00, keeping 3.9. Both halves of the last step are half-learned
and the larger loss is the first: **given an objective, the policy walks
off it.** Its critic was healthy (explained variance 0.55–0.68), so this
is not the reward-shape arms' failure. From deployment the arm reads
0.00 / 0.03 / 0.03, behind the original at every read, and its census
has the half-step's walk-off with a new variant on s1 — 9.6 of 18 bodies
on objectives by turn 5, the highest arrival on the half-step, stacked on
the near column. The schedule (share 0.75, advance 0.8 over eight
rollouts) was never exercised; a lower bar would have walked the level
onto a policy that holds a third of its starts, and whether that helps
is untested.

## What this says

- **On the five-objective spread the wall is the per-model trainer's.**
  Same config, same rounds: the whole-army trainer 0.81–0.95 with 4.8–4.95
  held; the per-model trainer from scratch 0.00–0.33 under five settings,
  and 0.60 at best from a warm start at twice the cap. #283's premise —
  that a step of one model beats a step of the whole army — has its first
  clear negative row on the A rungs.
- **A warm start from the objective rung below is the one per-model lever
  that moves a spread half-step**, as S3 said on A3 and T1 did not on A5.
  Read a transfer arm at matched rounds and at the extended budget; the
  per-seed spread (0.17–0.60) is wider than any from-scratch arm's and one
  seed of three carries the result.
- **The per-model policy under this reward does not keep an objective it
  is standing on.** Isolated by the backward start: four squads placed,
  one to two leave. The walk-off in every census on this half-step is
  this event, not a failure to arrive. Nothing per-model pays a body for
  standing (the travel term is zero inside, coverage is a broadcast mean,
  the bonus terminal); the A3 speed screen and A5e measured a hold term
  as null, under the mean and under actor credit, on other rungs, so the
  lever is not obviously a hold term either — but the next per-model arm
  on this half-step should be aimed at staying, not at arriving.
- **A start curriculum needs its advance bar calibrated on the policy's
  own level success.** 0.8 was never approached; the schedule never
  stepped and so measured only level four. Pre-register the bar against
  the bar's own success on the same starts (1.00 here) and the policy's
  first read at the top level.
- **Process.** Three seeds launched in one loop can land in run
  directories a second apart; a chain that derives every seed's directory
  from the first seed's waits forever (caught after an hour). Resolve each
  seed's directory separately. And gate a final read on the trainer
  exiting, never on `last.pt` existing.

## Where this lands

1. This report.
2. `reports/README.md` index row.
3. `CLAUDE.md`: three ladder rows (the control, the warm start, the
   backward start) and a rule bullet.
4. Live docs: the backward start's flags are documented in
   `wargame_rl/wargame/model/CLAUDE.md` and its placement in
   `wargame_rl/wargame/envs/CLAUDE.md` (`2cebc65`); no config added.

## Addendum — 2026-09-20 13:00: what exactly makes it hard, three checks and one more arm

Asked after the control's read: what in the per-model trainer's design
makes five objectives hard? Three candidates were checked
(`reports/2026-09-20-curriculum-A5-points-rounds-per-update-preregistration.md`
and its amendment carry the tables):

1. **Perception — ruled out by inspection.** The objective token carries
   our count on it (`row[2]`) and the enemy's; every body reads its
   offset to every objective as a relation. An empty objective is
   observable to the body that should walk to it.
2. **Outcomes per update — ruled out by an arm.** The whole-army trainer
   updates every 2,048 rounds (~200 episodes on this one-phase config),
   the per-model trainer every 128 (~13). At 2,048 rounds per update the
   per-model trainer from scratch reads **0.000 / 0.010 / 0.000** at
   122,880, behind the original at every read, with the displacement head
   diffuse and the walk-off at its largest (11.3 of 18 bodies on
   objectives after turn 3 and 0.1 at the end on one seed).
3. **The reward's silence on staying — measured, and shared by every
   arm.** A probe over the displacement decisions of a body standing
   inside an objective, on six trained per-model policies (the warm
   start's, the original's, the regime arm's): it stands still on
   **0.00–0.02** of them and walks out of the objective on **0.60–0.81**;
   the scripted bar stands still on 0.47. The step reward for leaving is
   −0.002 to −0.005 and for staying 0.000, against +0.003 to +0.014 for a
   step toward an objective from outside. Standing on an objective is the
   one state in which every action is paid alike, and the policy never
   learns the stand-still action at all. The whole-army trainer trains on
   the same terms but pays each model its own travel term in full every
   step and broadcasts coverage in full; the per-model retimer divides
   the action term by the army and pays coverage once at the close, so
   the per-step magnitudes a body sees are an order of magnitude smaller
   and, on an objective, indistinguishable.

The wall on the five-objective half-step is therefore neither the
horizon, the terminal payment, the credit's attribution, the update
regime nor perception. It is that nothing pays a body for keeping an
objective on its own step, and the backward start showed what follows:
handed four objectives, the policy keeps two or three. The next arm pays
a body for standing on an objective on its own decision step, through
the `actor` credit path, with a mechanism that is not the mean-credited
hold term S1 and A5e read as null.
