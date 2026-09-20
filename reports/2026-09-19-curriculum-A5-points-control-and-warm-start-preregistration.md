# Pre-registration: the whole-army control and the A4x warm start on A5-points — whose wall is it, and does a start help?

Written 2026-09-19 20:05, **before any training round on either arm**, on
branch `feature/per-model-actor-credit` (PR #383). Parent question #340.
Set as a goal by Sash 2026-09-19 20:00: "run the A5-points whole-army
control and the A4x warm-start beside it; then build and run the backward
start curriculum on A5-points".

## The question

A5-points (six squads of three over five objectives; the bar
`squad_march_take` 1.000 in 6.71 turns) has now been read from scratch
under four settings of the per-model trainer — the original recipe
(0.030 / 0.060 / 0.330), the discount at 0.99 (0.070 / 0.180 / 0.100),
the flat terminal bonus (0.000 / 0.030 / 0.010) and the per-objective
terminal bonus (0.020 / 0.030 / 0.000) — and the census is the same on
every one: a third of the bodies on objectives, one objective empty, the
clock run out. Two things the ladder's own contract asks for have not
been run on this half-step:

1. **The whole-army control.** Every A rung ran `train.py` on the same
   config as the pipeline control. On A5 (six objectives) the control
   read 0.80 / 0.94 / 0.98 at 120 epochs where the per-model arm read
   0.26 / 0.18 / 0.16. Five objectives has no control. If the whole-army
   trainer passes here, the wall is the per-model trainer's; if it fails,
   five objectives is hard for both trainers at this budget.
2. **The warm start from the objective rung below.** S3 (A3 from A2b)
   read AHEAD 3/3 and is the one lever that ever moved a spread rung;
   T1 (A5 from A4x) read ahead on every seed and passed on none. A4x —
   the same army's four-squad ancestor at three objectives, passed 3/3 at
   245,760 rounds — has not been tried as the start for five objectives.

## The arms

| arm | trainer | config | start | budget | tag |
|---|---|---|---|---|---|
| **CTL** | whole-army `train.py` | `a5_points.yaml` | scratch | 60 epochs (`--n-eval-episodes 30`, `--record-during-training`; defaults otherwise, `ent_coef` 0.03 as on every control), extended once to 120 if any seed is still rising at 60 | `a5pts-ctl` |
| **W** | per-model `train_per_model.py` | `a5_points.yaml` | `--warm-start-from` A4x seed-for-seed: `per-model-a4-2026-09-15-07-08-53-s{1,2,3}a4/pm-00245760.pt` (weights only; cold critic and optimiser as T1 ran) | 122,880 rounds, the original recipe (128 per update, `--ent-coef 0.003`, `gamma` 0.9, `mean` credit, eval and checkpoint every 512, recording and video on) | `a5pw` |

Three seeds each, on Wandb under `curriculum-a5`. The bridge on this
config was identical on 2026-09-18 (`squad_march_take` 1.000 on both
facades). The per-model comparator is the original A5-points from
scratch (`uagmul66` / `wedjjozy` / `ici5py46`) read at 40,960 / 81,920 /
122,880; the control is read against the script on its own facade.

## Criteria

**CTL** (the rung's letter, as on every A rung):
- **PASS:** success ≥ 0.95 on 3/3 at n=100 on 700000+ at 60 epochs (or at
  120 after the one extension), turns a readout.
- **FAIL:** any seed under 0.95 at the last read.
- **Readouts:** rounds-to-pass in-run (rolling 95% at n=30), turns, held,
  the by-turn census, which objective is left empty.

**W** (the transfer readouts T1 taught, beside the rounds bound):
- **PASS:** success ≥ 0.95 on 3/3 at 122,880, no in-run dip below 0.80
  after the first rolling pass.
- **MOVES:** ahead of the original 0.030 / 0.060 / 0.330 seed for seed by
  more than two binomial SE on 3/3 at 122,880, **and** ahead at 40,960
  (0.040 / 0.140 / 0.090) and 81,920 (0.220 / 0.220 / 0.110) on 3/3 — a
  start that is ahead at every read.
- **FAIL:** neither.
- **Readouts:** success at 20,480 (T1's read: where scratch ends, the warm
  start already was), the in-run peak, held, the census, the panel,
  sampled beside greedy.

**What the pair says.** CTL PASS and W FAIL: the wall is the per-model
trainer at five objectives, and #283's question has its first negative
row on the A rungs. CTL FAIL and W FAIL: five objectives is beyond both
trainers at this budget; the backward start curriculum (goal item 3) is
built on that. W MOVES or PASS: the three-objective policy carries to
five, and the C rungs' warm-start rule extends to the half-step. A W
seed that is ahead at 20k and behind at the end is T1's shape: transfer
onto a failure, reported as such.

## What I expect (a guess, written so it can be wrong)

CTL passes 2 of 3 at 60 epochs and 3 of 3 at 120, a round behind the
script, leaving the far column's second objective empty when it fails —
the A5 control's shape one objective smaller. W reads 0.3–0.5 at 20,480
(ahead of everything from scratch), plateaus at 0.4–0.6 with held
3.5–4.0, ahead of the original at every read on 3/3 (MOVES) and PASS on
none — T1's shape, the fifth objective still the residual. If CTL fails
too, the half-step is harder than the A5 control's numbers implied and
the backward start is the next thing on both trainers.

## Amendment 1 — written 2026-09-19 22:40, the control read at 60 and 120 epochs

(The header's "20:05" is wrong; the pre-registration was committed at
22:02, `c9f23d4`, and the arms launched at 22:03.)

**CTL: FAIL on the letter by two seeds at 120 epochs — and it is the best
read this half-step has had from any trainer.** Runs `pic9zjka` /
`ccheoveo` / `oct4luxd` (60 epochs, 22:03–22:14), extended once to 120
per the pre-registration as `ryk2bfuc` / `0wlaqa3l` / `xwsqlrtn`
(`--resume-ckpt-path`, 22:17–22:30). Greedy at n=100 on 700000+:

| row | success | turns | vs bar, paired | held of 5 | on objectives | coherent |
|---|---|---|---|---|---|---|
| `squad_march_take` | 1.000 | 6.71 | — | 5.00 | 0.889 | 0.832 |
| s1 at 60 | **0.980** | 7.22 | +0.51 ± 0.09 | 4.98 | 0.936 | 0.609 |
| s2 at 60 | 0.620 | 8.80 | +2.09 ± 0.11 | 4.61 | 0.948 | 0.664 |
| s3 at 60 | 0.870 | 8.27 | +1.56 ± 0.08 | 4.86 | 0.940 | 0.655 |
| s1 at 120 | 0.910 | 7.33 | +0.62 ± 0.10 | 4.91 | 0.965 | 0.645 |
| s2 at 120 | 0.810 | 7.73 | +1.02 ± 0.14 | 4.80 | 0.930 | 0.575 |
| s3 at 120 | **0.950** | 7.35 | +0.64 ± 0.10 | 4.95 | 0.955 | 0.570 |

- **PASS** needs ≥ 0.95 on 3/3: one seed at 120 (s3), one at 60 (s1).
  s1 fell 0.98 → 0.91 over the extension (a drift, as A2 and the
  whole-army A5 control showed); s2 rose 0.62 → 0.81, s3 0.87 → 0.95.
  **FAIL on the letter.**
- Against the per-model trainer on the same config and budget the
  control holds **4.8–4.95 of five objectives with 93–97% of its bodies on
  them**, half a round to a round behind the bar, where five per-model
  settings from scratch read 0.00–0.33 with a third of the bodies on
  objectives. The whole-army trainer solves the five-objective spread
  from scratch to within a seed of the letter; the per-model trainer
  does not approach it. **The wall on this half-step is the per-model
  trainer's**, on the reading the pre-registration wrote for this
  quadrant — pending W (a warm start that carries) and BS (a start
  curriculum that walks back), both in flight.
- Readouts still to take: the control's in-run curve (its Wandb file has
  no history rows readable by the local scanner; read after the runs
  exit through `just run-summary`), rounds-to-pass, and which objective
  the failing episodes leave empty (a whole-army census).

## Amendment 2 — written 2026-09-20 02:20, the warm start read at 20,480 / 40,960 / 81,920 / 122,880

**W: FAIL on the letter — and the best per-model result reward has
produced on this half-step, still climbing at the cap.** Runs
`8clgcy59` / `klv24tu2` / `8h4eq4lj`, from A4x seed-for-seed, exited
01:58–02:10. Greedy at n=100 on 700000+:

| rounds | W success | W held of 5 | the original at the same rounds |
|---|---|---|---|
| 20,480 | 0.030 / 0.120 / 0.180 | 2.94 / 3.37 / 3.31 | (0.03 / 0.06 / 0.33 is the original's END) |
| 40,960 | 0.150 / 0.110 / 0.250 | 3.23 / 3.40 / 3.75 | 0.040 / 0.140 / 0.090 |
| 81,920 | 0.100 / 0.250 / **0.520** | 2.98 / 3.53 / 4.22 | 0.220 / 0.220 / 0.110 |
| **122,880** | **0.460 / 0.260 / 0.350** | 3.96 / 3.53 / 3.62 | 0.030 / 0.060 / 0.330 |

- **PASS:** no seed near 0.95.
- **MOVES:** ahead of the original at the end by two SE on s1 (+0.43)
  and s2 (+0.20), not s3 (+0.02 against 0.330); at 81,920 s1 was behind
  (0.10 against 0.22). Two of three at the end, not three at every read.
  **FAIL on the letter.**
- Against every per-model read on this half-step it is the top row:
  0.46 with 3.96 held on s1, 0.52 with 4.22 held on s3 at 81,920, where
  five settings from scratch never passed 0.33 or 3.8 held. The in-run
  curve on s1 is still rising at the cap — success by quarter 4 / 13 /
  18 / **42%**, a rolling 50% with 4.45 held over the last eight
  evaluations — and s3 sits at 34–38%. The displacement head is
  concentrated (1.2–1.5 nats against 2.0–2.6 from scratch), explained
  variance 0.48–0.78, clip fraction 0.25–0.32.
- **T1's shape did not repeat.** T1 (A5 from A4x, six objectives) was
  ahead at 20k and flat after. Here the start carried the walk (six or
  seven bodies on objectives from turn 3, no walk-off at 20k) and reward
  then kept climbing on two seeds through the whole budget.
- **The ladder's symmetric cap rule applies.** The control on this
  half-step needed its once-only extension to 120 epochs (about 245k
  rounds) and still read 0.91 / 0.81 / 0.95, so the per-model arm under
  test gets the same rounds before the rung is called (A3x, A4x). The
  three runs are resumed in place to **245,760** for one read; this
  amendment records the 122,880 read as the pre-registered one, and the
  next records the extension.
