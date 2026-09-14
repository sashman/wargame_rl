# Pre-registration: curriculum rung A3 — four squads, four objectives, the spread rung

Written 2026-09-15, **before any number from the arm exists**, on branch
`feature/curriculum-a3` (one PR per rung, stacked on A2's PR #348; the
chain does not merge to `main` while exploratory, so the commit timestamp
of this file is what orders it before the numbers). Issue #349, parent
question #340, rung **A3** — axis: **points**. A2 (#347) is the rung
below and is still training when this is written; A3's per-model arm does
not launch until A2 has been read, and not at all if A2 fails.

## The one change

**Four objectives** (radius 4, a column at x = 35, y = 8 / 17 / 27 / 36)
where A2 had one, with the deployment band lengthened to the board's
height (`[4, 4, 10, 40]`) so each squad deploys nearest a distinct point.
Four units of three, no opponent, movement only, deployed as squads —
twelve bodies as A2. The disc goes back to A1's radius 4 (three bodies
per point), and success is the criterion this rung ships:
**`all_objectives_occupied`** — every objective has at least one alive
model inside it at the end — because the two criteria on file count
models and cannot tell twelve on one point from three on each of four.
Same reward (`closest_objective_v2` + `objective_coverage`,
`terminate_on_success`), 8 rounds, 60×44, Move 6, coherency a readout
(`configs/experiments/curriculum/a3.yaml`). Two arms of the same scenario,
three seeds each: per-model (under test) and whole-army (pipeline
control).

The question the rung asks: go to the closest point, one squad per point,
nobody crosses. A crossing policy still succeeds — it is slower, which the
turn readout shows — and a stacking policy fails the criterion outright.

## Comparator, measured first

`squad_march_take`, scored through both facades on seeds 700000+ at n=100
(`just measure-bridge configs/experiments/curriculum/a3.yaml 100`), bridge
identical on every shared field: success **1.000**, `held` **4.00** of 4,
turns **5.28** (sd 0.81, range 4–7), `on_obj` 0.892, coherent **0.927**,
vp 31.6 ± 1.2. In-run band (n=30, 500000+): success 1.000, turns 5.13,
held 4.00, coherent 0.970.

## Criteria

Read at the END of the budget from the last checkpoint (`last.pt` /
`last.ckpt`), greedy, no decode, n=100 on seeds 700000+ — the same seeds
the bar was measured on, so success and turns pair per episode.

- **PASS:** success (every objective occupied at the end) ≥ 0.95 on
  **every seed** of the per-model arm, within budget.
- **FAIL:** any per-model seed misses it at budget while the whole-army
  control passes on ≥ 2 of 3 seeds — the fault is in allocation across
  units under the per-model step: the unit pointer choosing which squad
  moves, and each squad's target.
- **NULL (scenario):** the whole-army control also fails on ≥ 2 of 3 seeds
  — redesign, do not read the per-model result.
- **Pass with a defect:** the bound holds but the health panel is red over
  the last quarter (the ratio line at 1.6–1.9 is this regime's normal, per
  A1; a fault is a value that moves with a failure).

**Readouts, not criteria:** turns-to-success paired against the script and
against the whole-army control's own mean (a crossing policy is slower);
`held` (how many of four, at n=100); the **spread statistic** — the share
of episodes in which every squad ends on the point nearest to it at
deployment — computed from a recording of the final checkpoints; `on_obj`;
coherency beside the bar's 0.927.

Power: success at n=100 has binomial SE 0.022 at p=0.95; a true 0.90
fails at ~2.3 SE. The bar sits at 1.000.

## Budget, regime, seeds

The whole-army control runs **first** and calibrates the budget (#340
contract 3, as amended by A1x): the per-model budget is **6× the control's
slowest rounds-to-pass** (the first in-run evaluation at which success
reaches 0.95 and stays there), with a floor of 20,480 rounds and a cap of
122,880 (60 epochs of 2048 steps). If the control has not passed by the
cap, the rung is NULL (scenario) and the per-model arm is not launched.
The control may run before A2 is read; the per-model arm does not.

| | per-model arm | whole-army control |
|---|---|---|
| budget | 6× the control's slowest rounds-to-pass, in [20,480, 122,880] | up to 60 epochs of 2048 steps |
| regime | `--rollout-rounds 32 --num-rollout-envs 4` = **128 rounds per update** | 2048 steps per update |
| seeds | 1, 2, 3; rollout layouts at seed×100+ | 1, 2, 3 |
| in-run eval | every 512 rounds, n=30, seeds 500000+, greedy, passive pair and `eval/mean_turns` logged | every epoch, n=30, seeds 500000+ |
| final score | n=100, seeds 700000+, `just measure-rung` | same |
| decode | none (K=1) | K=1, undecoded |
| other flags | defaults (`gamma` 0.9, `ent_coef` 0.03, `lr` 3e-4) | defaults |
| logging | Wandb group `curriculum-a3`, one run per seed | same group |
| start | from scratch (the transfer question is T1's) | from scratch |

Launch: `just train-curriculum-control 60 3 curriculum-a3 a3-ctl "" configs/experiments/curriculum/a3.yaml`,
then `just train-per-model-arm <6× rounds> 3 curriculum-a3 a3 "--num-rollout-envs 4 --rollout-rounds 32 --eval-every-rounds 512 --checkpoint-every-rounds 512 --n-eval-episodes 30" configs/experiments/curriculum/a3.yaml`.

## Primary readouts

Success rate (decides); turns paired against the script and against the
control; `held`, `on_obj`, the spread statistic, coherency; the passive
pair and `train/rounds_per_update` on every per-model row; the health
panel over the last quarter; rounds-to-pass on both arms.

## What I expect (a guess, written so it can be wrong)

Both facades pass. The per-model arm's turns land near the script's 5.28
again and the control's a round slower. The spread statistic is high on
both — the deployment band makes the nearest point unambiguous and the
progress reward pays each squad to close on its own nearest point — so
this rung will not separate the facades on allocation; A4's spare squad
is where that question lives.

## Amendment 1 — the control's read at the cap, and what follows (2026-09-15 01:10, before any per-model number exists)

The whole-army control ran to its 60-epoch cap (Wandb group `curriculum-a3`,
runs `78l7l9u6` s1 · `7rbs29pq` s2 · `z2arg0vf` s3). In-run success (n=30,
seeds 500000+) reaches 0.95 and stays there on **s2 only, at epoch 59**;
s1 sits at 80–93 from epoch 28 and s3 at 83–97 from epoch 28, both still
rising at the cap. Scored at n=100 on seeds 700000+ from `last.ckpt`
(epoch 60): success **0.850 / 0.980 / 0.930**, `held` 3.83 / 3.98 / 3.92,
turns 6.72 / 6.47 / 6.52 against the script's 5.28, coherent 0.770 /
0.733 / 0.668 against the bar's 0.927.

**On the letter, that is NULL (scenario): the control has not passed by
the cap on 2 of 3 seeds.** The clause was written to catch a
misconfigured rung — one the script cannot solve either — and this rung
is not that: the script reads 1.000, and the control is at 85–98% and
climbing when its budget runs out. The clause conflates "cannot learn it"
with "slower than the cap", which the A1x finding (the per-model arm is
2–4× slower than the control) should have warned would bite the control
too on a harder rung. Recorded as a defect of the clause, not of the
scenario.

What follows, written before any of it runs:

1. **The control is resumed to 120 epochs** (`--resume-ckpt-path`, same
   seeds, the same three checkpoints), the cap doubled once for the
   control only, to read its rounds-to-pass. If it has not passed 2 of 3
   by 120, the rung is NULL (scenario) for real.
2. **The per-model budget is the cap, 122,880 rounds**, whatever the
   control's rounds-to-pass turns out to be (6× anything ≥ 20,480 exceeds
   it). The per-model arm launches once A2 (#347) has been read, as this
   file already says, and is read at the end of its budget exactly as
   written above.
3. **Attribution at the read**: FAIL needs the control to pass on ≥ 2 of 3
   seeds at its extended budget; otherwise the rung is NULL and the
   per-model result is reported but not read as a verdict.

Nothing about the per-model arm was known when this was written.
