# Pre-registration: curriculum rung A5 — eight squads, six objectives

Written 2026-09-15, **before any number from the arm exists**, on branch
`feature/curriculum-a5` (one PR per rung, stacked on A4's PR #352; the
chain does not merge to `main` while exploratory, so the commit timestamp
of this file is what orders it before the numbers). Issue #353, parent
question #340, rung **A5** — axis: **bodies + points, same shape**. A4
(#351) is the rung below; its per-model arm has not run when this is
written. A5's whole-army control may run now; A5's per-model arm does not
launch until A4 has been read, and not at all if A4 fails. T1 warm-starts
this rung from A4's per-model checkpoint, paired by seed against these
from-scratch runs.

## The one change

**Eight units of three and six objectives** where A4 had four units and
three — the spare-squads shape the full game poses
(`configs/experiments/24v24_maps_spare_squads.yaml`) without the guns, the
terrain or the opponent. Two columns of points (x = 30 and x = 45, y = 8 /
22 / 36), the deployment band the height of the board (`[2, 2, 10, 42]`),
radius 4, the same reward and the same `all_objectives_occupied`
criterion, coherency a readout. **Ten rounds**, not eight: the far column
is 35–41" from the band at Move 6, the script's bar read 0.98 at eight
and 1.00 at ten, and a rung's clock is part of its shape
(`configs/experiments/curriculum/a5.yaml`). Two arms of the same scenario,
three seeds each: per-model (under test) and whole-army (pipeline
control).

## Comparator, measured first

`squad_march_take`, scored through both facades on seeds 700000+ at n=100
(`just measure-bridge configs/experiments/curriculum/a5.yaml 100`), bridge
identical on every shared field: success **1.000**, `held` **6.00** of 6,
turns **6.77** (sd 0.68, range 6–10), `on_obj` 0.856, coherent **0.822**,
vp 66.3 ± 1.0. In-run band (n=30, 500000+): success 1.000, turns 6.77,
held 6.00, coherent 0.835. At eight rounds the same script read 0.980 /
held 5.98 / 6.74 turns — the clock was set before any training number
existed.

## Criteria

Read at the END of the budget from the last checkpoint (`last.pt` /
`last.ckpt`), greedy, no decode, n=100 on seeds 700000+ — the same seeds
the bar was measured on, so success and turns pair per episode.

- **PASS:** success (every objective occupied at the end) ≥ 0.95 on
  **every seed** of the per-model arm, within budget.
- **FAIL:** any per-model seed misses it at budget while the whole-army
  control passes on ≥ 2 of 3 seeds at its budget — the set network at
  twenty-four bodies and eight units.
- **NULL (scenario):** the whole-army control fails on ≥ 2 of 3 seeds at
  its budget and is not still rising at the cap; a rising control is
  resumed once to 120 epochs (A3's amendment 1) and read there.
- **Pass with a defect:** the bound holds but the health panel is red over
  the last quarter (the ratio line at 1.6–1.9 is this regime's normal).

**Readouts, not criteria:** turns-to-success paired against the script and
against the control; `held`; `on_obj`; coherency beside the bar's 0.822;
per-model decisions per round (the throughput cost of the shape).

Power: success at n=100 has binomial SE 0.022 at p=0.95; a true 0.90
fails at ~2.3 SE. The bar sits at 1.000.

## Budget, regime, seeds

The whole-army control runs **first** and calibrates the budget: the
per-model budget is **6× the control's slowest rounds-to-pass**, floor
20,480, cap 122,880. If the control needs its 120-epoch extension, the
per-model budget is the cap.

| | per-model arm | whole-army control |
|---|---|---|
| budget | 6× the control's slowest rounds-to-pass, in [20,480, 122,880] | 60 epochs of 2048 steps, extended once to 120 if still rising |
| regime | `--rollout-rounds 32 --num-rollout-envs 4` = **128 rounds per update** | 2048 steps per update |
| seeds | 1, 2, 3; rollout layouts at seed×100+ | 1, 2, 3 |
| in-run eval | every 512 rounds, n=30, seeds 500000+, greedy, passive pair and `eval/mean_turns` logged | every epoch, n=30, seeds 500000+ |
| final score | n=100, seeds 700000+, `just measure-rung` | same |
| decode | none (K=1) | K=1, undecoded |
| other flags | defaults (`gamma` 0.9, `ent_coef` 0.03, `lr` 3e-4) | defaults |
| logging | Wandb group `curriculum-a5`, one run per seed | same group |
| start | from scratch (T1 is the warm-started pair) | from scratch |

Launch: `just train-curriculum-control 60 3 curriculum-a5 a5-ctl "" configs/experiments/curriculum/a5.yaml`,
then `just train-per-model-arm <6× rounds> 3 curriculum-a5 a5 "--num-rollout-envs 4 --rollout-rounds 32 --eval-every-rounds 512 --checkpoint-every-rounds 512 --n-eval-episodes 30" configs/experiments/curriculum/a5.yaml`.

## Primary readouts

Success rate (decides); turns paired against the script and against the
control; `held`, `on_obj`, coherency; the passive pair and
`train/rounds_per_update` on every per-model row; the health panel over
the last quarter; rounds-to-pass on both arms.

## What I expect (a guess, written so it can be wrong)

Both facades pass, the control needing its extension as it did on A3
(more bodies, more points, the same 60-epoch cap). The per-model arm's
budget is the cap and it is the slowest run of the A rungs by wall-clock,
eight units of three being eight decisions a round per unit-open plus
twenty-four moves. Coherency on both learned arms ends well below the
bar's 0.822.
