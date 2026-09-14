# Pre-registration: curriculum rung A4 — four squads, three objectives, a spare squad

Written 2026-09-15, **before any number from the arm exists**, on branch
`feature/curriculum-a4` (one PR per rung, stacked on A3's PR #350; the
chain does not merge to `main` while exploratory, so the commit timestamp
of this file is what orders it before the numbers). Issue #351, parent
question #340, rung **A4** — axis: **points**, a spare squad. A3 (#349)
is the rung below; its per-model arm has not run when this is written.
A4's whole-army control may run now; A4's per-model arm does not launch
until A3 has been read, and not at all if A3 fails.

## The one change

**Three objectives** (radius 4, a column at x = 35, y = 12 / 22 / 32)
where A3 had four. The same four units of three, the same deployment band
(`[4, 4, 10, 40]`), the same reward and the same `all_objectives_occupied`
success criterion; 8 rounds, 60×44, Move 6, coherency a readout
(`configs/experiments/curriculum/a4.yaml`). One squad has no point of its
own — the record says that is the smallest scenario in which "which squad
goes where" is a question at all (`CLAUDE.md` § Allocation: five squads
over five points pose none; a surplus poses one). Two arms of the same
scenario, three seeds each: per-model (under test) and whole-army
(pipeline control).

## Comparator, measured first

`squad_march_take`, scored through both facades on seeds 700000+ at n=100
(`just measure-bridge configs/experiments/curriculum/a4.yaml 100`), bridge
identical on every shared field: success **1.000**, `held` **3.00** of 3,
turns **4.70** (sd 0.63, range 4–6), `on_obj` 0.724 (the spare squad
stands off the points), coherent **0.923**, vp 21.4 ± 0.6. In-run band
(n=30, 500000+): success 1.000, turns 4.50, held 3.00, coherent 0.918.

## Criteria

Read at the END of the budget from the last checkpoint (`last.pt` /
`last.ckpt`), greedy, no decode, n=100 on seeds 700000+ — the same seeds
the bar was measured on, so success and turns pair per episode.

- **PASS:** success (every objective occupied at the end) ≥ 0.95 on
  **every seed** of the per-model arm, within budget.
- **FAIL:** any per-model seed misses it at budget while the whole-army
  control passes on ≥ 2 of 3 seeds at its budget — allocation with a
  surplus under the per-model step.
- **NULL (scenario):** the whole-army control fails on ≥ 2 of 3 seeds at
  its budget **and its in-run curves are not still rising at the cap**;
  a control that is climbing when its cap arrives is resumed once to 120
  epochs (A3's amendment 1) and read there. If it fails there too, NULL.
- **Pass with a defect:** the bound holds but the health panel is red over
  the last quarter (the ratio line at 1.6–1.9 is this regime's normal).

**Readouts, not criteria:** turns-to-success paired against the script and
against the control; `held`; `on_obj` (where the spare squad ends — the
script's 0.724 means it stands off); coherency beside the bar's 0.923.

Power: success at n=100 has binomial SE 0.022 at p=0.95; a true 0.90
fails at ~2.3 SE. The bar sits at 1.000.

## Budget, regime, seeds

The whole-army control runs **first** and calibrates the budget: the
per-model budget is **6× the control's slowest rounds-to-pass** (the first
in-run evaluation at which success reaches 0.95 and stays there), floor
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
| logging | Wandb group `curriculum-a4`, one run per seed | same group |
| start | from scratch (the transfer question is T1's, which warm-starts A5 from this rung's checkpoint) | from scratch |

Launch: `just train-curriculum-control 60 3 curriculum-a4 a4-ctl "" configs/experiments/curriculum/a4.yaml`,
then `just train-per-model-arm <6× rounds> 3 curriculum-a4 a4 "--num-rollout-envs 4 --rollout-rounds 32 --eval-every-rounds 512 --checkpoint-every-rounds 512 --n-eval-episodes 30" configs/experiments/curriculum/a4.yaml`.

## Primary readouts

Success rate (decides); turns paired against the script and against the
control; `held`, `on_obj`, coherency; the passive pair and
`train/rounds_per_update` on every per-model row; the health panel over
the last quarter; rounds-to-pass on both arms.

## What I expect (a guess, written so it can be wrong)

Both facades pass, the control slower than on A3 (a spare squad crowding
a point it is not needed at costs the others their approach). The
per-model arm's spare squad ends on a point, not off it — nothing pays it
to stand aside — and `on_obj` reads above the script's 0.724.
