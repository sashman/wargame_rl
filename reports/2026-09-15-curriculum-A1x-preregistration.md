# Pre-registration: curriculum rung A1, second arm (A1x) — the A1 runs resumed to twice the budget

Written 2026-09-15 00:32, **before any number from the arm exists**, on
branch `feature/curriculum-a1` (PR #344 carries the rung's arms). Issue
#345, parent question #340, rung **A1** — a follow-up to #343, which read
**FAIL as pre-registered** with 2 of 3 per-model seeds passing
([report](2026-09-15-curriculum-a1-fails-on-one-seed.md)).

## The one change

**Budget.** The three A1 per-model runs (Wandb `curriculum-a1`, `9irv0y7y`
s1 · `cm17c1wq` s2 · `ahcy2jl0` s3) are resumed **in place** with
`--resume-from <run_dir> --rounds 122880`: same seed, same knobs (the
driver refuses a changed one), optimizer and sampling generator restored,
`metrics.jsonl` appended — a resumed run is one run. 122,880 rounds is 2×
the A1 budget, 6× the control's slowest rounds-to-pass (20,480), and the
ladder's cap. Nothing else changes; the whole-army control is not re-run.

## Why this arm and not a diagnosis first

A1's failing seed did not collapse: it climbed from 0.50 to a plateau of
0.80–0.93 over the last quarter and was still moving at budget, and the two
passing seeds needed 32k and 53k rounds where the control needed 14k–20k.
A0's arm ran 2–4× the control's rounds too. The cheapest test of "the
budget rule, calibrated on a trainer that learns this rung in a fifth of
the rounds, is the defect" is to give the same runs the rounds. It costs
twelve minutes. A recording of the failing episodes is the diagnostic if
this does not resolve it.

## Comparator

`squad_march_take` on `configs/experiments/curriculum/a1.yaml`, seeds
700000+ at n=100: success 0.960, turns 4.96, coherent 0.993. And A1's own
read at 61,440 rounds on the same seeds: success **1.000 / 0.790 / 1.000**,
turns 5.01 / 6.10 / 4.90.

## Criteria

Read at 122,880 rounds from `last.pt`, greedy, no decode, n=100 on seeds
700000+ (`just measure-rung`).

- **PASS:** success ≥ 0.95 on **all three** seeds. A1's verdict stands as
  written (it was a pre-registered read at a pre-registered budget); the
  budget rule for the rungs above becomes a **floor of 6× the control's
  slowest rounds-to-pass** (cap unchanged), and the ladder continues to A2.
- **FAIL:** seed 2 still below 0.95 at 122,880 — the per-model handling of
  a unit (the unit open, the sequenced members, the membership tokens) is
  the suspect; the ladder stops at A1 and the next step is
  `just record-per-model-events` on the failing seeds.
- **Regression:** s1 or s3 falls below 0.95 on the resumed read — reported
  as such whatever s2 does; a policy that passes and then unlearns is its
  own finding.
- **Readouts:** turns paired vs the script and vs A1's read; coherency
  greedy and sampled; the health panel over the last quarter of the
  resumed half; s2's rounds-to-pass if it passes.

Power: binomial SE 0.022 at p=0.95, n=100; s2's 0.790 is 7 SE below the
bound, so a pass is a move, not noise.

## Budget, regime, seeds

| | per-model arm |
|---|---|
| budget | 122,880 rounds total, 61,440 more, 128 rounds per update |
| seeds | 1, 2, 3, continued |
| in-run eval | every 512 rounds, n=30, seeds 500000+ (as A1) |
| final score | n=100, seeds 700000+ |
| logging | the runs' own Wandb ids, continued |

Launch, per seed: `uv run train_per_model.py --resume-from checkpoints/per_model/<run> --rounds 122880`, detached with `setsid`, stdout under `checkpoints/per_model/logs/curriculum-a1/a1x-s<seed>.log`.

## What I expect (a guess, written so it can be wrong)

Seed 2 passes somewhere between 70k and 100k rounds and the budget rule
is what moves. Seeds 1 and 3 hold. The sampled coherency stays far below
the greedy one.
