# Pre-registration: curriculum rung E3 — eight units, six objectives, no opponent

Written 2026-09-14, **before any number from the arm exists**, on branch
`feature/per-model-pipeline-health` (the curriculum runs on one PR while it
is exploratory; the commit timestamp of this file is what orders it before
the numbers). Issue #335, parent question #328.

## The one change

Training on the **E3** rung: 8 units of 3 (24 models), 6 objectives in two columns at x=30 and x=45, no opponent, movement only, 12 rounds — the spare-squads shape (`configs/experiments/curriculum/e3.yaml`). Two
arms of the same scenario, three seeds each: the **per-model** trainer
(`train_per_model.py`, the thing under test) and the **whole-army** trainer
(`train.py`, the pipeline control). Nothing else varies.

## Comparator, measured first

`squad_march_take`, scored through both facades on seeds 700000+ at n=100
(`just measure-bridge configs/experiments/curriculum/e3.yaml 100`), bridge identical on every shared
field: success **0.990**, turns-to-success **8.78**, `held` **6.00**, `on_obj` 0.907, vp 89.8 ± 1.3. In-run band: not measured separately.

## Criteria

Read at the END of the budget from the last checkpoint (`last.pt` /
`last.ckpt`), greedy, no decode, n=100 on seeds 700000+ — the same seeds
the bar was measured on, so success and turns pair per episode.

- **PASS:** success rate ≥ 0.95 **and `held` ≥ 5.80** and mean
  turns-to-success ≤ 9.78 (the script's 8.78 + 1), on **every seed** of the per-model
  arm, within 153,600 rounds.
- **FAIL:** any seed misses either bound at budget while the whole-army
  control passes on ≥ 2 of 3 seeds — the fault is scaling of the set network (the whole-army control is the pipeline reference here too).
- **NULL (scenario):** the whole-army control also fails on ≥ 2 of 3 seeds —
  the rung is misconfigured; redesign, do not read the per-model result.
- **Pass with a defect:** the bounds hold but the health panel
  (`docs/metrics.md` § The per-model health panel) is red over the last
  quarter of the run — recorded as such, never as a clean pass.

Power: success at n=100 has binomial SE 0.022 at p=0.95, so a true 0.90
fails the bound at ~2.3 SE; turns-to-success at sd ≈ 0.7 has SE 0.07 at
n=100, so the +1 allowance is fourteen SE wide — this rung is decided by the
success rate, and the turn bound is there to catch a policy that idles until
the last round.

## Budget, regime, seeds

| | per-model arm | whole-army control |
|---|---|---|
| budget | 153,600 rounds | 75 epochs = 153,600 rounds |
| regime | `--rollout-rounds 32 --num-rollout-envs 4` = **128 rounds per update**, 1,200 updates | 2048 steps per update (the shipped loop) |
| seeds | 1, 2, 3 (`--seed`); rollout layouts at seed×100+ | 1, 2, 3 |
| in-run eval | every 512 rounds, n=30, seeds 500000+, greedy, passive pair logged | every epoch, n=30, seeds 500000+ |
| final score | n=100, seeds 700000+, `evaluate_spec` | same |
| decode | none (K=1) | K=1, `verify_moves` default — undecoded, the same regime |
| other flags | defaults (`gamma` 0.9, `ent_coef` 0.03, `lr` 3e-4), `--no-wandb` | defaults, `--no-wandb` |
| compute | GPU for the per-model arm; the control on CPU | |

Launch: `just train-per-model-arm <rounds> 3 curriculum-e3 e3 "--num-rollout-envs 4 --rollout-rounds 32 --eval-every-rounds 512 --checkpoint-every-rounds 512 --n-eval-episodes 30 --no-wandb" configs/experiments/curriculum/e3.yaml`
and `just train-curriculum-control <epochs> 3 curriculum-e3 e3-ctl "" configs/experiments/curriculum/e3.yaml`.

## Primary readouts

Success rate and turns-to-success (paired against the script per episode),
`held`, `on_obj`; the passive pair and `train/rounds_per_update` on every
per-model row; the health panel over the last quarter. Reported with the
provenance table in the arm issue at close.

## What I expect (a guess, written so it can be wrong)

The whole-army trainer passes E3 within budget. The per-model trainer
either passes at a comparable round count — in which case the rung says the
plumbing works and nothing about the architecture — or its declaration head
collapses to `stationary` early (the do-nothing fingerprint) and the greedy
eval never leaves the floor while the sampled score climbs, which
`measure-per-model-eval-mode` will show. I do not know which.
