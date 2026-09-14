# Pre-registration: curriculum rung A0 — three lone models, one objective, no opponent

Written 2026-09-14, **before any number from the arm exists**, on branch
`feature/curriculum-a0` (one PR per rung, stacked on PR #339; the chain does
not merge to `main` while exploratory, so the commit timestamp of this file
is what orders it before the numbers). Issue #341, parent question #340,
rung **A0** — the plumbing rung, axis: none.

## The one change

Training on the **A0** rung: 3 units of 1 model each, 1 objective of radius
4 sitting 22–28" from the deployment band, no opponent, movement only, 8
rounds, 60×44, Move 6 (`configs/experiments/curriculum/a0.yaml`). Two arms
of the same scenario, three seeds each: the **per-model** trainer
(`train_per_model.py`, the thing under test) and the **whole-army** trainer
(`train.py`, the pipeline control). Nothing else varies. Three lone models
have no formation to keep, so nothing but the optimiser and the reward is
under test; A1 binds these same three bodies into one squad.

## Comparator, measured first

`squad_march_take`, scored through both facades on seeds 700000+ at n=100
(`just measure-bridge configs/experiments/curriculum/a0.yaml 100`), bridge
identical on every shared field: success **1.000**, turns-to-success
**4.93** (sd 0.26, range 4–5), `held` 1.00, `on_obj` 1.000, vp 8.8 ± 0.2.
In-run band (n=30, seeds 500000+): success 1.000, turns 4.93, vp 9.7 ± 0.2.

⚠ **The scenario was changed once before this file, and why is recorded.**
At objective radius 3 the script itself scored success 0.950 (`on_obj`
0.983) at both 8 and 10 rounds: in 5 of 100 episodes one model never
arrived, the friendly-gridlock freeze (`CLAUDE.md` § Freezing is friendly
gridlock — a deterministic policy re-issues a blocked order forever). The
bar failing its own criterion sends a rung back to design, not to training
(#340 contract 1), so the disc was widened to radius 4, at which the bar is
1.000. No training number existed at either radius.

## Criteria

Read at the END of the budget from the last checkpoint (`last.pt` /
`last.ckpt`), greedy, no decode, n=100 on seeds 700000+ — the same seeds
the bar was measured on, so success and turns pair per episode.

- **PASS:** success rate ≥ 0.95 and mean turns-to-success ≤ 6.0 (the
  script's slowest episode, 5, plus one round) on **every seed** of the
  per-model arm, within budget.
- **FAIL:** any per-model seed misses either bound at budget while the
  whole-army control passes on ≥ 2 of 3 seeds — the fault is on the
  per-model side: optimiser or reward plumbing.
- **NULL (scenario):** the whole-army control also fails on ≥ 2 of 3 seeds —
  the rung is misconfigured; redesign, do not read the per-model result.
- **Pass with a defect:** the bounds hold but the health panel
  (`docs/metrics.md` § The per-model health panel) is red over the last
  quarter of the run — recorded as such, never as a clean pass.

Power: success at n=100 has binomial SE 0.022 at p=0.95, so a true 0.90
fails the bound at ~2.3 SE; turns-to-success at sd 0.26 has SE 0.03 at
n=100, so the turn bound is decided by the mean, not by its noise. The rung
is decided on the success rate; the turn bound is there to catch a policy
that idles and collects the terminal bonus on the last round it can.

## Budget, regime, seeds

The whole-army control runs **first** and calibrates the budget (#340
contract 3): the per-model budget is **3× the control's rounds-to-pass**
(the first epoch at which its in-run success rate reaches 0.95 and stays
there), with a floor of 20,480 rounds and a cap of 122,880 rounds (60
epochs of 2048 steps). If the control has not passed by the cap, the rung
is NULL (scenario) and the per-model arm is not launched.

| | per-model arm | whole-army control |
|---|---|---|
| budget | 3× the control's rounds-to-pass, in [20,480, 122,880] | up to 60 epochs of 2048 steps = 122,880 rounds |
| regime | `--rollout-rounds 32 --num-rollout-envs 4` = **128 rounds per update** | 2048 steps per update (the shipped loop) |
| seeds | 1, 2, 3 (`--seed`); rollout layouts at seed×100+ | 1, 2, 3 |
| in-run eval | every 512 rounds, n=30, seeds 500000+, greedy, passive pair logged | every epoch, n=30, seeds 500000+ |
| final score | n=100, seeds 700000+, `just measure-rung` | same |
| decode | none (K=1) | K=1, `verify_moves` default — undecoded, the regime it trains in |
| other flags | defaults (`gamma` 0.9, `ent_coef` 0.03, `lr` 3e-4) | defaults |
| logging | Wandb group `curriculum-a0`, one run per seed (#340 contract 7) | same group |
| compute | CPU-sized (episodes are ~15 decisions); run on the 4090 box for throughput | |

Launch: `just train-curriculum-control 60 3 curriculum-a0 a0-ctl "" configs/experiments/curriculum/a0.yaml`,
then, once the control's rounds-to-pass is read,
`just train-per-model-arm <3× rounds> 3 curriculum-a0 a0 "--num-rollout-envs 4 --rollout-rounds 32 --eval-every-rounds 512 --checkpoint-every-rounds 512 --n-eval-episodes 30" configs/experiments/curriculum/a0.yaml`.

## Primary readouts

Success rate and turns-to-success (paired against the script per episode),
`held`, `on_obj`; the passive pair and `train/rounds_per_update` on every
per-model row; the health panel over the last quarter; rounds-to-pass on
both arms from their in-run curves. Reported with the provenance table in
#341 at close.

## What I expect (a guess, written so it can be wrong)

The whole-army trainer passes A0 in well under the E1 control's ~60k rounds
— three lone models have no coherency chain to satisfy, which is the thing
that made E1's control slow. The per-model trainer passes at a comparable
round count, or its declaration head collapses to `stationary` early (the
do-nothing fingerprint) and the greedy eval never leaves the floor while the
sampled score climbs, which `measure-per-model-eval-mode` will show. I do
not know which.
