# Pre-registration: curriculum rung A2 — four squads, one objective

Written 2026-09-15, **before any number from the arm exists**, on branch
`feature/curriculum-a2` (one PR per rung, stacked on A1's PR #344; the
chain does not merge to `main` while exploratory, so the commit timestamp
of this file is what orders it before the numbers). Issue #347, parent
question #340, rung **A2** — axis: **bodies**. A1 (#343) is the rung
below.

## The one change

Four units of three (`group_id 0–3`, `max_groups 4`), **one** objective,
no opponent, movement only, deployed as squads — twelve bodies where A1
had three, all bound for the same point. The disc and the success
criterion scale with the body count and are part of the axis: twelve
bases cannot all stand on A1's radius-4 disc without the friendly-gridlock
freeze the script is known to suffer, so the disc is **radius 6** and
success is **≥ 75% of alive models on the objective** (9 of 12,
`fraction_at_objectives`). 8 rounds, 60×44, Move 6, the same reward
(`configs/experiments/curriculum/a2.yaml`). Two arms of the same scenario,
three seeds each: per-model (under test) and whole-army (pipeline
control).

## Comparator, measured first

`squad_march_take`, scored through both facades on seeds 700000+ at n=100
(`just measure-bridge configs/experiments/curriculum/a2.yaml 100`), bridge
identical on every shared field: success **0.990**, turns **4.55** (sd
0.66, range 4–8), `held` 1.00, `on_obj` 0.843, coherent **0.889**, vp
7.7 ± 0.3. In-run band (n=30, 500000+): success 1.000, turns 4.67,
coherent 0.839.

**What was rejected before the disc and the fraction were fixed**, all the
script at n=100 on 700000+, no training number existing at any of them:
`all_at_objectives` at radius 4 / 6 / 8 → success 0.11 / 0.33 / 0.66
(`on_obj` 0.855 / 0.901 / 0.960 — the bodies arrive, the last two or three
freeze at the rim); 85% at radius 4 / 6 / 8 / 10 → 0.47 / 0.61 / 0.90 /
0.96; 75% at radius 8 → 1.00 with turns 4.03, a disc so wide the approach
is the whole task. Radius 6 at 75% is the tightest setting at which the
bar clears its own criterion with margin.

## Criteria

Read at the END of the budget from the last checkpoint (`last.pt` /
`last.ckpt`), greedy, no decode, n=100 on seeds 700000+ — the same seeds
the bar was measured on, so success and turns pair per episode.

- **PASS:** success rate ≥ 0.95 on **every seed** of the per-model arm,
  within budget.
- **FAIL:** any per-model seed misses it at budget while the whole-army
  control passes on ≥ 2 of 3 seeds — the fault is in the per-model
  handling of several units and a crowd: the unit pointer over four
  squads, the sequenced members, twelve tokens contending for one disc.
- **NULL (scenario):** the whole-army control also fails on ≥ 2 of 3 seeds
  — redesign, do not read the per-model result.
- **Pass with a defect:** the bound holds but the health panel is red over
  the last quarter; the ratio tail is read against whatever A1 decided
  about the panel's ~1.5 line.

**Readouts, not criteria:** turns-to-success paired against the script and
against the whole-army control's own mean; `on_obj` (how many of twelve
arrive — the script leaves 1.9 behind on average); coherency beside the
bar's 0.889, since nothing on this rung pays for it.

Power: success at n=100 has binomial SE 0.022 at p=0.95; a true 0.90
fails at ~2.3 SE. The bar sits 1.8 SE above the bound.

## Budget, regime, seeds

The whole-army control runs **first** and calibrates the budget (#340
contract 3, as amended by A1x — the per-model arm needs 2–4× the control's rounds on the A rungs): the per-model budget is **6× the control's slowest
rounds-to-pass** (the first in-run evaluation at which success reaches
0.95 and stays there), with a floor of 20,480 rounds and a cap of 122,880
(60 epochs of 2048 steps). If the control has not passed by the cap, the
rung is NULL (scenario) and the per-model arm is not launched.

| | per-model arm | whole-army control |
|---|---|---|
| budget | 6× the control's slowest rounds-to-pass (the A1x amendment to #340 contract 3), in [20,480, 122,880] | up to 60 epochs of 2048 steps |
| regime | `--rollout-rounds 32 --num-rollout-envs 4` = **128 rounds per update** | 2048 steps per update |
| seeds | 1, 2, 3; rollout layouts at seed×100+ | 1, 2, 3 |
| in-run eval | every 512 rounds, n=30, seeds 500000+, greedy, passive pair and `eval/mean_turns` logged | every epoch, n=30, seeds 500000+ |
| final score | n=100, seeds 700000+, `just measure-rung` | same |
| decode | none (K=1) | K=1, undecoded |
| other flags | defaults (`gamma` 0.9, `ent_coef` 0.03, `lr` 3e-4) | defaults |
| logging | Wandb group `curriculum-a2`, one run per seed | same group |
| start | from scratch (the transfer question is T1's) | from scratch |

Launch: `just train-curriculum-control 60 3 curriculum-a2 a2-ctl "" configs/experiments/curriculum/a2.yaml`,
then `just train-per-model-arm <6× rounds> 3 curriculum-a2 a2 "--num-rollout-envs 4 --rollout-rounds 32 --eval-every-rounds 512 --checkpoint-every-rounds 512 --n-eval-episodes 30" configs/experiments/curriculum/a2.yaml`.

## Primary readouts

Success rate (decides); turns paired against the script and against the
control; `held`, `on_obj`, coherency; the passive pair and
`train/rounds_per_update` on every per-model row; the health panel over
the last quarter; rounds-to-pass on both arms.

## What I expect (a guess, written so it can be wrong)

Both facades pass, and the per-model arm's `on_obj` is higher than the
script's 0.843 — a learned policy has no reason to re-issue a blocked
order, which is what leaves the script's last bodies at the rim. The
per-model arm again needs 2–4× the control's rounds. Coherency on both
learned arms ends below the bar's 0.889, because nothing pays for it and
twelve bodies on one disc have every reason to spread.
