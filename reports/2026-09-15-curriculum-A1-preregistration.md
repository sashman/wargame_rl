# Pre-registration: curriculum rung A1 — the same three models as one squad

Written 2026-09-15, **before any number from the arm exists**, on branch
`feature/curriculum-a1` (one PR per rung, stacked on PR #342; the chain does
not merge to `main` while exploratory, so the commit timestamp of this file
is what orders it before the numbers). Issue #343, parent question #340,
rung **A1** — axis: **squads**. A0 (#341) passed with a defect on both
facades ([report](2026-09-15-curriculum-a0-passes.md)).

## The one change

The A0 scenario with its three models bound into **one unit of three**
(`group_id 0` ×3, `max_groups 1`), deployed as a squad
(`coherency.enforce_at_deployment: true`). Nothing pays for or referees
coherency on this rung — it is a readout — so what changes for the
per-model step is the unit itself: one open per unit, the members walked in
sequence, unit membership in the tokens. Radius 4, 8 rounds, 60×44, Move 6,
the same reward (`configs/experiments/curriculum/a1.yaml`). Two arms of the
same scenario, three seeds each: per-model (under test) and whole-army
(pipeline control).

## Comparator, measured first

`squad_march_take`, scored through both facades on seeds 700000+ at n=100
(`just measure-bridge configs/experiments/curriculum/a1.yaml 100`), bridge
identical on every shared field: success **0.960**, turns **4.96** (sd
0.74, range 4–8; the four failures are one model frozen behind a squadmate,
counted at the 8-round clock), `held` 1.00, `on_obj` 0.987, coherent
**0.993**, vp 9.1 ± 0.3. In-run band (n=30, 500000+): success 0.967, turns
5.10, coherent 0.989.

The bar clears its own criterion (≥ 0.95) by half a binomial SE. It is not
widened: the disc is A0's, and changing it would be a second axis. The
freeze is a property of a deterministic script re-issuing a blocked order,
not of the disc; a learned policy that freezes the same way will read it in
`on_obj` and in the turn count.

## Criteria

Read at the END of the budget from the last checkpoint (`last.pt` /
`last.ckpt`), greedy, no decode, n=100 on seeds 700000+ — the same seeds
the bar was measured on, so success and turns pair per episode.

- **PASS:** success rate ≥ 0.95 on **every seed** of the per-model arm,
  within budget.
- **FAIL:** any per-model seed misses it at budget while the whole-army
  control passes on ≥ 2 of 3 seeds — the fault is in the per-model
  handling of a unit: the unit open, the sequenced members, the membership
  tokens.
- **NULL (scenario):** the whole-army control also fails on ≥ 2 of 3 seeds
  — redesign, do not read the per-model result.
- **Pass with a defect:** the bound holds but the health panel is red over
  the last quarter. A0's open item is the ratio tail (`train/ratio_p99`
  1.65–1.79 on a passing run at 128 rounds per update): if it persists here
  on a pass, the panel's ~1.5 line is recalibrated for this regime; if it
  coincides with a failure, the update is the suspect.

**Readouts, not criteria** — the A0 lesson (`CLAUDE.md` § The per-model
curriculum): turns-to-success paired against the script and against the
whole-army control's own mean, reported and flagged if a seed is more than
one round slower than the control; coherency beside the bar's 0.993, since
nothing on this rung pays for it and a squad that learns to walk apart is a
finding about the E rungs' referee, not a failure here.

Power: success at n=100 has binomial SE 0.022 at p=0.95; a true 0.90 fails
at ~2.3 SE. The bar sits 0.5 SE above the bound.

## Budget, regime, seeds

The whole-army control runs **first** and calibrates the budget (#340
contract 3): the per-model budget is **3× the control's slowest
rounds-to-pass** (the first in-run evaluation at which success reaches 0.95
and stays there), with a floor of 20,480 rounds and a cap of 122,880 (60
epochs of 2048 steps). If the control has not passed by the cap, the rung
is NULL (scenario) and the per-model arm is not launched.

| | per-model arm | whole-army control |
|---|---|---|
| budget | 3× the control's slowest rounds-to-pass, in [20,480, 122,880] | up to 60 epochs of 2048 steps |
| regime | `--rollout-rounds 32 --num-rollout-envs 4` = **128 rounds per update** | 2048 steps per update |
| seeds | 1, 2, 3; rollout layouts at seed×100+ | 1, 2, 3 |
| in-run eval | every 512 rounds, n=30, seeds 500000+, greedy, passive pair and `eval/mean_turns` logged | every epoch, n=30, seeds 500000+ |
| final score | n=100, seeds 700000+, `just measure-rung` | same |
| decode | none (K=1) | K=1, undecoded |
| other flags | defaults (`gamma` 0.9, `ent_coef` 0.03, `lr` 3e-4) | defaults |
| logging | Wandb group `curriculum-a1`, one run per seed | same group |
| start | from scratch (the transfer question is T1's) | from scratch |

Launch: `just train-curriculum-control 60 3 curriculum-a1 a1-ctl "" configs/experiments/curriculum/a1.yaml`,
then `just train-per-model-arm <3× rounds> 3 curriculum-a1 a1 "--num-rollout-envs 4 --rollout-rounds 32 --eval-every-rounds 512 --checkpoint-every-rounds 512 --n-eval-episodes 30" configs/experiments/curriculum/a1.yaml`.

## Primary readouts

Success rate (decides); turns paired against the script and against the
control; `held`, `on_obj`, coherency; the passive pair and
`train/rounds_per_update` on every per-model row; the health panel over the
last quarter, the ratio tail in particular; rounds-to-pass on both arms.

## What I expect (a guess, written so it can be wrong)

Both facades pass. The per-model arm again takes 2–4× the control's rounds
and again lands near the script's turn count; the whole-army control is
again about a round slower. The ratio tail persists at 1.6–1.8 on the
per-model side, which would move the panel's line rather than indict the
update. Coherency on the per-model side ends below the bar's 0.993 —
nothing pays for it — and by how much is the number worth having.
