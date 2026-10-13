# Curriculum — the A rungs (A0–A5), T1 and A5's second pass

The movement-only rungs of the per-model curriculum (#340): three lone models up to eight squads over six objectives, the transfer rung T1, and A5's second pass. Each rung's pre-registration is followed by its result, word for word as first written.

## Contents

- [Pre-registration: curriculum rung A0 — three lone models, one objective, no opponent](#2026-09-14-curriculum-A0-preregistration)
- [Curriculum rung A0: the per-model pipeline learns the plumbing rung — PASS with a defect](#2026-09-15-curriculum-a0-passes)
- [Pre-registration: curriculum rung A1 — the same three models as one squad](#2026-09-15-curriculum-A1-preregistration)
- [Curriculum rung A1: two seeds pass at the script's speed, one plateaus at 0.79 — FAIL, as pre-registered](#2026-09-15-curriculum-a1-fails-on-one-seed)
- [Pre-registration: curriculum rung A1, second arm (A1x) — the A1 runs resumed to twice the budget](#2026-09-15-curriculum-A1x-preregistration)
- [Curriculum rung A1, second arm: the same runs given twice the rounds pass 3 of 3 — the budget rule was the defect](#2026-09-15-curriculum-a1x-passes)
- [Pre-registration: curriculum rung A2 — four squads, one objective](#2026-09-15-curriculum-A2-preregistration)
- [Curriculum rung A2: the per-model arm solves the rung by 10k rounds, then unlearns it — FAIL, as pre-registered](#2026-09-15-curriculum-a2-solved-then-unlearned)
- [Pre-registration: curriculum rung A2, second arm (A2b) — the entropy coefficient at 0.003](#2026-09-15-curriculum-A2b-preregistration)
- [Curriculum rung A2, second arm: at `ent_coef` 0.003 the per-model arm holds the rung — PASS with drift](#2026-09-15-curriculum-a2b-passes-with-drift)
- [Pre-registration: curriculum rung A3 — four squads, four objectives, the spread rung](#2026-09-15-curriculum-A3-preregistration)
- [Curriculum rung A3: the per-model arm is behind the control on the spread rung at the cap — FAIL, as pre-registered](#2026-09-15-curriculum-a3-behind-the-control-at-the-cap)
- [Pre-registration: three arms at why the spread rung learns slowly (A3 speed screen)](#2026-09-15-curriculum-A3-speed-preregistration)
- [A3 speed screen, arm S1: paying for standing on a point moves two seeds of three — NULL on the pre-registered letter](#2026-09-15-curriculum-a3-speed-s1-hold)
- [A3 speed screen, arm S2: one objective per squad in the travel assignment makes the rung SLOWER — HARMFUL, as the pre-registered clause names it](#2026-09-15-curriculum-a3-speed-s2-match)
- [A3 speed screen, arm S3: warm-starting the spread rung from the one-point rung is AHEAD on 3 of 3 — the transfer the ladder promised, one rung early, against the prediction](#2026-09-15-curriculum-a3-speed-s3-warm)
- [Pre-registration: curriculum rung A3, second arm (A3x) — the A3 runs resumed to the control's 245,760 rounds](#2026-09-15-curriculum-A3x-preregistration)
- [Curriculum rung A3, second arm: given the control's rounds, the per-model arm spreads over four points — PASS 3 of 3](#2026-09-15-curriculum-a3x-passes)
- [Pre-registration: curriculum rung A4 — four squads, three objectives, a spare squad](#2026-09-15-curriculum-A4-preregistration)
- [Curriculum rung A4: one standard error short at the cap on two seeds — FAIL, as pre-registered](#2026-09-15-curriculum-a4-one-se-short-at-the-cap)
- [Pre-registration: curriculum rung A4, second arm (A4x) — the A4 runs resumed to 245,760 rounds](#2026-09-15-curriculum-A4x-preregistration)
- [Curriculum rung A4, second arm: given twice the cap, the per-model arm places the spare squad — PASS 3 of 3, at the script's speed](#2026-09-15-curriculum-a4x-passes)
- [Pre-registration: curriculum rung A5 — eight squads, six objectives](#2026-09-15-curriculum-A5-preregistration)
- [Curriculum rung A5: the whole-army control cannot put a squad on every point — NULL (scenario) as pre-registered, and a defect in the clause](#2026-09-15-curriculum-a5-control-cannot-allocate)
- [Pre-registration: curriculum rung A5, second arm (A5b) — the per-model arm read beside the control, not gated by it](#2026-09-15-curriculum-A5b-preregistration)
- [Curriculum rung A5b: the per-model arm fails the allocation rung, and not the way the control did — FAIL as pre-registered](#2026-09-16-curriculum-a5b-fails-differently)
- [Pre-registration: curriculum rung T1 — A5 warm-started from A4x's checkpoint, against A5b from scratch (the transfer rung)](#2026-09-15-curriculum-T1-preregistration)
- [Curriculum rung T1: warm-starting A5 from A4x does not rescue it — FAIL as pre-registered, and the transfer is real](#2026-09-17-curriculum-t1-transfer-does-not-rescue-a5)
- [Pre-registration: curriculum rung A5c — A5b re-run with the actor credit](#2026-09-17-curriculum-A5c-preregistration)
- [Pre-registration: A5e (the hold term under the actor credit) and A5f (the bar cloned, then anchored PPO)](#2026-09-18-curriculum-A5e-A5f-preregistration)
- [Pre-registration: the A5 half-steps — A5-bodies and A5-points](#2026-09-18-curriculum-A5-half-steps-preregistration)
- [Curriculum rung A5, the second pass: the credit is not the wall, the clone is the ceiling, and the anchor holds it over the mark](#2026-09-19-curriculum-a5-passes-on-the-start-axis)

---

<a id="2026-09-14-curriculum-A0-preregistration"></a>

## Pre-registration: curriculum rung A0 — three lone models, one objective, no opponent

Written 2026-09-14, **before any number from the arm exists**, on branch
`feature/curriculum-a0` (one PR per rung, stacked on PR #339; the chain does
not merge to `main` while exploratory, so the commit timestamp of this file
is what orders it before the numbers). Issue #341, parent question #340,
rung **A0** — the plumbing rung, axis: none.

### The one change

Training on the **A0** rung: 3 units of 1 model each, 1 objective of radius
4 sitting 22–28" from the deployment band, no opponent, movement only, 8
rounds, 60×44, Move 6 (`configs/experiments/curriculum/a0.yaml`). Two arms
of the same scenario, three seeds each: the **per-model** trainer
(`train_per_model.py`, the thing under test) and the **whole-army** trainer
(`train.py`, the pipeline control). Nothing else varies. Three lone models
have no formation to keep, so nothing but the optimiser and the reward is
under test; A1 binds these same three bodies into one squad.

### Comparator, measured first

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

### Criteria

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

### Budget, regime, seeds

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

### Primary readouts

Success rate and turns-to-success (paired against the script per episode),
`held`, `on_obj`; the passive pair and `train/rounds_per_update` on every
per-model row; the health panel over the last quarter; rounds-to-pass on
both arms from their in-run curves. Reported with the provenance table in
#341 at close.

### What I expect (a guess, written so it can be wrong)

The whole-army trainer passes A0 in well under the E1 control's ~60k rounds
— three lone models have no coherency chain to satisfy, which is the thing
that made E1's control slow. The per-model trainer passes at a comparable
round count, or its declaration head collapses to `stationary` early (the
do-nothing fingerprint) and the greedy eval never leaves the floor while the
sampled score climbs, which `measure-per-model-eval-mode` will show. I do
not know which.

### Amendment 1 — the control's read, and the budget it sets (2026-09-14 23:50, before any per-model number exists)

The whole-army control ran to its 60-epoch cap (Wandb group `curriculum-a0`,
runs `v9uwm61r` s1 · `9w0rzj22` s2 · `1enz2ero` s3; the launch comment on
#341 had s2 and s3's ids swapped). In-run success (n=30, seeds 500000+)
first reaches 0.95 and stays there at epoch **7 / 5 / 8** (0-based), i.e.
rounds-to-pass **16,384 / 12,288 / 18,432**. Under the budget rule above the
per-model budget is 3× the slowest seed: **55,296 rounds** (108 updates of
512-round eval cadence), launched at that figure.

Scored at n=100 on seeds 700000+ from `last.ckpt` (epoch 60): success
**1.000 / 1.000 / 1.000**, turns **6.21 / 6.67 / 6.07** against the
script's 4.93 (paired +1.28 / +1.74 / +1.14, SE ≤ 0.05), `held` 1.00.

**So the control passes the success bound on 3 of 3 seeds and misses the
turn bound (≤ 6.0) on 3 of 3.** Read literally, the criteria above call
that NULL (scenario), and that reading is recorded as a **defect in the
rule, not in the scenario**: the whole-army trainer learned the task in
under ten epochs and is a round slower than a script that walks a straight
line — the same shape the E1 control showed (6.9–7.0 turns against 5.55).
A speed bound of "the script's slowest episode plus one" was written
without checking what a converged PPO policy on this reward reaches, which
`CLAUDE.md` § How to measure here already warns against ("power-check a
per-seed bound against the expected spread before writing it down").

The per-model arm's criteria are **unchanged** and will be read on both
bounds exactly as written. What this amendment changes is the attribution
clause: the control's success bound is what decides FAIL versus NULL
(scenario), and the turn bound is reported beside it for both facades
rather than deciding either. The rung's verdict will say so in as many
words. Nothing about the per-model arm was known when this was written.

---

<a id="2026-09-15-curriculum-a0-passes"></a>

## Curriculum rung A0: the per-model pipeline learns the plumbing rung — PASS with a defect

**Verdict first.** On the first rung of the ladder (#340, arm #341) — three
lone models, one objective, nobody shooting — the per-model trainer passes
the pre-registered bounds on **3 of 3 seeds**: greedy success **1.000 /
1.000 / 0.950** at n=100 and mean turns **4.93 / 4.96 / 5.11** against the
script's 4.93, where the whole-army control (success 1.000 ×3) arrives a
round later at **6.21 / 6.67 / 6.07**. It needs **2–4× the control's rounds**
to get there (29k / 52k / 41k against 16k / 12k / 18k). The health panel is
not clean over the last quarter — the importance ratio's p99 sits at
1.65–1.79 on every seed, past the panel's ~1.5 line — so the rung is recorded
as a **pass with a defect**, as the pre-registration says it must be. The
pipeline works; whether the ratio tail is a defect of the pipeline or of the
panel's threshold at 128 rounds per update is open and is carried to A1.

### Provenance

| field | value |
|---|---|
| date | 2026-09-15 (runs 2026-09-14 23:32–00:00) |
| GPU / no-GPU | GPU (RTX 4090), six processes concurrent |
| seeds | training 1 / 2 / 3 on both arms; rollout layouts at seed×100+ |
| n | 100 at seed base 700000 (final); 30 at 500000 (in-run) |
| config | `configs/experiments/curriculum/a0.yaml` — unrefereed, nothing to referee (three units of one model) |
| decode | none on the per-model facade; K=1 on the whole-army control (the regime each trains in) |
| paired | per episode against `squad_march_take` on identical seeds; per-model v whole-army unpaired |
| comparator | `squad_march_take`, fixed by name and measured first (bridge identical on every shared field) |
| opponent | none |
| budget | per-model 55,296 rounds at 128 rounds per update (`--rollout-rounds 32 --num-rollout-envs 4`); control 60 epochs of 2048 steps |
| code revision | `04329c5` on `feature/curriculum-a0` (PR #342, stacked on #339) |
| checkpoint | `last.pt` (at budget) / `last.ckpt` (epoch 60) |
| coherency | 1.000 by construction |
| Wandb | group `curriculum-a0`: per-model `vk5oj0g8` s1 · `iqsh3rym` s2 · `26lwv8fd` s3; control `v9uwm61r` s1 · `9w0rzj22` s2 · `1enz2ero` s3 |
| pre-registration | `reports/2026-09-15-curriculum-a-rungs.md#2026-09-14-curriculum-A0-preregistration`, committed at `df3e1c3` before any training number; amendment 1 at `04329c5` before any per-model number |

### The rung

3 units of 1 model, 1 objective of radius 4 sitting 22–28" from the
deployment band, no opponent, movement only, 8 rounds, 60×44, Move 6.
`closest_objective_v2` progress + `objective_coverage`, success
`all_at_objectives`, `terminate_on_success` with the terminal bonus scaled by
the rounds remaining. ⚠ At radius 3 the script itself failed 5 of 100
episodes with one model frozen behind a friend (friendly gridlock), so the
disc was widened before any training number existed; the pre-registration
records it.

**The bar**, n=100 on 700000+: success 1.000, turns 4.93 (sd 0.26, range
4–5), `held` 1.00, vp 8.8 ± 0.2.

### The read

| row | success | turns | vs bar, paired | held | on obj | stat | rounds-to-pass |
|---|---|---|---|---|---|---|---|
| `squad_march_take` | 1.000 | 4.93 | — | 1.00 | 1.000 | — | — |
| per-model s1 | **1.000** | 4.93 | +0.00 ± 0.02 | 1.00 | 1.000 | 0.00 | 29,184 |
| per-model s2 | **1.000** | 4.96 | +0.03 ± 0.03 | 1.00 | 1.000 | 0.01 | 52,224 |
| per-model s3 | **0.950** | 5.11 | +0.18 ± 0.07 | 1.00 | 0.977 | 0.02 | 40,960 |
| whole-army s1 | 1.000 | 6.21 | +1.28 ± 0.05 | 1.00 | 1.000 | — | 16,384 |
| whole-army s2 | 1.000 | 6.67 | +1.74 ± 0.04 | 1.00 | 1.000 | — | 12,288 |
| whole-army s3 | 1.000 | 6.07 | +1.14 ± 0.03 | 1.00 | 1.000 | — | 18,432 |

Rounds-to-pass is the first in-run evaluation (n=30, seeds 500000+) at
which success reaches 0.95 and stays there.

**Greedy against sampled** on the tuning band (900000+, n=100, paired):
−0.1 ± 0.4 / −0.8 ± 0.4 / −1.1 ± 0.4 vp, win 100% both ways, stationary share
0.00–0.04. The policy a score reports and the policy training rolled out are
the same policy here; there is no do-nothing fingerprint.

**Health panel over the last quarter** (108 updates per seed):

| key | s1 | s2 | s3 | healthy |
|---|---|---|---|---|
| `train/rounds_per_update` | 128 | 128 | 128 | stated |
| `train/advantage_std` | 0.24 | 0.46 | 0.27 | > 0 |
| `advantage_abs_max / std`, mean (max) | 4.1 (7.0) | 3.2 (5.6) | 3.9 (5.9) | within ~5× |
| `train/explained_variance` | 0.68 | 0.59 | 0.67 | up |
| `train/ratio_p99`, mean (max) | 1.65 (2.00) | 1.68 (2.39) | 1.79 (2.08) | **~1.5** |
| `train/clip_fraction` | 0.16 | 0.17 | 0.21 | pair with p99 |
| `train/approx_kl_per_1k_rounds` | 0.171 | 0.170 | 0.178 | a rate to compare |
| `train/entropy/head/declaration` | 0.04 | 0.14 | 0.05 | > 0, < 1.39 |
| `train/entropy/head/displacement` | 1.23 | 1.86 | 1.34 | > 0, < 4.57 |
| `eval/stationary_share` (last 12) | 0.00–0.02 | 0.00–0.05 | 0.00–0.03 | away from 1 |

Everything is green except the two rows in bold-adjacent territory: the
ratio tail is past 1.5 on all three seeds throughout the last quarter, and
the advantage abs-max exceeds 5× its std on some updates (the terminal bonus
scaled by remaining rounds is exactly the spike the panel row anticipates).

### What it says

- **The per-model pipeline learns.** Optimiser, reward re-timing, the
  declaration and displacement heads, the greedy read and the passive
  fingerprint all behave on the plumbing rung. The first per-model runs sat
  at the floor for regime, not for code (`reports/2026-09-14-one-episode-per-update.md`);
  at 128 rounds per update the same code passes.
- **It arrives at the script's speed; the whole-army control does not.**
  Every per-model seed lands within 0.2 turns of a script that walks a
  straight line; every whole-army seed is 1.1–1.7 turns slower with the
  same reward. That is the first behavioural difference between the two
  facades on record, and it is in the per-model facade's favour — but it is
  one rung, three models, no formation to keep. Read it as a note, not a
  result.
- **It is 2–4× slower in rounds to get there.** Rounds-to-pass 29k–52k
  against the control's 12k–18k. Each per-model round is three decisions
  and one close, so per decision the gap is larger; per gradient step it is
  not measured here. The budget rule (3× the control's slowest) held with
  little margin on s2 (52k of 55k).
- ⚠ **The speed bound written in the pre-registration was wrong for the
  control, and amendment 1 says so before the per-model numbers existed.**
  "The script's slowest episode plus one round" was missed 3/3 by a control
  that learned the task in under ten epochs — the same shape E1's control
  showed. The per-model arm then cleared it 3/3, which is why the rung
  reads as a pass on the letter of the criteria and not only on their
  spirit. The lesson is about bounds, not about A0: pre-register a speed
  bound against what a converged whole-army policy reaches, or make the
  turn bound a readout.
- **The ratio tail is the open item.** A p99 of 1.65–1.79 with clip fraction
  0.16–0.21 is the panel's "trust region no longer binds" signature, on a
  run that passed. Either the threshold — written from the whole-army
  trainer at 1024 rounds per update — does not transfer to 128, or the
  per-model update is genuinely over-stepping and A0 is too easy to show
  it. A1 (the same bodies bound into a squad) reads the same panel; if the
  tail persists on a passing rung there too, the threshold moves; if it
  coincides with a failure, the update does.

### What was not done

- No per-gradient-step comparison between facades; no epoch-time comparison.
- The whole-army control was not resumed past 60 epochs to see whether its
  turn count keeps falling; the per-model arm was not extended past budget.
- s3's 0.950 is exactly the bound; at n=100 that is 5 episodes, and a
  rerun on another seed band could land either side. The verdict does not
  rest on s3 alone.

---

<a id="2026-09-15-curriculum-A1-preregistration"></a>

## Pre-registration: curriculum rung A1 — the same three models as one squad

Written 2026-09-15, **before any number from the arm exists**, on branch
`feature/curriculum-a1` (one PR per rung, stacked on PR #342; the chain does
not merge to `main` while exploratory, so the commit timestamp of this file
is what orders it before the numbers). Issue #343, parent question #340,
rung **A1** — axis: **squads**. A0 (#341) passed with a defect on both
facades ([report](2026-09-15-curriculum-a-rungs.md#2026-09-15-curriculum-a0-passes)).

### The one change

The A0 scenario with its three models bound into **one unit of three**
(`group_id 0` ×3, `max_groups 1`), deployed as a squad
(`coherency.enforce_at_deployment: true`). Nothing pays for or referees
coherency on this rung — it is a readout — so what changes for the
per-model step is the unit itself: one open per unit, the members walked in
sequence, unit membership in the tokens. Radius 4, 8 rounds, 60×44, Move 6,
the same reward (`configs/experiments/curriculum/a1.yaml`). Two arms of the
same scenario, three seeds each: per-model (under test) and whole-army
(pipeline control).

### Comparator, measured first

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

### Criteria

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

### Budget, regime, seeds

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

### Primary readouts

Success rate (decides); turns paired against the script and against the
control; `held`, `on_obj`, coherency; the passive pair and
`train/rounds_per_update` on every per-model row; the health panel over the
last quarter, the ratio tail in particular; rounds-to-pass on both arms.

### What I expect (a guess, written so it can be wrong)

Both facades pass. The per-model arm again takes 2–4× the control's rounds
and again lands near the script's turn count; the whole-army control is
again about a round slower. The ratio tail persists at 1.6–1.8 on the
per-model side, which would move the panel's line rather than indict the
update. Coherency on the per-model side ends below the bar's 0.993 —
nothing pays for it — and by how much is the number worth having.

### Amendment 1 — the control's read, and the budget it sets (2026-09-15 00:12, before any per-model number exists)

The whole-army control ran to its 60-epoch cap (Wandb group `curriculum-a1`,
runs `h8hq4yj2` s1 · `553ljuz4` s2 · `r0bceg97` s3). In-run success (n=30,
seeds 500000+) first reaches 0.95 and stays there at epoch **6 / 9 / 8**
(0-based), i.e. rounds-to-pass **14,336 / 20,480 / 18,432**. Under the
budget rule the per-model budget is 3× the slowest seed: **61,440 rounds**
(120 evaluations at the 512-round cadence), launched at that figure.

Scored at n=100 on seeds 700000+ from `last.ckpt` (epoch 60): success
**1.000 / 1.000 / 1.000**, turns **6.20 / 7.43 / 6.10** against the
script's 4.96 (paired +1.24 / +2.47 / +1.14), `held` 1.00, coherent
**0.976 / 0.882 / 0.994** against the bar's 0.993. The control passes the
success bound 3 of 3, so a per-model miss reads FAIL and not NULL. Turns and
coherency are readouts on this rung and are quoted beside the per-model
row when it exists; the control is again 1.1–2.5 rounds slower than the
script, and one seed walks its squad apart (0.882) with nothing paying for
formation. Nothing about the per-model arm was known when this was written.

---

<a id="2026-09-15-curriculum-a1-fails-on-one-seed"></a>

## Curriculum rung A1: two seeds pass at the script's speed, one plateaus at 0.79 — FAIL, as pre-registered

**Verdict first.** On the second rung of the ladder (#340, arm #343) — A0's
three models bound into one squad — the per-model trainer passes the
pre-registered success bound on **2 of 3 seeds** (greedy success **1.000 /
0.790 / 1.000** at n=100, turns **5.01 / 6.10 / 4.90** against the script's
4.96) while the whole-army control passes on 3 of 3 (success 1.000 ×3,
turns 6.20 / 7.43 / 6.10). The pre-registration says any per-model seed
missing the bound at budget while the control passes is **FAIL**, and it
is recorded as one. What the record says beside it: the two seeds that
pass do so at the script's own speed where every control seed is 1.1–2.5
rounds slower; the seed that fails did not collapse — it climbed from 0.50
to a plateau of 0.80–0.93 in the last quarter of a budget that was set at
3× the control's slowest seed, and its in-run curve was still moving. The
follow-up the record's own rule prescribes ("treat a marginal result as
run it longer") is a second arm on this rung, pre-registered before it
runs: the same three runs resumed in place to twice the budget.

### Provenance

| field | value |
|---|---|
| date | 2026-09-15 (runs 00:06–00:23) |
| GPU / no-GPU | GPU (RTX 4090), six processes concurrent |
| seeds | training 1 / 2 / 3 on both arms; rollout layouts at seed×100+ |
| n | 100 at seed base 700000 (final); 30 at 500000 (in-run) |
| config | `configs/experiments/curriculum/a1.yaml` — unrefereed by design (coherency a readout) |
| decode | none on the per-model facade; K=1 on the control |
| paired | per episode against `squad_march_take` on identical seeds; per-model v whole-army unpaired |
| comparator | `squad_march_take`, fixed by name and measured first (bridge identical) |
| opponent | none |
| budget | per-model 61,440 rounds at 128 rounds per update (3× the control's slowest rounds-to-pass); control 60 epochs of 2048 steps |
| code revision | `7e217ad` on `feature/curriculum-a1` (PR #344, stacked on #342) |
| checkpoint | `last.pt` (at budget) / `last.ckpt` (epoch 60) |
| coherency | per-model greedy 0.760 / 0.889 / 0.896; control 0.976 / 0.882 / 0.994; bar 0.993 |
| Wandb | group `curriculum-a1`: per-model `9irv0y7y` s1 · `cm17c1wq` s2 · `ahcy2jl0` s3; control `h8hq4yj2` s1 · `553ljuz4` s2 · `r0bceg97` s3 |
| pre-registration | `reports/2026-09-15-curriculum-a-rungs.md#2026-09-15-curriculum-A1-preregistration` at `ba1dd5c`; amendment 1 (the control's read) at `7e217ad`, before any per-model number |

### The rung

A0's scenario with its three models as one unit of three, deployed as a
squad (`enforce_at_deployment`); nothing pays for or referees coherency.
Radius 4, 8 rounds, 60×44, Move 6. **The bar**, n=100 on 700000+: success
0.960 (four episodes with a model frozen behind a squadmate), turns 4.96,
coherent 0.993.

### The read

| row | success | turns | vs bar, paired | held | on obj | coherent | stat | rounds-to-pass |
|---|---|---|---|---|---|---|---|---|
| `squad_march_take` | 0.960 | 4.96 | — | 1.00 | 0.987 | 0.993 | — | — |
| per-model s1 | **1.000** | 5.01 | +0.05 ± 0.07 | 1.00 | 1.000 | 0.760 | 0.00 | 53,248 |
| per-model s2 | **0.790** | 6.10 | +1.14 ± 0.15 | 0.87 | 0.823 | 0.889 | 0.00 | — (never) |
| per-model s3 | **1.000** | 4.90 | −0.06 ± 0.07 | 1.00 | 1.000 | 0.896 | 0.00 | 31,744 |
| whole-army s1 | 1.000 | 6.20 | +1.24 ± 0.08 | 1.00 | 1.000 | 0.976 | — | 14,336 |
| whole-army s2 | 1.000 | 7.43 | +2.47 ± 0.09 | 1.00 | 1.000 | 0.882 | — | 20,480 |
| whole-army s3 | 1.000 | 6.10 | +1.14 ± 0.08 | 1.00 | 1.000 | 0.994 | — | 18,432 |

Seed 2's in-run curve (n=30, 500000+): 0.50–0.73 through 10k rounds, 0.80
from ~50k, its last ten evaluations 80 / 83 / 83 / 93 / 83 / 80 / 80 / 83 /
80 / 80 — a plateau, not a collapse, and not a pass. Its final n=100 read
fails 21 episodes with `on_obj` 0.823: in the failing episodes one of the
three does not arrive.

**Greedy against sampled** on the tuning band (900000+, n=100, paired):
−0.1 ± 0.4 / −0.3 ± 0.6 / −0.6 ± 0.3 vp, win 100 / 95 / 100 sampled. The
number to keep is the coherency gap: **greedy 0.844 / 0.895 / 0.942 against
sampled 0.555 / 0.232 / 0.620.** The policy training rolls out walks the
squad apart far more than the policy a score reports; nothing on this rung
pays for formation, so this is a readout, not a failure — but it is the
first sign of what the E rungs' referee will meet.

**Health panel over the last quarter** (120 updates per seed):

| key | s1 (pass) | s2 (fail) | s3 (pass) | healthy |
|---|---|---|---|---|
| `train/rounds_per_update` | 128 | 128 | 128 | stated |
| `train/advantage_std` | 0.30 | 0.60 | 0.25 | > 0 |
| `advantage_abs_max / std`, mean (max) | 4.2 (7.6) | 2.6 (4.1) | 4.4 (6.1) | within ~5× |
| `train/explained_variance` | 0.64 | 0.54 | 0.68 | up |
| `train/ratio_p99`, mean (max) | 1.83 (2.65) | 1.68 (2.28) | 1.70 (2.47) | ~1.5 |
| `train/clip_fraction` | 0.24 | 0.20 | 0.17 | pair with p99 |
| `train/approx_kl_per_1k_rounds` | 0.203 | 0.178 | 0.171 | a rate |
| `train/entropy/head/declaration` | 0.007 | 0.074 | 0.003 | > 0, < 1.39 |
| `train/entropy/head/displacement` | 1.37 | 1.80 | 1.05 | > 0, < 4.57 |
| `eval/stationary_share` (last 12) | 0.00 | 0.00 | 0.00 | away from 1 |

### What it says

- **FAIL on the letter, 2 of 3 on the numbers.** The criteria were written
  to attribute a miss to the per-model side when the control passes, and
  they do. What they cannot do is separate "the pipeline cannot learn a
  squad" from "one seed needed more than 3× the control's rounds", and the
  curve says the second is live: s2 was still rising at budget, and the
  two passing seeds needed 32k and 53k rounds where the control needed
  14k–20k. A0's per-model arm also ran 2–4× the control's rounds. **The
  budget rule may be the defect** — it is calibrated on a trainer that
  learns this rung in a fifth of the rounds.
- **The ratio tail does not track the failure, and the line moves.** A0's
  open item was `train/ratio_p99` at 1.65–1.79 on passing seeds. Here the
  two passing seeds read 1.83 and 1.70 and the failing seed reads the
  lowest, 1.68. The pre-registration said: persists on a pass → recalibrate
  the panel's line for this regime. It persists on a pass. At 128 rounds
  per update the regime's normal is 1.6–1.9 with clip fraction 0.17–0.24,
  and `docs/metrics.md` now says so; the line is not evidence of a fault
  unless it moves *with* a failure.
- **The declaration head runs near zero entropy on every seed** (0.003–0.07
  nats) with stationary share 0.00: the unit always declares a move. That
  is the right answer on a rung where standing still never pays, and the
  panel's "above zero" floor is met — but it is a head with nothing left
  to explore, and on the rungs where holding fire or standing still is
  correct it will have to come back up.
- **Speed, again.** Both passing per-model seeds land within 0.06 turns of
  the script; every control seed is a round or more slower, one by 2.5.
  Two rungs now show the same shape. It is still a note, not a result —
  but it is now a note twice.
- **Coherency is unpaid and it shows.** The learned arms on both facades
  end below the bar's 0.993 (per-model 0.76–0.90 greedy, control
  0.88–0.99), and the sampled per-model policy is at 0.23–0.62. Nothing on
  this rung asks for formation. The E rungs will, through the referee, and
  the gap between the policy trained and the policy scored will be priced
  there.

### What happens next

A second arm on this rung, **A1x**, pre-registered before it runs: the
three per-model runs resumed **in place** (`--resume-from`, same seed, same
knobs, optimizer and generator restored) to **122,880 rounds** — 2× the A1
budget, 6× the control's slowest rounds-to-pass, the ladder's cap. PASS is
3 of 3 at the same bound. If s2 passes, A1's verdict stands as written and
the budget rule is amended for the rungs above (the per-model arm gets a
floor of 6× the control); if it does not, the fault is in the per-model
handling of a unit and the ladder stops here until it is found.

### What was not done

- No inspection of *why* s2's third model does not arrive in 21 episodes
  (a recording of the failing seeds, `just record-per-model-events`, is the
  next diagnostic if A1x does not resolve it).
- The control was not resumed past 60 epochs.

---

<a id="2026-09-15-curriculum-A1x-preregistration"></a>

## Pre-registration: curriculum rung A1, second arm (A1x) — the A1 runs resumed to twice the budget

Written 2026-09-15 00:32, **before any number from the arm exists**, on
branch `feature/curriculum-a1` (PR #344 carries the rung's arms). Issue
#345, parent question #340, rung **A1** — a follow-up to #343, which read
**FAIL as pre-registered** with 2 of 3 per-model seeds passing
([report](2026-09-15-curriculum-a-rungs.md#2026-09-15-curriculum-a1-fails-on-one-seed)).

### The one change

**Budget.** The three A1 per-model runs (Wandb `curriculum-a1`, `9irv0y7y`
s1 · `cm17c1wq` s2 · `ahcy2jl0` s3) are resumed **in place** with
`--resume-from <run_dir> --rounds 122880`: same seed, same knobs (the
driver refuses a changed one), optimizer and sampling generator restored,
`metrics.jsonl` appended — a resumed run is one run. 122,880 rounds is 2×
the A1 budget, 6× the control's slowest rounds-to-pass (20,480), and the
ladder's cap. Nothing else changes; the whole-army control is not re-run.

### Why this arm and not a diagnosis first

A1's failing seed did not collapse: it climbed from 0.50 to a plateau of
0.80–0.93 over the last quarter and was still moving at budget, and the two
passing seeds needed 32k and 53k rounds where the control needed 14k–20k.
A0's arm ran 2–4× the control's rounds too. The cheapest test of "the
budget rule, calibrated on a trainer that learns this rung in a fifth of
the rounds, is the defect" is to give the same runs the rounds. It costs
twelve minutes. A recording of the failing episodes is the diagnostic if
this does not resolve it.

### Comparator

`squad_march_take` on `configs/experiments/curriculum/a1.yaml`, seeds
700000+ at n=100: success 0.960, turns 4.96, coherent 0.993. And A1's own
read at 61,440 rounds on the same seeds: success **1.000 / 0.790 / 1.000**,
turns 5.01 / 6.10 / 4.90.

### Criteria

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

### Budget, regime, seeds

| | per-model arm |
|---|---|
| budget | 122,880 rounds total, 61,440 more, 128 rounds per update |
| seeds | 1, 2, 3, continued |
| in-run eval | every 512 rounds, n=30, seeds 500000+ (as A1) |
| final score | n=100, seeds 700000+ |
| logging | the runs' own Wandb ids, continued |

Launch, per seed: `uv run train_per_model.py --resume-from checkpoints/per_model/<run> --rounds 122880`, detached with `setsid`, stdout under `checkpoints/per_model/logs/curriculum-a1/a1x-s<seed>.log`.

### What I expect (a guess, written so it can be wrong)

Seed 2 passes somewhere between 70k and 100k rounds and the budget rule
is what moves. Seeds 1 and 3 hold. The sampled coherency stays far below
the greedy one.

---

<a id="2026-09-15-curriculum-a1x-passes"></a>

## Curriculum rung A1, second arm: the same runs given twice the rounds pass 3 of 3 — the budget rule was the defect

**Verdict first.** The three A1 per-model runs, resumed in place from
61,440 to **122,880 rounds** (#345, pre-registered before the resume),
pass the bound on **3 of 3 seeds**: greedy success **1.000 / 1.000 /
1.000** at n=100, turns **4.93 / 4.88 / 4.93** against the script's 4.96 —
every seed now at or under the script's own speed, where every whole-army
control seed is 1.1–2.5 rounds slower. The seed that read 0.790 at the A1
budget passed on the first evaluation after the resume and never dipped;
the two that had already passed hold 1.000, each with one n=30 dip in 240
evaluations. A1's FAIL stands as written — it was a pre-registered read at
a pre-registered budget — and what it measured was the budget: **3× the
whole-army control's rounds-to-pass is too little for the per-model arm on
this rung, and 6× is enough.** The ladder's budget rule gains a floor of
6× from A2 on. Coherency, which nothing on this rung pays for, rose with
the extra rounds on every seed: greedy 0.968 / 0.980 / 0.972 (from
0.76–0.90), sampled 0.70 / 0.81 / 0.77 (from 0.23–0.62).

### Provenance

| field | value |
|---|---|
| date | 2026-09-15 (resumed 00:26, exited 00:39) |
| GPU / no-GPU | GPU (RTX 4090), three processes |
| seeds | 1 / 2 / 3, the A1 runs continued (`--resume-from`, optimizer and generator restored) |
| n | 100 at seed base 700000 (final); 30 at 500000 (in-run) |
| config | `configs/experiments/curriculum/a1.yaml` — unrefereed by design |
| decode | none |
| paired | per episode against `squad_march_take`; against A1's read on the same seeds and checkpoints |
| comparator | `squad_march_take`: success 0.960, turns 4.96, coherent 0.993; A1's read: 1.000 / 0.790 / 1.000 |
| opponent | none |
| budget | 122,880 rounds (61,440 more), 128 rounds per update |
| code revision | `3037b11` on `feature/curriculum-a1` (PR #344) |
| checkpoint | `last.pt` at 122,880 |
| coherency | greedy 0.968 / 0.980 / 0.972; sampled 0.703 / 0.808 / 0.769 |
| Wandb | `curriculum-a1`: `9irv0y7y` s1 · `cm17c1wq` s2 · `ahcy2jl0` s3, continued |
| pre-registration | `reports/2026-09-15-curriculum-a-rungs.md#2026-09-15-curriculum-A1x-preregistration` at `3037b11`, before the resume |

⚠ **The resume dropped the eval and checkpoint cadences** (512 → the 256
default) and overwrote `provenance.json` with the new values — #346. The
in-run curve is twice as dense after 61,440; nothing scored changes.

### The read

| row | success | turns | vs bar, paired | held | on obj | coherent greedy | coherent sampled | first pass, resumed half |
|---|---|---|---|---|---|---|---|---|
| `squad_march_take` | 0.960 | 4.96 | — | 1.00 | 0.987 | 0.993 | — | — |
| s1 at 61,440 (A1) | 1.000 | 5.01 | +0.05 | 1.00 | 1.000 | 0.760 | 0.555 | — |
| **s1 at 122,880** | **1.000** | 4.93 | −0.03 ± 0.07 | 1.00 | 1.000 | 0.968 | 0.703 | held (one dip, 85 at 94,464) |
| s2 at 61,440 (A1) | 0.790 | 6.10 | +1.14 | 0.87 | 0.823 | 0.889 | 0.232 | — |
| **s2 at 122,880** | **1.000** | 4.88 | −0.08 ± 0.07 | 1.00 | 1.000 | 0.980 | 0.808 | 61,696 — the first evaluation after the resume, never below 95 |
| s3 at 61,440 (A1) | 1.000 | 4.90 | −0.06 | 1.00 | 1.000 | 0.896 | 0.620 | — |
| **s3 at 122,880** | **1.000** | 4.93 | −0.03 ± 0.07 | 1.00 | 1.000 | 0.972 | 0.769 | held (one dip, 80 at 120,832) |

Greedy − sampled on the tuning band (900000+, n=100): −0.1 ± 0.4 / −0.1 ±
0.3 / −0.1 ± 0.3 vp, win 100% both ways.

**Health panel over the last quarter of the run** (240 updates per seed):
advantage std 0.21–0.22, explained variance 0.72 on all three (up from
0.54–0.68), ratio p99 1.72–1.78 with clip fraction 0.16–0.21 (the regime's
normal), declaration entropy 0.001–0.005, displacement entropy 0.87–1.03.
Nothing red.

### What it says

- **The budget rule was the defect, not the pipeline.** A1 read FAIL at a
  budget of 3× the control's slowest rounds-to-pass; the same runs at 6×
  pass 3/3 with every seed at the script's speed. Per-model rounds-to-pass
  on this rung: 53k / ~62k / 32k against the control's 14k–20k, i.e.
  **2–4× the control**, the same band as A0. From A2 on the per-model
  budget is **6× the control's slowest rounds-to-pass**, floor 20,480, cap
  122,880 (#340 contract 3, amended here).
- **s2 was at the threshold, not stuck.** Its n=30 in-run curve read
  80–93 at budget and its n=100 read 0.790; the first evaluation after the
  resume was ≥ 95 and it never dipped again. The plateau was the last few
  percent, which a 128-round update takes thousands of rounds to close.
- **Coherency rises with training even though nothing pays for it.**
  Greedy 0.76–0.90 → 0.97–0.98, sampled 0.23–0.62 → 0.70–0.81, on every
  seed. A policy that has converged on "walk straight at the point" keeps
  the squad together as a by-product; a policy still exploring does not.
  The greedy/sampled gap is still 0.17–0.27 and is the number the E rungs'
  referee will meet.
- **Speed, a third time.** Three per-model seeds at 4.88–4.93 turns against
  a script at 4.96 and a control at 6.1–7.4. A0 and A1 together are six
  seeds, all within 0.2 turns of the script; six control seeds, all a
  round or more behind. This is now a pattern about the two trainers on
  this reward, and it goes in the rule, still labelled as movement-only
  rungs with three bodies.
- **The verdict discipline held.** A1 was called FAIL on the letter, the
  follow-up was pre-registered with its own PASS / FAIL / regression
  clauses before it ran, and the regression clause was checked (two
  single-evaluation dips at n=30, no regression). Nothing was re-read
  until the criterion was in git.

### What was not done

- No recording of what s2's third model did in the 21 failing episodes at
  61,440; the question dissolved when it passed.
- The whole-army control was not extended; its 6.1–7.4 turns stand at 60
  epochs.

---

<a id="2026-09-15-curriculum-A2-preregistration"></a>

## Pre-registration: curriculum rung A2 — four squads, one objective

Written 2026-09-15, **before any number from the arm exists**, on branch
`feature/curriculum-a2` (one PR per rung, stacked on A1's PR #344; the
chain does not merge to `main` while exploratory, so the commit timestamp
of this file is what orders it before the numbers). Issue #347, parent
question #340, rung **A2** — axis: **bodies**. A1 (#343) is the rung
below.

### The one change

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

### Comparator, measured first

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

### Criteria

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

### Budget, regime, seeds

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

### Primary readouts

Success rate (decides); turns paired against the script and against the
control; `held`, `on_obj`, coherency; the passive pair and
`train/rounds_per_update` on every per-model row; the health panel over
the last quarter; rounds-to-pass on both arms.

### What I expect (a guess, written so it can be wrong)

Both facades pass, and the per-model arm's `on_obj` is higher than the
script's 0.843 — a learned policy has no reason to re-issue a blocked
order, which is what leaves the script's last bodies at the rim. The
per-model arm again needs 2–4× the control's rounds. Coherency on both
learned arms ends below the bar's 0.889, because nothing pays for it and
twelve bodies on one disc have every reason to spread.

### Amendment 1 — the control's read, and the budget it sets (2026-09-15 00:52, before any per-model number exists)

The whole-army control ran to its 60-epoch cap (Wandb group `curriculum-a2`,
runs `oz1qq4lq` s1 · `0jrgewh3` s2 · `ewun7j0y` s3). In-run success (n=30,
seeds 500000+) first reaches 0.95 and stays there at epoch **9 / 0 / 0**
(0-based; s2 and s3 read 100 on their first evaluation, s1 sat at 0 for
four epochs and dipped to 83 at epoch 8), i.e. rounds-to-pass **20,480 /
2,048 / 2,048**. Under the 6× rule the per-model budget is 6 × 20,480 =
**122,880 rounds**, the cap, launched at that figure.

Scored at n=100 on seeds 700000+ from `last.ckpt` (epoch 60): success
**1.000 / 1.000 / 1.000**, turns **6.01 / 5.95 / 6.00** against the
script's 4.55 (paired +1.46 / +1.40 / +1.45), `on_obj` 0.948 / 0.972 /
0.972 against the script's 0.843 — the learned policy leaves fewer bodies
at the rim — coherent 0.703 / 0.778 / 0.740 against the bar's 0.889. The
control passes the success bound 3 of 3, so a per-model miss reads FAIL
and not NULL. Nothing about the per-model arm was known when this was
written.

---

<a id="2026-09-15-curriculum-a2-solved-then-unlearned"></a>

## Curriculum rung A2: the per-model arm solves the rung by 10k rounds, then unlearns it — FAIL, as pre-registered

**Verdict first.** On the third rung of the ladder (#340, arm #347) — four
squads of three walking onto one point — the per-model trainer reads
greedy success **0.370 / 0.880 / 0.980** at n=100 at the end of its
122,880-round budget, against a whole-army control at **1.000 ×3**. That
is **FAIL** on the pre-registered criteria, and it is the first per-model
fail on the ladder that more rounds would not have fixed: every seed had
**already passed**. Scored from their periodic checkpoints on the same
n=100 seeds, all three read **success 1.000, turns 4.02 (half a round
faster than the script's 4.55), coherency 1.000 at 10,240 rounds**, held
it through 20k, and then drifted — coherency first (1.00 → 0.25–0.48 by
41k–82k), then success. The displacement head's entropy fell from 4.4 to
about 3.3 nats in the first 12k rounds and never sharpened further,
while the clip fraction climbed from 0.15 to 0.26–0.40. The control at
the same entropy coefficient passed within its first ten epochs and held
100% for sixty. A second arm, A2b, is pre-registered and running with
`ent_coef` 0.003 — the setting the record already prefers for the
whole-army trainer on formation — and nothing above A2 launches until it
answers.

### Provenance

| field | value |
|---|---|
| date | 2026-09-15 (runs 00:50–02:38) |
| GPU / no-GPU | GPU (RTX 4090); the A3–A5 controls shared the box for part of the run |
| seeds | 1 / 2 / 3 on both arms; rollout layouts at seed×100+ |
| n | 100 at seed base 700000 (final and periodic); 30 at 500000 (in-run) |
| config | `configs/experiments/curriculum/a2.yaml` — radius 6, success at 75% on the point (9 of 12), unrefereed by design |
| decode | none on the per-model facade; K=1 on the control |
| paired | per episode against `squad_march_take` on identical seeds |
| comparator | `squad_march_take`: success 0.990, turns 4.55, `on_obj` 0.843, coherent 0.889 |
| opponent | none |
| budget | per-model 122,880 rounds at 128 rounds per update (6× the control's slowest rounds-to-pass, 20,480, = the cap); control 60 epochs of 2048 steps |
| code revision | `1024960` on `feature/curriculum-a2` (PR #348, stacked on #344) |
| checkpoint | `last.pt` at 122,880, and `pm-*.pt` at 10,240 / 20,480 / 40,960 / 81,920; `last.ckpt` at epoch 60 |
| coherency | per-model greedy 0.079 / 0.359 / 0.245 at the end (1.000 ×3 at 10k); control 0.703 / 0.778 / 0.740 |
| Wandb | `curriculum-a2`: per-model `oadxfmi7` s1 · `n40pvlrw` s2 · `eo2kfx1a` s3; control `oz1qq4lq` s1 · `0jrgewh3` s2 · `ewun7j0y` s3 |
| pre-registration | `reports/2026-09-15-curriculum-a-rungs.md#2026-09-15-curriculum-A2-preregistration` at `8f66737`; amendment 1 (the control's read and the budget) at `1024960`, before any per-model number |

### The read

**Final, at budget** (n=100, 700000+, greedy):

| row | success | turns | vs bar, paired | held | on obj | coherent | stat |
|---|---|---|---|---|---|---|---|
| `squad_march_take` | 0.990 | 4.55 | — | 1.00 | 0.843 | 0.889 | — |
| per-model s1 | **0.370** | 7.61 | +3.06 ± 0.11 | 1.00 | 0.654 | 0.079 | 0.17 |
| per-model s2 | **0.880** | 5.63 | +1.08 ± 0.12 | 1.00 | 0.737 | 0.359 | 0.00 |
| per-model s3 | **0.980** | 6.13 | +1.58 ± 0.08 | 1.00 | 0.870 | 0.245 | 0.03 |
| whole-army s1 | 1.000 | 6.01 | +1.46 ± 0.07 | 1.00 | 0.948 | 0.703 | — |
| whole-army s2 | 1.000 | 5.95 | +1.40 ± 0.07 | 1.00 | 0.972 | 0.778 | — |
| whole-army s3 | 1.000 | 6.00 | +1.45 ± 0.07 | 1.00 | 0.972 | 0.740 | — |

**The periodic checkpoints, same seeds, same n** — the row the verdict
turns on:

| rounds | s1 success / turns / coherent | s2 | s3 |
|---|---|---|---|
| 10,240 | **1.000** / 4.02 / 1.000 | **1.000** / 4.02 / 1.000 | **1.000** / 4.02 / 1.000 |
| 20,480 | 1.000 / 4.44 / 0.710 | 1.000 / 4.48 / 0.507 | 1.000 / 4.02 / 1.000 |
| 40,960 | 0.900 / 5.30 / 0.359 | 1.000 / 4.15 / 0.966 | 0.940 / 6.86 / 0.247 |
| 81,920 | 0.930 / 6.25 / 0.434 | 1.000 / 5.37 / 0.480 | 1.000 / 6.23 / 0.255 |
| 122,880 | 0.370 / 7.61 / 0.079 | 0.880 / 5.63 / 0.359 | 0.980 / 6.13 / 0.245 |

The three 10k rows are identical to the digit: the greedy policy each
seed had converged on by then is the same straight walk, and on identical
seeds it produces identical episodes.

**In-run** (n=30, 500000+): 100% with coherency 1.00 on every seed from
6k rounds; s1 breaks at ~25k, s2 at ~37k, s3 at ~25k (to 20–63%), and
none holds 95% again for long — s3 recovers to 100% in-run from ~68k but
reads 0.980 at n=100.

**Greedy against sampled** (900000+, n=100, paired): +0.7 ± 0.4 / −5.5 ±
0.4 / −2.6 ± 0.4 vp — on two seeds the *sampled* policy scores better
than the argmax, which A0 and A1 never showed; coherency greedy 0.08 /
0.37 / 0.28 against sampled 0.04 / 0.04 / 0.10; decisions per episode
89–104 greedy against 122–128 sampled.

**Health panel, last quarter** (240 updates per seed): advantage std
0.16–0.22, explained variance **0.88–0.90** (the ladder's highest — the
critic is not the problem), ratio p99 1.70–1.93, clip fraction
**0.26–0.37** (the ladder's highest), declaration entropy 0.08–0.18,
displacement entropy **2.98–3.31 of a 4.57 ceiling**, `eval/stationary_share`
0.00–0.21 (s1 ends at 0.17 — some units decline to move).

**Over the run** (every 96th update): displacement entropy s1 4.40 → 3.31
(12k) → 3.28 (110k); s2 4.38 → 3.44 → 3.19; s3 4.42 → 3.26 → 2.87. Clip
fraction s1 0.36 → 0.13 (12k) → 0.25; s2 0.48 → 0.15 → 0.23; s3 0.37 →
0.15 → 0.36.

### What it says

- **The per-model trainer learns this rung faster and better than
  anything on the ladder so far, and then does not hold it.** At 10k
  rounds every seed beats the script's own speed with perfect formation.
  The control needs 2k–20k rounds to pass and never drops. The failure
  is not learning; it is **staying learned**.
- **The signature is entropy, not the critic.** Explained variance is
  0.88–0.90 throughout the last quarter; the value function knows the
  task. The displacement head plateaus at ~3.3 nats — of a 97-way head's
  4.57 — from 12k rounds onward and never sharpens, while coherency,
  which only a sharp shared heading can produce, decays from 1.00. Once
  every episode succeeds the advantage signal has little variance and
  the entropy bonus does not; at `ent_coef` 0.03 the update is left
  optimising the bonus. The whole-army trainer's one head per model
  under a shared trunk tolerated the same coefficient; the per-model
  network sums three heads' entropies under it.
- ⚠ **Reading only at the end of the budget is what made this visible,
  and also what made it a fail.** A0 and A1 read at budget too; their
  policies did not drift. A rule that read "first pass and hold" would
  have called A2 a pass at 10k and never seen the drift. The
  pre-registration's clause is right — a policy that cannot hold a solved
  rung has not passed it — and A2b's PASS clause now says so explicitly:
  ≥ 0.95 at the end **and** no dip below 0.80 after the first pass.
- **The clip fraction is the panel row that tracked it.** Ratio p99 read
  1.7–1.9 (the regime's normal); clip fraction rose from 0.13–0.15 at 12k
  to 0.23–0.37 by the end, the ladder's highest, as the policy drifted.
  Read the two together: p99 alone did not move with the failure, clip
  fraction did.
- **The stationary share re-appeared.** s1 ends with 0.17 of unit opens
  declared stationary, the passive fingerprint the void runs showed at
  0.94. Here it is a symptom of the drift, not its cause — it was 0.00
  through the first 60k rounds.

### What happens next

**A2b** (#356), pre-registered at `c2bcf26`
before it ran and launched the same minute: the same three seeds, the
same scenario, budget and regime, with `--ent-coef 0.003`. PASS is 3/3 at
the end **and** no in-run dip below 0.80 after the first pass; FAIL says
the drift is not the entropy bonus and the update itself (the clip
fraction) is the next suspect. The ladder does not open A3's per-model
arm until A2b is read — A3's and A4's and A5's controls have already run
and are recorded on their pre-registrations.

### What was not done

- No recording of what a drifted policy does (which squad walks apart,
  whether the stationary units are the same ones).
- No sweep of `ent_coef`; A2b is one value, the one the record already
  measured on the other trainer.
- The control was not extended; it never needed to be.

---

<a id="2026-09-15-curriculum-A2b-preregistration"></a>

## Pre-registration: curriculum rung A2, second arm (A2b) — the entropy coefficient at 0.003

Written 2026-09-15, **before any number from the arm exists**, on branch
`feature/curriculum-a2` (PR #348 carries the rung's arms). Issue #356,
parent question #340, rung **A2** — a follow-up to #347, which read
**FAIL as pre-registered**: greedy success **0.370 / 0.880 / 0.980** at
n=100 at 122,880 rounds against a whole-army control at 1.000 ×3.

### Why this arm

A2's three per-model seeds all read 100% in-run success with coherency
1.00 by 6k rounds and held it to 18k–31k; then coherency collapsed to
0.1–0.4 and success drifted (s1 down to 0–57 in its last quarter, s2
oscillating 43–100, s3 recovering to 100 in-run but 0.980 at n=100). The
displacement head's entropy fell from 4.4 to ~3.3 nats in the first 12k
rounds and never sharpened further — 3.0–3.4 of a 4.57 ceiling for the
whole run — while the clip fraction rose from 0.15 to 0.26–0.40. Once
every episode succeeds the advantage signal is small and the entropy
bonus is not; at `ent_coef` 0.03 over a 97-way head the bonus is what
the update is left optimising. The whole-army control, at the same
coefficient, passed by epoch 0–9 and held 100% for 60 epochs; its policy
is one 97-way head per model on a shared trunk, the per-model network's
is a declaration, a displacement and a unit pointer, and the sum of head
entropies is what `ent_coef` scales here.

The record already says 0.003 is the better setting for the whole-army
trainer on formation (`CLAUDE.md` § Coherency: paired +5.9 ± 2.5, and it
"concentrated the policy exactly as predicted"). This arm asks the same
of the per-model trainer on the first rung where its policy drifted.

### The one change

`--ent-coef 0.003` (default 0.03). Same scenario and config
(`configs/experiments/curriculum/a2.yaml`, unchanged), same seeds, same
regime (128 rounds per update), same budget (122,880 rounds), same
in-run and final seed bands. Nothing else varies.

### Comparator

`squad_march_take` on `a2.yaml`, seeds 700000+ at n=100: success 0.990,
turns 4.55, `on_obj` 0.843, coherent 0.889. Beside it, A2's own read at
`ent_coef` 0.03: success 0.370 / 0.880 / 0.980, coherent 0.079 / 0.359 /
0.245; and the whole-army control: 1.000 ×3, turns 5.95–6.01, coherent
0.70–0.78.

### Criteria

Read at 122,880 rounds from `last.pt`, greedy, no decode, n=100 on seeds
700000+ (`just measure-rung`).

- **PASS:** success ≥ 0.95 on **all three** seeds at the end of the
  budget, **and** no seed's in-run success (n=30, 500000+) drops below
  0.80 on any evaluation after its first pass — the drift A2 showed does
  not recur. The entropy setting was the cause; the ladder continues to
  A3 with `--ent-coef 0.003` on every per-model arm from here, recorded
  as an amendment to #340 contract 3.
- **PASS with drift:** ≥ 0.95 on all three at the end but a dip below
  0.80 after the first pass on some seed — the setting helps and does not
  cure; reported as such and the ladder continues with a watch on the
  in-run curve.
- **FAIL:** any seed below 0.95 at the end — the drift is not the entropy
  bonus. The next suspect is the update itself (clip fraction 0.26–0.40
  on A2, the ladder's highest), and the ladder stops at A2 until it is
  found.
- **Readouts:** the in-run curve after the first pass; coherency greedy
  and sampled; the displacement head's entropy over the run; turns paired
  vs the script and the control; the paired-by-seed difference against
  A2's read (same seeds, same init, one scalar).

Power: binomial SE 0.022 at p=0.95, n=100. A2's seeds sit 0.370 / 0.880 /
0.980, so a 3/3 pass is a move on two of them and a hold on the third.

### Budget, regime, seeds

| | per-model arm |
|---|---|
| budget | 122,880 rounds, 128 rounds per update (`--rollout-rounds 32 --num-rollout-envs 4`) |
| seeds | 1, 2, 3; rollout layouts at seed×100+ |
| in-run eval | every 512 rounds, n=30, seeds 500000+, greedy, passive pair and `eval/mean_turns` logged |
| final score | n=100, seeds 700000+ |
| flags | `--ent-coef 0.003`; otherwise defaults (`gamma` 0.9, `lr` 3e-4) |
| logging | Wandb group `curriculum-a2`, tag `a2b` |

Launch: `just train-per-model-arm 122880 3 curriculum-a2 a2b "--num-rollout-envs 4 --rollout-rounds 32 --eval-every-rounds 512 --checkpoint-every-rounds 512 --n-eval-episodes 30 --ent-coef 0.003" configs/experiments/curriculum/a2.yaml`.

### What I expect (a guess, written so it can be wrong)

All three seeds pass by ~10k rounds as before, the displacement entropy
keeps falling past 3 nats instead of plateauing, coherency stays near
1.0, and the final read is ≥ 0.95 on all three. If the drift is the
update rather than the bonus, the curve will look like A2's with a
delay.

---

<a id="2026-09-15-curriculum-a2b-passes-with-drift"></a>

## Curriculum rung A2, second arm: at `ent_coef` 0.003 the per-model arm holds the rung — PASS with drift

**Verdict first.** The same three A2 seeds, scenario, regime and budget
with the head entropy coefficient at **0.003** instead of 0.03 (#356)
read greedy success **1.000 / 1.000 / 0.990** at n=100 at the end of
122,880 rounds, turns **4.48 / 4.19 / 4.11** against the script's 4.55,
coherency 0.74–0.79 — where A2 at 0.03 had drifted to 0.370 / 0.880 /
0.980. The displacement head's entropy in the last quarter is **0.31–0.39
nats** against A2's 2.98–3.31, and the clip fraction **0.12–0.15**
against 0.26–0.37. The pre-registered PASS clause also required no in-run
dip below 0.80 after the first pass, and every seed had some: 8 / 10 / 7
of ~230 evaluations, including two transient collapses to 0 at n=30 (s2
at 17–19k, s3 at 35–36k) from which the policy recovered within a few
thousand rounds. On the pre-registered clauses that is **PASS with
drift**: the setting is the cause of the terminal drift A2 showed, and
it does not make the policy monotone. The ladder continues with
`--ent-coef 0.003` on every per-model arm and the in-run curve watched.

### Provenance

| field | value |
|---|---|
| date | 2026-09-15 (runs 02:44–04:08) |
| GPU / no-GPU | GPU (RTX 4090), the box to itself |
| seeds | 1 / 2 / 3; rollout layouts at seed×100+ |
| n | 100 at seed base 700000 (final); 30 at 500000 (in-run) |
| config | `configs/experiments/curriculum/a2.yaml`, unchanged; unrefereed by design |
| decode | none |
| paired | per episode against `squad_march_take`; by seed against A2's read (same seeds, same init, one scalar) |
| comparator | `squad_march_take`: success 0.990, turns 4.55, coherent 0.889; A2 at `ent_coef` 0.03: 0.370 / 0.880 / 0.980; the control: 1.000 ×3 |
| opponent | none |
| budget | 122,880 rounds at 128 rounds per update |
| code revision | `c2bcf26` on `feature/curriculum-a2` (PR #348) |
| checkpoint | `last.pt` |
| coherency | greedy 0.742 / 0.788 / 0.747; sampled 0.620 / 0.713 / 0.688 |
| Wandb | `curriculum-a2`: `ynqw9xqj` s1 · `shly1te0` s2 · `hue4q51a` s3 |
| pre-registration | `reports/2026-09-15-curriculum-a-rungs.md#2026-09-15-curriculum-A2b-preregistration` at `c2bcf26`, before the launch |

### The read

| row | success | turns | vs bar, paired | held | on obj | coherent greedy / sampled | first pass | dips < 0.80 after |
|---|---|---|---|---|---|---|---|---|
| `squad_march_take` | 0.990 | 4.55 | — | 1.00 | 0.843 | 0.889 / — | — | — |
| A2 s1 (0.03) | 0.370 | 7.61 | +3.06 | 1.00 | 0.654 | 0.079 / 0.038 | ~6k | terminal |
| **A2b s1 (0.003)** | **1.000** | 4.48 | −0.07 ± 0.08 | 1.00 | 0.865 | 0.742 / 0.620 | 512 | 8 of 239 (worst 20 at 28k; last at 52k) |
| A2 s2 (0.03) | 0.880 | 5.63 | +1.08 | 1.00 | 0.737 | 0.359 / 0.040 | ~6k | terminal |
| **A2b s2 (0.003)** | **1.000** | 4.19 | −0.36 ± 0.07 | 1.00 | 0.854 | 0.788 / 0.713 | 6.7k | 10 of 227 (0 / 0 / 0 / 10 / 7 at 17–19k; last at 63k) |
| A2 s3 (0.03) | 0.980 | 6.13 | +1.58 | 1.00 | 0.870 | 0.245 / 0.103 | ~6k | terminal |
| **A2b s3 (0.003)** | **0.990** | 4.11 | −0.44 ± 0.08 | 1.00 | 0.827 | 0.747 / 0.688 | 4.6k | 7 of 231 (0 / 0 / 63 at 35–36k; last at 51k) |

Greedy − sampled on the tuning band (900000+, n=100): −0.7 ± 0.3 / −1.2
± 0.3 / −0.8 ± 0.3 vp, win 100% both ways — the two policies are the same
policy again, as on A0 and A1.

**Health panel, last quarter** (240 updates per seed): displacement
entropy **0.34 / 0.39 / 0.31** (A2: 3.31 / 3.06 / 2.98), declaration
entropy 0.006 / 0.010 / 0.019, clip fraction **0.15 / 0.14 / 0.12** (A2:
0.26 / 0.28 / 0.37), ratio p99 1.71 / 1.65 / 1.55, explained variance
0.71 / 0.72 / 0.70 (A2: 0.90 — lower here because the return has less
variance to explain once success is near-certain), advantage std
0.17–0.25.

### What it says

- **The entropy bonus was the cause of A2's terminal drift.** One scalar,
  same seeds and init: the head sharpens by a factor of ten, the clip
  fraction halves, and the final read goes from 0.37 / 0.88 / 0.98 to
  1.00 / 1.00 / 0.99 at the script's speed or better. The record's
  whole-army finding (0.003 "concentrated the policy exactly as
  predicted") transfers to the per-model trainer.
- **It is not monotone, and the pre-registration was right to ask.** The
  in-run curve still dips — three to ten evaluations per seed below 0.80
  in 230, two of them full transient collapses at n=30 — and recovers
  every time, with no dip after 63k rounds on any seed. A2's dips did not
  recover. The distinction the PASS-with-drift clause drew is the one the
  data drew.
- **From here every per-model arm runs at `--ent-coef 0.003`**, recorded
  as an amendment to #340 contract 3 and to A3's pre-registration before
  its per-model arm launches. Whether the whole-army control should move
  too is a separate arm; its passes at 0.03 stand.
- **Coherency is the number the E rungs will price.** Greedy 0.74–0.79,
  sampled 0.62–0.71, against a bar at 0.889 and A2's 10k checkpoints at
  1.000: the sharpened policy is faster than the script and less coherent
  than its own early self. Nothing pays for formation on the A rungs.

### What was not done

- No sweep between 0.003 and 0.03; no lower value.
- The control was not re-run at 0.003.
- No recording of the transient collapses.

---

<a id="2026-09-15-curriculum-A3-preregistration"></a>

## Pre-registration: curriculum rung A3 — four squads, four objectives, the spread rung

Written 2026-09-15, **before any number from the arm exists**, on branch
`feature/curriculum-a3` (one PR per rung, stacked on A2's PR #348; the
chain does not merge to `main` while exploratory, so the commit timestamp
of this file is what orders it before the numbers). Issue #349, parent
question #340, rung **A3** — axis: **points**. A2 (#347) is the rung
below and is still training when this is written; A3's per-model arm does
not launch until A2 has been read, and not at all if A2 fails.

### The one change

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

### Comparator, measured first

`squad_march_take`, scored through both facades on seeds 700000+ at n=100
(`just measure-bridge configs/experiments/curriculum/a3.yaml 100`), bridge
identical on every shared field: success **1.000**, `held` **4.00** of 4,
turns **5.28** (sd 0.81, range 4–7), `on_obj` 0.892, coherent **0.927**,
vp 31.6 ± 1.2. In-run band (n=30, 500000+): success 1.000, turns 5.13,
held 4.00, coherent 0.970.

### Criteria

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

### Budget, regime, seeds

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

### Primary readouts

Success rate (decides); turns paired against the script and against the
control; `held`, `on_obj`, the spread statistic, coherency; the passive
pair and `train/rounds_per_update` on every per-model row; the health
panel over the last quarter; rounds-to-pass on both arms.

### What I expect (a guess, written so it can be wrong)

Both facades pass. The per-model arm's turns land near the script's 5.28
again and the control's a round slower. The spread statistic is high on
both — the deployment band makes the nearest point unambiguous and the
progress reward pays each squad to close on its own nearest point — so
this rung will not separate the facades on allocation; A4's spare squad
is where that question lives.

### Amendment 1 — the control's read at the cap, and what follows (2026-09-15 01:10, before any per-model number exists)

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

### Amendment 2 — the control's read at 120 epochs (2026-09-15, before any per-model number exists)

Resumed from its epoch-60 checkpoints to 120 epochs (Wandb `curriculum-a3`,
runs `p9znz3i5` s1 · `sbh0lltv` s2 · `mror78kv` s3, suffix `-x2`), scored
at n=100 on seeds 700000+ from `last.ckpt`: success **0.950 / 0.950 /
0.960**, `held` 3.95 ×3, turns 6.26 / 6.33 / 6.24 against the script's
5.28 (paired +0.98 / +1.05 / +0.96), coherent 0.735 / 0.785 / 0.688
against the bar's 0.927. **The control passes the success bound on 3 of
3 seeds at its extended budget**, so amendment 1's clause 3 makes a
per-model miss read FAIL, not NULL. Its in-run rounds-to-pass is read from
Wandb once no run is training (the A2 per-model arm is live) and goes in
the report; the per-model budget is the cap regardless. Nothing about the
per-model arm was known when this was written.

### Amendment 3 — the entropy coefficient, and the launch (2026-09-15, before any per-model number exists)

A2 (#347) read FAIL at the default `ent_coef` 0.03 — every seed solved its
rung by 10k rounds and unlearned it by 123k — and A2b (#356) at 0.003
read PASS with drift on the same seeds ([report](2026-09-15-curriculum-a-rungs.md#2026-09-15-curriculum-a2b-passes-with-drift)).
#340 contract 3 is amended: every per-model arm from A3 on runs
`--ent-coef 0.003`. The "other flags" row above therefore reads `gamma`
0.9, `lr` 3e-4, **`ent_coef` 0.003**; nothing else changes. A2b's strict
clause is adopted as a readout here: the in-run curve after the first
pass, with any dip below 0.80 reported.

The per-model arm launches now at the cap, **122,880 rounds** (amendment
1, clause 2), the control having passed 3 of 3 at 120 epochs (amendment
2; rounds-to-pass 118k / 237k / 209k — s2 and s3 never held 0.95 in-run
at n=30 until epochs 115 and 103, s1 never did, while all three read
≥ 0.95 at n=100). A2 (#347 / #356) has been read, so the ladder's order
holds. Nothing about the per-model arm was known when this was written.

---

<a id="2026-09-15-curriculum-a3-behind-the-control-at-the-cap"></a>

## Curriculum rung A3: the per-model arm is behind the control on the spread rung at the cap — FAIL, as pre-registered

**Verdict first.** On the spread rung (#340, arm #349) — four squads of
three, four objectives in a column, success only when every point has a
body on it — the per-model trainer at the 122,880-round cap reads greedy
success **0.700 / 0.800 / 0.770** at n=100, `held` **3.49 / 3.60 / 3.65**
of 4, every seed still climbing (last six in-run evaluations 53–90%).
The whole-army control read 0.850 / 0.980 / 0.930 at the same rounds and
**0.950 / 0.950 / 0.960** after its once-only extension to 245,760, so
under the pre-registration (amendment 1, clause 3) this is **FAIL**. It
is the first rung on which the per-model arm is behind the control at
equal rounds, and the first on which the control itself needed twice the
cap. The pre-registered follow-up, A3x (#357), resumes the same three
runs to the control's 245,760 rounds; the ladder waits on it.

### Provenance

| field | value |
|---|---|
| date | 2026-09-15 (per-model 04:12–05:37; control 00:54–01:50 with its extension) |
| GPU / no-GPU | GPU (RTX 4090), the per-model arm with the box to itself |
| seeds | 1 / 2 / 3 on both arms; rollout layouts at seed×100+ |
| n | 100 at seed base 700000 (final); 30 at 500000 (in-run) |
| config | `configs/experiments/curriculum/a3.yaml` — unrefereed by design; success `all_objectives_occupied` |
| decode | none on the per-model facade; K=1 on the control |
| paired | per episode against `squad_march_take` on identical seeds |
| comparator | `squad_march_take`: success 1.000, held 4.00, turns 5.28, coherent 0.927 |
| opponent | none |
| budget | per-model 122,880 rounds at 128 rounds per update, `ent_coef` 0.003 (A2b's amendment); control 60 epochs extended once to 120 |
| code revision | `a6afc74` on `feature/curriculum-a3` (PR #350) |
| checkpoint | `last.pt` at 122,880; `last.ckpt` at epochs 60 and 120 |
| coherency | per-model greedy 0.566 / 0.480 / 0.442, sampled 0.281 / 0.276 / 0.221; control 0.735 / 0.785 / 0.688 at 120 |
| Wandb | `curriculum-a3`: per-model `wzalbuic` s1 · `4x0p1q4a` s2 · `jzmpi6pl` s3; control `78l7l9u6` / `7rbs29pq` / `z2arg0vf` (to 60), `p9znz3i5` / `sbh0lltv` / `mror78kv` (60–120) |
| pre-registration | `reports/2026-09-15-curriculum-a-rungs.md#2026-09-15-curriculum-A3-preregistration` at `070c422`; amendments 1–3 at `471eed2`, `14c8db8`, `a6afc74`, each before the number it precedes |

### The read

| row | success | turns | vs bar, paired | held | on obj | coherent greedy / sampled | stat |
|---|---|---|---|---|---|---|---|
| `squad_march_take` | 1.000 | 5.28 | — | 4.00 | 0.892 | 0.927 / — | — |
| per-model s1 | **0.700** | 6.05 | +0.77 ± 0.16 | 3.49 | 0.654 | 0.566 / 0.281 | 0.09 |
| per-model s2 | **0.800** | 5.95 | +0.67 ± 0.13 | 3.60 | 0.652 | 0.480 / 0.276 | 0.00 |
| per-model s3 | **0.770** | 5.81 | +0.53 ± 0.14 | 3.65 | 0.740 | 0.442 / 0.221 | 0.01 |
| control s1 at 123k | 0.850 | 6.72 | +1.44 | 3.83 | 0.855 | 0.770 | — |
| control s2 at 123k | 0.980 | 6.47 | +1.19 | 3.98 | 0.848 | 0.733 | — |
| control s3 at 123k | 0.930 | 6.52 | +1.24 | 3.92 | 0.865 | 0.668 | — |
| control s1 at 245k | 0.950 | 6.26 | +0.98 | 3.95 | 0.868 | 0.735 | — |
| control s2 at 245k | 0.950 | 6.33 | +1.05 | 3.95 | 0.875 | 0.785 | — |
| control s3 at 245k | 0.960 | 6.24 | +0.96 | 3.95 | 0.883 | 0.688 | — |

In-run (n=30, 500000+), per-model: 40–47% with `held` 1.6–2.1 through
~30k rounds; `held` 3.2–3.8 from ~70k; the last six evaluations 53–73 /
80–90 / 70–87%. Control in-run rounds-to-pass: s1 never at n=30, s2 at
epoch 115, s3 at 103 — the control's n=30 curve wobbled at 83–97 for
sixty epochs while its n=100 read cleared 0.95.

**Greedy against sampled** (900000+, n=100): −1.2 ± 1.6 / −0.3 ± 1.6 /
+0.1 ± 1.3 vp; `held` 3.47 / 3.69 / 3.74 greedy against 3.43 / 3.63 /
3.76 sampled; coherency 0.52 / 0.51 / 0.47 against 0.28 / 0.28 / 0.22.

**Health panel, last quarter** (240 updates per seed): displacement
entropy 1.75 / 1.72 / 1.81 (neither collapsed nor sharp), declaration
0.05, clip fraction 0.22 / 0.32 / 0.32, ratio p99 1.72 / 2.02 / 1.97,
explained variance **0.34 / 0.45 / 0.43** — the ladder's lowest —
advantage std 0.57–0.63.

### What it says

- **The per-model arm is behind the control on this rung at equal
  rounds, and it is the first time.** A0 and A1 it matched or beat on
  speed and needed 2–4× the rounds; A2 it solved faster and then lost;
  here at 123k it holds 3.5–3.7 points where the control held 3.8–4.0,
  and its success is 0.70–0.80 against the control's 0.85–0.98. Both
  trainers find allocation over four points hard; the whole-army one
  found it less hard.
- **The bodies arrive; one point stays short.** `on_obj` 0.65–0.74 with
  `held` 3.5–3.7: in the failing episodes the squads are on points, just
  not on all four. That is the spread question the rung was built to
  ask, and the per-model arm has not answered it at the cap. Whether it
  does with the control's rounds is A3x's question; whether it sends two
  squads to one point is the recording that follows a FAIL there.
- **The critic is the panel row that moved.** Explained variance 0.34–0.45
  against 0.70–0.90 on every earlier rung: the value of a four-point
  episode, where success is a conjunction over points, is harder to
  predict from a decision's state, and the advantage std doubled. That
  is the per-step credit assignment #283 argued for, meeting its first
  conjunction.
- **The cap rule was written for a control that passes inside it.** A3's
  control needed 245k rounds and was granted them by amendment; the arm
  under test was read at 123k against that. The A3x pre-registration
  makes the extension symmetric from here.
- **Coherency keeps falling down the ladder**: greedy 0.44–0.57 here
  (A2b 0.74–0.79, A1x 0.97), sampled 0.22–0.28. Unpaid on the A rungs;
  the E rungs' referee will price it.

### What was not done

- No recording of the failing episodes yet (which point, and whether two
  squads share one).
- No sweep of anything; A3x changes only the budget.

---

<a id="2026-09-15-curriculum-A3-speed-preregistration"></a>

## Pre-registration: three arms at why the spread rung learns slowly (A3 speed screen)

Written 2026-09-15, **before any number from any arm exists**, on branch
`feature/curriculum-a3-speed` (stacked on PR #354). Parent question #340,
rung **A3**. One issue per arm; this file carries the three criteria and
the reads that motivate them, because the three share one comparator and
one launch.

The ladder is paused (the user's instruction, 2026-09-15). These arms are
the investigation, not the ladder: nothing here changes A3's verdict, and
a lever that pays here is a proposal for the rungs above, not a rewrite of
the rungs below.

### The reads that motivate the arms (no GPU, 2026-09-15, A3's own board)

All on `configs/experiments/curriculum/a3.yaml`, greedy, seeds 700000+,
the census scripts in the session scratchpad (a copy of the travel-gate
instrumentation from `scripts/measure_shaping_gates.py`, run without the
map tables it always overlays, plus a by-turn read of bodies on objectives
and points held).

**1. The rung is not the rung its config describes.** The comment says
each squad deploys nearest a distinct point. Deployment is uniform along a
36-inch band, and in **24 of 200** deployments (12%) are the four squads'
nearest points four distinct points. In the other 88% at least one squad
must cross to a point that is nobody's nearest. A3 is an assignment
problem from a random start, every episode.

**2. The travel term points half the army at the wrong thing.**
`closest_objective_v2` assigns each candidate objective to the group with
the closest model, so one group can own several points and the group left
without one drops through `fallback_to_nearest` onto a point somebody
else is already taking. The scripted bar, n=100: one unit owns two or more
objectives on **49.1%** of steps; **50.1%** of model-steps are paid toward
the nearest point by fallback rather than toward an assigned one; squads
are paid toward different targets on 11.3% of squad-steps. The learned
policy (A3x `last.pt`, three seeds) reads the same gate numbers (51–61%,
44–50%) and splits its squads on **20–32%** of squad-steps.

**3. The bottleneck along the path is bodies that arrive and stay, not
allocation between points.** Seed 1's periodic checkpoints, n=30, bodies
on objectives (of 12) and points held (of 4) at turn 5 then turn 8:

| rounds | success | on objectives, turn 5 → 8 | held, turn 5 → 8 |
|---|---|---|---|
| 20,480 | 0.13 | 5.0 → **2.3** | 1.9 → **1.3** |
| 61,440 | 0.10 | 3.2 → 4.4 | 1.8 → 2.1 |
| 122,880 | 0.77 | 8.2 → 8.0 | 3.5 → 3.7 |
| 184,320 | 0.90 | 8.5 → 8.6 | 3.5 → 3.8 |
| 245,760 (n=100) | 0.96 | 9.6 → 9.7 | 3.9 → 3.9 |
| `squad_march_take` | 1.00 | 9.9 → 10.7 | 3.7 → 4.0 |

At 20k rounds the policy reaches points and **walks off them**. Bodies per
held point stay at 2.2–2.4 throughout, the script's own ratio, so there is
no stack-then-peel signature; the prediction that there would be one was
made before this read and is refuted by it. What the travel term pays
saturates at the boundary (progress zero once inside), the coverage term
is 0.3 over four points broadcast to twelve bodies alike, and the terminal
bonus never fires until all four are held. Nothing pays a body for
standing on a point it reached.

**4. A3's own in-run curve, the speed comparator** (n=30 at 500000+ every
512 rounds, rolling mean of five): rounds to 50% — 38,400 / 23,040 /
36,864; rounds to 80% — never / 90,624 / 101,888; rounds to 95% — never
on any seed inside 122,880. Final read at 122,880, n=100: **0.700 / 0.800
/ 0.770**, held 3.49 / 3.60 / 3.65, turns 6.05 / 5.95 / 5.81.

### The three arms

Each is `a3.yaml` with **one** change, three seeds, **122,880 rounds** at
128 rounds per update (`--rollout-rounds 32 --num-rollout-envs 4`),
`--ent-coef 0.003`, in-run eval every 512 rounds at n=30 on 500000+,
checkpoints every 512, Wandb group `curriculum-a3-speed`. The comparator
for all three is **A3's own three runs at 122,880** (Wandb `curriculum-a3`:
`wzalbuic` / `4x0p1q4a` / `jzmpi6pl`), read on identical seeds.

| arm | tag | the one change | config | paired on init? |
|---|---|---|---|---|
| **S1 hold** | `a3h` | `objective_hold` weight 1.0, `crowding_exponent` 1.0, added beside the two terms A3 has | `curriculum/a3_hold.yaml` | yes — same seed, same head, so the init is A3's |
| **S2 match** | `a3m` | `closest_objective_v2` `one_objective_per_group: true` (new flag, default off, shipped with this branch) | `curriculum/a3_match.yaml` | yes |
| **S3 warm** | `a3w` | `--warm-start-from` the A2b checkpoint of the same seed (`per-model-a2-2026-09-15-02-44-23-s{1,2,3}a2b/last.pt`) | `curriculum/a3.yaml` unchanged | **no** — the init is A2b's; layouts and seeds shared |

Bar checks before launch: `squad_march_take` on `a3_hold.yaml` reads
1.000 / held 4.00 / 5.28 turns, identical to `a3.yaml` (the term does not
touch the script). On `a3_match.yaml` the gate census reads one unit
owning two or more objectives on **0.0%** of steps (was 49.1%), squads
split on 1.1% (was 11.3%), every squad given a point of its own on the
first paid step in 100 of 100 episodes (was 9 of 100). A warm start from
A2b's `last.pt` into `a3.yaml` loads and trains (smoke, 256 rounds).

### Criteria, per arm, identical

Read at 122,880 rounds from `last.pt`, greedy, no decode, n=100 on
700000+, beside A3's same-seed read.

- **FASTER:** success ≥ 0.95 on **all three** seeds — the arm passes the
  rung inside the cap where A3 from scratch did not.
- **AHEAD:** not FASTER, but every seed ≥ its A3 read + 0.10. Binomial SE
  at p≈0.75 is 0.043 per seed, so +0.10 is 2.3 SE and a null lever clears
  it on 3 of 3 with probability under 1%.
- **NULL:** otherwise.
- **HARMFUL:** any seed below its A3 read by more than 0.10.

Readouts, all of them, none decides: rounds to rolling-5 in-run 50 / 80 /
95% beside A3's; `held`, `on_obj`, turns; coherency greedy and sampled
(`just measure-per-model-eval-mode`); the by-turn census at 20,480 /
61,440 / 122,880 (the walk-off signature at 20k: bodies on objectives
turn 5 → 8); the gate census on `last.pt`; the health panel over the last
quarter, explained variance in particular (A3: 0.34–0.45 at the cap).

### Predictions, written so they can be wrong

- **S1 hold** is the one aimed at the mechanism read 3 found. Prediction:
  the turn-5-to-8 drop in bodies on objectives is gone at 20k rounds, and
  the arm is **FASTER** on at least two seeds, with rounds to 80% under
  60k on all three.
- **S2 match** fixes read 2. Prediction: **AHEAD** but not FASTER, and the
  larger effect is on coherency — squad-steps split under 5%, greedy
  coherency above 0.6 (A3: 0.38–0.52).
- **S3 warm** starts from a policy that walks everyone to one point.
  Prediction: faster to the first bodies on objectives, then **NULL** on
  success — an A2b policy has to unlearn stacking, and the transfer
  question (T1) is better asked from A4 to A5 as the ladder says.

### What this screen cannot say

- Nothing about the rungs above: a term that pays here is a proposal for
  A4/A5, to be pre-registered there.
- S1 changes the reward, so its episode reward is not comparable to A3's;
  success, held and turns are.
- S3 is unpaired on init. Its per-seed differences carry init variance the
  other two do not.
- Three seeds at n=100 is a screen. A lever that reads AHEAD on one seed
  and NULL on two is a null.

---

<a id="2026-09-15-curriculum-a3-speed-s1-hold"></a>

## A3 speed screen, arm S1: paying for standing on a point moves two seeds of three — NULL on the pre-registered letter

**Verdict first.** `a3.yaml` plus `objective_hold` at `crowding_exponent`
1.0 (#359), three seeds, 122,880 rounds, read greedy at n=100 on
700000+: success **0.940 / 0.810 / 0.900** against A3's own runs at the
same rounds and seeds, **0.700 / 0.800 / 0.770**. Seeds 1 and 3 are ahead
by 0.24 and 0.13; seed 2 by 0.01. That misses FASTER (0.95 on all three)
and AHEAD (+0.10 on all three), so the pre-registered verdict is
**NULL** — and the shape of the miss matters: two seeds moved by two to
five binomial SE, one did not move at all. The prediction on file
("FASTER on at least two seeds, rounds to 80% under 60k on all three")
is wrong on both counts: no seed passes at n=100, and rounds to 80% were
92k / 80k / 62k.

### Provenance

| field | value |
|---|---|
| date | 2026-09-15 (launched 10:00, exited 15:16–15:22) |
| GPU / no-GPU | GPU (RTX 4090), shared with eleven other per-model runs (the S2, S3 and A5b arms) |
| seeds | 1 / 2 / 3; rollout layouts at seed×100+; paired on init with A3 (same seed, same head) |
| n | 100 at seed base 700000 (final); 30 at 500000 (in-run, every 512 rounds); 100 at 900000 (greedy against sampled) |
| config | `configs/experiments/curriculum/a3_hold.yaml` — `a3.yaml` with the one added term; unrefereed by design |
| decode | none |
| paired | per episode against `squad_march_take`; per seed against A3's read at 122,880 |
| comparator | A3's per-model runs at 122,880 (Wandb `curriculum-a3`: `wzalbuic` / `4x0p1q4a` / `jzmpi6pl`): 0.700 / 0.800 / 0.770, held 3.49 / 3.60 / 3.65, turns 6.05 / 5.95 / 5.81; the bar `squad_march_take`: 1.000, held 4.00, 5.28 turns, coherent 0.927 (identical on `a3_hold.yaml`) |
| opponent | none |
| budget | 122,880 rounds at 128 rounds per update (`--rollout-rounds 32 --num-rollout-envs 4`), `--ent-coef 0.003` |
| code revision | launched from the `bd15db9` working tree of `feature/curriculum-a3-speed` (PR #362), rebased the same day to `71c0f8f` with an identical tree |
| checkpoint | `last.pt` at 122,880; `checkpoints/per_model/per-model-a3_hold-2026-09-15-10-00-58-s{1,2,3}a3h` |
| coherency | greedy 0.453 / 0.429 / 0.280 at 700000+; 0.41 / 0.44 / 0.29 greedy against 0.23 / 0.22 / 0.16 sampled at 900000+ |
| Wandb | `curriculum-a3-speed`: `g55n5x9h` s1 · `1eygass8` s2 · `ud74ov2g` s3 |
| pre-registration | `reports/2026-09-15-curriculum-a-rungs.md#2026-09-15-curriculum-A3-speed-preregistration` at `bd15db9` (10:00, before the launch) |

### The read

| row | success | turns | vs bar, paired | held | on obj | coherent | rounds to rolling 50 / 80 / 95% |
|---|---|---|---|---|---|---|---|
| `squad_march_take` | 1.000 | 5.28 | — | 4.00 | 0.892 | 0.927 | — |
| A3 s1 / s2 / s3 at 122,880 | 0.700 / 0.800 / 0.770 | 6.05 / 5.95 / 5.81 | +0.77 / +0.67 / +0.53 | 3.49 / 3.60 / 3.65 | 0.65–0.74 | 0.57 / 0.48 / 0.44 | 38k, never, never / 23k, 91k, never / 37k, 102k, never |
| **S1 s1** | **0.940** | 5.56 | +0.28 ± 0.11 | 3.93 | 0.738 | 0.453 | 22k / 92k / never (peak 97 at 100k) |
| **S1 s2** | **0.810** | 5.89 | +0.61 ± 0.13 | 3.77 | 0.686 | 0.429 | 18k / 80k / never (peak 93 at 95k) |
| **S1 s3** | **0.900** | 5.93 | +0.65 ± 0.13 | 3.87 | 0.664 | 0.280 | 49k / 62k / 80k, then 5 dips below 80 in 84 evaluations |

**Greedy against sampled** (900000+, n=100, paired): +0.9 ± 1.4 / +2.3 ± 1.5 / +0.3 ± 1.5 vp; `held` 3.77 / 3.86 / 3.83 greedy against 3.80 / 3.85 / 3.78 sampled; coherency 0.41 / 0.44 / 0.29 greedy against **0.23 / 0.22 / 0.16** sampled — the lowest sampled coherency on the ladder.

**Health panel, last quarter** (240 updates per seed): explained variance
**0.61 / 0.59 / 0.68** against A3's 0.34–0.45 at the same rounds; clip
fraction 0.29 / 0.31 / 0.33; ratio p99 1.90 / 2.01 / 2.13; displacement
entropy 1.63 / 1.53 / 1.49; advantage std 0.41–0.50.

**The by-turn census** (bodies on objectives of 12 and points held of 4,
turn 5 → turn 8, n=20 on 700000+; A3 seed 1 at 20k read 5.0 → 2.3 bodies
and 1.9 → 1.3 held, the walk-off the arm was aimed at):

| checkpoint | success (n=20) | on objectives, turn 5 → 6 → 8 | held, turn 5 → 8 |
|---|---|---|---|
| s1 at 20,480 | 0.55 | 5.1 → 7.8 → 7.3 | 2.2 → 3.1 |
| s2 at 20,480 | 0.35 | 7.2 → 9.0 → **3.2** | 2.9 → **1.4** |
| s3 at 20,480 | 0.15 | 6.8 → **2.4** → 3.7 | 2.2 → 1.4 |
| s1 at 61,440 | 0.55 | 7.5 → 8.4 → 7.4 | 2.9 → 3.1 |
| s2 at 61,440 | 0.50 | 4.9 → 6.4 → 6.2 | 2.5 → 3.2 |
| s3 at 61,440 | 1.00 | 8.6 → 9.2 → 9.4 | 3.5 → 4.0 |
| s1 at 122,880 | 1.00 | 8.5 → 9.0 → 9.0 | 3.5 → 4.0 |
| s2 at 122,880 | 0.90 | 8.9 → 9.3 → 8.7 | 3.8 → 3.9 |
| s3 at 122,880 | 0.85 | 5.7 → 7.9 → 8.2 | 3.0 → 3.9 |

**The walk-off is still there at 20k on two seeds** — seed 2 puts nine
bodies on points at turn 6 and has three by turn 8; seed 3 drops from
6.8 to 2.4 in one turn. The term aimed at it did not remove it. Travel
gates on this arm read as A3's (one unit owning 2+ points 49–62% of
steps, fallback 40–53%), as they must: the travel term is unchanged.

### What it says

- **The term does what it was built to do and it is not enough.** The
  critic values the hold-paid episode far better (explained variance
  0.59–0.68 against 0.34–0.45), two seeds gain 0.13–0.24 at the cap, and
  the in-run curves reach 80% on all three seeds where A3's reached it on
  two. But no seed holds 0.95 at n=100 and seed 2 does not move. Under
  the pre-registered rule, three seeds at n=100 with one flat is a null.
- **Seed 2 is the informative one.** It reaches 80% in-run at the same
  rounds as A3's seed 2 (80k against 91k) and reads 0.810 against 0.800.
  Paying for standing changed nothing on that init, which says the
  walk-off at 20k is not the whole of what A3's ~200k rounds buy.
- **Coherency fell** (0.28–0.45 greedy against A3's 0.44–0.57, and
  0.16–0.23 sampled): nothing here pays for a squad staying together,
  and whatever the added term changed about the path, it did not help
  formation. Read the next section before attributing it to the pot.
- **The prediction was wrong twice.** Written before the run: FASTER on
  two seeds, rounds to 80% under 60k on all three. Read: no seed passes,
  62k–92k to 80%.

### Why the term could not do what the pre-registration asked of it

Read in `envs/per_model/reward_timing.py` after the numbers, not before.
On the per-model facade a **per-model state term is paid at the turn
close as the MEAN over alive models** — one scalar on the close step
(`_pay_close`: `weight × Σ calc(i) / n_alive`). The pot-split's total
still moves (three on each of two points earn more than six on one), but
no body receives its own share: the thirteenth model on a point and the
first are paid the same scalar at the same step, which is exactly the
flat-term defect the exponent exists to fix on the phase facade. The only
per-decision credit this facade pays is the action-class travel term, on
the step, to the model that acted. So S1 tested "the pot, as a mean, at
the close", not per-model credit for standing — and that is what a NULL
with two seeds moving looks like. ⚠ The same holds for `group_cohesion`
and every other per-model state term on this facade; a per-model
curriculum lever built on one of them is a global lever here until the
retimer pays state terms to the model that produced them.

### What was not done

- Not scored at 245,760 (the A3x budget); the criterion was the cap.
- No weight sweep; 1.0 was the one value pre-registered.

---

<a id="2026-09-15-curriculum-a3-speed-s2-match"></a>

## A3 speed screen, arm S2: one objective per squad in the travel assignment makes the rung SLOWER — HARMFUL, as the pre-registered clause names it

**Verdict first.** `a3.yaml` with `closest_objective_v2`'s assignment
made a matching (`one_objective_per_group: true`, #360), three seeds,
122,880 rounds, read greedy at n=100 on 700000+: success **0.830 /
0.690 / 0.610** against A3's own runs at the same rounds and seeds,
**0.700 / 0.800 / 0.770**. Seed 1 is ahead by 0.13; seeds 2 and 3 are
behind by 0.11 and 0.16, and `held` is 3.77 / 3.64 / **3.25** of 4 against
A3's 3.49–3.65. Under the pre-registered clauses that is **HARMFUL**: a
seed more than 0.10 below its A3 read. The gate the change was aimed at
moved exactly as measured before launch — no squad owns two points on
any step, squads are paid toward one target — and the policy learned
the rung more slowly for it. The prediction on file ("AHEAD but not
FASTER, the larger effect on coherency") is wrong on the verdict and
half-right on coherency (0.29–0.43 greedy, no better than A3's).

### Provenance

| field | value |
|---|---|
| date | 2026-09-15 (launched 10:00, exited 15:16–15:22) |
| GPU / no-GPU | GPU (RTX 4090), shared with eleven other per-model runs (the S1, S3 and A5b arms) |
| seeds | 1 / 2 / 3; rollout layouts at seed×100+; paired on init with A3 (same seed, same head) |
| n | 100 at seed base 700000 (final); 30 at 500000 (in-run, every 512 rounds); 100 at 900000 (greedy against sampled) |
| config | `configs/experiments/curriculum/a3_match.yaml` — `a3.yaml` with the one flag; unrefereed by design |
| decode | none |
| paired | per episode against `squad_march_take`; per seed against A3's read at 122,880 |
| comparator | A3's per-model runs at 122,880 (Wandb `curriculum-a3`: `wzalbuic` / `4x0p1q4a` / `jzmpi6pl`): 0.700 / 0.800 / 0.770, held 3.49 / 3.60 / 3.65, turns 6.05 / 5.95 / 5.81; the bar `squad_march_take`: 1.000, held 4.00, 5.28 turns, coherent 0.927 (identical on `a3_match.yaml`) |
| opponent | none |
| budget | 122,880 rounds at 128 rounds per update (`--rollout-rounds 32 --num-rollout-envs 4`), `--ent-coef 0.003` |
| code revision | launched from the `bd15db9` working tree of `feature/curriculum-a3-speed` (PR #362), rebased the same day to `71c0f8f` with an identical tree; the flag ships on this branch with `tests/test_shaping_gates.py::TestTheAssignment::test_one_objective_per_group_makes_the_assignment_a_matching` |
| checkpoint | `last.pt` at 122,880; `checkpoints/per_model/per-model-a3_match-2026-09-15-10-00-58-s{1,2,3}a3m` |
| coherency | greedy 0.339 / 0.294 / 0.425 at 700000+; 0.33 / 0.29 / 0.43 greedy against 0.18 / 0.15 / 0.13 sampled at 900000+ |
| Wandb | `curriculum-a3-speed`: `q432pqkj` s1 · `sgrae329` s2 · `u1943s80` s3 |
| pre-registration | `reports/2026-09-15-curriculum-a-rungs.md#2026-09-15-curriculum-A3-speed-preregistration` at `bd15db9` (10:00, before the launch) |

### The read

| row | success | turns | vs bar, paired | held | on obj | coherent | rounds to rolling 50 / 80 / 95% |
|---|---|---|---|---|---|---|---|
| `squad_march_take` | 1.000 | 5.28 | — | 4.00 | 0.892 | 0.927 | — |
| A3 s1 / s2 / s3 at 122,880 | 0.700 / 0.800 / 0.770 | 6.05 / 5.95 / 5.81 | +0.77 / +0.67 / +0.53 | 3.49 / 3.60 / 3.65 | 0.65–0.74 | 0.57 / 0.48 / 0.44 | 38k, never, never / 23k, 91k, never / 37k, 102k, never |
| **S2 s1** | **0.830** | 6.10 | +0.82 ± 0.12 | 3.77 | 0.608 | 0.339 | 71k / 77k / never (peak 97 at 106k; last ten 93–70) |
| **S2 s2** | **0.690** | 6.52 | +1.24 ± 0.14 | 3.64 | 0.563 | 0.294 | 14k / 80k / never (peak 87 at 79k; **last ten 30–63**) |
| **S2 s3** | **0.610** | 6.96 | +1.68 ± 0.13 | 3.25 | 0.474 | 0.425 | 39k / never / never (peak 70 at 95k; last ten 30–70) |

**Greedy against sampled** (900000+, n=100, paired): −5.3 ± 1.7 / +0.8 ± 1.6 / +5.8 ± 1.3 vp; `held` 3.82 / 3.63 / 3.27 greedy against 3.67 / 3.47 / 2.98 sampled; coherency 0.33 / 0.29 / 0.43 greedy against **0.18 / 0.15 / 0.13** sampled; the sampled policy takes 104–119 decisions an episode against the greedy 96–111 — it is still wandering at the end.

**Health panel, last quarter** (240 updates per seed): explained variance
**0.26 / 0.28 / 0.38** — below A3's 0.34–0.45 at the same rounds; clip
fraction 0.26 / 0.35 / 0.35; ratio p99 1.79 / 2.06 / 2.05; displacement
entropy 1.75 / 1.86 / 1.88 (A3 at its cap: 1.7–1.8); advantage std
0.47–0.62.

**The by-turn census and the travel gates** (n=20 on 700000+):

| checkpoint | success (n=20) | on objectives, turn 5 → 6 → 8 | held, turn 5 → 8 | one unit owns 2+ | fallback | squads split |
|---|---|---|---|---|---|---|
| s1 at 20,480 | 0.35 | 7.6 → **3.4** → 3.3 | 2.7 → 1.5 | 0.0% | 19.0% | 5.5% |
| s2 at 20,480 | 0.55 | 6.7 → 6.4 → 4.6 | 3.1 → 2.3 | 0.0% | 21.7% | 13.4% |
| s3 at 20,480 | 0.65 | 6.4 → 9.2 → 6.1 | 3.0 → 2.7 | 0.0% | 22.3% | 8.7% |
| s1 at 61,440 | 0.45 | 5.9 → 5.1 → 5.0 | 2.8 → 2.8 | 0.0% | 29.1% | 10.3% |
| s2 at 61,440 | 0.50 | 6.4 → 4.5 → 4.8 | 3.0 → 2.8 | 0.0% | 26.9% | 7.6% |
| s3 at 61,440 | 0.45 | 6.0 → 6.2 → 6.5 | 2.6 → 3.0 | 0.0% | 28.0% | 11.9% |
| s1 at 122,880 | 0.90 | 6.6 → 7.6 → 7.7 | 3.2 → 3.9 | 0.0% | 29.6% | 15.5% |
| s2 at 122,880 | 0.80 | 5.6 → 6.7 → 7.1 | 3.0 → 3.7 | 0.0% | 33.1% | 14.6% |
| s3 at 122,880 | 0.60 | 5.2 → 5.6 → 5.3 | 2.7 → 3.2 | 0.0% | 31.5% | 13.7% |

The gate did what it was built to do (no unit owns two points on any
step, against 49–62% on the other arms). Two things the table adds: the
walk-off at 20k is here too (seed 1: 7.6 bodies at turn 5, 3.4 at turn
6), so it is not the assignment's doing either; and at the end the
policies put only **5.3–7.7 of 12 bodies** on points (A3 at the cap: 8.0;
A3x: 9.7; the bar: 10.7) while splitting squads on 14–16% of squad-steps
against the bar's 1.1% under the same rule — the learned policy sends
fewer bodies and sends them apart.

### What it says

- **The change is real and it hurts.** Before launch the bar's gate
  census on this config read one unit owning 2+ points on 0.0% of steps
  (49.1% under the per-objective rule) and squads split on 1.1% (11.3%).
  Trained under it, two seeds finish 0.11–0.16 behind A3, one point
  short on seed 3 (`held` 3.25), and the in-run curves of seeds 2 and 3
  fall back to 30–70% over the last ten evaluations after peaking at 87
  and 70.
- **A matching makes the target switch more often, and a switch is
  free.** Under the per-objective rule an objective's owner is whichever
  group has the closest model, which is stable while squads walk
  straight. Under a matching the assignment re-solves every step over
  all four squads, so one squad's move can flip two other squads' targets
  — and `closest_objective_v2` pays zero on the step a target changes and
  re-anchors, so a policy whose targets keep flipping is paid less for
  the same walking. The gate census before launch counted assignments,
  not switches; the switch rate is the statistic that would have caught
  this, and it was not on the list.
- **The critic is worse under it** (explained variance 0.26–0.38 against
  0.34–0.45): the travel income is less predictable from the state when
  the target it pays toward depends on everyone else's position.
- **Nothing about coherency improved** (0.29–0.43 greedy against A3's
  0.44–0.57), against the prediction. Paying a squad toward one target
  did not keep it together.
- **Stop nominating `closest_objective_v2`'s assignment rule.** This is
  the record's fifth empty-or-worse result on that term (`CLAUDE.md` § The
  travel reward, audited, lists four). A matching is the natural fix for
  the defect the gate census shows, and it lost; the flag stays, default
  off, because the census that motivated it is real, but the next lever
  on this rung is not this term.

### What was not done

- No measurement of the target-switch rate; the mechanism above is the
  reading that fits the numbers, not a measured cause.
- Not scored at 245,760.

---

<a id="2026-09-15-curriculum-a3-speed-s3-warm"></a>

## A3 speed screen, arm S3: warm-starting the spread rung from the one-point rung is AHEAD on 3 of 3 — the transfer the ladder promised, one rung early, against the prediction

**Verdict first.** `a3.yaml` unchanged, each seed started from the A2b
checkpoint of the same seed (`--warm-start-from`, #361), three seeds,
122,880 rounds, read greedy at n=100 on 700000+: success **0.910 / 0.940
/ 0.960** against A3 from scratch at the same rounds and seeds, **0.700 /
0.800 / 0.770** — ahead by **+0.21 / +0.14 / +0.19**, every seed past the
+0.10 bound. Two seeds sit under 0.95, so it is not FASTER on the letter;
it is **AHEAD** on all three, the only arm of the screen to clear a
clause. In-run it reached 80% at **22k / 11k / 19k** rounds where scratch
took 91k, 102k and never, and two seeds crossed 95% rolling at 62k and
71k. The prediction on file — faster to the first bodies and then NULL on
success, because an A2b policy has to unlearn sending everyone to one
point — is wrong: the one-point policy carried over, and what it had to
learn was only the spreading.

### Provenance

| field | value |
|---|---|
| date | 2026-09-15 (launched 10:03, exited 15:37–15:40) |
| GPU / no-GPU | GPU (RTX 4090), shared with eleven other per-model runs (the S1, S2 and A5b arms) |
| seeds | 1 / 2 / 3; rollout layouts at seed×100+; **not paired on init** — the init is A2b's `last.pt` of the same seed (122,880 rounds at `ent_coef` 0.003, revision `c2bcf26`), so per-seed differences carry init variance the other two arms do not |
| n | 100 at seed base 700000 (final); 30 at 500000 (in-run, every 512 rounds); 100 at 900000 (greedy against sampled) |
| config | `configs/experiments/curriculum/a3.yaml`, unchanged; unrefereed by design |
| decode | none |
| paired | per episode against `squad_march_take`; per seed against A3's read at 122,880 (layouts and seeds shared, init not) |
| comparator | A3's per-model runs at 122,880 (Wandb `curriculum-a3`: `wzalbuic` / `4x0p1q4a` / `jzmpi6pl`): 0.700 / 0.800 / 0.770, held 3.49 / 3.60 / 3.65, turns 6.05 / 5.95 / 5.81; the bar `squad_march_take`: 1.000, held 4.00, 5.28 turns, coherent 0.927 |
| opponent | none |
| budget | 122,880 rounds at 128 rounds per update (`--rollout-rounds 32 --num-rollout-envs 4`), `--ent-coef 0.003`, fresh optimizer |
| warm start | `checkpoints/per_model/per-model-a2-2026-09-15-02-44-23-s{1,2,3}a2b/last.pt` (A2b, PASS with drift: 1.000 / 1.000 / 0.990 on one objective of radius 6); the set network loads across the objective count unchanged, the displacement head is the same |
| code revision | launched from the `bd15db9` working tree of `feature/curriculum-a3-speed` (PR #362), rebased the same day to `71c0f8f` with an identical tree |
| checkpoint | `last.pt` at 122,880; `checkpoints/per_model/per-model-a3-2026-09-15-10-03-16-s{1,2,3}a3w` |
| coherency | greedy 0.451 / 0.445 / 0.438 at 700000+; 0.43 / 0.44 / 0.45 greedy against 0.26 / 0.29 / 0.22 sampled at 900000+ |
| Wandb | `curriculum-a3-speed`: `zgc92is1` s1 · `w55z85n1` s2 · `z4tr3mfh` s3 |
| pre-registration | `reports/2026-09-15-curriculum-a-rungs.md#2026-09-15-curriculum-A3-speed-preregistration` at `bd15db9` (10:00, before the launch; the first launch at 10:00 died on the shell and started no run, relaunched 10:03) |

### The read

| row | success | turns | vs bar, paired | held | on obj | coherent | rounds to rolling 50 / 80 / 95% |
|---|---|---|---|---|---|---|---|
| `squad_march_take` | 1.000 | 5.28 | — | 4.00 | 0.892 | 0.927 | — |
| A3 s1 / s2 / s3 at 122,880 | 0.700 / 0.800 / 0.770 | 6.05 / 5.95 / 5.81 | +0.77 / +0.67 / +0.53 | 3.49 / 3.60 / 3.65 | 0.65–0.74 | 0.57 / 0.48 / 0.44 | 38k, never, never / 23k, 91k, never / 37k, 102k, never |
| **S3 s1** | **0.910** | 5.85 | +0.57 ± 0.11 | 3.88 | 0.741 | 0.451 | **8k / 22k** / never (peak 93; last ten 80–93) |
| **S3 s2** | **0.940** | 5.54 | +0.26 ± 0.12 | 3.91 | 0.763 | 0.445 | **7k / 11k / 62k**; one dip below 80 in 119 evaluations after |
| **S3 s3** | **0.960** | 5.17 | −0.11 ± 0.10 | 3.96 | 0.800 | 0.438 | **11k / 19k / 71k**; two dips below 80 in 102 after |

Seed 3 arrives faster than the script (5.17 against 5.28 turns) with 3.96
points held, the A3x read at 245,760 in half the rounds.

**Greedy against sampled** (900000+, n=100, paired): −1.8 ± 1.5 / +1.8 ± 1.4 / −2.0 ± 1.2 vp; `held` 3.90 / 3.90 / 3.99 greedy against 3.73 / 3.84 / 3.88 sampled; coherency 0.43 / 0.44 / 0.45 greedy against **0.26 / 0.29 / 0.22** sampled. The sampled policy holds the points nearly as well as the greedy one, as on A3x and A4x.

**Health panel, last quarter** (240 updates per seed): explained variance
0.53 / 0.56 / 0.47 (A3 at the cap: 0.34–0.45); clip fraction 0.24 / 0.27
/ 0.30; ratio p99 1.81 / 1.86 / 1.96; displacement entropy **1.18 / 1.43
/ 1.18** (A3 at the cap: 1.7–1.8, A3x at the end: 1.4); advantage std
0.48–0.57.

**The by-turn census** (n=20 on 700000+):

| checkpoint | success (n=20) | on objectives, turn 5 → 6 → 8 | held, turn 5 → 8 | squads split |
|---|---|---|---|---|
| s1 at 20,480 | 0.85 | 8.0 → 8.8 → 8.9 | 3.3 → 3.8 | 37.5% |
| s2 at 20,480 | 0.95 | 9.0 → 9.1 → 9.1 | 3.2 → 4.0 | 30.2% |
| s3 at 20,480 | 0.60 | 5.5 → 6.3 → 7.5 | 2.3 → 3.6 | 25.2% |
| s1 at 61,440 | 0.85 | 8.6 → 9.0 → 8.7 | 3.4 → 3.9 | 25.0% |
| s2 at 61,440 | 0.95 | 9.2 → 9.4 → 9.3 | 3.8 → 4.0 | 35.2% |
| s3 at 61,440 | 0.95 | 9.3 → 9.3 → 8.9 | 3.9 → 3.9 | 23.3% |
| s1 at 122,880 | 0.95 | 8.2 → 9.0 → 9.0 | 3.5 → 4.0 | 20.6% |
| s2 at 122,880 | 0.95 | 8.5 → 8.6 → 8.8 | 3.6 → 4.0 | 28.5% |
| s3 at 122,880 | 1.00 | 9.5 → 9.5 → 9.5 | 4.0 → 4.0 | 27.8% |

**No walk-off and no stacking, from 20k on.** Bodies on objectives are
8.0–9.0 at turn 5 and rise or hold through turn 8 on every seed and
every checkpoint, against scratch's 5.0 → 2.3 at 20k; points held are
3.3–3.8 at 20k where scratch had 1.9. The A2b policy brought "arrive and
stay" with it and the 20k rounds bought the fourth point. The travel
gates read as A3's (one unit owning 2+ points 45–59% of steps) and the
squads split on 21–38% of squad-steps — the highest on the screen, and
the one thing the warm start did not carry.

### What it says

- **What A2b learned transfers.** The one-point policy walks bodies onto
  a point and keeps them there; started from it, the spread rung needs
  only the assignment, and the in-run curve reaches 80% in 11k–22k rounds
  against scratch's 91k–never. The prediction that it would have to
  unlearn stacking was wrong: at 20k rounds there is no stacking to see
  (3.3–3.8 points held, 8–9 bodies on them, no walk-off), so whatever
  unlearning happened was over inside the first 20k.
- **Two seeds are within noise of the pass mark and one passes**: 0.94
  and 0.91 at n=100 (SE 0.024–0.029) against 0.95, with the in-run curve
  holding 80–96 over the last ten evaluations. At the A3x budget this
  arm would very likely read 3/3; that is not measured.
- **The transfer rung T1 is now backed by a measurement one rung
  earlier than planned.** #340's T1 asks whether A5 warm-started from A4
  beats A5 from scratch; A3 from A2b does, on the rung where scratch was
  slowest. T1 keeps its own pre-registration; this is the prior.
- **Unpaired on init, as pre-registered.** Three seeds each from their
  own A2b checkpoint agree at +0.14 to +0.21, so the init variance is not
  what carries the result; but the old rule stands that seeds off one
  warm start are not independent samples, and each of these is off a
  different one.
- **Coherency is A3's** (0.44–0.45 greedy against 0.44–0.57): the
  warm start neither helps nor hurts formation; nothing here pays for it.

### What was not done

- Not scored at 245,760, where the two 0.91–0.94 seeds would be expected
  to clear 0.95.
- No control that warm-starts from a *random* A2-shaped init or from an
  A1x checkpoint, which would separate "any trained start" from "the
  one-point skill". One arm, one change.

---

<a id="2026-09-15-curriculum-A3x-preregistration"></a>

## Pre-registration: curriculum rung A3, second arm (A3x) — the A3 runs resumed to the control's 245,760 rounds

Written 2026-09-15, **before any number from the arm exists**, on branch
`feature/curriculum-a3` (PR #350 carries the rung's arms). Issue #357,
parent question #340, rung **A3** — a follow-up to #349, which read
**FAIL as pre-registered** at the cap: greedy success **0.700 / 0.800 /
0.770** at 122,880 rounds, `held` 3.49 / 3.60 / 3.65 of 4, still rising,
against a control at 0.950 / 0.950 / 0.960 after its once-only extension
to 120 epochs (245,760 rounds).

### The one change

**Budget.** The three A3 per-model runs (Wandb `curriculum-a3`,
`wzalbuic` s1 · `4x0p1q4a` s2 · `jzmpi6pl` s3) resumed **in place** with
`--resume-from <run_dir> --rounds 245760 --eval-every-rounds 512
--checkpoint-every-rounds 512` — same seeds, same knobs (`ent_coef` 0.003
carried by the checkpoint and refused if changed), optimizer and
generator restored; the cadences are passed explicitly because a resume
drops them (#346). 245,760 rounds is the control's extended budget in
rounds: the same once-only doubling the control was granted on this
rung, applied to the arm under test.

### Why this arm

A3 is the first rung on which the whole-army control needed its
extension — at the cap it read 0.850 / 0.980 / 0.930 and passed 3 of 3
only at 245k rounds. The per-model arm at the cap reads 0.70 / 0.80 /
0.77 with every seed's in-run curve still climbing (last six evaluations
53–90%, `held` 3.2–3.8). Reading it against a control that had twice
the rounds is the comparison the cap rule did not anticipate; giving the
arm the same rounds is the symmetric test, and it costs one more run.

### Comparator

`squad_march_take` on `a3.yaml`, seeds 700000+ at n=100: success 1.000,
`held` 4.00, turns 5.28, coherent 0.927. The control: 0.850 / 0.980 /
0.930 at 122,880 rounds, **0.950 / 0.950 / 0.960 at 245,760**. A3's read
at 122,880: 0.700 / 0.800 / 0.770.

### Criteria

Read at 245,760 rounds from `last.pt`, greedy, no decode, n=100 on seeds
700000+.

- **PASS:** success ≥ 0.95 on **all three** seeds. A3's verdict stands as
  a read at the cap; the per-model arm allocates four squads over four
  points within the rounds the control needed, and the cap rule gains the
  control's symmetry from here: when the control needs its extension, the
  per-model arm gets the same one.
- **FAIL:** any seed below 0.95 at 245,760 — the per-model arm is worse
  than the whole-army trainer at spreading over points at equal rounds,
  and the ladder stops at A3; the next step is a recording of the failing
  episodes, asking which point is left empty and whether the unit
  pointer sends two squads to one.
- **Readouts:** `held` and `on_obj`; the equal-rounds comparison to the
  control (at 123k: per-model 0.70 / 0.80 / 0.77 v control 0.85 / 0.98 /
  0.93; at 245k: this read v 0.95 / 0.95 / 0.96); coherency greedy and
  sampled; the panel over the last quarter of the resumed half; the
  in-run curve after any first pass.

Power: binomial SE 0.022 at p=0.95, n=100. The seeds sit 7–11 SE below
the bound, so a pass is a move.

### Budget, regime, seeds

| | per-model arm |
|---|---|
| budget | 245,760 rounds total (122,880 more), 128 rounds per update |
| seeds | 1, 2, 3, continued |
| in-run eval | every 512 rounds, n=30, seeds 500000+ |
| final score | n=100, seeds 700000+ |
| flags | carried from the checkpoint (`ent_coef` 0.003); cadences 512 / 512 passed explicitly |
| logging | the runs' own Wandb ids, continued |

### What I expect (a guess, written so it can be wrong)

Two of three seeds pass by ~200k rounds and the third sits at 0.90–0.95:
a marginal read. `held` ends near 3.9 on every seed. Coherency stays
below 0.6 greedy.

---

<a id="2026-09-15-curriculum-a3x-passes"></a>

## Curriculum rung A3, second arm: given the control's rounds, the per-model arm spreads over four points — PASS 3 of 3

**Verdict first.** The three A3 per-model runs resumed from 122,880 to
**245,760 rounds** (#357) — the rounds the whole-army control needed on
this rung — read greedy success **0.960 / 0.960 / 0.970** at n=100,
`held` **3.94 / 3.96 / 3.96** of 4, turns **5.19 / 5.24 / 5.33** against
the script's 5.28, where at 122,880 they read 0.700 / 0.800 / 0.770 with
3.5–3.7 points held. That is **PASS** on the pre-registered clause, and
it matches the control's own read at the same rounds (0.950 / 0.950 /
0.960, held 3.95) to within the binomial noise. A3's FAIL stands as a
read at the cap; the answer to it is that on the spread rung the
per-model arm needs about the rounds the whole-army trainer needs, and
arrives at the script's speed where the control is a round slower
(6.24–6.33). From here the cap rule is symmetric: when the control needs
its once-only extension, the arm under test gets the same rounds before
the rung is called.

### Provenance

| field | value |
|---|---|
| date | 2026-09-15 (resumed 05:40, exited 07:05) |
| GPU / no-GPU | GPU (RTX 4090), the box to itself |
| seeds | 1 / 2 / 3, the A3 runs continued (`--resume-from`, optimizer and generator restored, `ent_coef` 0.003 carried; cadences 512 / 512 passed explicitly per #346) |
| n | 100 at seed base 700000 (final); 30 at 500000 (in-run) |
| config | `configs/experiments/curriculum/a3.yaml`, unrefereed by design |
| decode | none |
| paired | per episode against `squad_march_take`; against A3's read on the same seeds and checkpoints |
| comparator | `squad_march_take`: 1.000, held 4.00, 5.28 turns, coherent 0.927; the control at 245,760: 0.950 / 0.950 / 0.960, held 3.95, 6.24–6.33 turns, coherent 0.69–0.79 |
| opponent | none |
| budget | 245,760 rounds (122,880 more), 128 rounds per update |
| code revision | `07c2ac6` on `feature/curriculum-a3` (PR #350) |
| checkpoint | `last.pt` at 245,760 |
| coherency | greedy 0.443 / 0.516 / 0.375 |
| Wandb | `curriculum-a3`: `wzalbuic` s1 · `4x0p1q4a` s2 · `jzmpi6pl` s3, continued |
| pre-registration | `reports/2026-09-15-curriculum-a-rungs.md#2026-09-15-curriculum-A3x-preregistration` at `07c2ac6`, before the resume |

### The read

| row | success | turns | vs bar, paired | held | on obj | coherent | last 12 in-run (n=30) |
|---|---|---|---|---|---|---|---|
| `squad_march_take` | 1.000 | 5.28 | — | 4.00 | 0.892 | 0.927 | — |
| s1 at 122,880 (A3) | 0.700 | 6.05 | +0.77 | 3.49 | 0.654 | 0.566 | 53–73 |
| **s1 at 245,760** | **0.960** | 5.19 | −0.09 ± 0.09 | 3.94 | 0.804 | 0.443 | 85–100, held 3.8–3.9 |
| s2 at 122,880 (A3) | 0.800 | 5.95 | +0.67 | 3.60 | 0.652 | 0.480 | 80–90 |
| **s2 at 245,760** | **0.960** | 5.24 | −0.04 ± 0.10 | 3.96 | 0.814 | 0.516 | 85–100, held 3.75–3.95 |
| s3 at 122,880 (A3) | 0.770 | 5.81 | +0.53 | 3.65 | 0.740 | 0.442 | 70–87 |
| **s3 at 245,760** | **0.970** | 5.33 | +0.05 ± 0.11 | 3.96 | 0.738 | 0.375 | 85–100, held 3.9 |
| control s1–s3 at 245,760 | 0.950 / 0.950 / 0.960 | 6.26 / 6.33 / 6.24 | +0.96 to +1.05 | 3.95 ×3 | 0.87–0.88 | 0.69–0.79 | wobbled 83–97 for sixty epochs |

The in-run n=30 curve wobbles at 85–100 on every seed through the last
quarter, exactly as the control's did at 83–97; neither trainer's n=30
curve "holds 0.95" on this rung while both clear it at n=100.

**Health panel over the last quarter** (240 updates per seed):
displacement entropy 1.37–1.43 (down from 1.7–1.8 at the cap), clip
fraction 0.29–0.33, ratio p99 1.88–2.03, explained variance **0.48 /
0.48 / 0.57** (up from 0.34–0.45), advantage std 0.49–0.53.

### What it says

- **The per-model arm solves the spread rung in the rounds the
  whole-army trainer needs**, not fewer and not many more: at 123k it
  trailed (0.70–0.80 v 0.85–0.98), at 245k it matches (0.96–0.97 v
  0.95–0.96) and arrives a round faster. The rung asks for a conjunction
  over four points, and both trainers take ~200k rounds to learn it.
- **The critic caught up as the policy did.** Explained variance 0.34–0.45
  at the cap, 0.48–0.57 at the end; the value of a four-point episode
  became predictable once the policy reliably produced it.
- **The cap rule is now symmetric.** A control granted its once-only
  extension is read at 245k rounds; reading the arm under test at 123k
  against it was the asymmetry A3's FAIL measured. From A4 on, if the
  control needs its extension the per-model arm gets the same rounds
  before the rung is called (an amendment to #340 contract 3).
- **Coherency is the lowest on the ladder** (greedy 0.38–0.52 against the
  control's 0.69–0.79 and the bar's 0.927). The per-model policy spreads
  its bodies over points without keeping squads together — nothing here
  pays for that — and the E rungs' referee will meet it.
- **On n=30 neither trainer "holds 0.95" on this rung**; the in-run curve
  wobbles 85–100 for both, and the verdict has to be read at n=100. The
  strict no-dip clause A2b adopted is for drift, not for this.

### What was not done

- No recording of the residual 3–4% of failing episodes (which point is
  short); the spread statistic the pre-registration named as a readout
  is not computed — the rung passed on `held` 3.94–3.96, which bounds
  it.
- The control was not extended further; it never needed to be.

---

<a id="2026-09-15-curriculum-A4-preregistration"></a>

## Pre-registration: curriculum rung A4 — four squads, three objectives, a spare squad

Written 2026-09-15, **before any number from the arm exists**, on branch
`feature/curriculum-a4` (one PR per rung, stacked on A3's PR #350; the
chain does not merge to `main` while exploratory, so the commit timestamp
of this file is what orders it before the numbers). Issue #351, parent
question #340, rung **A4** — axis: **points**, a spare squad. A3 (#349)
is the rung below; its per-model arm has not run when this is written.
A4's whole-army control may run now; A4's per-model arm does not launch
until A3 has been read, and not at all if A3 fails.

### The one change

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

### Comparator, measured first

`squad_march_take`, scored through both facades on seeds 700000+ at n=100
(`just measure-bridge configs/experiments/curriculum/a4.yaml 100`), bridge
identical on every shared field: success **1.000**, `held` **3.00** of 3,
turns **4.70** (sd 0.63, range 4–6), `on_obj` 0.724 (the spare squad
stands off the points), coherent **0.923**, vp 21.4 ± 0.6. In-run band
(n=30, 500000+): success 1.000, turns 4.50, held 3.00, coherent 0.918.

### Criteria

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

### Budget, regime, seeds

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

### Primary readouts

Success rate (decides); turns paired against the script and against the
control; `held`, `on_obj`, coherency; the passive pair and
`train/rounds_per_update` on every per-model row; the health panel over
the last quarter; rounds-to-pass on both arms.

### What I expect (a guess, written so it can be wrong)

Both facades pass, the control slower than on A3 (a spare squad crowding
a point it is not needed at costs the others their approach). The
per-model arm's spare squad ends on a point, not off it — nothing pays it
to stand aside — and `on_obj` reads above the script's 0.724.

### Amendment 1 — the control's read (2026-09-15, before any per-model number exists)

The whole-army control ran to its 60-epoch cap (Wandb group `curriculum-a4`,
runs `3bgdwu1e` s1 · `w1su1gqg` s2 · `64koq60w` s3). Scored at n=100 on
seeds 700000+ from `last.ckpt` (epoch 60): success **0.980 / 0.980 /
0.990**, `held` 2.98 / 2.98 / 2.99, turns 6.12 / 6.23 / 6.16 against the
script's 4.70 (paired +1.42 / +1.53 / +1.46), `on_obj` 0.78 / 0.73 / 0.73,
coherent 0.739 / 0.864 / 0.776 against the bar's 0.923. **The control
passes the success bound on 3 of 3 seeds**, so a per-model miss reads
FAIL. Its in-run rounds-to-pass is read from Wandb once no run is
training (the A2 per-model arm is live) and sets the per-model budget at
6×, in [20,480, 122,880]; the per-model arm launches once A3 (#349) has
been read. Nothing about the per-model arm was known when this was
written.

### Amendment 2 — the entropy coefficient, the symmetric cap, and the launch (2026-09-15, before any per-model number exists)

Two contract amendments since this file was written apply here: every
per-model arm runs `--ent-coef 0.003` (A2b, #356), and the cap is
symmetric — if the control needed its once-only extension the per-model
arm gets the same rounds before the rung is called (A3x, #357). A4's
control passed 3 of 3 at 60 epochs without an extension: in-run success
first reaches 0.95 and stays at epoch **46 / 58 / 44** (0-based), i.e.
rounds-to-pass **96,256 / 120,832 / 92,160**; 6× the slowest exceeds the
cap, so the per-model budget is **122,880 rounds**. The "other flags" row
reads `gamma` 0.9, `lr` 3e-4, **`ent_coef` 0.003**. A3 has been read
(FAIL at the cap, PASS at the control's rounds under A3x), so the ladder's
order holds and the arm launches now. A2b's strict clause is a readout
here: the in-run curve after the first pass, any dip below 0.80 reported.
Nothing about the per-model arm was known when this was written.

---

<a id="2026-09-15-curriculum-a4-one-se-short-at-the-cap"></a>

## Curriculum rung A4: one standard error short at the cap on two seeds — FAIL, as pre-registered

**Verdict first.** On the spare-squad rung (#340, arm #351) — four squads
of three, three objectives, success only when every point has a body on
it — the per-model trainer at the 122,880-round cap reads greedy success
**0.930 / 0.930 / 0.970** at n=100, `held` **2.91 / 2.93 / 2.97** of 3,
turns 5.36 / 5.46 / 5.21 against the script's 4.70 and the control's
6.12–6.23. The control read 0.980 / 0.980 / 0.990 at the same rounds, so
this is **FAIL** on the pre-registered clause. Two seeds sit one binomial
SE below the bound with their in-run curves still rising (70–93 and 66–93
over the last twelve evaluations); the third passes. The control itself
passed only at epochs 44–58 of 60 — 92k–121k of 123k rounds — and on
every earlier rung the per-model arm has needed 2–4× the control's
rounds, so 6× the control is the cap here and the cap binds. A4x (#358),
pre-registered before it runs, resumes the same runs to 245,760 rounds
and writes the amendment that implies: when 6× the control exceeds the
cap, the per-model budget is 2× the cap.

### Provenance

| field | value |
|---|---|
| date | 2026-09-15 (per-model 07:08–08:33; control 01:15–01:24) |
| GPU / no-GPU | GPU (RTX 4090), the box to itself |
| seeds | 1 / 2 / 3 on both arms; rollout layouts at seed×100+ |
| n | 100 at seed base 700000 (final); 30 at 500000 (in-run) |
| config | `configs/experiments/curriculum/a4.yaml` — unrefereed by design; success `all_objectives_occupied` |
| decode | none on the per-model facade; K=1 on the control |
| paired | per episode against `squad_march_take` on identical seeds |
| comparator | `squad_march_take`: success 1.000, held 3.00, turns 4.70, `on_obj` 0.724, coherent 0.923 |
| opponent | none |
| budget | per-model 122,880 rounds at 128 rounds per update, `ent_coef` 0.003; control 60 epochs of 2048 steps |
| code revision | `7171679` on `feature/curriculum-a4` (PR #352) |
| checkpoint | `last.pt` at 122,880; `last.ckpt` at epoch 60 |
| coherency | per-model greedy 0.441 / 0.431 / 0.528, sampled 0.168 / 0.187 / 0.239; control 0.739 / 0.864 / 0.776 |
| Wandb | `curriculum-a4`: per-model `la0ztx25` s1 · `262q9d0k` s2 · `rfqku890` s3; control `3bgdwu1e` / `w1su1gqg` / `64koq60w` |
| pre-registration | `reports/2026-09-15-curriculum-a-rungs.md#2026-09-15-curriculum-A4-preregistration` at `67bfb6f`; amendments 1–2 at `8c52e83`, `7171679`, each before the number it precedes |

### The read

| row | success | turns | vs bar, paired | held | on obj | coherent greedy / sampled | last 12 in-run |
|---|---|---|---|---|---|---|---|
| `squad_march_take` | 1.000 | 4.70 | — | 3.00 | 0.724 | 0.923 / — | — |
| per-model s1 | **0.930** | 5.36 | +0.66 ± 0.09 | 2.91 | 0.717 | 0.441 / 0.168 | 70–93, held 2.7–2.9 |
| per-model s2 | **0.930** | 5.46 | +0.76 ± 0.10 | 2.93 | 0.603 | 0.431 / 0.187 | 66–93, held 2.7–2.9 |
| per-model s3 | **0.970** | 5.21 | +0.51 ± 0.09 | 2.97 | 0.607 | 0.528 / 0.239 | 90–100, held 3.0 |
| control s1 | 0.980 | 6.12 | +1.42 | 2.98 | 0.782 | 0.739 | passed at epoch 46 |
| control s2 | 0.980 | 6.23 | +1.53 | 2.98 | 0.731 | 0.864 | passed at epoch 58 |
| control s3 | 0.990 | 6.16 | +1.46 | 2.99 | 0.733 | 0.776 | passed at epoch 44 |

In-run, per-model: 40–57% with `held` 1.3–1.8 through ~60k rounds; `held`
2.5–3.0 from ~70k; s3 at 96–100 from ~86k; s1 and s2 climbing through
the last quarter without holding 0.95.

**Greedy against sampled** (900000+, n=100): −0.7 ± 0.8 / −0.3 ± 0.7 /
−2.5 ± 0.7 vp; `held` 2.93 / 2.88 / 2.98 greedy against 2.89 / 2.86 /
2.90 sampled; coherency 0.48 / 0.46 / 0.54 against 0.17 / 0.19 / 0.24.

**Health panel, last quarter** (240 updates per seed): displacement
entropy 1.26 / 1.64 / 1.77, declaration 0.02–0.05, clip fraction 0.22 /
0.30 / 0.33, ratio p99 1.86–1.99, explained variance **0.54 / 0.43 /
0.42**, advantage std 0.44–0.52. The same shape as A3 at its cap.

### What it says

- **A4 at the cap reads like A3 at the cap, one rung's worth closer.** A3
  read 0.70–0.80 at 123k and 0.96–0.97 at 245k; A4 reads 0.93–0.97 at
  123k with the same rising curves and the same critic (explained
  variance 0.42–0.54). The spare squad makes the rung no harder for the
  per-model arm than four matched points did — its `held` at the cap is
  2.9 of 3 where A3's was 3.5 of 4.
- **The cap binds on a rung the control passes inside it.** The
  symmetric-cap rule (A3x) keyed the extension on the control needing
  one; here the control passed at 92k–121k rounds and the per-model arm,
  which has needed 2–4× the control's rounds on every rung below, was
  read at 123k. A4x writes the rule that follows: when 6× the control
  exceeds the cap, the arm's budget is 2× the cap.
- **The spare squad ends on a point, not off it.** `on_obj` 0.60–0.72 on
  the per-model arm against the script's 0.724 and the control's
  0.73–0.78: with three points and four squads, the learned policies put
  the surplus on a point already held, which the criterion neither
  rewards nor punishes. The E rungs' VP will.
- **Coherency: 0.43–0.53 greedy, 0.17–0.24 sampled.** The lowest sampled
  figure on the ladder; unpaid here, priced at the E rungs.

### What was not done

- No recording of the failing episodes (which point is short).
- No sweep; A4x changes only the budget.

---

<a id="2026-09-15-curriculum-A4x-preregistration"></a>

## Pre-registration: curriculum rung A4, second arm (A4x) — the A4 runs resumed to 245,760 rounds

Written 2026-09-15, **before any number from the arm exists**, on branch
`feature/curriculum-a4` (PR #352 carries the rung's arms). Issue #358,
parent question #340, rung **A4** — a follow-up to #351, which read
**FAIL as pre-registered** at the cap: greedy success **0.930 / 0.930 /
0.970** at 122,880 rounds, `held` 2.91 / 2.93 / 2.97 of 3, the two short
seeds still rising (last twelve in-run evaluations 70–93 and 66–93),
against a control at 0.980 / 0.980 / 0.990.

### The one change

**Budget.** The three A4 per-model runs (Wandb `curriculum-a4`,
`la0ztx25` s1 · `262q9d0k` s2 · `rfqku890` s3) resumed **in place** with
`--resume-from <run_dir> --rounds 245760 --eval-every-rounds 512
--checkpoint-every-rounds 512` — same seeds, same knobs (`ent_coef` 0.003
carried and refused if changed), optimizer and generator restored,
cadences passed explicitly (#346). 245,760 rounds is the once-only
doubling A3x was granted.

### Why this arm, and the rule it amends

The symmetric-cap rule (A3x) grants the per-model arm the control's
extension when the control needed one. A4's control did not: it passed
inside its 60 epochs — but only at epochs 44 / 58 / 46, i.e. 92k–121k of
its 123k rounds, and on every earlier rung the per-model arm has needed
2–4× the control's rounds. Six times the control's slowest pass is
725k; the cap is 123k; the cap binds, and a read at the cap measures the
cap. This arm tests that, and the pre-registration writes the amendment
it implies **before the run**: when 6× the control's slowest
rounds-to-pass exceeds the cap, the per-model arm's budget is 2× the cap
(245,760), read once at the end. A4's FAIL stands as a read at the cap
whatever this arm says.

### Comparator

`squad_march_take` on `a4.yaml`, seeds 700000+ at n=100: success 1.000,
`held` 3.00, turns 4.70, coherent 0.923. The control at 122,880:
0.980 / 0.980 / 0.990, `held` 2.98–2.99, turns 6.12–6.23. A4's read at
122,880: 0.930 / 0.930 / 0.970, `held` 2.91 / 2.93 / 2.97, turns 5.36 /
5.46 / 5.21.

### Criteria

Read at 245,760 rounds from `last.pt`, greedy, no decode, n=100 on seeds
700000+.

- **PASS:** success ≥ 0.95 on **all three** seeds. The amendment above
  stands for the rungs above; the ladder continues to A5b.
- **FAIL:** any seed below 0.95 at 245,760 — the per-model arm is worse
  than the whole-army trainer at allocation with a spare squad; the
  ladder stops at A4 and the next step is a recording of the failing
  episodes (which point is short, where the spare squad ends).
- **Readouts:** `held`, `on_obj` (the script's spare squad stands off at
  0.724; the per-model arm's reads 0.60–0.72 at the cap); coherency
  greedy and sampled; the panel over the last quarter of the resumed
  half; the in-run curve after any first pass.

Power: binomial SE 0.022 at p=0.95, n=100. Two seeds sit one SE below
the bound; this arm reads the direction of travel over 123k more rounds,
and the in-run curves are the readout that says whether the last few
percent arrive or the seeds plateau.

### Budget, regime, seeds

| | per-model arm |
|---|---|
| budget | 245,760 rounds total (122,880 more), 128 rounds per update |
| seeds | 1, 2, 3, continued |
| in-run eval | every 512 rounds, n=30, seeds 500000+ |
| final score | n=100, seeds 700000+ |
| flags | carried from the checkpoint (`ent_coef` 0.003); cadences 512 / 512 passed |
| logging | the runs' own Wandb ids, continued |

### What I expect (a guess, written so it can be wrong)

All three pass by ~180k rounds with `held` ≥ 2.95; the spare squad ends
on a point rather than off it (`on_obj` above the script's 0.724 on at
least two seeds). If instead s1 and s2 plateau at 0.90–0.93, the spare
squad is the thing the per-model arm cannot place, and the recording is
the next step.

---

<a id="2026-09-15-curriculum-a4x-passes"></a>

## Curriculum rung A4, second arm: given twice the cap, the per-model arm places the spare squad — PASS 3 of 3, at the script's speed

**Verdict first.** The three A4 per-model runs resumed from 122,880 to
**245,760 rounds** (#358) read greedy success **0.980 / 1.000 / 1.000** at
n=100, `held` **2.96 / 3.00 / 3.00** of 3, turns **4.75 / 4.72 / 4.73**
against the script's 4.70 and the control's 6.12–6.23, where at 122,880
they read 0.930 / 0.930 / 0.970 with 2.91–2.97 held. That is **PASS** on
the pre-registered clause, and the rule the pre-registration wrote
stands: when 6× the control's slowest rounds-to-pass exceeds the cap, the
per-model budget is 2× the cap. A4's FAIL stands as a read at the cap.
The per-model arm is the only trainer on this rung that arrives at the
script's speed; the control passes a round and a half behind it.

### Provenance

| field | value |
|---|---|
| date | 2026-09-15 (resumed 08:36, exited 10:00) |
| GPU / no-GPU | GPU (RTX 4090), the box to itself |
| seeds | 1 / 2 / 3, the A4 runs continued (`--resume-from`, optimizer and generator restored, `ent_coef` 0.003 carried; cadences 512 / 512 passed explicitly per #346) |
| n | 100 at seed base 700000 (final); 30 at 500000 (in-run); 100 at 900000 (greedy against sampled) |
| config | `configs/experiments/curriculum/a4.yaml`, unrefereed by design |
| decode | none |
| paired | per episode against `squad_march_take`; against A4's read on the same seeds and checkpoints |
| comparator | `squad_march_take`: 1.000, held 3.00, 4.70 turns, `on_obj` 0.724, coherent 0.923; the control at 122,880: 0.980 / 0.980 / 0.990, held 2.98–2.99, 6.12–6.23 turns |
| opponent | none |
| budget | 245,760 rounds (122,880 more), 128 rounds per update |
| code revision | `d9cb482` on `feature/curriculum-a4` (PR #352), the pre-registration's own commit, 08:36:52 |
| checkpoint | `last.pt` at 245,760, written 10:00 on exit |
| coherency | greedy 0.701 / 0.518 / 0.668 at 700000+; 0.73 / 0.57 / 0.68 greedy against 0.47 / 0.42 / 0.45 sampled at 900000+ |
| Wandb | `curriculum-a4`: `la0ztx25` s1 · `262q9d0k` s2 · `rfqku890` s3, continued |
| pre-registration | `reports/2026-09-15-curriculum-a-rungs.md#2026-09-15-curriculum-A4x-preregistration` at `d9cb482`, before the resume |

### The read

| row | success | turns | vs bar, paired | held | on obj | coherent | in-run: rolling-5 ≥ 95%, and stays |
|---|---|---|---|---|---|---|---|
| `squad_march_take` | 1.000 | 4.70 | — | 3.00 | 0.724 | 0.923 | — |
| s1 at 122,880 (A4) | 0.930 | 5.36 | +0.66 | 2.91 | 0.717 | 0.441 | — |
| **s1 at 245,760** | **0.980** | 4.75 | +0.05 ± 0.07 | 2.96 | 0.685 | 0.701 | 125k, stays from 245k |
| s2 at 122,880 (A4) | 0.930 | 5.46 | +0.76 | 2.93 | 0.603 | 0.431 | — |
| **s2 at 245,760** | **1.000** | 4.72 | +0.02 ± 0.07 | 3.00 | 0.652 | 0.518 | 88k, stays from 241k |
| s3 at 122,880 (A4) | 0.970 | 5.21 | +0.51 | 2.97 | 0.607 | 0.528 | — |
| **s3 at 245,760** | **1.000** | 4.73 | +0.03 ± 0.06 | 3.00 | 0.734 | 0.668 | 88k, stays from 198k |
| control s1–s3 at 122,880 | 0.980 / 0.980 / 0.990 | 6.12 / 6.23 / 6.16 | +1.42 to +1.53 | 2.98–2.99 | 0.73–0.78 | 0.74–0.86 | passed at epochs 44–58 |

In-run, rolling mean of five at n=30: 50% at 17k / 29k / 11k, 80% at 89k
/ 69k / 38k, 95% at 125k / 88k / 88k. The n=30 curve then wobbles 90–100
on s1 and s2 through most of the resumed half (the "stays" column is the
last dip below 95), as it did on A3x and on A3's control; the verdict is
read at n=100.

**Greedy against sampled** (900000+, n=100, paired): −0.7 ± 0.5 / −0.5 ± 0.6 / −0.8 ± 0.5 vp; `held` 2.99 / 3.00 / 2.99 greedy against 2.98 / 3.00 / 2.99 sampled; coherency 0.73 / 0.57 / 0.68 greedy against **0.47 / 0.42 / 0.45** sampled. The sampled policy holds the points as well as the greedy one and walks the squads further apart, as on every A rung.

**Health panel, last quarter** (480 updates per seed): explained variance
**0.56 / 0.58 / 0.59** (up from 0.54 / 0.43 / 0.42 at the cap), clip
fraction 0.21 / 0.28 / 0.26, ratio p99 1.84 / 2.01 / 2.03, displacement
entropy 0.75 / 0.89 / 0.68, declaration entropy ≤ 0.02, advantage std
0.24–0.36.

### What it says

- **The rule the cap implied is confirmed.** A4's control passed at
  92k–121k of its 123k, so 6× it was 725k and the cap bound at 123k; the
  per-model arm read 0.93 / 0.93 / 0.97 there and reads 0.98 / 1.00 /
  1.00 at 2× the cap. When 6× the control exceeds the cap, read the
  per-model arm at 2× the cap, once.
- **At the script's speed, a round and a half ahead of the control.**
  Turns 4.72–4.75 against 4.70 and 6.12–6.23 — the same pattern as A0,
  A1x and A3x: the per-model arm matches the straight-line walk where the
  whole-army trainer is a round or more behind.
- **The spare squad ends on a point, as on A4.** `on_obj` 0.65–0.73
  against the script's 0.724: the surplus stands on a held point rather
  than off it, which this criterion neither rewards nor punishes.
- **Coherency rose with the pass** — greedy 0.52–0.70 against 0.43–0.53
  at the cap — without anything paying for it; still below the bar's 0.92
  and the E rungs' referee will price it.
- **The critic followed the policy**, as on A3x: explained variance
  0.42–0.54 at the cap, 0.56–0.59 at the end.

### What was not done

- No recording of the residual 2% of failing episodes on s1.
- Nothing about the rungs above: the ladder is paused (the user's
  instruction, 2026-09-15) for the A3 speed screen (PR #362, arms
  #359–#361), so A5b does not launch on this pass.

**Amendment, 10:14 the same morning.** The pause was lifted while this
was being written ("continue with the plan, get through the ladder"):
A5b (#355) launched at 10:07 at 245,760 rounds and `ent_coef` 0.003
(its pre-registration's amendment 1), beside the speed-screen runs. The
last bullet above records what was true when the read was taken.

---

<a id="2026-09-15-curriculum-A5-preregistration"></a>

## Pre-registration: curriculum rung A5 — eight squads, six objectives

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

### The one change

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

### Comparator, measured first

`squad_march_take`, scored through both facades on seeds 700000+ at n=100
(`just measure-bridge configs/experiments/curriculum/a5.yaml 100`), bridge
identical on every shared field: success **1.000**, `held` **6.00** of 6,
turns **6.77** (sd 0.68, range 6–10), `on_obj` 0.856, coherent **0.822**,
vp 66.3 ± 1.0. In-run band (n=30, 500000+): success 1.000, turns 6.77,
held 6.00, coherent 0.835. At eight rounds the same script read 0.980 /
held 5.98 / 6.74 turns — the clock was set before any training number
existed.

### Criteria

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

### Budget, regime, seeds

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

### Primary readouts

Success rate (decides); turns paired against the script and against the
control; `held`, `on_obj`, coherency; the passive pair and
`train/rounds_per_update` on every per-model row; the health panel over
the last quarter; rounds-to-pass on both arms.

### What I expect (a guess, written so it can be wrong)

Both facades pass, the control needing its extension as it did on A3
(more bodies, more points, the same 60-epoch cap). The per-model arm's
budget is the cap and it is the slowest run of the A rungs by wall-clock,
eight units of three being eight decisions a round per unit-open plus
twenty-four moves. Coherency on both learned arms ends well below the
bar's 0.822.

### Amendment 1 — the control's read at the cap (2026-09-15, before any per-model number exists)

The whole-army control ran to its 60-epoch cap (Wandb group `curriculum-a5`,
runs `koti5sv5` s1 · `jemeom83` s2 · `49n4y1l5` s3). Scored at n=100 on
seeds 700000+ from `last.ckpt` (epoch 60): success **0.890 / 0.820 /
0.600**, `held` 5.88 / 5.81 / 5.58 of 6, `on_obj` 0.93 / 0.92 / 0.95,
turns 8.31 / 8.50 / 8.79 against the script's 6.77, coherent 0.58 / 0.64 /
0.58 against the bar's 0.822. The control fails the bound on 3 of 3 at
the cap with nearly every body on a point and five to six points held —
the shape of a control still climbing, as A3's was. Under the criteria
above it is **resumed once to 120 epochs** (`--resume-ckpt-path`, suffix
`-x2`, the same three checkpoints) and read there; its in-run curve is
read from Wandb once nothing is training. If it fails ≥ 2 of 3 at 120,
the rung is NULL (scenario). The per-model budget, if the arm launches, is
the cap. Nothing about the per-model arm was known when this was written.

---

<a id="2026-09-15-curriculum-a5-control-cannot-allocate"></a>

## Curriculum rung A5: the whole-army control cannot put a squad on every point — NULL (scenario) as pre-registered, and a defect in the clause

**Verdict first.** On the last movement-only rung of the ladder (#340, arm
#353) — eight squads of three, six objectives, ten rounds, success only
when every point has a body on it — the whole-army control reads success
**0.890 / 0.820 / 0.600** at its 60-epoch cap and **0.800 / 0.940 /
0.980** after the pre-registered extension to 120 epochs (245,760
rounds). Two of three seeds miss the bound at the extended budget, which
the pre-registration's letter calls **NULL (scenario)**, and that verdict
is recorded. The per-model arm was not launched under it. But the
scenario is not misconfigured: the script scores **1.000** on it, and the
control's failure has a shape — `held` 5.76 / 5.93 / 5.98 of 6 with
`on_obj` 0.93 / 0.93 / 0.96, i.e. **twenty-two of twenty-four bodies on
points and one point empty** in the failing episodes — that is the
allocation failure the record already attributes to the whole-army
trainer at this exact shape (`CLAUDE.md` § Allocation, § Holding pays —
"the agent stacks"). A NULL clause that fires when the control cannot do
what the script can is measuring the control, and the ladder was built
to measure the per-model arm against it. A second arm on this rung,
A5b, is pre-registered with the clause corrected: the control gates the
budget, the script gates the scenario, and the per-model arm is read
against both.

### Provenance

| field | value |
|---|---|
| date | 2026-09-15 (control 01:26–01:38; extension 01:39–01:50) |
| GPU / no-GPU | GPU (RTX 4090), alongside the A2 per-model arm |
| seeds | 1 / 2 / 3, extended in place with `--resume-ckpt-path` |
| n | 100 at seed base 700000 (final); 30 at 500000 (in-run) |
| config | `configs/experiments/curriculum/a5.yaml` — unrefereed by design |
| decode | K=1, undecoded |
| paired | per episode against `squad_march_take` on identical seeds |
| comparator | `squad_march_take`: success 1.000, held 6.00, turns 6.77, coherent 0.822 |
| opponent | none |
| budget | 60 epochs of 2048 steps, extended once to 120 |
| code revision | `5845e1c` on `feature/curriculum-a5` (PR #354) |
| checkpoint | `last.ckpt` at 60 and at 120 |
| coherency | 0.58 / 0.64 / 0.58 at 60; 0.62 / 0.57 / 0.47 at 120 |
| Wandb | `curriculum-a5`: `koti5sv5` / `jemeom83` / `49n4y1l5` (to 60), `5wh7plaj` / `gmonofrt` / `jlnshwd0` (60–120) |
| pre-registration | `reports/2026-09-15-curriculum-a-rungs.md#2026-09-15-curriculum-A5-preregistration` at `297b252`; amendment 1 (the cap read and the extension) at `5845e1c` |

### The read

| row | success | turns | vs bar | held | on obj | coherent |
|---|---|---|---|---|---|---|
| `squad_march_take` | 1.000 | 6.77 | — | 6.00 | 0.856 | 0.822 |
| control s1 at 60 | 0.890 | 8.31 | +1.54 | 5.88 | 0.934 | 0.576 |
| control s2 at 60 | 0.820 | 8.50 | +1.73 | 5.81 | 0.922 | 0.637 |
| control s3 at 60 | 0.600 | 8.79 | +2.02 | 5.58 | 0.945 | 0.577 |
| **control s1 at 120** | **0.800** | 8.32 | +1.55 ± 0.12 | 5.76 | 0.927 | 0.619 |
| **control s2 at 120** | **0.940** | 7.96 | +1.19 ± 0.10 | 5.93 | 0.930 | 0.570 |
| **control s3 at 120** | **0.980** | 7.21 | +0.44 ± 0.09 | 5.98 | 0.964 | 0.474 |

The in-run curves and rounds-to-pass are read from Wandb once the A2
per-model arm has exited and are appended to #353. Seed 1 moved
*backwards* between 60 and 120 (0.890 → 0.800) while seeds 2 and 3 moved
up (0.820 → 0.940, 0.600 → 0.980); at n=100 the SE is 0.02–0.05, so the
first is a real move and the three seeds are not converging on one
answer.

### What it says

- **The verdict is NULL (scenario) on the letter, and the letter is
  wrong here.** The clause was written for a rung nobody can solve, and
  the script solves this one every time. What failed is the pipeline
  control, on the shape the record already says it fails on: more bodies
  than points, and a policy that stacks. A5 is the first rung where the
  control does not simply learn the task slower than the script; it
  does not learn it at all in 245k rounds, and the way it fails is the
  full game's known failure in miniature.
- **The control's failure is a finding, not a null.** Every earlier rung
  read the control as a yardstick. Here the yardstick is the thing the
  ladder was built to improve on. A per-model arm that passes A5 would be
  the first evidence for #283's premise — per-step credit against a
  shared allocation problem — and a per-model arm that fails the same
  way would say the architecture does not buy it. The letter of the
  pre-registration forbids launching it; the point of the ladder demands
  it. The resolution is a second arm with the clause corrected before it
  runs, not a reading of the first arm against a clause it did not sign.
- **Coherency collapsed with the extension** (0.58–0.64 → 0.47–0.62),
  with nothing paying for it and twenty-four bodies squeezing onto six
  radius-4 discs. It is a readout on this rung; the E rungs' referee will
  price it.
- **The clause moves.** From A5b on: the whole-army control calibrates
  the budget (6× its slowest pass, or the cap if it never passes) and is
  read beside the per-model arm as the comparison it is; NULL (scenario)
  requires the **script** to fail its own criterion; a control that fails
  while the per-model arm passes is reported as exactly that. This goes
  in #340 as an amendment to its contract, and in `CLAUDE.md`.

### What was not done

- The per-model arm did not run under this arm.
- The control was not extended past 120; the once-only extension was
  the pre-registered limit.
- No recording of which point the control leaves empty, or whether it is
  the same point across episodes.

---

<a id="2026-09-15-curriculum-A5b-preregistration"></a>

## Pre-registration: curriculum rung A5, second arm (A5b) — the per-model arm read beside the control, not gated by it

Written 2026-09-15, **before any per-model number on this rung exists**,
on branch `feature/curriculum-a5` (PR #354 carries the rung's arms).
Issue #355, parent question #340, rung **A5** — a follow-up to #353,
which read **NULL (scenario)** as pre-registered when the whole-army
control missed the bound on 2 of 3 seeds at its extended budget
([report](2026-09-15-curriculum-a-rungs.md#2026-09-15-curriculum-a5-control-cannot-allocate)).

### The one change

**The attribution clause.** The scenario, the config
(`configs/experiments/curriculum/a5.yaml`, unchanged at `297b252`), the
bar, the seeds and the per-model regime are A5's. What changes is how the
per-model read is attributed: the whole-army control calibrates the
budget and is read *beside* the per-model arm as the comparison it is;
NULL (scenario) requires the **script** to fail its own criterion. A
control that cannot do what the script can is a finding about the
control, and this rung is where the ladder first meets one.

### Comparator

`squad_march_take`, both facades, seeds 700000+ at n=100: success
**1.000**, `held` 6.00, turns 6.77 (sd 0.68), `on_obj` 0.856, coherent
0.822. Beside it, the whole-army control at 120 epochs (245,760 rounds):
success **0.800 / 0.940 / 0.980**, `held` 5.76 / 5.93 / 5.98, `on_obj`
0.93 / 0.93 / 0.96, turns 8.32 / 7.96 / 7.21, coherent 0.62 / 0.57 / 0.47.

### Criteria

Read at the END of the budget from `last.pt`, greedy, no decode, n=100
on seeds 700000+ (`just measure-rung`).

- **PASS:** success (every objective occupied at the end) ≥ 0.95 on
  **all three** seeds. Reported beside the control's 1 of 3: the
  per-model arm allocates eight squads over six points within 122,880
  rounds where the whole-army trainer does not within 245,760.
- **FAIL:** any seed below 0.95 at budget. Read beside the control's
  shape: if the per-model arm fails the same way — `held` below 6 with
  `on_obj` above 0.9, one point empty while bodies stack on another —
  neither trainer solves allocation at this shape and per-step credit did
  not buy it here; if it fails differently, the report says how.
- **NULL (scenario):** only if the script fails its own criterion. It
  does not.
- **Pass with a defect:** the bound holds but the health panel is red
  over the last quarter (the ratio line at 1.6–1.9 is this regime's
  normal).

**Readouts, not criteria:** turns paired against the script and the
control; `held` and `on_obj`; coherency greedy and sampled; per-model
decisions per round.

Power: binomial SE 0.022 at p=0.95, n=100. The control's best seed sits
0.98 at n=100 and its worst 0.80 — nine SE apart — so a 3/3 per-model
pass is a move, not noise, and so is a 0/3.

### Budget, regime, seeds

| | per-model arm |
|---|---|
| budget | **122,880 rounds** (the cap; 6× anything the control needs exceeds it), 128 rounds per update (`--rollout-rounds 32 --num-rollout-envs 4`) |
| seeds | 1, 2, 3; rollout layouts at seed×100+ |
| in-run eval | every 512 rounds, n=30, seeds 500000+, greedy, passive pair and `eval/mean_turns` logged |
| final score | n=100, seeds 700000+ |
| other flags | defaults (`gamma` 0.9, `ent_coef` 0.03, `lr` 3e-4) |
| logging | Wandb group `curriculum-a5`, one run per seed |
| start | from scratch (T1 is the warm-started pair) |
| order | launches after A4's per-model arm (#351) has been read — the ladder's order holds |

Launch: `just train-per-model-arm 122880 3 curriculum-a5 a5 "--num-rollout-envs 4 --rollout-rounds 32 --eval-every-rounds 512 --checkpoint-every-rounds 512 --n-eval-episodes 30" configs/experiments/curriculum/a5.yaml`.

### What I expect (a guess, written so it can be wrong)

The per-model arm passes on 2 or 3 seeds and needs most of the cap to do
it. Its `held` at the end is higher than the control's on every seed —
per-step credit against a shared coverage term is the mechanism #283
argued for, and this is the first rung that tests it. Coherency, unpaid,
ends below the control's.

### Amendment 1 — written 2026-09-15 before the launch: the budget and the entropy coefficient

Two rows in the table above contradict rules the ladder had already
adopted when this was written, and both are corrected here before any
number exists.

- **Budget: 245,760 rounds, read once at the end**, not 122,880. The
  cap rule is symmetric (A3x): the control on this rung ran its once-only
  extension to 120 epochs, 245,760 rounds, so the arm under test gets the
  same rounds before the rung is called — and A4x (read this morning:
  0.980 / 1.000 / 1.000 at 245,760 where 122,880 read 0.93 / 0.93 / 0.97)
  confirmed that when 6× the control exceeds the cap the per-model budget
  is 2× the cap. Reading A5b at 122,880 would repeat the asymmetry A3's
  FAIL measured.
- **`--ent-coef 0.003`**, not the default 0.03. Every per-model arm runs
  at 0.003 from A3 on (A2b: at 0.03 the arm solves a rung and then
  unlearns it). The "other flags" row above was carried from A5's
  control-side table.

Launch, as amended: `just train-per-model-arm 245760 3 curriculum-a5 a5b
"--num-rollout-envs 4 --rollout-rounds 32 --eval-every-rounds 512
--checkpoint-every-rounds 512 --n-eval-episodes 30 --ent-coef 0.003"
configs/experiments/curriculum/a5.yaml`, on the code of PR #362's tip
(`bd15db9`: the A5 branch plus the travel term's matching flag, which
`a5.yaml` does not set), beside the nine A3 speed-screen runs already on
the GPU. The criteria are unchanged; the in-run curve is read for drift
after any first pass, as the A2b clause requires.

---

<a id="2026-09-16-curriculum-a5b-fails-differently"></a>

## Curriculum rung A5b: the per-model arm fails the allocation rung, and not the way the control did — FAIL as pre-registered

**Verdict first.** On the last movement-only rung of the ladder (#340,
arm #355) — eight squads of three, six objectives, ten rounds, success
only when every point has a body on it — the per-model arm from scratch
reads greedy success **0.260 / 0.180 / 0.160** at 245,760 rounds, n=100,
`held` 4.57 / 4.10 / 4.21 of 6, `on_obj` **0.37 / 0.32 / 0.33**, turns
9.70 / 9.51 / 9.70 of a ten-round game against the script's 6.77,
coherent 0.10 / 0.23 / 0.10. **FAIL** on the pre-registered clause, on
every seed, and the arm never reached a rolling 50% in-run (peaks 48 /
43 / 30%). The whole-army control read 0.800 / 0.940 / 0.980 at the same
rounds by putting twenty-two of twenty-four bodies on points and leaving
one point empty. The per-model arm holds four points with a third of its
bodies on any point at all: this is not the control's allocation error,
it is the walk-off the A3 speed screen saw at 20k rounds, at full scale
and never outgrown. The pre-registration's guess — a pass on two or
three seeds with `held` above the control's — is wrong on every count.

### Provenance

| field | value |
|---|---|
| date | launched 2026-09-15 10:07; killed by the box crash at 16:38 (90,624 / 90,624 / 90,112 rounds); resumed in place 2026-09-16 12:43; exited 23:10 / 23:06 / 23:15 |
| GPU / no-GPU | GPU (RTX 4090), beside the A3 speed screen, then T1, C1 and C2 — up to twelve trainers |
| seeds | 1 / 2 / 3, from scratch; rollout layouts at seed×100+ |
| n | 100 at seed base 700000 (final); 30 at 500000 (in-run, every 512 rounds); 100 at 900000 (greedy against sampled); 20 at 700000 (census) |
| config | `configs/experiments/curriculum/a5.yaml` — unrefereed by design; success `all_objectives_occupied` |
| decode | none on the per-model facade; K=1 on the control |
| paired | per episode against `squad_march_take` on identical seeds |
| comparator | `squad_march_take`, both facades: success 1.000, held 6.00, turns 6.77, `on_obj` 0.856, coherent 0.822 |
| opponent | none |
| budget | 245,760 rounds at 128 per update (`--rollout-rounds 32 --num-rollout-envs 4`), `--ent-coef 0.003`; resumed at ~90k with cadences explicit (#346); the control's 120 epochs is the same rounds |
| code revision | first leg `bd15db9` (PR #362's tip); resumed leg `ed791f1` (PR #366's tree) — docs and the travel term's off-by-default flag differ, nothing `a5.yaml` sets |
| checkpoint | `last.pt` at 245,760; `checkpoints/per_model/per-model-a5-2026-09-15-10-07-{48,49,51}-s{1,2,3}a5b` |
| coherency | greedy 0.099 / 0.226 / 0.096 at 700000+; 0.10 / 0.21 / 0.09 greedy against 0.06 / 0.06 / 0.06 sampled at 900000+ |
| Wandb | `curriculum-a5`: `b9zwoiib` / `2t2nj8tb` / `g9pnffwx` (to the crash), `gegkkxnl` / `ipjuvfu8` / `jdyb1j5a` (resumed); control `koti5sv5` / `jemeom83` / `49n4y1l5` and `5wh7plaj` / `gmonofrt` / `jlnshwd0` |
| pre-registration | `reports/2026-09-15-curriculum-a-rungs.md#2026-09-15-curriculum-A5b-preregistration` (2026-09-15, before launch); amendment 1 (245,760 rounds, `ent_coef` 0.003) at `80b1234`, before launch |

### The read

| row | success | turns | vs bar, paired | held | on obj | coherent | stat | in-run: rounds to rolling 50 / 80 / 95%, peak |
|---|---|---|---|---|---|---|---|---|
| `squad_march_take` | 1.000 | 6.77 | — | 6.00 | 0.856 | 0.822 | — | — |
| **s1 at 245,760** | **0.260** | 9.70 | +2.93 ± 0.09 | 4.57 | 0.371 | 0.099 | 0.02 | never / never / never; peak 48% at 174k |
| **s2 at 245,760** | **0.180** | 9.51 | +2.74 ± 0.13 | 4.10 | 0.316 | 0.226 | 0.11 | never / never / never; peak 43% at 133k |
| **s3 at 245,760** | **0.160** | 9.70 | +2.93 ± 0.10 | 4.21 | 0.327 | 0.096 | 0.01 | never / never / never; peak 30% at 211k |
| control at 60 epochs | 0.890 / 0.820 / 0.600 | 8.31–8.79 | +1.54 to +2.02 | 5.58–5.88 | 0.92–0.95 | 0.58–0.64 | — | — |
| control at 120 epochs | 0.800 / 0.940 / 0.980 | 8.32 / 7.96 / 7.21 | +1.55 / +1.19 / +0.44 | 5.76 / 5.93 / 5.98 | 0.93 / 0.93 / 0.96 | 0.62 / 0.57 / 0.47 | — | — |
| s1 / s2 / s3 at their in-run peak (174k / 133k / 211k, `pm-*.pt`) | 0.330 / 0.360 / 0.220 | 9.40 / 9.44 / 9.76 | +2.63 / +2.67 / +2.99 | 4.68 / 4.79 / 4.29 | 0.37 / 0.37 / 0.32 | 0.14 / 0.13 / 0.11 | 0.02–0.04 | — |

**Greedy against sampled** (900000+, n=100, paired): vp +1.7 ± 0.9 /
+5.7 ± 1.3 / +3.1 ± 1.3; `held` 4.70 / 4.02 / 4.58 greedy against 4.62 /
4.23 / 4.42 sampled; coherency 0.10 / 0.21 / 0.09 greedy against 0.06 ×3
sampled; 285–311 decisions per episode either way. The policy a score
reports and the policy the rollouts train are the same policy here —
both hold four and a half points, and neither keeps a squad together.

**Health panel, last quarter** (~480 updates per seed): explained
variance 0.69 / 0.62 / 0.75, clip fraction **0.37 / 0.40 / 0.42**, ratio
p99 **2.25 / 2.29 / 2.40**, displacement entropy 1.47 / 1.63 / 1.66,
declaration entropy 0.04–0.06, advantage std 0.25–0.33. The last forty
in-run evaluations average 23.9 / 21.9 / 12.5%. Red on the clip line
and the tail together (A2's drift read 0.26–0.37 and 1.7–1.8), with the
critic fine — the update is fighting a policy that is not settling.

**The by-turn census** (n=20 on 700000+, bodies on objectives of 24 and
points held of 6, turn 3 → 5 → 7 → 10; which points are empty at the
end, by index 0–5):

| policy | success (n=20) | turns | on objectives, 3 → 5 → 7 → 10 | held, 3 → 5 → 7 → 10 | episodes with each point empty at the end | max stack |
|---|---|---|---|---|---|---|
| `squad_march_take` | 1.00 | 6.55 | 3.7 → 16.5 → 20.1 → 20.1 | 2.3 → 3.0 → 6.0 → 6.0 | 0 / 0 / 0 / 0 / 0 / 0 | 5.8 |
| control at 120 (from #353) | 0.80 / 0.94 / 0.98 | 7.2–8.3 | 22 of 24 on points at the end | 5.8–6.0 | one point, in the failing episodes | — |
| s1 | 0.20 | 9.85 | 0.4 → 5.2 → 9.1 → **8.7** | 0.4 → 2.5 → 3.4 → 4.4 | 0.25 / 0.25 / 0.20 / 0.35 / 0.25 / 0.30 | 3.4 |
| s2 | 0.20 | 9.40 | 1.3 → 7.9 → 8.3 → **7.5** | 1.1 → 2.6 → 4.6 → 4.2 | 0.35 / 0.40 / 0.65 / 0.05 / 0.05 / 0.30 | 3.0 |
| s3 | 0.15 | 9.75 | 0.4 → 5.9 → 8.2 → **8.5** | 0.4 → 2.7 → 3.8 → 4.5 | 0.10 / 0.40 / 0.20 / 0.40 / 0.35 / 0.10 | 3.5 |

**Two thirds of the army never arrives.** The script has 16.5 bodies on
points at turn 5 and 20 by turn 7, on all six; the arm has 5–8 at turn
5 and never more than ten, spread over four and a half points at two
bodies each. The empty point is a different one each episode (every
index is empty in 5–65% of episodes, no favourite), the max stack is
3.0–3.5 against the script's 5.8, and the sampled and greedy policies
agree. So this is neither the control's failure (bodies on points, one
point short) nor a stacking failure: it is **under-arrival** — squads
broken up (coherency 0.10), bodies drifting toward the points slowly,
`on_obj` flat or falling over the last three rounds. The A3 speed
screen's walk-off at 20k rounds is the same signature, and here it was
never outgrown.

### What it says

- **FAIL on every seed, and the failure is the arm's own.** The
  pre-registration's FAIL clause asked which way: if the arm fails the
  control's way (bodies on points, one empty), neither trainer solves
  allocation at this shape; if differently, say how. It fails
  differently. The control puts twenty-two bodies on points and leaves
  one empty; the arm puts eight to ten on points and holds four and a
  half. The control has an allocation problem; the arm has an
  **arrival** problem, and on this rung it is the larger one by far.
- **This is the ladder's first rung where the per-model arm is worse
  than the whole-army control at equal rounds, and by a wide margin**:
  0.16–0.26 against 0.80–0.98. On A3 it trailed at 123k and matched at
  245k; here at 245k it has not reached a rolling 50% in-run on any
  seed, and its peak was at 133k–211k with no climb after. The peak
  checkpoints score 0.33 / 0.36 / 0.22 at n=100 with the same `on_obj`
  (0.32–0.37) — the best this arm ever was is the same policy, a little
  less of it. Never learned, not drifted; more rounds is not the reading.
- **What changed from A4 to A5 is the distance and the count, and on
  this facade the count dilutes the credit.** A4's bar walks 4.70 turns,
  A5's 6.77 — farther points, twice the bodies. `gamma` counts rounds
  here (0.9 per round spans the ten-round game), so the horizon is not
  it. What the record does say is that every per-model state term on
  this facade — the travel term included — is paid at the turn close as
  the **mean over alive models** (§ The per-model curriculum): the
  scalar a body sees for its own inch closed is one twenty-fourth of the
  army's, against one twelfth on A3–A4 and one third on A1. The A rungs
  passed as that dilution doubled, and this is the rung where it doubled
  again and the arm stopped arriving. A hypothesis, not a finding — the
  build that would test it (pay a state term to the model that produced
  it, `envs/per_model/reward_timing.py`) is the one the speed screen
  already named.
- **The health panel is red and says the same thing.** Clip fraction
  0.37–0.42 and ratio p99 2.25–2.40 over the last quarter, with the
  critic at 0.62–0.75 explained variance — A2's drift signature, at the
  entropy coefficient that fixed A2. The critic can value the states the
  policy reaches; the policy is not settling on how to reach better ones.
- **Coherency 0.10 greedy, 0.06 sampled — the lowest on the ladder.**
  Unpaid on every rung, and here the squads dissolve. Whether formation
  would help arrival is a question for a rung that pays for it (the E
  rungs' referee); the record says nothing pays for it here.
- **The control's pass is not a pass either.** Read beside each other:
  one trainer leaves a point empty, the other leaves most of the army in
  the open. Neither solves eight squads over six points from scratch,
  and the script does at 1.000 in 6.77 turns. #340's answer for this
  axis is that per-step credit did not buy allocation here; T1 says
  whether a warm start buys arrival.

### What was not done

- No second budget: the symmetric cap already put the arm at 245,760.
- No arm at `ent_coef` 0.03 on this rung (A2's lesson stands; the clip
  fraction here is already A2's failing range at 0.003).
- The control's in-run curve was read from Wandb after its runs exited
  and is on #353; it is not re-read here.

---

<a id="2026-09-15-curriculum-T1-preregistration"></a>

## Pre-registration: curriculum rung T1 — A5 warm-started from A4x's checkpoint, against A5b from scratch (the transfer rung)

Written 2026-09-15, **before any number from the arm exists**, on branch
`feature/curriculum-t1` (stacked on PR #362). Issue #363, parent
question #340, rung **T1**. The comparator, A5b (#355), is in flight at
the time of writing: 85k of 245,760 rounds, in-run 5–19%.

### The one change

**The starting weights.** `configs/experiments/curriculum/a5.yaml`
unchanged (eight squads of three, six objectives at radius 4, ten
rounds), each seed started with `--warm-start-from
checkpoints/per_model/per-model-a4-2026-09-15-07-08-53-s{1,2,3}a4/last.pt`
— the A4x checkpoints at 245,760 rounds that read 0.980 / 1.000 / 1.000
on A4. Fresh optimizer. The set network loads across the army size
(twelve bodies to twenty-four, four squads to eight, three objectives to
six): smoke-tested for 256 rounds before this was written. Everything
else is A5b's — 245,760 rounds at 128 per update (`--rollout-rounds 32
--num-rollout-envs 4`), `--ent-coef 0.003`, in-run eval every 512 rounds
at n=30 on 500000+, checkpoints every 512, Wandb group `curriculum-t1`.

### Why this rung, and why now

#340's T1 asks whether the size-independence the set network claims
delivers in practice: a rung warm-started from the rung below should
pass in fewer rounds than the same rung from scratch. This morning's A3
speed screen (S3, #361) measured that one rung early: A3 from A2b read
AHEAD on 3 of 3 (0.91 / 0.94 / 0.96 against scratch's 0.70 / 0.80 / 0.77),
with in-run 80% at 11k–22k rounds against scratch's 91k–never and no
walk-off from 20k on. That is the prior. T1 launches beside A5b rather
than after it because the arm does not need A5b's numbers to run, only
to be read, and the GPU would otherwise sit half idle for five hours.

### Comparator

A5b's three runs from scratch (Wandb `curriculum-a5`: `b9zwoiib` s1 ·
`2t2nj8tb` s2 · `g9pnffwx` s3), read at 245,760 on the same seeds:
their in-run rounds-to-pass and their n=100 final read. Paired on
layouts and seeds, **not on init**. The bar: `squad_march_take`, both
facades, seeds 700000+ at n=100: success 1.000, `held` 6.00, turns
6.77, `on_obj` 0.856, coherent 0.822. The whole-army control at 245,760:
0.800 / 0.940 / 0.980, `held` 5.76 / 5.93 / 5.98.

### Criteria

**Rounds-to-pass** = the first in-run evaluation at which the rolling
mean of five (n=30 on 500000+, every 512 rounds) reaches 95%, capped at
the budget; a seed that never reaches it in-run has rounds-to-pass
245,760.

- **PASS:** on every seed, T1's rounds-to-pass ≤ half of A5b's same-seed
  rounds-to-pass (a seed on which A5b never passes in-run bounds T1 at
  122,880), **and** T1 reads success ≥ 0.95 at n=100 on 700000+ from
  `last.pt` at 245,760 on all three seeds. #340's letter is "rounds-to-pass
  ≤ half of scratch, 3/3 seeds"; the n=100 clause is added so that a
  "pass" is a pass by the rungs' own standard and not an n=30 wobble.
- **FAIL:** any seed misses the halving bound, or any seed reads below
  0.95 at 245,760. #340 names the consequence: the size-independence
  claim is not delivering in practice, every later rung trains from
  scratch, and #283's premise gets a report.
- **NULL (scenario):** only if the script fails its own criterion. It
  does not.
- **Pass with a defect:** the bound holds but the health panel is red
  over the last quarter.

**Readouts, not criteria:** success / `held` / turns at the 122,880
checkpoint (`pm-00122880.pt`), where A5's cap would have read;
coherency greedy and sampled (`just measure-per-model-eval-mode`, 900000+);
the by-turn census at 20,480 (bodies on objectives and points held, turn
5 → 8: the walk-off signature); the health panel over the last quarter;
A5b beside every row.

Power: the bound is a factor of two on a curve sampled every 512 rounds
at n=30 — coarse, and deliberately so; on A3 the same lever read a factor
of four or more on every seed. Binomial SE at p=0.95, n=100 is 0.022.

### Budget, regime, seeds

| | T1 arm |
|---|---|
| budget | 245,760 rounds, 128 rounds per update |
| seeds | 1, 2, 3; rollout layouts at seed×100+ |
| start | A4x `last.pt` of the same seed (245,760 rounds, revision `d9cb482`) |
| in-run eval | every 512 rounds, n=30, seeds 500000+ |
| final score | n=100, seeds 700000+ |
| flags | `--num-rollout-envs 4 --rollout-rounds 32 --eval-every-rounds 512 --checkpoint-every-rounds 512 --n-eval-episodes 30 --ent-coef 0.003 --warm-start-from <A4x last.pt>` |
| logging | Wandb group `curriculum-t1`, one run per seed |
| code | the `feature/curriculum-t1` working tree at this commit (PR #362's tree plus this file) |

### What I expect (a guess, written so it can be wrong)

PASS 3/3: in-run 95% by 60k–90k rounds on every seed against an A5b
that reaches it late or not at all (its control read 0.80 / 0.94 / 0.98
at 245,760). The doubling of the army costs the transfer something — the
A4x policy has never seen eight squads or six points — so the factor is
smaller than A3's four-plus, but above two. If instead T1 reads under
0.95 on a seed at 245,760, the likely shape is the control's: one point
left empty while bodies stack, and the report says whether that seed's
A5b run did the same.

---

<a id="2026-09-17-curriculum-t1-transfer-does-not-rescue-a5"></a>

## Curriculum rung T1: warm-starting A5 from A4x does not rescue it — FAIL as pre-registered, and the transfer is real

**Verdict first.** On the transfer rung (#340, arm #363) — A5 (eight
squads of three, six objectives, ten rounds) started from A4x's
checkpoints instead of from scratch, seed for seed, read against A5b —
no seed reaches a rolling 95% in-run at any point in 245,760 rounds, so
every seed's rounds-to-pass is the budget, and the halving bound is
missed on all three; at n=100 from `last.pt` the arm reads greedy
success ****0.450 / 0.290 / 0.200** (A5b: 0.260 / 0.180 / 0.160)**. **FAIL** on both clauses of the pre-registration.
What the warm start *does* buy is visible in every readout: T1 is ahead
of A5b at matched rounds on every seed and holds more points at the end
(4.91 / 4.25 / 4.53 of 6 against A5b's 4.57 / 4.10 / 4.21), with the health
panel in the normal range where A5b's is red — the transfer carries
something, and the something is not enough on a rung neither trainer
solves. The consequence #340 names is applied: every later rung trains
from scratch unless the warm start is the rung's own axis, and the
size-independence claim goes to #283 with this number on it.

### Provenance

| field | value |
|---|---|
| date | launched 2026-09-15 16:28; killed by the box crash at 16:38 (~4k rounds); resumed in place 2026-09-16 12:43; exited 2026-09-17 02:31 / 02:51 / 02:44 |
| GPU / no-GPU | GPU (RTX 4090), beside A5b, C1 and C2 — up to twelve trainers |
| seeds | 1 / 2 / 3; rollout layouts at seed×100+; each seed from A4x `last.pt` of the same seed (245,760 rounds on A4: 0.980 / 1.000 / 1.000), fresh optimizer — **not paired on init with A5b** |
| n | 100 at seed base 700000 (final and the 122,880 readout); 30 at 500000 (in-run, every 512 rounds); 100 at 900000 (greedy against sampled); 20 at 700000 (census, at 20,480 and at the end) |
| config | `configs/experiments/curriculum/a5.yaml` — unrefereed by design; success `all_objectives_occupied` |
| decode | none on the per-model facade |
| paired | per episode against `squad_march_take` on identical seeds; against A5b on identical seeds and layouts, not on init |
| comparator | A5b's three runs from scratch (0.260 / 0.180 / 0.160 at 245,760, never a rolling 50% in-run); the bar `squad_march_take`: success 1.000, held 6.00, turns 6.77, `on_obj` 0.856, coherent 0.822; the whole-army control at 245,760: 0.800 / 0.940 / 0.980 |
| opponent | none |
| budget | 245,760 rounds at 128 per update (`--rollout-rounds 32 --num-rollout-envs 4`), `--ent-coef 0.003`; resumed at ~4k with cadences explicit (#346) |
| code revision | `969ec81` on `feature/curriculum-t1` (PR #364) at launch; resumed leg on `ed791f1` (PR #366's tree) — docs only between them for this config |
| checkpoint | `last.pt` at 245,760; `checkpoints/per_model/per-model-a5-2026-09-15-16-28-45-s{1,2,3}t1` |
| coherency | greedy 0.225 / 0.157 / 0.193 at 700000+; 0.21 / 0.17 / 0.19 greedy against 0.12 / 0.11 / 0.13 sampled at 900000+ |
| Wandb | `curriculum-t1`: `v7lt4mab` / `k3azl8f4` / `lxxheeb2` (to the crash), `34o8gvat` / `2zf3ef3m` / `134z8z21` (resumed); A5b `b9zwoiib` / `2t2nj8tb` / `g9pnffwx` + `gegkkxnl` / `ipjuvfu8` / `jdyb1j5a` |
| pre-registration | `reports/2026-09-15-curriculum-a-rungs.md#2026-09-15-curriculum-T1-preregistration` at `969ec81` (2026-09-15 16:28), before any number |

### The read

| row | success | turns | vs bar, paired | held | on obj | coherent | stat |
|---|---|---|---|---|---|---|---|
| `squad_march_take` | 1.000 | 6.77 | — | 6.00 | 0.856 | 0.822 | — |
| **T1 s1 at 245,760** | **0.450** | 8.85 | +2.08 ± 0.16 | 4.91 | 0.423 | 0.225 | 0.02 |
| **T1 s2 at 245,760** | **0.290** | 9.13 | +2.36 ± 0.16 | 4.25 | 0.317 | 0.157 | 0.06 |
| **T1 s3 at 245,760** | **0.200** | 9.69 | +2.92 ± 0.11 | 4.53 | 0.370 | 0.193 | 0.10 |
| T1 s1 / s2 / s3 at 122,880 (readout) | 0.240 / 0.140 / 0.590 | 9.41 / 9.72 / 8.63 | +2.64 / +2.95 / +1.86 | 4.41 / 4.07 / 5.19 | 0.39 / 0.34 / 0.43 | 0.14 / 0.13 / 0.21 | 0.02–0.07 |
| T1 s1 / s2 / s3 at their in-run peak (233k / 239k / 128k, `pm-*.pt`, readout) | 0.470 / 0.370 / **0.640** | 9.03 / 9.11 / 8.44 | +2.26 / +2.34 / +1.67 | 4.95 / 4.79 / 5.37 | 0.43 / 0.40 / 0.48 | 0.15 / 0.17 / 0.23 | 0.00–0.04 |
| A5b s1 / s2 / s3 at 245,760 | 0.260 / 0.180 / 0.160 | 9.70 / 9.51 / 9.70 | +2.93 / +2.74 / +2.93 | 4.57 / 4.10 / 4.21 | 0.37 / 0.32 / 0.33 | 0.10 / 0.23 / 0.10 | 0.01–0.11 |
| control at 120 epochs (245,760) | 0.800 / 0.940 / 0.980 | 8.32 / 7.96 / 7.21 | +1.55 / +1.19 / +0.44 | 5.76 / 5.93 / 5.98 | 0.93 / 0.93 / 0.96 | 0.62 / 0.57 / 0.47 | — |

Binomial SE at n=100 is 0.04–0.05 at these rates: T1 is above A5b on
every seed (+0.19 / +0.11 / +0.04), the third within one SE. Seed 3's
122,880 readout (0.590) is its best read and its final (0.200) its
worst — it peaked in-run at 128k and fell.

**In-run, rolling five of n=30, T1 against A5b at matched rounds:**

| rounds | T1 s1 / s2 / s3 | A5b s1 / s2 / s3 |
|---|---|---|
| 20,480 | 17 / 3 / 34 | — |
| 61,440 | 46 / 11 / 27 | — |
| 122,880 | 17 / 19 / 58 | — |
| 184,320 | 15 / 27 / 30 | — |
| peak (at) | **64** (233k) / **51** (239k) / **64** (128k) | 48 (174k) / 43 (133k) / 30 (211k) |
| rolling ≥ 50% first at | 62k / 238k / 121k | never / never / never |
| mean of the last 40 | 46.4 / 40.8 / 25.8 | 23.9 / 21.9 / 12.5 |
| rounds-to-pass (rolling 95%) | 245,760 ×3 (never) | 245,760 ×3 (never) |

The bound was rounds-to-pass ≤ half of A5b's, i.e. ≤ 122,880 on a seed
where A5b never passes. No T1 seed passes in-run at all.

**Greedy against sampled** (900000+, n=100, paired): vp −0.1 ± 1.6 /
+3.7 ± 1.6 / −0.2 ± 1.3; `held` 5.05 / 4.04 / 4.54 greedy against 5.01 /
4.65 / 4.32 sampled; coherency 0.21 / 0.17 / 0.19 against 0.12 / 0.11 /
0.13; 273–288 decisions per episode. As on A5b, the trained policy and
the scored one are the same policy.

**Health panel, last quarter** (~480 updates per seed): explained
variance 0.49 / 0.58 / 0.48, clip fraction **0.22 / 0.27 / 0.32**, ratio
p99 **1.76 / 1.91 / 2.00**, displacement entropy 1.13 / 1.16 / 1.42,
declaration entropy 0.06–0.07, advantage std 0.39–0.44. In the regime's
normal range on the clip line and the tail (A5b: 0.37–0.42 and
2.25–2.40, red), with a weaker critic (A5b 0.62–0.75).

**The by-turn census** (n=20 on 700000+, bodies on objectives of 24 and
points held of 6, turn 3 → 5 → 7 → 10, at 20,480 rounds and at the end):

| policy | success (n=20) | turns | on objectives, 3 → 5 → 7 → 10 | held, 3 → 5 → 7 → 10 | max stack |
|---|---|---|---|---|---|
| `squad_march_take` | 1.00 | 6.55 | 3.7 → 16.5 → 20.1 → 20.1 | 2.3 → 3.0 → 6.0 → 6.0 | 5.8 |
| s1 at 20,480 | 0.25 | 9.10 | 0.9 → 10.8 → 11.3 → 10.4 | 0.7 → 2.8 → 4.7 → 4.6 | 4.0 |
| s2 at 20,480 | 0.00 | 10.00 | 0.7 → 8.6 → 7.2 → 7.5 | 0.6 → 2.7 → 3.5 → 3.9 | 3.2 |
| s3 at 20,480 | 0.15 | 9.60 | 1.6 → **13.0 → 8.2 → 6.6** | 1.2 → 3.0 → 4.2 → 3.8 | 2.9 |
| s1 at 245,760 | 0.25 | 9.25 | 0.5 → 10.2 → 10.9 → 10.0 | 0.4 → 2.8 → 5.1 → 4.8 | 3.9 |
| s2 at 245,760 | 0.40 | 8.80 | 1.4 → 9.2 → 8.6 → 8.3 | 1.2 → 2.8 → 4.9 → 4.8 | 3.0 |
| s3 at 245,760 | 0.25 | 9.60 | 1.0 → 9.3 → 9.3 → 8.8 | 0.8 → 2.8 → 4.0 → 4.5 | 3.7 |
| A5b at 245,760 (s1 / s2 / s3) | 0.20 / 0.20 / 0.15 | 9.4–9.9 | 8.7 / 7.5 / 8.5 at the end | 4.4 / 4.2 / 4.5 at the end | 3.0–3.5 |

**At 20,480 rounds the warm start is already where A5b ends at
245,760** — eight to eleven bodies on points, four points held — and
seed 3 shows the walk-off the pre-registration asked about, thirteen
bodies on points at turn 5 down to seven by the end. **225k more rounds
buy almost nothing**: the end-of-run rows hold 4.5–4.8 with eight to
ten bodies on points, the walk-off is gone on seed 3 (flat from turn 5)
and seed 2 has climbed from 0.00 to 0.40, but no seed sends the other
fourteen bodies anywhere near a point. The A4x policy knows how to walk
a squad to the nearest point; it did not know, and did not learn, to
send eight squads to six.

### What it says

- **FAIL on the letter, twice over.** #340's clause is rounds-to-pass
  at most half of scratch on every seed; the n=100 clause added by the
  pre-registration is 0.95 at the end. No seed passes in-run and none
  passes at n=100. The consequence the question names is applied from
  here: **later rungs train from scratch unless the start is the rung's
  own axis** (C1 and C2 already warm-start by the C-rung rule, written
  and measured before this; that rule stands on its own S3 and C1
  evidence, and this result does not touch it — see the next bullet).
- **The transfer is real and it is not enough.** Every readout puts T1
  ahead of A5b: at n=100 +0.19 / +0.11 / +0.04 at the end; in-run peaks 64 / 51 / 64 against 48 / 43 / 30; a rolling 50% reached on every seed (62k / 238k / 121k) where A5b never reached it; `held` 4.9 / 4.3 / 4.5 against 4.6 / 4.1 / 4.2; and the panel green where A5b's is red. The A4x policy carries onto twenty-four
  bodies and six points — the network does load across the army size and
  the loaded weights do something useful — but on a rung neither trainer
  solves from scratch, a head start on a curve that never reaches the
  mark cannot halve a rounds-to-pass that is infinite. The halving bound
  presupposed that scratch passes; A5b did not, so the rung measured
  transfer onto a failure. That is what the prediction on file missed:
  it guessed 95% by 60k–90k against an A5b that "reaches it late or not
  at all", and the "not at all" case makes the bound unsatisfiable.
- **So the size-independence claim gets a number, not a verdict.** What
  T1 shows is that A4's twelve-body, three-point policy transfers to
  A5's twenty-four and six with an advantage of +16 to +34 points of rolling in-run success at the peak and
  ~20 points of success in the last quarter, over a scratch run that
  reads 0.16–0.26. It does not show the transfer is worth a factor of two
  in rounds, because there is no rounds-to-pass to halve. The S3 screen
  (A3 from A2b) remains the clean measurement of the lever: there the
  scratch run passes eventually and the warm start is 4× faster.
- **A5 is the wall, and both routes up it fail differently again.** The
  control mis-allocates (22 of 24 bodies on points, one empty); A5b
  under-arrives (8–10 bodies on points); T1 sits between: eight to eleven bodies on points and four and a half held from 20k rounds on, the same shape as A5b's end state reached 225k rounds earlier and then held flat.
  The rung's own report (A5b) carries the hypothesis on file — the
  per-model state terms are paid as the mean over alive models, diluting
  a body's own credit with the army — and T1 is consistent with it: a
  policy that already knows how to walk to a point keeps some of that
  skill and the training signal at this army size does not sharpen it.
- **Coherency 0.16–0.23 greedy, 0.11–0.13 sampled, twice A5b's and a quarter of the bar's** — unpaid, as on every rung.

### What was not done

- No paired-on-init comparison (the warm start changes the init by
  construction).
- No second budget, no other warm-start source (A3x, A2b): the rung has
  one change, and it is read once.
- The peak checkpoints were scored as a readout (`pm-*.pt`), not as a
  criterion: 0.47 / 0.37 / 0.64, the best any per-model run has read on
  this rung, and still a third of the bodies short.

---

<a id="2026-09-17-curriculum-A5c-preregistration"></a>

## Pre-registration: curriculum rung A5c — A5b re-run with the actor credit

Written 2026-09-17 23:55, **before any training round under the actor
credit exists**, on branch `feature/per-model-actor-credit` (stacked on
PR #382). Parent question #340, rung **A5**, third per-model arm. No
issue was opened for it, by decision; this file is its record.

### The one change

`--credit actor` on `train_per_model.py`, built today: the retimer pays a
model its action term **undivided** (under the default `mean` it is
divided by the alive count, 1/24 here) and pays each state term's
per-model value at the close as a **credit** landed on that model's own
step of the turn (under `mean` the close pays the army mean, one scalar
to everyone). Globals, delta globals and the terminal bonus are unchanged.
Everything else is A5b: `configs/experiments/curriculum/a5.yaml`
(eight squads of three, six points, radius 4, ten rounds, the `arrive`
reward — travel term at `progress_scale` 6.0 with the fallback, coverage
0.3, terminal bonus 5.0 on `all_objectives_occupied`,
`terminate_on_success` on), from scratch, seeds 1 / 2 / 3, 128 rounds
per update (`--rollout-rounds 32 --num-rollout-envs 4`), `--ent-coef
0.003`, in-run eval every 512 rounds at n=30 on 500000+, Wandb
`curriculum-a5`, tag `a5c`.

Under `actor` A5's two terms become: the travel term to the mover, in
full, on its own step; coverage at the close, global as before. There is
no state term in A5's reward, so on this rung the build's second half is
inert and the arm tests the first half alone — the scale of the actor's
own credit. That is deliberate: it is the smallest change to the arm
that failed, and the rung with the clearest under-arrival to move.

### The comparator

A5b's own runs at the same rounds: **0.260 / 0.180 / 0.160** at
245,760, never a rolling 50% in-run, 8–10 of 24 bodies on points, `held`
4.1–4.6, turns 9.5–9.7 of 10, coherency 0.10. The bar is
`squad_march_take` at 1.000 / 6.77 turns. The whole-army control read
0.800 / 0.940 / 0.980 at 120 epochs (245k rounds).

### Budget and reads

245,760 rounds straight through (A5b's budget, so the rows pair by
rounds), a readout at the 122,880 checkpoint. Read greedy at n=100 on
700000+ with `just measure-rung`, the census beside it (bodies on points,
`held`, max stack, turns, coherency greedy and sampled, the panel).

### Criteria

- **PASS (the rung):** success ≥ 0.95 on all three seeds at 245,760, and
  no in-run dip below 0.80 after the first rolling pass.
- **MOVES (the build):** success at 245,760 ahead of A5b's same seed on
  3/3 AND the mean of the three ≥ 0.50, with bodies on points ≥ 16 of 24
  on every seed. This is the reading that decides whether the credit
  scale is the mechanism; a pass is not required for it.
- **FAIL:** neither; then the second half of the build (the state credit)
  has no test on this rung, and the next arm is A3 speed's hold term
  under `actor`, where it does.
- Power: binomial SE 0.03–0.05 at n=100; A5b's seeds sit at 0.16–0.26,
  so MOVES needs a shift of about 0.3, ten SE.

### What I expect (a guess, written so it can be wrong)

Arrival improves substantially: 16–22 bodies on points and success
0.5–0.8, without a pass — the last empty point is the control's
allocation failure, which a bigger travel credit does not address. If
success does not move at all, the dilution hypothesis is wrong on the
scale half and the state-credit half is the only one left.

### Amendment 1 — written 2026-09-18 00:50, at forty thousand of 245,760 rounds, before any final read

**A confound in the build, found on the panel.** Over rounds 30,000–41,500 the
three A5c seeds read success 1.2 / 0.3 / 0.1% in-run against A5b's 6.2 /
4.6 / 4.3% at the same rounds, `held` 2.3 / 2.2 / 2.7 against 3.3 / 2.1 /
3.3. The panel says why: the actor stream's returns are 13× the mean's
(return mean 12–14 against 0.9–1.1, return sd 6.6–8.5 against 0.41–0.49),
and while `ppo_update` normalises advantages it does not normalise value
targets, so the value loss grew by the square of that and the pre-clip
gradient norm rose from 1.1–1.3 to 4.7–5.9 with `max_grad_norm` 0.5
clipping every step — most of each update is now the value head's, and
the policy gradient's share fell about five-fold. Explained variance is
high (0.93–0.96) and the entropy heads unchanged, so this is scale, not a
broken signal. A reward-scale change is a PPO change on this trainer.

**The fix, and the arm.** `Credit.actor` now pays every payment over the
model count, a constant: the actor's action term is where the mean put it
(alive count against model count, identical on A5 where nobody dies) and
the common payments — coverage, the terminal bonus — shrink by 24. The
policy gradient sees exactly the ratio the unscaled cut gave it, since
advantages are normalised; the return scale is the mean's order. **A5d**
is A5b with that credit (tag `a5d`, everything else as above), launched
now; **A5c continues to its budget as the unscaled control**, so the pair
reads the scale confound directly. The criteria above apply to A5d;
A5c is a readout.

### Amendment 2 — written 2026-09-18 02:55, A5c stopped at ~75k rounds

The unscaled control was stopped at about 75,000 rounds (146 periodic
checkpoints kept under `per-model-a5-2026-09-17-23-4*-s{1,2,3}a5c`): the
box was out of memory with nine trainers and the 1,200-game clone fit
(16 GB) beside them, 9.6 GB into swap, and of the four runs A5c is the
one that cannot contribute to the goal of a pass. Its reading stands as
a readout, not a comparison to 245,760: at 60k rounds it read success
1.3% rolling with 2.6–3.3 held on seed 2 and 0% on the others — the
scale confound's control to that point, flat where A5b at the same
rounds was at 4–6%.

---

<a id="2026-09-18-curriculum-A5e-A5f-preregistration"></a>

## Pre-registration: A5e (the hold term under the actor credit) and A5f (the bar cloned, then anchored PPO)

Written 2026-09-18 01:15, **before any number for either exists**, on
branch `feature/per-model-actor-credit` (PR #383). Parent question #340,
rung **A5**. Goal set by Sash: make A5 pass. A5d (the scaled actor
credit on A5's own reward) is at ~20k of 245,760 rounds beside these.

### A5e — the state-credit half of the build, on A5

**The one change relative to A5d:** `objective_hold` at
`crowding_exponent` 1.0, weight 1.0 (`configs/experiments/curriculum/
a5_hold.yaml`), the term the A3 speed screen's S1 read NULL under the
mean credit. Under `--credit actor` the close pays each model its own
value of it on the step it acted, so a body standing on a point it
reached is paid for standing there and a point pays a pot. Everything
else is A5d: from scratch, seeds 1 / 2 / 3, 245,760 rounds, 128 rounds
per update, `--ent-coef 0.003`, Wandb `curriculum-a5`, tag `a5e`.

Comparators: A5d (same credit, no hold term) seed for seed at the same
rounds — the paired reading of the term; A5b (0.260 / 0.180 / 0.160) as
the rung's record. Criteria: **PASS** success ≥ 0.95 on all three seeds
at 245,760 (greedy, n=100 on 700000+) with no in-run dip below 0.80
after the first rolling pass; **MOVES** ahead of A5d on 3/3 at 245,760;
**FAIL** neither. Prediction: the hold term stops the walk-off (`held`
rises past A5d's) and arrival improves; a pass is unlikely, since the
last empty point is the allocation failure the control also has.

### A5f — the bar cloned into the set network, then anchored PPO

The D-route applied to A5: `squad_march_take` solves the rung at 1.000
in 6.77 turns, and the record says the set network holds a scripted plan
from imitation that reward could not teach it (D1b: 0.96 on C3b from
1,200 games) and that anchored PPO holds it (D2: 0.94–0.96).

1. **The clone:** `just behaviour-clone-per-model squad_march_take
   configs/experiments/curriculum/a5.yaml 1200 40
   checkpoints/per_model/clones/take-a5-1200-s0.pt 0`, seeds 800000+,
   the last 240 games held out. Scored greedy at n=100 on 700000+.
   **PASS (by imitation)** if success ≥ 0.95, `held` ≥ 5.8, turns within
   a round of the script's 6.77; reported as what it is — the "start"
   axis, not a reward result.
2. **Anchored PPO from it** (only if the clone passes): A5's reward,
   `--warm-start-from` the clone, `--kl-ref-coef 10 --kl-ref-target
   0.03`, `--credit actor`, 3 seeds, 122,880 rounds, tag `a5f`.
   **HOLDS** if success ≥ 0.95 on 3/3 at the end; **IMPROVES** if turns
   are ahead of the clone's paired on 3/3.

What a pass here would mean: the architecture can play A5; the per-model
trainer cannot yet learn it from reward. Both halves go in the ladder
row.

### Amendment 1 — written 2026-09-18 04:05, after the clone's read and before any PPO round from it

**The clone (A5f step 1).** `take-a5-1200-s0.pt`, 1,200 games × 40
epochs, held-out match displacement 0.791 / declaration 0.971 / selector
0.639 / joint 0.513 (the opening order at chance, as with the escort).
Greedy at n=100 on 700000+: success **0.930**, `held` 5.93, `on_obj`
0.831, turns 6.96 (+0.19 ± 0.12 paired against the script's 6.77), vp
+68.0, coherent 0.744, stationary 0.18. Two hundredths under the
imitation pass mark, within one binomial SE (0.026) of it: **FAIL on the
letter, and the best policy on the per-model facade on this rung by a
wide margin** (A5b's best seed 0.260; A5e's best in-run 12% at 40k).

**Step 2 launched regardless, under the goal, as two arms**, both
`--warm-start-from` the clone, three seeds, 122,880 rounds, 128 per
update, `--ent-coef 0.003`, `--credit actor`, Wandb `curriculum-a5`:

- **A5f** as pre-registered: A5's reward, the anchor at `--kl-ref-coef
  10 --kl-ref-target 0.03` (D2's setting, which held the escort clone
  and improved nothing). Criteria as above: HOLDS ≥ 0.95 on 3/3 at the
  end; IMPROVES if turns are ahead of the clone's paired on 3/3.
- **A5g**, one change on top of A5f's recipe in each of two places, so
  it is not one change and is read as a screen: A5e's reward (the hold
  term under the actor credit, which is paying — A5e reads 8–12% rolling
  at 40k on two seeds where A5b never left 6%) and a looser anchor
  (`--kl-ref-coef 1 --kl-ref-target 0.10`), so reward can move the
  policy where D2's anchor pinned it. Tag `a5g`. Criterion: **PASS** ≥
  0.95 on 3/3 at the end with no dip below 0.80 after the first pass;
  a pass here is a pass by imitation plus reward, reported as such.

Prediction: A5f holds at 0.92–0.94 and does not improve (D2's result
again); A5g moves — the hold term fixes the walk-off that is the
clone's residual — and lands 0.93–0.97, a coin flip on the letter.

### Amendment 2 — written 2026-09-18 08:40, at 40k of 122,880 rounds on A5f/A5g, before any number for A5h

**Reads so far**, greedy at n=100 on 700000+ (the measure that decides),
the clone at 0.930:

| rounds | A5f s1 / s2 / s3 | A5g s1 / s2 / s3 |
|---|---|---|
| 20,480 | 0.950 / 0.920 / 0.930 | 0.920 / 0.910 / 0.960 |
| 40,960 | **0.850** / 0.930 / 0.940 | 0.910 / **0.950** / 0.940 |

Both arms sit in the clone's noise band (binomial SE 0.026); no seed set
clears 0.95 on all three. In-run reads on 500000+ run higher (A5f s1 at
96% rolling where its 40k checkpoint reads 0.850 held-out), so the
in-run curve is not the read. A5d at 80k: 1–5%; A5e at 80k: 8–9% on all
three seeds with 3.4–3.8 held — the hold term is a real but small edge.

**A5h, a third arm from the clone, launched now**: A5g's recipe with the
anchor loosened an order of magnitude (`--kl-ref-coef 0.1
--kl-ref-target 0.3`), the D2 finding's own next step ("the IMPROVES
question is the anchor's coefficient"). Under A5g's coefficient of 1 the
drift sits at its 0.10 target and reward moves the policy inside the
clone's band; under 0.1 it has room to move the residual — the point
left empty in six episodes of a hundred — and room to fall apart, which
plain PPO did by 2,560 rounds on D2. Seeds 1 / 2 / 3, 122,880 rounds,
`--credit actor`, `a5_hold.yaml`, tag `a5h`. Criterion as A5g's.
Prediction: it moves further than A5g in both directions across seeds —
one seed above 0.95 and one below 0.90 by 40k.

### Amendment 3 — written 2026-09-18 11:05, before any number for the second clone

**The 60k reads**, greedy at n=100 on 700000+: A5f **0.880 / 0.910 /
0.950**, A5g **0.900 / 0.970 / 0.900** (20k: 0.950 / 0.920 / 0.930 and
0.920 / 0.910 / 0.960; 40k: 0.850 / 0.930 / 0.940 and 0.910 / 0.950 /
0.940). Half the budget in, both arms still sit in the clone's band, one
seed touching 0.95–0.97 at a time and never the set. The anchored PPO
neither fixes nor breaks the clone's residual.

**A5d stopped at ~103k rounds** (216 periodic checkpoints kept under
`per-model-a5-2026-09-18-00-57-2*-s{1,2,3}a5d`): 1–6% rolling in-run
with 3–4 held at 100k, flat like A5b, and the box needs its memory for
the next clone. Its reading against the pre-registered MOVES criterion
is a FAIL at 100k: the scaled actor credit alone does not move A5's
arrival. The build's state-credit half is read from A5e, still running.

**A5f step 1 again, bigger: `take-a5-2000-s0`.** D1 → D1b moved the
escort clone 0.83 → 0.96 by quadrupling the games; this clone's residual
is a coordination error at the start (two squads choosing the same
point, the opening order being unlearnable) that more demonstrations of
the script's assignment should narrow. 2,000 games × 60 epochs, seeds
800000+, the last 400 held out, fit seed 0. Demonstrations are now
recorded at half precision (`compact_tokens`, pinned by a test) so the
fit holds ~14 GB instead of the ~27 GB the float32 recording would.
Scored greedy at n=100 on 700000+: PASS (by imitation) at ≥ 0.95,
`held` ≥ 5.8; if it passes, anchored PPO from it replaces A5f's start.
Prediction: 0.95–0.97.

### Amendment 4 — written 2026-09-18 12:50, all nine PPO-from-clone runs stopped

**The 80k reads**, greedy at n=100 on 700000+: A5f **0.930 / 0.910 /
0.960**, A5g **0.960 / 0.900 / 0.920**. Five reads over 20k–80k rounds put
every seed of both arms in the clone's band (0.85–0.97, the clone
0.930, binomial SE 0.026), one seed at 0.95–0.96 on each read and the
set never above the mark. A5h under the tenfold-looser anchor drifted
down instead: 84 / 89 / 88% rolling in-run at 40k against A5g's 90–96%
at the same rounds with the same reward.

**Stopped by decision** at ~89k (A5f, A5g) and ~44k (A5h) rounds: the
chance of a 0.95 × 3 read at the end from a true rate near 0.93 is
about one in fifty and would not survive the next read, so the end
reads would have served the record's completeness and not the goal;
the box's memory goes to the 2,000-game clone fit and to what follows
it. **A5f and A5g are read at 80k as their final row: HOLDS (the clone's
level on every seed), IMPROVES not shown. A5h: the anchor an order of
magnitude looser loses the clone slowly — the coefficient question's
answer is that 1 holds and 0.1 does not, and neither improves.**
Checkpoints every 512 rounds kept under the run directories.

**A5e** (stopped at ~110k, 11:30, for the same memory) reads 1–14%
rolling in-run over 60k–110k with 3.4–4.2 of six held on every seed,
against A5d's 1–6% and 2.5–4.0 at the same rounds: the hold term under
the actor credit is a small, consistent edge in arrival and nowhere
near a pass. A5d (stopped at ~103k) FAILS its MOVES criterion. Together:
**the actor credit build does not move A5's arrival; the per-model
trainer's failure on this rung is not a credit-scale problem.**

### Amendment 5 — written 2026-09-18 18:10, the reboot, and the stopped arms' final reads

**The machine rebooted at about 15:20** (a crash) and killed the
2,000-game clone fit of amendment 3 four and a half hours in, before
its first epoch had been written. It was relaunched at 17:01 with the
same recipe (2,000 games of `squad_march_take` on a5.yaml, seeds
800000+, 60 epochs, fit seed 0, `compact_tokens`; 312,672 decision
steps to fit, 78,108 held out) and is expected to finish around 23:00.
Its read and the criteria of amendment 3 are unchanged.

**Final reads of the arms stopped under amendment 4**, greedy at n=100
on 700000+ at each run's last checkpoint, written before the second
clone exists: A5d (~103k) **0.050 / 0.070 / 0.010**, A5e (~108k)
**0.170 / 0.060 / 0.060**, A5h (at 40,960) **0.950 / 0.840 / 0.820**.
A5f and A5g stand at their 80k rows. **The 1,200-game clone's seven
failures in a hundred are one event**: in every one the script's
assigned squad went to a point another squad was already taking, so
two squads shared a point and one stayed empty — never a late
arrival. The teacher breaks that tie by a rule the observation does not
carry (its own assignment order), which is the clone residual that a
bigger fit can shrink only if the tie-break is a function of the board;
the second clone's read decides whether it is. Every number in this
amendment is at the stops named in amendment 4 and at the memory's
prices — none is a read at budget.

### Amendment 6 — written 2026-09-18 20:55, the 2,000-game clone read; A5i launched from it

**The 2,000-game clone PASSES by imitation: 0.960**, held 5.96 of six,
turns 6.97 (+0.20 ± 0.10 against the bar's 6.77, paired), `on_obj`
0.838, coherent 0.771 (bar 0.822), greedy at n=100 on 700000+, beside
the 1,200-game clone re-read on the same seeds at 0.930 / 5.93 / 6.96.
Both criteria of amendment 3 are met (≥ 0.95, held ≥ 5.8) and the
prediction (0.95–0.97) was right. Held-out per-head match: declaration
0.977, displacement **0.821** (the 1,200-game clone 0.72), selector
0.639, joint 0.532 (the opening order, at chance as D1 recorded). The
fit finished at 20:42, earlier than the ~23:00 estimate, at 15 GB.

The census (n=100): 20.1 of 24 bodies on points and all six held by
turn 7, max stack 5.8 (the script 5.8). Four failures in a hundred:
point 4 empty in two, points 1 and 5 in one each — the 1,200-game
clone's seven were six at point 4 and one at point 0. Fewer failures,
less concentrated on one point; the event (two squads on one point)
is unchanged and the read does not separate "the tie is on the board"
from "the fit averages the tie better". More games moved the read by
three hundredths, which is the whole of the residual the goal needs.

**A5i launched 20:46**: A5f's pre-registered step 2 with the start
replaced — anchored PPO from `clones/take-a5-2000-s0.pt` on a5.yaml,
`--kl-ref-coef 10 --kl-ref-target 0.03`, `--credit actor`, `--ent-coef
0.003`, seeds 1 / 2 / 3, 122,880 rounds at 128 per update, in-run eval
every 512 rounds at n=30 on 500000+, Wandb `curriculum-a5` runs
`cb7pmoks` / `7hvtp7ju` / `m2ujugw1`, run dirs
`per-model-a5-2026-09-18-20-46-09-s{1,2,3}a5i`. Criteria as A5f's:
**HOLDS** if success ≥ 0.95 on 3/3 at the end (greedy, n=100 on
700000+); **IMPROVES** if turns are ahead of the clone's paired on 3/3;
read at 20k / 40k / 60k as A5f was. A HOLDS here is three seeds at or
above the rung's mark from a start that is itself at 0.960 — the
rung's PASS on the letter, reported as what it is: the "start" axis,
the bar imitated and held, not learned from reward. Prediction, from
A5f/A5g's twenty-four reads in the clone's band: every seed in
0.93–0.99, HOLDS on 3/3 with probability near one half (binomial at
0.96, n=100, three draws), IMPROVES not shown.

### Amendment 7 — written 2026-09-18 22:10, A5i at 40k of 122,880

The 20k and 40k reads of A5i, greedy at n=100 on 700000+:

| rounds | s1 | s2 | s3 | turns | held of 6 | coherent |
|---|---|---|---|---|---|---|
| the clone | 0.960 | — | — | 6.97 | 5.96 | 0.771 |
| 20,480 | **0.980** | 0.940 | **0.970** | 6.87 / 7.09 / 6.90 | 5.98 / 5.94 / 5.97 | 0.795 ×3 |
| 40,960 | **0.970** | **0.970** | **0.970** | 6.94 / 6.94 / 6.89 | 5.97 / 5.97 / 5.97 | 0.78–0.79 |

At 40k every seed is at 0.970 — one to two hundredths above the start,
inside its binomial band (0.960 ± 0.020), the first read on this rung
with all three seeds at or above the mark. Turns are within a tenth of
the clone's on every seed and a tenth or two behind the bar; the
in-run curve (n=30 on 500000+) sits at 96–99% rolling, three to four
points above the held-out row as on every arm before. Nothing here is
a verdict: HOLDS is read at 122,880, and A5f's seeds crossed 0.95 and
came back down between reads from a 0.930 start. Written before the
60k read.

### Amendment 8 — written 2026-09-19 00:55, A5i read at 122,880: HOLDS 3/3, the rung's mark on every seed

**A5i HOLDS on every seed: 0.960 / 0.980 / 0.960**, greedy at n=100 on
700000+ at 122,880 rounds (`last.pt`; runs `cb7pmoks` / `7hvtp7ju` /
`m2ujugw1`):

| row | success | turns | vs bar, paired | vs clone, paired | held of 6 | on obj | coherent | stat |
|---|---|---|---|---|---|---|---|---|
| `squad_march_take` | 1.000 | 6.77 | — | — | 6.00 | 0.856 | 0.822 | — |
| the 2,000-game clone | 0.960 | 6.97 | +0.20 ± 0.10 | — | 5.96 | 0.838 | 0.771 | 0.17 |
| s1 | **0.960** | 6.88 | +0.11 ± 0.10 | −0.09 ± 0.10 | 5.96 | 0.835 | 0.78 | 0.16 |
| s2 | **0.980** | 6.81 | +0.04 ± 0.08 | **−0.16 ± 0.10** | 5.98 | 0.851 | 0.79 | 0.16 |
| s3 | **0.960** | 7.04 | +0.27 ± 0.10 | +0.07 ± 0.13 | 5.96 | 0.845 | 0.79 | 0.16 |

Every read from 40k on had all three seeds at or above the mark (40k
0.970 ×3; 60k 1.000 / 0.970 / 0.970; 80k 1.000 / 0.980 / 0.960; end
0.960 / 0.980 / 0.960). **The in-run dip clause holds**: the rolling
in-run success (n=30 on 500000+, eight-evaluation window) first reached
95% within the first two evaluations on every seed and never fell below
94.0 / 95.8 / 95.0 after; the lowest single evaluation was 87 / 87 / 90.
**IMPROVES is not shown**: turns are ahead of the clone's paired on two
seeds (−0.09, −0.16) and behind on one (+0.07), against a 3/3 clause.
Sampled play (n=100 on 900000+, paired against greedy) is the same
policy: −0.1 / −1.2 / +0.1 ± 1.2 vp, held 5.96–5.98, coherent
0.775–0.791 against greedy's 0.807–0.818 — no do-nothing fingerprint
(stationary 0.16 either way, the clone's 0.17).

The census (n=100 on 700000+): 20.1–20.4 of 24 bodies on points and all
six held by turn 7 on every seed (the clone 20.1 / 6.0, the script
20.1 / 6.0), max stack 5.7–5.8 (the script 5.8). Failures: 4 / 2 / 4 in
a hundred, each a single short point, different points on different
seeds — the clone's event, at the clone's rate.

**Verdict on the arm as pre-registered: HOLDS 3/3, IMPROVES not shown.
Verdict on the rung: A5 PASSES on the letter** — success ≥ 0.95 on all
three seeds at the end, greedy at n=100 on 700000+, no in-run dip below
0.80 after the first rolling pass — **on the start axis**: the bar
cloned from 2,000 games (0.960) and held by PPO under a KL anchor
(coef 10, target 0.03) with the actor credit, on A5's own reward. The
reward added at most a tenth of a turn on two seeds and nothing to the
success rate beyond the clone's band. What it is not: the per-model
trainer learning A5 from reward, which read 0.26 / 0.18 / 0.16 at twice
this budget from scratch (A5b), 0.05 / 0.07 / 0.01 with the actor credit
(A5d), and 0.03–0.33 one point down (A5-points). The report carries
both halves in the ladder row.

---

<a id="2026-09-18-curriculum-A5-half-steps-preregistration"></a>

## Pre-registration: the A5 half-steps — A5-bodies and A5-points

Written 2026-09-18 13:40, **before any training round on either exists**,
on branch `feature/per-model-actor-credit` (PR #383). Parent question
#340. Asked by Sash: is the A4 → A5 jump too big? A4 is four squads over
three points (twelve bodies); A5 doubled both axes in one rung, against
the ladder's one-axis rule, and fails every per-model arm on file
(A5b 0.26 / 0.18 / 0.16; T1 from A4x 0.45 / 0.29 / 0.20; A5d and A5e
under the actor credit 1–14% in-run). These two rungs separate the axes.

### The rungs

| | A5-bodies | A5-points |
|---|---|---|
| config | `configs/experiments/curriculum/a5_bodies.yaml` | `configs/experiments/curriculum/a5_points.yaml` |
| army | 24 bodies, eight squads of three (A5's) | 18 bodies, six squads of three |
| points | THREE, A4's column at x=35 | FIVE: A5's near column of three, a far column of two |
| everything else | A5's band, ten rounds, radius 4, A5's `arrive` reward, `all_objectives_occupied`, `terminate_on_success` | the same |
| bar (`squad_march_take`, n=100, 700000+) | **1.000**, held 3.00, 4.43 turns, coherent 0.759 | **1.000**, held 5.00, 6.71 turns, coherent 0.832 |
| bridge | identical on every shared field (n=3) | identical |

### The arms

Per-model from scratch, the ladder's standard recipe (128 rounds per
update, `--ent-coef 0.003`, the default `mean` credit so the rows pair
with A4x and A5b), seeds 1 / 2 / 3, **122,880 rounds** (the cap; a rung
that fails at the cap is a real fail), in-run eval every 512 rounds at
n=30 on 500000+, Wandb `curriculum-a5`, tags `a5bodies` / `a5points`.
A5-bodies launches now beside the 2,000-game clone fit; A5-points when
the fit releases its memory. No whole-army control on these — they are
diagnostic half-steps, not ladder rungs, and the budget is the cap.

### Criteria

Read greedy at n=100 on 700000+ at 122,880.

- **PASS:** success ≥ 0.95 on all three seeds, no in-run dip below 0.80
  after the first rolling pass.
- **FAIL:** any seed below 0.95; then the census (bodies on points,
  held, max stack, turns).
- **NULL:** only if the bar fails its own criterion; both read 1.000.

### What the pair says

- bodies PASS, points FAIL: the wall is the number of points to cover at
  once — the conjunction the critic first struggled to value on A3.
- bodies FAIL, points PASS: the wall is the army — twenty-four bodies
  dissolve their squads (A5b's coherency 0.10) before they arrive.
- both PASS: A5's difficulty is the interaction, and A5 warm-started
  from A5-points is the next arm.
- both FAIL: the jump was not the problem; A5's wall is already present
  one half-step up from A4 on either axis.

### Prediction

Bodies passes (0.95–1.00 by 60k rounds: three points with eight squads
is A2's one-point problem three times over); points fails on at least
one seed (0.85–0.95), the far column's points the ones left short.

### Amendment 1 — written 2026-09-18 18:10, A5-bodies at ~106k rounds after a reboot, A5-points at ~29k

**The machine rebooted at about 15:20** (a crash, cause not recorded)
and killed every process: the three A5-bodies runs at about 84k rounds
and the 2,000-game clone fit four and a half hours in. At 17:01–17:05
A5-bodies was **resumed seed-for-seed from each run's `last.pt`**
(`--resume-from`; the first evaluation after the resume at 84,480 /
81,920 / 84,992 rounds, so under three thousand rounds of each run are
replayed), the clone fit was relaunched from the start, and
**A5-points launched from scratch beside them** rather than waiting for
the fit to release its memory as the plan above said — the box carries
the six trainers and the fit together at 128 rounds per update with
about four gigabytes to spare, and the points half-step is the one this
pair cannot be read without. Same recipe as written: 122,880 rounds,
`--ent-coef 0.003`, the `mean` credit, in-run eval every 512 rounds on
500000+, Wandb `curriculum-a5` (A5-bodies resumed `j1e0mpam` /
`0ioij69h` / `nazcv56j`, the pre-reboot runs `5qbyx99r` / `w15c7xih` /
`3vb272ip`; A5-points `uagmul66` / `ici5py46` / `wedjjozy`).

A resume replays a few thousand rounds under a fresh rollout seed and
a fresh Adam state, so the resumed curve is not the same trajectory the
run would have followed; on the criteria it changes nothing — the read
is at 122,880 greedy on 700000+, and the dip rule counts in-run
evaluations from the first rolling pass, which no seed had reached
before the reboot (in-run at 80k: 91 / 62 / 71%, held 2.6–2.9 of 3).
Written before either half-step's end read: A5-bodies' rolling in-run
success over the last eight evaluations at ~106k is 94 / 96 / 98%
(held 2.90–3.00), climbing from the 80k plateau; A5-points at ~29k
reads 0% in-run with 0.7–1.5 of five points held. Neither number is a
result.

### Amendment 2 — written 2026-09-18 19:50, A5-bodies read at 122,880; A5-points at ~85k, unread

**A5-bodies: FAIL as pre-registered, one seed over the mark.** Greedy
at n=100 on 700000+ at 122,880 rounds (`last.pt`; the resumed runs
`j1e0mpam` / `0ioij69h` / `nazcv56j`):

| row | success | turns | vs bar, paired | held of 3 | on points | coherent | stat |
|---|---|---|---|---|---|---|---|
| `squad_march_take` | 1.000 | 4.43 | — | 3.00 | 0.405 | 0.759 | — |
| s1 | **0.960** | 5.46 | +1.03 ± 0.12 | 2.94 | 0.353 | 0.357 | 0.01 |
| s2 | **0.930** | 5.80 | +1.37 ± 0.13 | 2.90 | 0.293 | 0.187 | 0.04 |
| s3 | **0.860** | 6.23 | +1.80 ± 0.18 | 2.70 | 0.291 | 0.255 | 0.07 |

The by-turn census (n=20 on 700000+; bodies on points of 24 and points
held of 3 after turns 3 → 5 → 7 → end; share of episodes with each point
empty at the end, all episodes and failures only; max stack):

| policy | success | turns | on points | held | empty by point | max stack | empty by point, failures |
|---|---|---|---|---|---|---|---|
| `squad_march_take` | 1.00 | 4.20 | 5.2 → 7.8 → 7.8 → 7.8 | 2.7 → 3.0 → 3.0 → 3.0 | 0 / 0 / 0 | 3.4 | — |
| s1 | 0.95 | 5.45 | 0.8 → 9.2 → 9.1 → 9.0 | 0.6 → 2.9 → 3.0 → 3.0 | 0.05 / 0 / 0 | 4.6 | 1.00 / 0 / 0 |
| s2 | 0.95 | 5.80 | 0.5 → 6.7 → 7.2 → 7.3 | 0.5 → 2.8 → 2.9 → 3.0 | 0 / 0.05 / 0 | 3.5 | 0 / 1.00 / 0 |
| s3 | 0.75 | 6.60 | 0.9 → 5.3 → 5.5 → 5.5 | 0.7 → 2.2 → 2.4 → 2.4 | 0.20 / 0.20 / 0.25 | 2.9 | 0.80 / 0.80 / 1.00 |

Three points of radius 4 hold about eight bodies between them, so "on
points" saturates near eight for every policy and the script parks the
rest beside them. What separates the rows is the clock: the script has
five bodies on points after turn 3 and every point held after turn 5;
the arm has under one body on a point after turn 3 on every seed, and
then arrives — s1 and s2 hold all three by turn 7 and finish a turn to
a turn and a half behind the bar, s3 holds 2.4 and never gets further.
s3's failures are not one short point: in its five failed episodes 2.6
of three points are empty on average and 5.5 bodies are on points at
the end — the army does not arrive, A5b's signature on a fifth of
the episodes. The in-run curve (n=30 on 500000+) over the last eight
evaluations read 96 / 93 / 90%, a point or two above the held-out row
on each seed; no seed reached a rolling 95% before ~100k rounds, so the
dip clause never engaged.

**Read against the prediction:** bodies was predicted to pass at 0.95–1.00
by 60k rounds ("A2's one-point problem three times over"). It did not:
one seed at 0.96 at the cap, the others 0.93 and 0.86, all a turn or
more slow. The bodies axis alone, with A4's point count, is not learned
to the mark within the cap. The pair's quadrant waits on A5-points; the
"bodies PASS" half that would have put the wall on the point count is
not available. What is available: twenty-four bodies over three points
reaches the high nineties in-run on every seed and 0.93–0.96 held-out on
two, where the same trainer on A5 never reached a rolling 50% — so most
of A5's difficulty is not the army size, and the points read decides how
much of it is the point count.

### Amendment 3 — written 2026-09-18 20:45, A5-points read at 122,880; the pair's quadrant

**A5-points: FAIL as pre-registered, on every seed, by a wide margin.**
Greedy at n=100 on 700000+ at 122,880 rounds (`last.pt`; runs
`uagmul66` / `ici5py46` / `wedjjozy`):

| row | success | turns | vs bar, paired | held of 5 | on points | coherent | stat |
|---|---|---|---|---|---|---|---|
| `squad_march_take` | 1.000 | 6.71 | — | 5.00 | 0.889 | 0.832 | — |
| s1 | **0.030** | 9.94 | +3.23 ± 0.07 | 2.47 | 0.263 | 0.194 | 0.04 |
| s2 | **0.060** | 9.96 | +3.25 ± 0.06 | 2.92 | 0.321 | 0.195 | 0.04 |
| s3 | **0.330** | 9.56 | +2.85 ± 0.09 | 3.83 | 0.377 | 0.211 | 0.06 |

The by-turn census (n=20 on 700000+; bodies on points of 18 and points
held of 5 after turns 3 → 5 → 7 → end; share of episodes with each point
empty at the end, by index 0–4; max stack):

| policy | success | turns | on points | held | empty by point | max stack |
|---|---|---|---|---|---|---|
| `squad_march_take` | 1.00 | 6.70 | 11.7 → 12.8 → 15.8 → 15.8 | 3.0 → 4.2 → 5.0 → 5.0 | 0 / 0 / 0 / 0 / 0 | 5.0 |
| s1 | 0.05 | 9.95 | 7.1 → **3.6** → 6.1 → 5.1 | 2.6 → 2.1 → 2.7 → 2.6 | 0.60 / 0.85 / 0.65 / 0.10 / 0.15 | 2.7 |
| s2 | 0.05 | 10.00 | 4.7 → 4.2 → 4.3 → 5.9 | 1.7 → 2.1 → 2.4 → 3.1 | 0.25 / 0.50 / 0.90 / 0.10 / 0.15 | 2.9 |
| s3 | 0.30 | 9.85 | 2.0 → 6.3 → 6.8 → 7.3 | 1.6 → 2.5 → 3.6 → 3.9 | 0.05 / 0.15 / 0.20 / 0.55 / 0.20 | 3.1 |

This is A5's failure at six squads: a third of the bodies on points at
the end against the script's seven eighths, whole points abandoned by
seed (s1 leaves points 0–2 empty in 60–85% of episodes, s2 point 2 in
90%, s3 point 3 in 55%), max stack under three, the clock run out on
every seed (turns 9.6–10.0 of ten; the script 6.71), coherency 0.19–0.21.
s1's bodies on points *fall* from 7.1 after turn 3 to 3.6 after turn 5 —
squads walk off points they reached. The in-run curve never reached a
rolling 10% on s1 or s2 and peaked in the thirties on s3.

**The pair's quadrant is "both FAIL" on the letter**, and the letter's
reading — "the jump was not the problem; A5's wall is already present
one half-step up from A4 on either axis" — under-describes what the
two rows say together, so the magnitudes go on the record beside it:

| axis | rung | success at the cap | held / points | turns v bar |
|---|---|---|---|---|
| — | A4 (four squads, three points) | 0.930 / 0.930 / 0.970 | 2.91–2.97 / 3 | — |
| bodies ×2 | A5-bodies (eight squads, three points) | 0.960 / 0.930 / 0.860 | 2.70–2.94 / 3 | +1.0 to +1.8 |
| points +2 | A5-points (six squads, five points) | **0.030 / 0.060 / 0.330** | 2.47–3.83 / 5 | +2.9 to +3.3 |
| both | A5 (eight squads, six points; A5b at 2× the cap) | 0.260 / 0.180 / 0.160 | 4.1–4.6 / 6 | +2.7 to +2.9 |

Doubling the army at A4's point count costs a turn and one seed's pass;
adding two points to A4's army takes the same trainer from the low
nineties to near zero at the same rounds, with A5's census. **The wall
is the number of points to cover at once**, and it is steep: at the cap
the spread rungs read 0.70–0.80 at four points (A3), 0.03–0.33 at five,
and A5b never reached a rolling 50% at six. The army size is a real but
second-order cost. Neither half-step is a rung and neither is repeated;
the pre-registered consequence for this quadrant (no next arm named)
stands, and what to do about a conjunction the trainer cannot value at
five points is the report's question, not this amendment's.

---

<a id="2026-09-19-curriculum-a5-passes-on-the-start-axis"></a>

## Curriculum rung A5, the second pass: the credit is not the wall, the clone is the ceiling, and the anchor holds it over the mark

**Verdict first. A5 PASSES on the letter, on the start axis** — the bar
cloned from 2,000 games (0.960) and held by anchored PPO on A5's own
reward at **0.960 / 0.980 / 0.960** (A5i, 122,880 rounds, greedy at
n=100 on 700000+, no in-run dip, sampled play identical) — **and
nothing on this rung was learned from reward.** Ten per-model arms and
two half-step rungs on the allocation rung (#340; eight squads of three over six points, success =
every point occupied, the bar `squad_march_take` at 1.000 in 6.77
turns), run 2026-09-17 23:42 to 2026-09-19 00:46, read greedy at n=100
on 700000+:

| arm | change | verdict | success | where read |
|---|---|---|---|---|
| **A5c** | A5b with the actor credit, undivided | **FAIL (a PPO change)** — returns 13×, the pre-clip gradient norm 5×, clipped on every step | 0–1.3% in-run | stopped at ~75k |
| **A5d** | A5b with the actor credit divided by the model count | **FAIL, MOVES not shown** | 0.050 / 0.070 / 0.010, held 3.9 / 3.8 / 2.7 | ~110k of 245,760 |
| **A5e** | A5d + `objective_hold` (crowding 1.0) paid per model on the actor's step | **FAIL; a small, consistent edge over A5d, nowhere near a pass** | 0.170 / 0.060 / 0.060, held 4.0 / 3.7 / 2.8 | ~108k of 245,760 |
| **clone, 1,200 games** | the bar cloned into the set network | **FAIL by imitation, by 0.02** | **0.930**, held 5.93, turns 6.96 (bar 6.77) | — |
| **A5f** | anchored PPO from the clone (coef 10, target 0.03), actor credit | **HOLDS, IMPROVES not shown** | 0.930 / 0.910 / 0.960 at 80k (0.95 / 0.92 / 0.93 at 20k) | stopped at ~89k of 122,880 |
| **A5g** | A5f + the hold term (anchor 1, target 0.10) | **HOLDS, IMPROVES not shown** | 0.960 / 0.900 / 0.920 at 80k | stopped at ~89k |
| **A5h** | A5g with the anchor loosened tenfold (0.1, target 0.3) | **loses the clone slowly** | 0.950 / 0.840 / 0.820 at 40,960; in-run 84–89% against A5g's 90–96% | stopped at ~44k |
| **clone, 2,000 games** | 1.67× the games, 60 epochs | **PASS by imitation** | **0.960**, held 5.96, turns 6.97; displacement match 0.72 → 0.821 | — |
| **A5i** | anchored PPO from the 2,000-game clone (coef 10 / 0.03), actor credit | **HOLDS 3/3 — the rung's PASS; IMPROVES not shown** | **0.960 / 0.980 / 0.960**, held 5.96–5.98, turns 6.88 / 6.81 / 7.04 (clone 6.97, bar 6.77); 40k 0.970 ×3, 60k 1.000 / 0.970 / 0.970, 80k 1.000 / 0.980 / 0.960 | 122,880 |
| **A5-bodies** | eight squads over THREE points (the bodies axis alone) | **FAIL, one seed over the mark; a turn or more slow** | 0.960 / 0.930 / 0.860, turns 5.46 / 5.80 / 6.23 (bar 4.43), held 2.94 / 2.90 / 2.70 | 122,880 |
| **A5-points** | SIX squads over five points (the points axis alone) | **FAIL on every seed — A5's failure at six squads** | 0.030 / 0.060 / 0.330, held 2.47 / 2.92 / 3.83 of 5, turns 9.6–10.0 (bar 6.71) | 122,880 |

Four things the pass says, each against the record's own hypothesis:

1. **The credit is not the wall.** The hypothesis on file since the A3
   speed screen — a body's own credit diluted 1/24 by the mean over the
   army — was built (`--credit actor`) and run three ways. The first cut
   changed PPO, not the reward: paying the actor undivided scaled the
   returns thirteenfold and the value loss swamped the clipped policy
   gradient. Scaled back to a constant, the credit moved nothing
   (A5d 0.01–0.07 against A5b's 0.16–0.26 at less than half the rounds),
   and adding the state term it was built to carry moved arrival by a
   tenth (A5e). The per-model trainer's failure on A5 is not a
   credit-scale problem.
2. **The clone is the ceiling, and it is 0.93.** The set network holds
   the bar's plan from 1,200 games at 0.930 with 5.93 of six points and
   within a fifth of a turn of the script — and every one of its seven
   failures in a hundred is the same event: the script's assigned squad
   walks to a point another squad is already taking, two squads share
   it, one point stays empty. Never a late arrival. The teacher breaks
   that tie by its own assignment order, which is not in the
   observation; a bigger fit closes the gap only if the tie-break is a
   function of the board. **The 2,000-game clone reads 0.960** (held 5.96, turns 6.97, coherent
0.771; displacement match 0.821 held-out), the first policy of any kind
on the per-model facade to clear A5's mark. Four failures in a hundred,
the same event, spread over three points where the smaller clone's
seven were six on one — more games shrank the residual by three
hundredths without changing its kind.
3. **Anchored PPO sits on the clone and does not lift it.** Five reads
   over 20k–80k rounds on two arms put every seed in the clone's band
   (0.85–0.97, binomial SE 0.026 at 0.93); one seed at 0.95–0.96 on each
   read, never the set. The coefficient question the D2 report left
   open has its answer: 10 holds, 1 holds, 0.1 loses, and none improves.
   From the 2,000-game clone the same recipe (A5i) holds 0.960 at
   0.960–0.980 on every read from 40k to the end — which is the rung's
   mark on three seeds, and is the clone's band.
4. **The wall is the number of points to cover at once.** Two half-steps
   from A4 (0.93–0.97 at the cap): doubling the army at A4's three
   points reads 0.960 / 0.930 / 0.860, a turn slow; adding two points to
   A4's army reads **0.030 / 0.060 / 0.330** with A5's census. The
   spread rungs at the cap go 0.70–0.80 at four points, 0.03–0.33 at
   five, ~0 at six. Army size is a second-order cost.

So the ladder row reads PASS with its provenance in the same cell: the
per-model architecture plays A5 at the bar's level from imitation and
keeps it under reward; the per-model trainer does not learn it from
reward at twice this budget, with or without the credit build, and
fails one point down. Both halves are the finding.

### Provenance

| field | value |
|---|---|
| date | A5c launched 2026-09-17 23:42; A5d 2026-09-18 00:57; A5e 01:16; the 1,200-game clone read 04:05; A5f and A5g 03:46; A5h 08:22; every PPO arm stopped by 12:50; A5-bodies 12:56 (resumed 17:01 after a reboot at ~15:20), read 19:40; A5-points 17:05, read 20:35; the 2,000-game clone fitted 17:01–20:42 (a first fit lost to the reboot), read 20:45; A5i 20:46–2026-09-19 00:37, read 00:46 |
| GPU / no-GPU | GPU (RTX 4090), up to fifteen trainers and a clone fit on a 31 GB box — memory, not the GPU, set the stops |
| seeds | 1 / 2 / 3 on every PPO arm; the clone at fit seed 0 on games 800000+ with the last 240 (400 for the 2,000-game fit) held out; A5f–A5h are three seeds off one warm start, so their agreement is the clone's and not three samples |
| n | 100 at 700000+ (every final and mark read, greedy); 30 at 500000+ in-run every 512 rounds |
| config | `configs/experiments/curriculum/a5.yaml` (A5c, A5d, A5f, A5h's twin; the clone); `a5_hold.yaml` (A5e, A5g, A5h: a5 + `objective_hold` w1.0 crowding 1.0); `a5_bodies.yaml` (eight squads, three points at x=35); `a5_points.yaml` (six squads, five points); every bar 1.000 |
| decode | none on the per-model facade |
| paired | per seed by rounds against A5b (A5c, A5d) and A5d (A5e); A5f–A5h against the clone on identical seeds |
| comparator | `squad_march_take` 1.000 / held 6 / 6.77 turns on a5 (4.43 on bodies, 6.71 on points); A5b 0.260 / 0.180 / 0.160 at 245,760; the 1,200-game clone 0.930 / 5.93 / 6.96 |
| opponent | none |
| budget | 245,760 rounds at 128 per update for A5c–A5e (stopped at 75k / 110k / 108k for memory, see the amendments); 122,880 for A5f–A5h (stopped at 89k / 89k / 44k by decision), A5i (run to the end) and the half-steps; `--ent-coef 0.003` throughout; `--credit actor` on every A5 PPO arm after A5b |
| code revision | A5c on `14998f2` (the credit); A5d–A5h on `ba622a4` (the scale fix) and after; the clones on `e632084` (`compact_tokens`); branch `feature/per-model-actor-credit`, PR #383 stacked on #382 |
| checkpoint | every 512 rounds under `checkpoints/per_model/per-model-a5*-2026-09-1[78]-*-s{1,2,3}a5{c,d,e,f,g,h,i,bodies,points}` (A5i and the half-steps read at `last.pt` = 122,880; the stopped arms at their last `pm-*.pt`); clones `checkpoints/per_model/clones/take-a5-{1200,2000}-s0.pt` |
| coherency | greedy at 700000+: A5d 0.21–0.27, A5e (n/a), the clones 0.744 / 0.771, A5i 0.78–0.79 (sampled 0.775–0.791 at 900000+), A5-bodies 0.19–0.36, A5-points 0.19–0.21; the bar 0.822 on a5, 0.759 on bodies, 0.832 on points |
| Wandb | `curriculum-a5`: A5c `lly5o8n4` / `5hy0hl7s` / `orcdnor7`; A5d `hemrqmbj` / `x10xp0iw` / `wt4cz7xs`; A5e `jeqtuhnq` / `5mynmsc2` / `77rr4xyi`; A5f `j8u2qkep` / `qomreieo` / `3p9ydybu`; A5g `hsmeo4ut` / `auwr7xyy` / `kvjuhpiu`; A5h `sorye78j` / `kmkmyqxi` / `76c3y9lr`; A5-bodies `5qbyx99r` / `w15c7xih` / `3vb272ip` then `j1e0mpam` / `0ioij69h` / `nazcv56j` after the resume; A5-points `uagmul66` / `ici5py46` / `wedjjozy`; A5i `cb7pmoks` / `7hvtp7ju` / `m2ujugw1` |
| pre-registration | `reports/2026-09-15-curriculum-a-rungs.md#2026-09-17-curriculum-A5c-preregistration` (+2 amendments), `reports/2026-09-15-curriculum-a-rungs.md#2026-09-18-curriculum-A5e-A5f-preregistration` (+8), `reports/2026-09-15-curriculum-a-rungs.md#2026-09-18-curriculum-A5-half-steps-preregistration` (+3) |

### The read

#### The actor credit: a PPO change first, then nothing

The build (`Credit.actor` in `envs/per_model/reward_timing.py`) was the
speed screen's named test: pay a model its own action term where the
mean had divided it by the alive count, and land each per-model state
term's value on the model's own step of the turn instead of paying the
army mean once at the close. A5's reward has no state term — the travel
term to the mover and a global coverage term — so A5c tested the first
half alone.

**A5c, the first cut, paid the actor undivided and was a PPO change.**
PPO here normalises advantages but not value targets, so a payment 24×
larger is a return 24× larger: at 20k rounds the returns read 13× A5b's,
the pre-clip gradient norm 5×, and the clip was binding on every update
— the value loss owned the gradient. In-run success 0–1.3% at 75k where
A5b was at 15–25%. The credit was rewritten the same night (`ba622a4`)
to divide every payment by the model count: the actor's action term
lands where the mean put it, the common payments shrink by the army
size, and the state credits are per model. A5c ran on as the control
until memory took it at ~75k.

**A5d, the scaled credit, moved nothing.** At ~110k rounds it reads
0.050 / 0.070 / 0.010 with 3.9 / 3.8 / 2.7 of six held, coherency
0.21–0.27, turns at the ten-round cap on every seed — it never finishes.
Its census is A5b's: whole points abandoned rather than crowded, seed 3
leaving the near column's point empty in 92–98% of episodes. In-run it
averaged 1–6% over 60k–110k where A5b's in-run evaluations averaged
23.9 / 21.9 / 12.5% over its run with peaks of 48 / 43 / 30% at
174k / 133k / 211k. MOVES — ahead of A5b on 3/3
— is not shown at any read; the pre-registered FAIL applies.

**A5e, the state half, is a tenth.** `objective_hold` at crowding 1.0,
paid per model on the actor's step, the term the speed screen's S1 read
NULL under the mean: 0.170 / 0.060 / 0.060 at ~108k with 4.0 / 3.7 / 2.8
held, against A5d's 0.050 / 0.070 / 0.010 and 3.9 / 3.8 / 2.7 at the same
rounds. In-run 1–14% over 60k–110k against A5d's 1–6%. Ahead of A5d on
held on every seed and on success on two — a consistent, small edge in
arrival, and the first upward reward signal on this rung from any
per-model arm. Nowhere near a pass, and MOVES asks for 245,760 rounds
neither arm reached.

#### The clone: 0.93, and one failure mode

`squad_march_take` cloned into the set network from 1,200 games on
800000+ (the last 240 held out, 40 epochs, fit seed 0): **0.930**, held
5.93 of six, 6.96 turns against the script's 6.77, about twenty of
twenty-four bodies on points, max stack 5.7 (the script's 5.8 — it
stacks two squads where it has a spare). One point is short: index 4 in
6% of episodes, index 0 in 1%. The D-route's clone reached the rung's
bound on C3b (D1b, 0.96); on A5 it is two hundredths under.

**Every one of its seven failures in a hundred is the same event.** A
census of the failed episodes (per-squad targets by turn against the
script's assignment on the same seeds): in all seven the clone's
assigned squad went to a point another squad was already taking, so two
squads shared a point and the assigned one stayed empty. None is a late
arrival, a dissolved squad or a body stuck behind a friend. The
script's assignment is a greedy matching with a tie-break by squad
order — its own plan order, not a function of the board — and D1
already recorded the network at chance on a teacher's plan order
held-out. So the residual is an assignment tie the observation cannot
carry, and more games close it only where the tie is in fact decided by
something on the board. The 2,000-game fit (60 epochs, the demonstrations
recorded at half precision to fit the box) was that test: **0.960**,
held 5.96, four failures of the same kind spread over three points —
the residual narrowed by three hundredths without changing its event.

#### Anchored PPO from the clone: five reads, one band

Anchored PPO from the 1,200-game clone, three seeds each, read greedy at
n=100 on 700000+ at every 20k checkpoint:

| rounds | A5f (a5, anchor 10 / 0.03) | A5g (a5_hold, anchor 1 / 0.10) |
|---|---|---|
| clone | 0.930 | 0.930 |
| 20,480 | 0.950 / 0.920 / 0.930 | 0.920 / 0.910 / 0.960 |
| 40,960 | 0.850 / 0.930 / 0.940 | 0.910 / 0.950 / 0.940 |
| 61,440 | 0.880 / 0.910 / 0.950 | 0.900 / 0.970 / 0.900 |
| 81,920 | 0.930 / 0.910 / 0.960 | 0.960 / 0.900 / 0.920 |

Twenty-four reads, every one in 0.85–0.97, the clone's binomial band at
n=100 (0.930 ± 0.026); one seed at or above 0.95 on each row, never the
set; no trend over 60k rounds on either arm. HOLDS on both; IMPROVES —
turns ahead of the clone's paired on 3/3 — on neither. **A5h**, A5g with
the anchor an order of magnitude looser (coef 0.1, target 0.3), drifted
the other way: rolling in-run 84 / 89 / 88% at 40k against A5g's 90–96%
at the same rounds with the same reward, greedy 0.950 / 0.840 / 0.820 at
40,960. The coefficient question D2 left open reads: 10 holds, 1 holds,
0.1 loses the clone slowly, and none of the three lifts it.

Two readings that go with the table. In-run success (n=30 on 500000+)
ran 5–10 points above the greedy held-out read on every arm and every
row, so an in-run curve at 95% is a held-out read near 0.90 — never call
a pass from the curve. And the three seeds of each arm are three seeds
off one warm start: their agreement is the clone's, and a fourth seed
would say nothing the census does not.

#### A5i: the same recipe from the bigger clone, read to the end

| rounds | s1 | s2 | s3 | turns | held of 6 |
|---|---|---|---|---|---|
| the clone | 0.960 | — | — | 6.97 | 5.96 |
| 20,480 | 0.980 | 0.940 | 0.970 | 6.87 / 7.09 / 6.90 | 5.98 / 5.94 / 5.97 |
| 40,960 | 0.970 | 0.970 | 0.970 | 6.94 / 6.94 / 6.89 | 5.97 ×3 |
| 61,440 | 1.000 | 0.970 | 0.970 | 6.80 / 6.93 / 6.99 | 6.00 / 5.97 / 5.97 |
| 81,920 | 1.000 | 0.980 | 0.960 | 6.80 / 6.87 / 6.91 | 6.00 / 5.97 / 5.96 |
| **122,880** | **0.960** | **0.980** | **0.960** | 6.88 / 6.81 / 7.04 | 5.96 / 5.98 / 5.96 |

Every read from 40k on has all three seeds at or above 0.95, and the
final one does: **HOLDS 3/3**, the rung's PASS clause. The in-run
rolling success (n=30 on 500000+) reached 95% within two evaluations
on every seed and never fell below 94.0 / 95.8 / 95.0 after — the dip
clause holds. Sampled play at n=100 on 900000+, paired against greedy:
−0.1 / −1.2 / +0.1 ± 1.2 vp, held 5.96–5.98, no do-nothing fingerprint
(stationary 0.16 either way). Paired against the clone on identical
seeds the turns read −0.09 ± 0.10 / **−0.16 ± 0.10** / +0.07 ± 0.13 —
ahead on two, behind on one, so **IMPROVES is not shown** on its 3/3
clause, and the two that are ahead are ahead by a tenth of a turn. The
census (n=100) is the clone's: 20.1–20.4 of 24 bodies on points and all
six held by turn 7, max stack 5.7–5.8, 4 / 2 / 4 failures each a single
short point. What reward did from a 0.960 start over 122,880 rounds is
a tenth of a turn on two seeds and nothing to the success rate outside
the clone's band.

#### The half-steps

**A5-bodies** (eight squads of three over three points at the middle
column; the bar 1.000 in 4.43 turns) reads **0.960 / 0.930 / 0.860** at
122,880 with 2.94 / 2.90 / 2.70 of three held, turns 5.46 / 5.80 / 6.23,
coherency 0.19–0.36 (bar 0.759). The by-turn census (n=20): three
points of radius 4 hold about eight bodies between them, so "on points"
saturates near eight for every policy; the rows differ on the clock.
The script has five bodies on points after turn 3 and every point held
after turn 5. The arm has under one body on a point after turn 3 on
every seed, then arrives — s1 and s2 hold all three by turn 7, a turn
to a turn and a half behind the bar; s3 holds 2.4 and stops, its five
failures in twenty having 2.6 of three points empty and 5.5 bodies on
points at the end, A5b's under-arrival on a fifth of the episodes. The
prediction was a pass by 60k; the axis alone is not learned to the mark
within the cap. It does reach the high nineties in-run on every seed
and 0.93–0.96 held-out on two where A5 never reached a rolling 50%.

**A5-points** (six squads of three over five points; the bar 1.000 in
6.71 turns) reads **0.030 / 0.060 / 0.330** at 122,880 with 2.47 / 2.92 /
3.83 of five held, the clock run out on every seed (9.6–10.0 of ten
turns), coherency 0.19–0.21. The census is A5b's: a third of the bodies
on points at the end against the script's seven eighths, whole points
abandoned by seed (s1 leaves three of the five empty in 60–85% of
episodes, s2 one point in 90%, s3 one in 55%), max stack under three,
and on s1 the bodies on points *fall* from 7.1 after turn 3 to 3.6 after
turn 5 — squads walk off points they reached. In-run it never reached a
rolling 10% on two seeds and peaked in the thirties on the third.

**Together the pair puts the wall on the point count.** At the same
rounds and with the same trainer: A4 (four squads, three points)
0.93–0.97; A5-bodies (twice the army, the same three points) 0.86–0.96,
a turn slower; A5-points (A4's army plus two squads, five points)
0.03–0.33 with A5's census; A5 itself never a rolling 50%. Doubling the
army costs a turn and one seed's pass; two more points cost everything.
The curve in the point count is steep — 0.70–0.80 at four points (A3 at
the cap), 0.03–0.33 at five, ~0 at six — and the army size is a real but
second-order cost. On the letter the quadrant is "both FAIL"; the
magnitudes are the finding.

### What it says

- **The credit hypothesis is closed.** Since the A3 speed screen the
  record carried "a body's own credit is diluted 1/24 by the mean" as
  the untested explanation for A5's under-arrival. Built and run three
  ways it moved nothing, and its first cut was a lesson of its own: on
  this trainer advantages are normalised and value targets are not, so
  a reward whose scale changes is a PPO change, and it has to be read
  as one (returns, gradient norm, clip fraction) before its arm is read
  at all. Do not fund another attribution build against A5's arrival.
- **A clone's residual is read by censusing its failures, not its
  match.** The 1,200-game clone matches the teacher's displacement head
  at 0.72 held-out and reads 0.930 by the rung's criterion; both numbers
  say "close" and neither says why. Twenty failed episodes say: one
  event, every time, a tie the observation does not carry. That is the
  reading that tells whether more games can help (only if the tie is
  decided by something on the board) — the D1 rule "score a clone by
  the rung's criterion and the ordering" gains "and census the
  failures". The 2,000-game fit narrowed it (0.930 → 0.960), so the tie is at least partly on the board.
- **Anchored PPO from a clone holds it and does not lift it, at every
  coefficient that holds.** D2 left the coefficient as the IMPROVES
  question; it is answered — 10 holds, 1 holds, 0.1 loses the clone
  slowly, none improves, twenty-four reads in one band. A policy that
  the anchor keeps within 0.03 nats of its start cannot move a success
  rate two hundredths; a policy free to move loses the plan. The lever
  that lifts a clone is not the anchor's strength and is not more
  rounds; it is unnamed, and until it is named the D-route's ceiling
  on a rung is the clone's read.
- **A5's wall is the number of points to cover at once, and the curve
  is steep.** With one trainer at one budget the spread rungs read
  0.70–0.80 at four points, 0.03–0.33 at five, ~0 at six, while
  doubling the army at three points costs a turn and a seed. A success
  that is a conjunction over N points is what the per-model critic has
  struggled to value since A3 (explained variance falling from 0.7–0.9
  to 0.3–0.4 there), and the census at five points shows the policy
  side of the same thing — points reached and then walked off. The
  next lever on A5 is about valuing or decomposing a conjunction, not
  about bodies, credit or the start.
- **Never call a pass from the in-run curve.** Across every arm here the
  n=30 in-run read on 500000+ ran 5–10 points above the greedy n=100
  read on 700000+; A5-bodies' seeds sat at 93–98% rolling and read
  0.86–0.96 held-out. The curve says when to read; the read decides.
- **Three seeds off one warm start agree because they share the start.**
  A5f–A5h's per-seed spread is the clone's binomial band, not seed
  variance; a fourth seed from the same clone would add nothing the
  census does not. Vary the clone before counting seeds.

### What was not done

- No whole-army control on any of these arms — A5's own control (0.80 / 0.94 / 0.98 at 245k) stands as the rung's.
- A5c–A5e were not run to budget: memory on a 31 GB box carrying up to fifteen trainers and a clone fit, then a reboot; each stop is an amendment with the read at the stop.
- A5f–A5h were stopped by decision at 89k / 89k / 44k; the chance of a 0.95 × 3 read from a true rate near 0.93 is about one in fifty.
- The 2,000-game fit ran twice: the first was killed by the reboot 4.5 hours in with nothing saved.
- A5's IMPROVES was never shown by any lever (A5i: ahead of the clone on two seeds by a tenth of a turn); the anchor's coefficient is answered, the lever that lifts a clone is not.
- No whole-army control was run for the imitation route; the D rungs established it on C3b and the phase facade has its own clone tooling, but a whole-army clone of the bar on A5 was not fitted.
- A5-points warm-started from A4x, at twice the cap, is the one reward-only arm the half-steps name and it was not run.
- One fit seed per clone. Two fit seeds agreed to a thousandth on D1; here the second clone differs from the first in games as well as seed, so the 0.930 → 0.960 step carries a fit-seed term nobody measured.
