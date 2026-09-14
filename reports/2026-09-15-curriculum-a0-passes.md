# Curriculum rung A0: the per-model pipeline learns the plumbing rung — PASS with a defect

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

## Provenance

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
| pre-registration | `reports/2026-09-14-curriculum-A0-preregistration.md`, committed at `df3e1c3` before any training number; amendment 1 at `04329c5` before any per-model number |

## The rung

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

## The read

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

## What it says

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

## What was not done

- No per-gradient-step comparison between facades; no epoch-time comparison.
- The whole-army control was not resumed past 60 epochs to see whether its
  turn count keeps falling; the per-model arm was not extended past budget.
- s3's 0.950 is exactly the bound; at n=100 that is 5 episodes, and a
  rerun on another seed band could land either side. The verdict does not
  rest on s3 alone.
