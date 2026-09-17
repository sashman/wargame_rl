# Curriculum rung C1: with an enemy squad standing on a point, the per-model arm passes on two seeds of three — FAIL as pre-registered

**Verdict first.** On the enemy rung (#340, arm #365) — A3 plus one
enemy unit of three standing on one of the four points, nobody shooting
— the per-model arm warm-started from A3x reads greedy success **0.990 /
0.930 / 0.990** at 245,760 rounds, n=100, `held` 3.15 / 2.96 / 3.06 of
3 (the enemy's point can be occupied but never held), turns **5.10 / 5.37
/ 5.01** against the script's 4.98 and the whole-army control's
6.82–7.09. Seed 2 sits under 0.95 at the budget, so this is **FAIL** on
the pre-registered clause. It is a fail by one seed on a plateau: seed 2
crossed 95% in-run at 4,608 rounds, never dipped below 80 after, and sat
at 85–95 for the remaining 240k. The control passes 3 of 3 at 120 epochs
(0.980 / 0.990 / 0.990) two rounds slower than the script. The
prediction on file — a pass inside 40k rounds on all three — is wrong on
one seed and right on the speed.

## Provenance

| field | value |
|---|---|
| date | 2026-09-16 (launched 12:43 after the 2026-09-15 crash killed the first launch at 384 rounds; first leg exited 17:28–17:30, resumed 17:30, exited 22:00–22:02) |
| GPU / no-GPU | GPU (RTX 4090), shared with eight other per-model runs (A5b, T1) and, for the first hour, the control |
| seeds | 1 / 2 / 3; rollout layouts at seed×100+; **not paired on init** — each seed starts from A3x's `last.pt` of the same seed (245,760 rounds, PASS 3/3) |
| n | 100 at seed base 700000 (final and the 122,880 readout); 30 at 500000 (in-run, every 512 rounds); 100 at 900000 (greedy against sampled) |
| config | `configs/experiments/curriculum/c1.yaml` — unrefereed by design; success `all_objectives_occupied` counts our bodies |
| decode | none on the per-model facade; K=1 on the control |
| paired | per episode against `squad_march_take` on identical seeds |
| comparator | `squad_march_take`, both facades identical: success 0.990, held 3.00, turns 4.98, `on_obj` 0.871, vp +8.2 ± 0.6, coherent 0.933 |
| opponent | `scripted_baseline` wrapping `hold_deployment`: three models, one declared unit, on the point at (35, 27), unarmed |
| budget | per-model 245,760 rounds at 128 per update (`--rollout-rounds 32 --num-rollout-envs 4`), `--ent-coef 0.003`, resumed in place at 122,880 (#346: cadences passed explicitly); control 60 epochs extended once to 120 |
| code revision | `2a33b50` on `feature/curriculum-c1` (PR #366) for the launch; docs-only commits after |
| checkpoint | `last.pt` at 245,760; `checkpoints/per_model/per-model-c1-2026-09-16-12-43-32-s{1,2,3}c1` |
| coherency | greedy 0.524 / 0.500 / 0.632 at 700000+; 0.54 / 0.56 / 0.64 greedy against 0.34 / 0.34 / 0.47 sampled at 900000+ |
| Wandb | `curriculum-c1`: per-model `s6t9rpw3` / `an6yas9o` / `sypr0o0v`, resumed halves `fk7v2efa` / `t3wak5rd` / `6kqsj7n2`; control `dt6uu1f4` / `fj4sybie` / `s30qxkjv`, extended `w69w9t6h` / `wbd13z9q` / `5yk8p1il` |
| pre-registration | `reports/2026-09-15-curriculum-C1-preregistration.md` at `2a33b50` (2026-09-15 16:38), amendment 1 at `4dc1c5f` (2026-09-16 13:01, after the control's 60-epoch read and before any per-model number) |

## The read

| row | success | turns | vs bar, paired | held | on obj | coherent | rounds to rolling 50 / 80 / 95% |
|---|---|---|---|---|---|---|---|
| `squad_march_take` | 0.990 | 4.98 | — | 3.00 | 0.871 | 0.933 | — |
| **s1 at 245,760** | **0.990** | 5.10 | +0.12 ± 0.07 | 3.15 | 0.801 | 0.524 | 2.6k / 2.6k / 40k; last ten 95–100 |
| **s2 at 245,760** | **0.930** | 5.37 | +0.39 ± 0.10 | 2.96 | 0.843 | 0.500 | 2.6k / 2.6k / 4.6k; last ten 85–95 |
| **s3 at 245,760** | **0.990** | 5.01 | +0.03 ± 0.06 | 3.06 | 0.813 | 0.632 | 2.6k / 2.6k / 10k; last ten 90–100 |
| s1 / s2 / s3 at 122,880 (readout) | 0.870 / 0.980 / 0.940 | 5.56 / 5.37 / 5.43 | +0.58 / +0.39 / +0.45 | 2.86 / 3.09 / 3.04 | 0.73–0.81 | 0.47–0.52 | — |
| control at 60 epochs | 0.920 / 0.960 / 0.970 | 6.92 ×3 | +1.94 | 3.02–3.12 | 0.90–0.94 | 0.71–0.78 | 95% at epoch 55 / never / 51 |
| control at 120 epochs | 0.980 / 0.990 / 0.990 | 7.09 / 6.82 / 6.83 | +1.84 to +2.11 | 2.99–3.00 | 0.93–0.98 | 0.72–0.79 | 95% at epoch 64 / 75 / 65 |

**Greedy against sampled** (900000+, n=100, paired): +0.5 ± 0.8 / +1.0 ± 0.7 / +0.5 ± 0.6 vp; `held` 3.20 / 3.09 / 3.01 greedy against 3.13 / 3.00 / 3.05 sampled; coherency 0.54 / 0.56 / 0.64 greedy against **0.34 / 0.34 / 0.47** sampled. The sampled policy holds the points as well as the greedy one and walks the squads further apart, as on every rung.

**Health panel, last quarter** (480 updates per seed): explained variance
0.52 / 0.46 / 0.50, clip fraction 0.24 / 0.27 / 0.26, ratio p99
1.88–1.91, displacement entropy 0.85 / 1.10 / 1.18, declaration entropy
≤ 0.02, advantage std 0.34–0.40. In-run dips below 80 after the first
rolling pass: 3 / 0 / 8 of 402 / 472 / 461 evaluations.

**The by-turn census** (n=20 on 700000+, bodies on objectives of 12 and
points held of 3, turn 5 → 8, with which point is empty at the end):

| policy | success (n=20) | turns | on objectives, turn 5 → 8 | held, turn 5 → 8 | point empty at the end (y = 8 / 17 / **27, the enemy's** / 36) | max stack |
|---|---|---|---|---|---|---|
| `squad_march_take` | 0.95 | 5.20 | 9.4 → 10.0 | 3.0 → 3.0 | 0 / 0 / **0.05** / 0 | 2.8 |
| s1 | 1.00 | 5.00 | 10.0 → 10.0 | 3.2 → 3.3 | 0 / 0 / 0 / 0 | 3.6 |
| s2 | 0.95 | 5.40 | 10.6 → 10.4 | 3.1 → 3.0 | 0 / 0.05 / 0 / 0.05 | 3.9 |
| s3 | 1.00 | 4.90 | 9.5 → 9.5 | 3.1 → 3.1 | 0 / 0 / 0 / 0 | 3.5 |

**The short point is not the enemy's.** The script's own one miss in
twenty is at the enemy's disc — the friendly gridlock the pre-registration
named. The learned policies get onto that disc every time; seed 2's
misses are the neighbouring points at y=17 and y=36, i.e. allocation
between empty points, the A3 failure mode, not the engagement-range
endpoint rule. The arm also out-numbers the enemy on its point often
enough to *hold* it on some episodes (`held` 3.0–3.3 of a 3.0 ceiling
for the script), stacking up to 3.5–3.9 on one point against the
script's 2.8.

## What it says

- **The warm start carries the rung's skill and the enemy's disc costs a
  turn of shuffling, as predicted** — every seed is at 80% in-run by
  2,560 rounds and two pass at n=100 within 0.1 turns of the script.
  The whole-army control needed 131k–154k rounds to reach 95% in-run and
  arrives two rounds behind the script.
- **One seed plateaus at 0.93 and that is the verdict.** Seed 2 passed
  first (4.6k) and then sat at 85–95 in-run for 240k rounds without a
  dip and without a climb; its 122,880 readout was the best of the three
  (0.980) and its 245,760 read the worst (0.930). Binomial SE at n=100
  is 0.026 at that rate, so the two reads are two SE apart: a plateau
  around 0.95 read twice, not a drift. The rule reads it as a fail, as
  A1 and A3 were read at their caps, and the record says what happened
  when those runs were given more rounds — and this one had 245,760.
- **The per-model arm is faster than the control on every seed and
  slightly below it on the bar**: turns 5.0–5.4 against 6.8–7.1, success
  0.93–0.99 against 0.98–0.99. Both had 245,760 rounds on this scenario;
  the arm also had A3x's 245,760 on the empty table behind it, which is
  what the C rungs' warm-start rule spends. Read at equal rounds on this
  scenario, the arm was at 80% in-run by 2.6k where the control reached
  it at 55k–60k.
- **The enemy's disc is not the failure; allocation is.** The
  pre-registration's FAIL clause asked for a recording of which point is
  short: it is the neighbours of the enemy's point, never the enemy's.
  The engagement-range endpoint rule costs the bound squad nothing the
  policy cannot learn in a few thousand rounds; what remains on seed 2
  is the residual assignment error A3 carries at 0.96–0.97.
- **Coherency 0.50–0.63 greedy** — unpaid, as on every rung so far.

## What was not done

- No second budget: the symmetric cap already put the arm at 245,760.
- No recording of seed 2's failing seven episodes beyond the census.
