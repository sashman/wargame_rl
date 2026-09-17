# Curriculum rung T1: warm-starting A5 from A4x does not rescue it — FAIL as pre-registered, and the transfer is real

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

## Provenance

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
| pre-registration | `reports/2026-09-15-curriculum-T1-preregistration.md` at `969ec81` (2026-09-15 16:28), before any number |

## The read

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

## What it says

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

## What was not done

- No paired-on-init comparison (the warm start changes the init by
  construction).
- No second budget, no other warm-start source (A3x, A2b): the rung has
  one change, and it is read once.
- The peak checkpoints were scored as a readout (`pm-*.pt`), not as a
  criterion: 0.47 / 0.37 / 0.64, the best any per-model run has read on
  this rung, and still a third of the bodies short.
