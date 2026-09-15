# Curriculum rung A3, second arm: given the control's rounds, the per-model arm spreads over four points — PASS 3 of 3

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

## Provenance

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
| pre-registration | `reports/2026-09-15-curriculum-A3x-preregistration.md` at `07c2ac6`, before the resume |

## The read

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

## What it says

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

## What was not done

- No recording of the residual 3–4% of failing episodes (which point is
  short); the spread statistic the pre-registration named as a readout
  is not computed — the rung passed on `held` 3.94–3.96, which bounds
  it.
- The control was not extended further; it never needed to be.
