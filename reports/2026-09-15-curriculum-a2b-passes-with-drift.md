# Curriculum rung A2, second arm: at `ent_coef` 0.003 the per-model arm holds the rung — PASS with drift

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

## Provenance

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
| pre-registration | `reports/2026-09-15-curriculum-A2b-preregistration.md` at `c2bcf26`, before the launch |

## The read

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

## What it says

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

## What was not done

- No sweep between 0.003 and 0.03; no lower value.
- The control was not re-run at 0.003.
- No recording of the transient collapses.
