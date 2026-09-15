# Curriculum rung A2: the per-model arm solves the rung by 10k rounds, then unlearns it — FAIL, as pre-registered

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

## Provenance

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
| pre-registration | `reports/2026-09-15-curriculum-A2-preregistration.md` at `8f66737`; amendment 1 (the control's read and the budget) at `1024960`, before any per-model number |

## The read

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

## What it says

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

## What happens next

**A2b** (#356), pre-registered at `c2bcf26`
before it ran and launched the same minute: the same three seeds, the
same scenario, budget and regime, with `--ent-coef 0.003`. PASS is 3/3 at
the end **and** no in-run dip below 0.80 after the first pass; FAIL says
the drift is not the entropy bonus and the update itself (the clip
fraction) is the next suspect. The ladder does not open A3's per-model
arm until A2b is read — A3's and A4's and A5's controls have already run
and are recorded on their pre-registrations.

## What was not done

- No recording of what a drifted policy does (which squad walks apart,
  whether the stationary units are the same ones).
- No sweep of `ent_coef`; A2b is one value, the one the record already
  measured on the other trainer.
- The control was not extended; it never needed to be.
