# Curriculum rung A1: two seeds pass at the script's speed, one plateaus at 0.79 — FAIL, as pre-registered

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

## Provenance

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
| pre-registration | `reports/2026-09-15-curriculum-A1-preregistration.md` at `ba1dd5c`; amendment 1 (the control's read) at `7e217ad`, before any per-model number |

## The rung

A0's scenario with its three models as one unit of three, deployed as a
squad (`enforce_at_deployment`); nothing pays for or referees coherency.
Radius 4, 8 rounds, 60×44, Move 6. **The bar**, n=100 on 700000+: success
0.960 (four episodes with a model frozen behind a squadmate), turns 4.96,
coherent 0.993.

## The read

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

## What it says

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

## What happens next

A second arm on this rung, **A1x**, pre-registered before it runs: the
three per-model runs resumed **in place** (`--resume-from`, same seed, same
knobs, optimizer and generator restored) to **122,880 rounds** — 2× the A1
budget, 6× the control's slowest rounds-to-pass, the ladder's cap. PASS is
3 of 3 at the same bound. If s2 passes, A1's verdict stands as written and
the budget rule is amended for the rungs above (the per-model arm gets a
floor of 6× the control); if it does not, the fault is in the per-model
handling of a unit and the ladder stops here until it is found.

## What was not done

- No inspection of *why* s2's third model does not arrive in 21 episodes
  (a recording of the failing seeds, `just record-per-model-events`, is the
  next diagnostic if A1x does not resolve it).
- The control was not resumed past 60 epochs.
