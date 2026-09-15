# Curriculum rung A4: one standard error short at the cap on two seeds — FAIL, as pre-registered

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

## Provenance

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
| pre-registration | `reports/2026-09-15-curriculum-A4-preregistration.md` at `67bfb6f`; amendments 1–2 at `8c52e83`, `7171679`, each before the number it precedes |

## The read

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

## What it says

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

## What was not done

- No recording of the failing episodes (which point is short).
- No sweep; A4x changes only the budget.
