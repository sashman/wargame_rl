# Curriculum rung A3: the per-model arm is behind the control on the spread rung at the cap — FAIL, as pre-registered

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

## Provenance

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
| pre-registration | `reports/2026-09-15-curriculum-A3-preregistration.md` at `070c422`; amendments 1–3 at `471eed2`, `14c8db8`, `a6afc74`, each before the number it precedes |

## The read

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

## What it says

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

## What was not done

- No recording of the failing episodes yet (which point, and whether two
  squads share one).
- No sweep of anything; A3x changes only the budget.
