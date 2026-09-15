# Curriculum rung A4, second arm: given twice the cap, the per-model arm places the spare squad — PASS 3 of 3, at the script's speed

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

## Provenance

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
| pre-registration | `reports/2026-09-15-curriculum-A4x-preregistration.md` at `d9cb482`, before the resume |

## The read

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

## What it says

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

## What was not done

- No recording of the residual 2% of failing episodes on s1.
- Nothing about the rungs above: the ladder is paused (the user's
  instruction, 2026-09-15) for the A3 speed screen (PR #362, arms
  #359–#361), so A5b does not launch on this pass.

**Amendment, 10:14 the same morning.** The pause was lifted while this
was being written ("continue with the plan, get through the ladder"):
A5b (#355) launched at 10:07 at 245,760 rounds and `ent_coef` 0.003
(its pre-registration's amendment 1), beside the speed-screen runs. The
last bullet above records what was true when the read was taken.
