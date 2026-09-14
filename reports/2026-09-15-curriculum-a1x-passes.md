# Curriculum rung A1, second arm: the same runs given twice the rounds pass 3 of 3 — the budget rule was the defect

**Verdict first.** The three A1 per-model runs, resumed in place from
61,440 to **122,880 rounds** (#345, pre-registered before the resume),
pass the bound on **3 of 3 seeds**: greedy success **1.000 / 1.000 /
1.000** at n=100, turns **4.93 / 4.88 / 4.93** against the script's 4.96 —
every seed now at or under the script's own speed, where every whole-army
control seed is 1.1–2.5 rounds slower. The seed that read 0.790 at the A1
budget passed on the first evaluation after the resume and never dipped;
the two that had already passed hold 1.000, each with one n=30 dip in 240
evaluations. A1's FAIL stands as written — it was a pre-registered read at
a pre-registered budget — and what it measured was the budget: **3× the
whole-army control's rounds-to-pass is too little for the per-model arm on
this rung, and 6× is enough.** The ladder's budget rule gains a floor of
6× from A2 on. Coherency, which nothing on this rung pays for, rose with
the extra rounds on every seed: greedy 0.968 / 0.980 / 0.972 (from
0.76–0.90), sampled 0.70 / 0.81 / 0.77 (from 0.23–0.62).

## Provenance

| field | value |
|---|---|
| date | 2026-09-15 (resumed 00:26, exited 00:39) |
| GPU / no-GPU | GPU (RTX 4090), three processes |
| seeds | 1 / 2 / 3, the A1 runs continued (`--resume-from`, optimizer and generator restored) |
| n | 100 at seed base 700000 (final); 30 at 500000 (in-run) |
| config | `configs/experiments/curriculum/a1.yaml` — unrefereed by design |
| decode | none |
| paired | per episode against `squad_march_take`; against A1's read on the same seeds and checkpoints |
| comparator | `squad_march_take`: success 0.960, turns 4.96, coherent 0.993; A1's read: 1.000 / 0.790 / 1.000 |
| opponent | none |
| budget | 122,880 rounds (61,440 more), 128 rounds per update |
| code revision | `3037b11` on `feature/curriculum-a1` (PR #344) |
| checkpoint | `last.pt` at 122,880 |
| coherency | greedy 0.968 / 0.980 / 0.972; sampled 0.703 / 0.808 / 0.769 |
| Wandb | `curriculum-a1`: `9irv0y7y` s1 · `cm17c1wq` s2 · `ahcy2jl0` s3, continued |
| pre-registration | `reports/2026-09-15-curriculum-A1x-preregistration.md` at `3037b11`, before the resume |

⚠ **The resume dropped the eval and checkpoint cadences** (512 → the 256
default) and overwrote `provenance.json` with the new values — #346. The
in-run curve is twice as dense after 61,440; nothing scored changes.

## The read

| row | success | turns | vs bar, paired | held | on obj | coherent greedy | coherent sampled | first pass, resumed half |
|---|---|---|---|---|---|---|---|---|
| `squad_march_take` | 0.960 | 4.96 | — | 1.00 | 0.987 | 0.993 | — | — |
| s1 at 61,440 (A1) | 1.000 | 5.01 | +0.05 | 1.00 | 1.000 | 0.760 | 0.555 | — |
| **s1 at 122,880** | **1.000** | 4.93 | −0.03 ± 0.07 | 1.00 | 1.000 | 0.968 | 0.703 | held (one dip, 85 at 94,464) |
| s2 at 61,440 (A1) | 0.790 | 6.10 | +1.14 | 0.87 | 0.823 | 0.889 | 0.232 | — |
| **s2 at 122,880** | **1.000** | 4.88 | −0.08 ± 0.07 | 1.00 | 1.000 | 0.980 | 0.808 | 61,696 — the first evaluation after the resume, never below 95 |
| s3 at 61,440 (A1) | 1.000 | 4.90 | −0.06 | 1.00 | 1.000 | 0.896 | 0.620 | — |
| **s3 at 122,880** | **1.000** | 4.93 | −0.03 ± 0.07 | 1.00 | 1.000 | 0.972 | 0.769 | held (one dip, 80 at 120,832) |

Greedy − sampled on the tuning band (900000+, n=100): −0.1 ± 0.4 / −0.1 ±
0.3 / −0.1 ± 0.3 vp, win 100% both ways.

**Health panel over the last quarter of the run** (240 updates per seed):
advantage std 0.21–0.22, explained variance 0.72 on all three (up from
0.54–0.68), ratio p99 1.72–1.78 with clip fraction 0.16–0.21 (the regime's
normal), declaration entropy 0.001–0.005, displacement entropy 0.87–1.03.
Nothing red.

## What it says

- **The budget rule was the defect, not the pipeline.** A1 read FAIL at a
  budget of 3× the control's slowest rounds-to-pass; the same runs at 6×
  pass 3/3 with every seed at the script's speed. Per-model rounds-to-pass
  on this rung: 53k / ~62k / 32k against the control's 14k–20k, i.e.
  **2–4× the control**, the same band as A0. From A2 on the per-model
  budget is **6× the control's slowest rounds-to-pass**, floor 20,480, cap
  122,880 (#340 contract 3, amended here).
- **s2 was at the threshold, not stuck.** Its n=30 in-run curve read
  80–93 at budget and its n=100 read 0.790; the first evaluation after the
  resume was ≥ 95 and it never dipped again. The plateau was the last few
  percent, which a 128-round update takes thousands of rounds to close.
- **Coherency rises with training even though nothing pays for it.**
  Greedy 0.76–0.90 → 0.97–0.98, sampled 0.23–0.62 → 0.70–0.81, on every
  seed. A policy that has converged on "walk straight at the point" keeps
  the squad together as a by-product; a policy still exploring does not.
  The greedy/sampled gap is still 0.17–0.27 and is the number the E rungs'
  referee will meet.
- **Speed, a third time.** Three per-model seeds at 4.88–4.93 turns against
  a script at 4.96 and a control at 6.1–7.4. A0 and A1 together are six
  seeds, all within 0.2 turns of the script; six control seeds, all a
  round or more behind. This is now a pattern about the two trainers on
  this reward, and it goes in the rule, still labelled as movement-only
  rungs with three bodies.
- **The verdict discipline held.** A1 was called FAIL on the letter, the
  follow-up was pre-registered with its own PASS / FAIL / regression
  clauses before it ran, and the regression clause was checked (two
  single-evaluation dips at n=30, no regression). Nothing was re-read
  until the criterion was in git.

## What was not done

- No recording of what s2's third model did in the 21 failing episodes at
  61,440; the question dissolved when it passed.
- The whole-army control was not extended; its 6.1–7.4 turns stand at 60
  epochs.
