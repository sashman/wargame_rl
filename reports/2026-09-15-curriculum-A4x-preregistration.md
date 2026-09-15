# Pre-registration: curriculum rung A4, second arm (A4x) — the A4 runs resumed to 245,760 rounds

Written 2026-09-15, **before any number from the arm exists**, on branch
`feature/curriculum-a4` (PR #352 carries the rung's arms). Issue #358,
parent question #340, rung **A4** — a follow-up to #351, which read
**FAIL as pre-registered** at the cap: greedy success **0.930 / 0.930 /
0.970** at 122,880 rounds, `held` 2.91 / 2.93 / 2.97 of 3, the two short
seeds still rising (last twelve in-run evaluations 70–93 and 66–93),
against a control at 0.980 / 0.980 / 0.990.

## The one change

**Budget.** The three A4 per-model runs (Wandb `curriculum-a4`,
`la0ztx25` s1 · `262q9d0k` s2 · `rfqku890` s3) resumed **in place** with
`--resume-from <run_dir> --rounds 245760 --eval-every-rounds 512
--checkpoint-every-rounds 512` — same seeds, same knobs (`ent_coef` 0.003
carried and refused if changed), optimizer and generator restored,
cadences passed explicitly (#346). 245,760 rounds is the once-only
doubling A3x was granted.

## Why this arm, and the rule it amends

The symmetric-cap rule (A3x) grants the per-model arm the control's
extension when the control needed one. A4's control did not: it passed
inside its 60 epochs — but only at epochs 44 / 58 / 46, i.e. 92k–121k of
its 123k rounds, and on every earlier rung the per-model arm has needed
2–4× the control's rounds. Six times the control's slowest pass is
725k; the cap is 123k; the cap binds, and a read at the cap measures the
cap. This arm tests that, and the pre-registration writes the amendment
it implies **before the run**: when 6× the control's slowest
rounds-to-pass exceeds the cap, the per-model arm's budget is 2× the cap
(245,760), read once at the end. A4's FAIL stands as a read at the cap
whatever this arm says.

## Comparator

`squad_march_take` on `a4.yaml`, seeds 700000+ at n=100: success 1.000,
`held` 3.00, turns 4.70, coherent 0.923. The control at 122,880:
0.980 / 0.980 / 0.990, `held` 2.98–2.99, turns 6.12–6.23. A4's read at
122,880: 0.930 / 0.930 / 0.970, `held` 2.91 / 2.93 / 2.97, turns 5.36 /
5.46 / 5.21.

## Criteria

Read at 245,760 rounds from `last.pt`, greedy, no decode, n=100 on seeds
700000+.

- **PASS:** success ≥ 0.95 on **all three** seeds. The amendment above
  stands for the rungs above; the ladder continues to A5b.
- **FAIL:** any seed below 0.95 at 245,760 — the per-model arm is worse
  than the whole-army trainer at allocation with a spare squad; the
  ladder stops at A4 and the next step is a recording of the failing
  episodes (which point is short, where the spare squad ends).
- **Readouts:** `held`, `on_obj` (the script's spare squad stands off at
  0.724; the per-model arm's reads 0.60–0.72 at the cap); coherency
  greedy and sampled; the panel over the last quarter of the resumed
  half; the in-run curve after any first pass.

Power: binomial SE 0.022 at p=0.95, n=100. Two seeds sit one SE below
the bound; this arm reads the direction of travel over 123k more rounds,
and the in-run curves are the readout that says whether the last few
percent arrive or the seeds plateau.

## Budget, regime, seeds

| | per-model arm |
|---|---|
| budget | 245,760 rounds total (122,880 more), 128 rounds per update |
| seeds | 1, 2, 3, continued |
| in-run eval | every 512 rounds, n=30, seeds 500000+ |
| final score | n=100, seeds 700000+ |
| flags | carried from the checkpoint (`ent_coef` 0.003); cadences 512 / 512 passed |
| logging | the runs' own Wandb ids, continued |

## What I expect (a guess, written so it can be wrong)

All three pass by ~180k rounds with `held` ≥ 2.95; the spare squad ends
on a point rather than off it (`on_obj` above the script's 0.724 on at
least two seeds). If instead s1 and s2 plateau at 0.90–0.93, the spare
squad is the thing the per-model arm cannot place, and the recording is
the next step.
