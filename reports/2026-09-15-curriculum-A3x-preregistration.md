# Pre-registration: curriculum rung A3, second arm (A3x) — the A3 runs resumed to the control's 245,760 rounds

Written 2026-09-15, **before any number from the arm exists**, on branch
`feature/curriculum-a3` (PR #350 carries the rung's arms). Issue #357,
parent question #340, rung **A3** — a follow-up to #349, which read
**FAIL as pre-registered** at the cap: greedy success **0.700 / 0.800 /
0.770** at 122,880 rounds, `held` 3.49 / 3.60 / 3.65 of 4, still rising,
against a control at 0.950 / 0.950 / 0.960 after its once-only extension
to 120 epochs (245,760 rounds).

## The one change

**Budget.** The three A3 per-model runs (Wandb `curriculum-a3`,
`wzalbuic` s1 · `4x0p1q4a` s2 · `jzmpi6pl` s3) resumed **in place** with
`--resume-from <run_dir> --rounds 245760 --eval-every-rounds 512
--checkpoint-every-rounds 512` — same seeds, same knobs (`ent_coef` 0.003
carried by the checkpoint and refused if changed), optimizer and
generator restored; the cadences are passed explicitly because a resume
drops them (#346). 245,760 rounds is the control's extended budget in
rounds: the same once-only doubling the control was granted on this
rung, applied to the arm under test.

## Why this arm

A3 is the first rung on which the whole-army control needed its
extension — at the cap it read 0.850 / 0.980 / 0.930 and passed 3 of 3
only at 245k rounds. The per-model arm at the cap reads 0.70 / 0.80 /
0.77 with every seed's in-run curve still climbing (last six evaluations
53–90%, `held` 3.2–3.8). Reading it against a control that had twice
the rounds is the comparison the cap rule did not anticipate; giving the
arm the same rounds is the symmetric test, and it costs one more run.

## Comparator

`squad_march_take` on `a3.yaml`, seeds 700000+ at n=100: success 1.000,
`held` 4.00, turns 5.28, coherent 0.927. The control: 0.850 / 0.980 /
0.930 at 122,880 rounds, **0.950 / 0.950 / 0.960 at 245,760**. A3's read
at 122,880: 0.700 / 0.800 / 0.770.

## Criteria

Read at 245,760 rounds from `last.pt`, greedy, no decode, n=100 on seeds
700000+.

- **PASS:** success ≥ 0.95 on **all three** seeds. A3's verdict stands as
  a read at the cap; the per-model arm allocates four squads over four
  points within the rounds the control needed, and the cap rule gains the
  control's symmetry from here: when the control needs its extension, the
  per-model arm gets the same one.
- **FAIL:** any seed below 0.95 at 245,760 — the per-model arm is worse
  than the whole-army trainer at spreading over points at equal rounds,
  and the ladder stops at A3; the next step is a recording of the failing
  episodes, asking which point is left empty and whether the unit
  pointer sends two squads to one.
- **Readouts:** `held` and `on_obj`; the equal-rounds comparison to the
  control (at 123k: per-model 0.70 / 0.80 / 0.77 v control 0.85 / 0.98 /
  0.93; at 245k: this read v 0.95 / 0.95 / 0.96); coherency greedy and
  sampled; the panel over the last quarter of the resumed half; the
  in-run curve after any first pass.

Power: binomial SE 0.022 at p=0.95, n=100. The seeds sit 7–11 SE below
the bound, so a pass is a move.

## Budget, regime, seeds

| | per-model arm |
|---|---|
| budget | 245,760 rounds total (122,880 more), 128 rounds per update |
| seeds | 1, 2, 3, continued |
| in-run eval | every 512 rounds, n=30, seeds 500000+ |
| final score | n=100, seeds 700000+ |
| flags | carried from the checkpoint (`ent_coef` 0.003); cadences 512 / 512 passed explicitly |
| logging | the runs' own Wandb ids, continued |

## What I expect (a guess, written so it can be wrong)

Two of three seeds pass by ~200k rounds and the third sits at 0.90–0.95:
a marginal read. `held` ends near 3.9 on every seed. Coherency stays
below 0.6 greedy.
