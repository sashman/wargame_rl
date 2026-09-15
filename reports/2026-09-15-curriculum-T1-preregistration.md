# Pre-registration: curriculum rung T1 — A5 warm-started from A4x's checkpoint, against A5b from scratch (the transfer rung)

Written 2026-09-15, **before any number from the arm exists**, on branch
`feature/curriculum-t1` (stacked on PR #362). Issue #363, parent
question #340, rung **T1**. The comparator, A5b (#355), is in flight at
the time of writing: 85k of 245,760 rounds, in-run 5–19%.

## The one change

**The starting weights.** `configs/experiments/curriculum/a5.yaml`
unchanged (eight squads of three, six objectives at radius 4, ten
rounds), each seed started with `--warm-start-from
checkpoints/per_model/per-model-a4-2026-09-15-07-08-53-s{1,2,3}a4/last.pt`
— the A4x checkpoints at 245,760 rounds that read 0.980 / 1.000 / 1.000
on A4. Fresh optimizer. The set network loads across the army size
(twelve bodies to twenty-four, four squads to eight, three objectives to
six): smoke-tested for 256 rounds before this was written. Everything
else is A5b's — 245,760 rounds at 128 per update (`--rollout-rounds 32
--num-rollout-envs 4`), `--ent-coef 0.003`, in-run eval every 512 rounds
at n=30 on 500000+, checkpoints every 512, Wandb group `curriculum-t1`.

## Why this rung, and why now

#340's T1 asks whether the size-independence the set network claims
delivers in practice: a rung warm-started from the rung below should
pass in fewer rounds than the same rung from scratch. This morning's A3
speed screen (S3, #361) measured that one rung early: A3 from A2b read
AHEAD on 3 of 3 (0.91 / 0.94 / 0.96 against scratch's 0.70 / 0.80 / 0.77),
with in-run 80% at 11k–22k rounds against scratch's 91k–never and no
walk-off from 20k on. That is the prior. T1 launches beside A5b rather
than after it because the arm does not need A5b's numbers to run, only
to be read, and the GPU would otherwise sit half idle for five hours.

## Comparator

A5b's three runs from scratch (Wandb `curriculum-a5`: `b9zwoiib` s1 ·
`2t2nj8tb` s2 · `g9pnffwx` s3), read at 245,760 on the same seeds:
their in-run rounds-to-pass and their n=100 final read. Paired on
layouts and seeds, **not on init**. The bar: `squad_march_take`, both
facades, seeds 700000+ at n=100: success 1.000, `held` 6.00, turns
6.77, `on_obj` 0.856, coherent 0.822. The whole-army control at 245,760:
0.800 / 0.940 / 0.980, `held` 5.76 / 5.93 / 5.98.

## Criteria

**Rounds-to-pass** = the first in-run evaluation at which the rolling
mean of five (n=30 on 500000+, every 512 rounds) reaches 95%, capped at
the budget; a seed that never reaches it in-run has rounds-to-pass
245,760.

- **PASS:** on every seed, T1's rounds-to-pass ≤ half of A5b's same-seed
  rounds-to-pass (a seed on which A5b never passes in-run bounds T1 at
  122,880), **and** T1 reads success ≥ 0.95 at n=100 on 700000+ from
  `last.pt` at 245,760 on all three seeds. #340's letter is "rounds-to-pass
  ≤ half of scratch, 3/3 seeds"; the n=100 clause is added so that a
  "pass" is a pass by the rungs' own standard and not an n=30 wobble.
- **FAIL:** any seed misses the halving bound, or any seed reads below
  0.95 at 245,760. #340 names the consequence: the size-independence
  claim is not delivering in practice, every later rung trains from
  scratch, and #283's premise gets a report.
- **NULL (scenario):** only if the script fails its own criterion. It
  does not.
- **Pass with a defect:** the bound holds but the health panel is red
  over the last quarter.

**Readouts, not criteria:** success / `held` / turns at the 122,880
checkpoint (`pm-00122880.pt`), where A5's cap would have read;
coherency greedy and sampled (`just measure-per-model-eval-mode`, 900000+);
the by-turn census at 20,480 (bodies on objectives and points held, turn
5 → 8: the walk-off signature); the health panel over the last quarter;
A5b beside every row.

Power: the bound is a factor of two on a curve sampled every 512 rounds
at n=30 — coarse, and deliberately so; on A3 the same lever read a factor
of four or more on every seed. Binomial SE at p=0.95, n=100 is 0.022.

## Budget, regime, seeds

| | T1 arm |
|---|---|
| budget | 245,760 rounds, 128 rounds per update |
| seeds | 1, 2, 3; rollout layouts at seed×100+ |
| start | A4x `last.pt` of the same seed (245,760 rounds, revision `d9cb482`) |
| in-run eval | every 512 rounds, n=30, seeds 500000+ |
| final score | n=100, seeds 700000+ |
| flags | `--num-rollout-envs 4 --rollout-rounds 32 --eval-every-rounds 512 --checkpoint-every-rounds 512 --n-eval-episodes 30 --ent-coef 0.003 --warm-start-from <A4x last.pt>` |
| logging | Wandb group `curriculum-t1`, one run per seed |
| code | the `feature/curriculum-t1` working tree at this commit (PR #362's tree plus this file) |

## What I expect (a guess, written so it can be wrong)

PASS 3/3: in-run 95% by 60k–90k rounds on every seed against an A5b
that reaches it late or not at all (its control read 0.80 / 0.94 / 0.98
at 245,760). The doubling of the army costs the transfer something — the
A4x policy has never seen eight squads or six points — so the factor is
smaller than A3's four-plus, but above two. If instead T1 reads under
0.95 on a seed at 245,760, the likely shape is the control's: one point
left empty while bodies stack, and the report says whether that seed's
A5b run did the same.
