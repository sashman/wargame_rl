# Pre-registration: curriculum rung A2, second arm (A2b) — the entropy coefficient at 0.003

Written 2026-09-15, **before any number from the arm exists**, on branch
`feature/curriculum-a2` (PR #348 carries the rung's arms). Issue #356,
parent question #340, rung **A2** — a follow-up to #347, which read
**FAIL as pre-registered**: greedy success **0.370 / 0.880 / 0.980** at
n=100 at 122,880 rounds against a whole-army control at 1.000 ×3.

## Why this arm

A2's three per-model seeds all read 100% in-run success with coherency
1.00 by 6k rounds and held it to 18k–31k; then coherency collapsed to
0.1–0.4 and success drifted (s1 down to 0–57 in its last quarter, s2
oscillating 43–100, s3 recovering to 100 in-run but 0.980 at n=100). The
displacement head's entropy fell from 4.4 to ~3.3 nats in the first 12k
rounds and never sharpened further — 3.0–3.4 of a 4.57 ceiling for the
whole run — while the clip fraction rose from 0.15 to 0.26–0.40. Once
every episode succeeds the advantage signal is small and the entropy
bonus is not; at `ent_coef` 0.03 over a 97-way head the bonus is what
the update is left optimising. The whole-army control, at the same
coefficient, passed by epoch 0–9 and held 100% for 60 epochs; its policy
is one 97-way head per model on a shared trunk, the per-model network's
is a declaration, a displacement and a unit pointer, and the sum of head
entropies is what `ent_coef` scales here.

The record already says 0.003 is the better setting for the whole-army
trainer on formation (`CLAUDE.md` § Coherency: paired +5.9 ± 2.5, and it
"concentrated the policy exactly as predicted"). This arm asks the same
of the per-model trainer on the first rung where its policy drifted.

## The one change

`--ent-coef 0.003` (default 0.03). Same scenario and config
(`configs/experiments/curriculum/a2.yaml`, unchanged), same seeds, same
regime (128 rounds per update), same budget (122,880 rounds), same
in-run and final seed bands. Nothing else varies.

## Comparator

`squad_march_take` on `a2.yaml`, seeds 700000+ at n=100: success 0.990,
turns 4.55, `on_obj` 0.843, coherent 0.889. Beside it, A2's own read at
`ent_coef` 0.03: success 0.370 / 0.880 / 0.980, coherent 0.079 / 0.359 /
0.245; and the whole-army control: 1.000 ×3, turns 5.95–6.01, coherent
0.70–0.78.

## Criteria

Read at 122,880 rounds from `last.pt`, greedy, no decode, n=100 on seeds
700000+ (`just measure-rung`).

- **PASS:** success ≥ 0.95 on **all three** seeds at the end of the
  budget, **and** no seed's in-run success (n=30, 500000+) drops below
  0.80 on any evaluation after its first pass — the drift A2 showed does
  not recur. The entropy setting was the cause; the ladder continues to
  A3 with `--ent-coef 0.003` on every per-model arm from here, recorded
  as an amendment to #340 contract 3.
- **PASS with drift:** ≥ 0.95 on all three at the end but a dip below
  0.80 after the first pass on some seed — the setting helps and does not
  cure; reported as such and the ladder continues with a watch on the
  in-run curve.
- **FAIL:** any seed below 0.95 at the end — the drift is not the entropy
  bonus. The next suspect is the update itself (clip fraction 0.26–0.40
  on A2, the ladder's highest), and the ladder stops at A2 until it is
  found.
- **Readouts:** the in-run curve after the first pass; coherency greedy
  and sampled; the displacement head's entropy over the run; turns paired
  vs the script and the control; the paired-by-seed difference against
  A2's read (same seeds, same init, one scalar).

Power: binomial SE 0.022 at p=0.95, n=100. A2's seeds sit 0.370 / 0.880 /
0.980, so a 3/3 pass is a move on two of them and a hold on the third.

## Budget, regime, seeds

| | per-model arm |
|---|---|
| budget | 122,880 rounds, 128 rounds per update (`--rollout-rounds 32 --num-rollout-envs 4`) |
| seeds | 1, 2, 3; rollout layouts at seed×100+ |
| in-run eval | every 512 rounds, n=30, seeds 500000+, greedy, passive pair and `eval/mean_turns` logged |
| final score | n=100, seeds 700000+ |
| flags | `--ent-coef 0.003`; otherwise defaults (`gamma` 0.9, `lr` 3e-4) |
| logging | Wandb group `curriculum-a2`, tag `a2b` |

Launch: `just train-per-model-arm 122880 3 curriculum-a2 a2b "--num-rollout-envs 4 --rollout-rounds 32 --eval-every-rounds 512 --checkpoint-every-rounds 512 --n-eval-episodes 30 --ent-coef 0.003" configs/experiments/curriculum/a2.yaml`.

## What I expect (a guess, written so it can be wrong)

All three seeds pass by ~10k rounds as before, the displacement entropy
keeps falling past 3 nats instead of plateauing, coherency stays near
1.0, and the final read is ≥ 0.95 on all three. If the drift is the
update rather than the bonus, the curve will look like A2's with a
delay.
