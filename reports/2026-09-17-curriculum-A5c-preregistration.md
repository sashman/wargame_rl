# Pre-registration: curriculum rung A5c — A5b re-run with the actor credit

Written 2026-09-17 23:55, **before any training round under the actor
credit exists**, on branch `feature/per-model-actor-credit` (stacked on
PR #382). Parent question #340, rung **A5**, third per-model arm. No
issue was opened for it, by decision; this file is its record.

## The one change

`--credit actor` on `train_per_model.py`, built today: the retimer pays a
model its action term **undivided** (under the default `mean` it is
divided by the alive count, 1/24 here) and pays each state term's
per-model value at the close as a **credit** landed on that model's own
step of the turn (under `mean` the close pays the army mean, one scalar
to everyone). Globals, delta globals and the terminal bonus are unchanged.
Everything else is A5b: `configs/experiments/curriculum/a5.yaml`
(eight squads of three, six points, radius 4, ten rounds, the `arrive`
reward — travel term at `progress_scale` 6.0 with the fallback, coverage
0.3, terminal bonus 5.0 on `all_objectives_occupied`,
`terminate_on_success` on), from scratch, seeds 1 / 2 / 3, 128 rounds
per update (`--rollout-rounds 32 --num-rollout-envs 4`), `--ent-coef
0.003`, in-run eval every 512 rounds at n=30 on 500000+, Wandb
`curriculum-a5`, tag `a5c`.

Under `actor` A5's two terms become: the travel term to the mover, in
full, on its own step; coverage at the close, global as before. There is
no state term in A5's reward, so on this rung the build's second half is
inert and the arm tests the first half alone — the scale of the actor's
own credit. That is deliberate: it is the smallest change to the arm
that failed, and the rung with the clearest under-arrival to move.

## The comparator

A5b's own runs at the same rounds: **0.260 / 0.180 / 0.160** at
245,760, never a rolling 50% in-run, 8–10 of 24 bodies on points, `held`
4.1–4.6, turns 9.5–9.7 of 10, coherency 0.10. The bar is
`squad_march_take` at 1.000 / 6.77 turns. The whole-army control read
0.800 / 0.940 / 0.980 at 120 epochs (245k rounds).

## Budget and reads

245,760 rounds straight through (A5b's budget, so the rows pair by
rounds), a readout at the 122,880 checkpoint. Read greedy at n=100 on
700000+ with `just measure-rung`, the census beside it (bodies on points,
`held`, max stack, turns, coherency greedy and sampled, the panel).

## Criteria

- **PASS (the rung):** success ≥ 0.95 on all three seeds at 245,760, and
  no in-run dip below 0.80 after the first rolling pass.
- **MOVES (the build):** success at 245,760 ahead of A5b's same seed on
  3/3 AND the mean of the three ≥ 0.50, with bodies on points ≥ 16 of 24
  on every seed. This is the reading that decides whether the credit
  scale is the mechanism; a pass is not required for it.
- **FAIL:** neither; then the second half of the build (the state credit)
  has no test on this rung, and the next arm is A3 speed's hold term
  under `actor`, where it does.
- Power: binomial SE 0.03–0.05 at n=100; A5b's seeds sit at 0.16–0.26,
  so MOVES needs a shift of about 0.3, ten SE.

## What I expect (a guess, written so it can be wrong)

Arrival improves substantially: 16–22 bodies on points and success
0.5–0.8, without a pass — the last empty point is the control's
allocation failure, which a bigger travel credit does not address. If
success does not move at all, the dilution hypothesis is wrong on the
scale half and the state-credit half is the only one left.
