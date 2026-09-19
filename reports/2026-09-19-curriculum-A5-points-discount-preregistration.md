# Pre-registration: A5-points at gamma 0.99 — is the conjunction wall the horizon?

Written 2026-09-19 12:05, **before any training round at this discount**,
on branch `feature/per-model-actor-credit` (PR #383). Parent question
#340; the first optimisation-side arm of the ladder's new rule (a rung
that fails at the cap gets one pre-registered optimiser arm before the
ladder moves). Set as a goal by Sash 2026-09-19 12:00: "run the discount
arm on A5-points from scratch".

## The question

A5-points (six squads of three over five points; the bar
`squad_march_take` 1.000 in 6.71 turns) read **0.030 / 0.060 / 0.330** at
122,880 rounds from scratch under the ladder's recipe, with A5's census:
a third of the bodies on points, whole points abandoned by seed, squads
walking off points they had reached. The D3 addendum measured why late
payoffs are invisible to this trainer: PPO discounts at `gamma` 0.9 per
round, so a bonus at the end of a game reaches the first decisions at a
quarter to a third of its value while the travel term pays in full now.
On A5-points the bar's reward stream says the same: the same 4.38 of
episode reward is worth **2.79** at the first decision under 0.9 and
**4.19** under 0.99 (n=40 on 700000+, the terminal bonus 2.16 of it, the
coverage term 0.91, the travel term 1.31). **The arm asks whether the
conjunction over five points becomes learnable when its payoff reaches
the decisions that produce it.**

## The one change

`--gamma 0.99` (the discount per close; `gae_lambda` stays 0.95).
Everything else is A5-points as pre-registered on 2026-09-18: from
scratch, `configs/experiments/curriculum/a5_points.yaml`, seeds 1 / 2 /
3, 122,880 rounds at 128 per update (`--rollout-rounds 32
--num-rollout-envs 4`), `--ent-coef 0.003`, the `mean` credit, in-run
eval every 512 rounds at n=30 on 500000+, Wandb `curriculum-a5`, tag
`a5pg`. No whole-army control (a diagnostic arm on a half-step, as
A5-points was). Read greedy at n=100 on 700000+ at 40,960 / 81,920 /
122,880 with the by-turn census, sampled beside greedy at the end.

## Criteria

- **PASS (the half-step):** success ≥ 0.95 on all three seeds at
  122,880, no in-run dip below 0.80 after the first rolling pass.
- **MOVES:** ahead of A5-points' 0.030 / 0.060 / 0.330 seed for seed at
  122,880 by more than two binomial SE (0.05 at 0.1; 0.09 at 0.33), 3/3.
- **FAIL:** neither.
- **Readouts:** held of 5, bodies on points by turn, points abandoned by
  seed, turns, coherency, explained variance and return std on the
  panel (a higher discount raises the return scale and the value
  target's variance — read the critic before reading the arm).

**What each answer says.** PASS or MOVES: the wall on the conjunction
rungs is the horizon, and the A5 rung gets the same arm before any
reward redesign. FAIL: the conjunction is hard for per-step credit at
any horizon this trainer can value, and the reward-shape arm (per-point
payment) is next. A red critic with a moved policy is reported as a
move with a defect.

## What I expect (a guess, written so it can be wrong)

MOVES on 3/3 and PASS on none: held rises from 2.5–3.8 toward 4.5 and
the walk-offs stop (the point once reached stays worth holding to the
end), success 0.4–0.8 with the fifth point still the residual; the
critic's explained variance falls from A5-points' 0.63–0.65 under the
longer horizon. If it passes I am wrong in the useful direction; if it
does not move, the horizon was not the wall.
