# Pre-registration: curriculum rung D2b — PPO from the escort clone with its critic fitted first

Written 2026-09-17 16:55, **before any PPO round from the critic-fitted
clone exists**, on branch `feature/curriculum-d2` (PR #377). Issue #379,
parent question #340, rung **D2** — second arm, one change from #376.

## The one change

The starting checkpoint: `checkpoints/per_model/clones/escort-c3b-1200-s0-critic.pt`
— D1b's clone (fit seed 0) with **only its value head fitted** to the
escort's discounted returns (`just fit-per-model-critic`, 300 games at
800000+ × 20 epochs, gamma 0.9 per round, held-out explained variance
**−0.17 → 0.74**; the trunk and the policy heads bit-identical, checked
at save). Everything else is D2's: `--rounds 122880 --seed {1,2,3}
--num-rollout-envs 4 --rollout-rounds 32 --eval-every-rounds 512
--checkpoint-every-rounds 512 --n-eval-episodes 30 --ent-coef 0.003`,
no anchor, Wandb `curriculum-d2` tag `d2b`.

## Why

D2 (#376) reads 90 / 80 / 73% at its first evaluation and 6 / 0 / 46%
by 2,560 rounds: the whole-army record's outcome — PPO from a clone with
a cold critic destroys it — reproduced. The record's own diagnosis was
the critic: a value head at initialisation makes the first advantages
noise, and the whole-army clone that carried its teacher was fitted with
one. This arm changes that alone.

## Comparator

The clone, paired per episode: success 0.960, held 3.93, alive 0.953,
vp +59.6, kill before arrival 1.00 (n=100; the n=180 read is taken
beside the arm's). `scripted_escort` 0.990 / +63.5. D2's arm beside
every row.

## Criteria

D2's, unchanged, read greedy at n=180 on 700000+ from `last.pt` at
122,880 and in-run at n=30 on 500000+ every 512 rounds:

- **IMPROVES:** vp paired against the clone > 0 with t ≥ 2 on every
  seed, success ≥ 0.96 and kill before arrival ≥ 0.95 on every seed.
- **HOLDS:** success ≥ 0.94 with ordering ≥ 0.95 on every seed and vp
  paired not significant on any.
- **DESTROYS:** success < 0.90 at the end on any seed.
- **NULL (scenario):** only if the escort fails its own criterion.
- **Readouts:** the in-run curve; `train/explained_variance` from the
  first update; `alive`, `held`, coherency greedy and sampled; the
  panel; the census by phase; the final policy's held-out match against
  the escort's demonstrations beside the clone's.

Power: paired vp SE about 2 at n=180; binomial SE 0.015 at p=0.96.

## What I expect (a guess, written so it can be wrong)

HOLDS on two seeds and DESTROYS on one: a fitted critic removes the
first updates' noise, and what is left is PPO's own drift at 128 rounds
per update, slower but in the same direction. If it HOLDS 3/3 the cold
critic was the whole of D2's failure and the anchor is a refinement; if
it DESTROYS 3/3 the critic was not it and D2c is the arm that matters.
