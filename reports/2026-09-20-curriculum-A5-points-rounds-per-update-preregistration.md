# Pre-registration: A5-points at the whole-army trainer's rounds per update — is the per-model trainer's wall the number of games each update sees?

Written 2026-09-20 05:20, **before any training round**, on branch
`feature/per-model-actor-credit` (PR #383). Parent question #340; asked by
Sash 2026-09-20 05:00: "can you investigate what exactly makes it hard to
learn?" after the control on this half-step read 0.91 / 0.81 / 0.95 where
the per-model trainer from scratch reads 0.00–0.33 under five settings.

## The question

The two trainers on the five-objective half-step differ in what each
soldier sees (checked: the objective token carries our count on it and
every soldier reads its offset to every objective — an empty objective is
observable), in how the reward reaches a soldier standing on an objective
(measured: a trained per-model policy stands still on 0% of its decisions
on an objective and walks out on 60–62%; the step reward for leaving is
−0.002 to −0.004 and for staying 0, while a step toward an objective from
outside pays +0.008 to +0.014), and in **how many finished games each
update sees**: `train.py` updates every `n_steps` = 2,048 env steps, one
per round on this one-phase config, so about **200 episodes per update**;
`train_per_model.py` updates every `rollout_rounds × num_rollout_envs` =
128 rounds, about **13 episodes per update** — sixteen times fewer
outcomes for a success that happens in a few percent of games, and the
closest the ladder has come to the one-episode-per-update regime it voided
in September. This arm changes that number and nothing else.

## The arm

| arm | config | the one change | budget | tag |
|---|---|---|---|---|
| **RR** | `a5_points.yaml` (unchanged) | `--rollout-rounds 512` with `--num-rollout-envs 4`: **2,048 rounds per update**, the whole-army trainer's; eval and checkpoint every 2,048 (the cadence must be a multiple of the rollout) | 122,880 rounds (60 updates instead of 960; the same transitions, `n_epochs` 5 and `batch_size` 128 so the same order of gradient steps), `--ent-coef 0.003`, `gamma` 0.9, `mean` credit, from scratch, recording and video on | `a5rr` |

Three seeds. Comparator: the original A5-points from scratch at 128 rounds
per update (`uagmul66` / `wedjjozy` / `ici5py46`), 0.030 / 0.060 / 0.330 at
122,880, 0.040 / 0.140 / 0.090 at 40,960, 0.220 / 0.220 / 0.110 at 81,920.
Read greedy at n=100 on 700000+ at 40,960 / 81,920 / 122,880 with the
by-turn census and the walk-off probe (stay share on an objective), sampled
beside greedy at the end.

## Criteria

- **PASS:** success ≥ 0.95 on 3/3 at 122,880, no in-run dip below 0.80
  after the first rolling pass.
- **MOVES:** ahead of the original seed for seed by more than two binomial
  SE on 3/3 at 122,880.
- **FAIL:** neither.
- **Readouts:** held, the census, the walk-off probe's stay share and step
  pay on an objective, the panel (explained variance, clip fraction — at
  2,048 rounds per update the ratio tail and clip fraction are a different
  quantity from the 128-round rows and are quoted as such), sampled beside
  greedy, the in-run curve at 60 evaluations.

**What each answer says.** PASS or MOVES: the per-model trainer's wall on
this half-step was its update regime — 128 rounds per update is too few
outcomes for a conjunction — and every from-scratch per-model row on the
A5 rungs is re-read at 2,048. FAIL with the same census: the regime is not
it, and what is left is the reward's silence on staying (the walk-off
probe's zero) — the next arm pays a body for standing on an objective on
its own step, which needs the actor credit path and a term that is not
the mean-credited hold term the A3 screen and A5e read as null.

## What I expect (a guess, written so it can be wrong)

MOVES on 2/3, PASS on none: success 0.2–0.5, held 3.5–4.2, the walk-off
reduced but not gone (the reward still pays nothing for staying). If it
passes, the September regime finding was under-scoped — 128 rounds per
update was enough for one, three and four objectives and not for five.
