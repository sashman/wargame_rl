# Pre-registration: the backward start curriculum on A5-points — start with the objectives held and walk the start back to deployment

Written 2026-09-19 22:14, **before any training round under the schedule**,
on branch `feature/per-model-backward-start` (built in a worktree off
`c9f23d4` while the control and the A4x warm start train; to be merged into
`feature/per-model-actor-credit`, PR #383, once those exit). Parent question
#340. Goal item 3 of Sash's 2026-09-19 22:00 goal.

## The question

Every read of the half-step A5-points (six squads of three over five
objectives; the bar `squad_march_take` 1.000 in 6.71 turns) says the same
thing: the per-model policy reaches three or four objectives and never
holds them long enough to find the fifth. Four settings of the reward and
optimiser from scratch (the original, the discount, the flat terminal
bonus, the per-objective bonus) read 0.00–0.33 with that census. The
whole-army critic probe of August and the D3 addendum both name the
failure as **search**: the policy does not visit the states where the
last objective pays. **This arm puts those states in the training
distribution directly**: episodes begin with k squads already standing on
k distinct objectives, so what is left to learn is the last steps of the
plan; once those are learned the start is walked back one squad at a
time until episodes begin from deployment. No script supplies actions;
the placement is a teleport at deployment (the same kind of augmentation
`start_on_objective_probability` has been since August) and the policy
learns from reward alone.

## The build (default off, goldens and every evaluation bit-identical)

- `start_groups_on_objectives` (`domain/battlefield/placement.py`): k
  distinct squads onto k distinct objectives, drawn without replacement,
  the same fitting as the one-squad augmentation; draws nothing at 0.
- `PerModelEnv.reset(options={"start_groups": k})` passes it to
  `place_for_episode`; no evaluation asks for it.
- `collect_rollout(start_groups=…)`: each inline reset draws its level;
  `EpisodeOutcome` carries `success` and `start_groups`.
- `train_per_model.py --backward-start k0`: `BackwardStart` — at level k a
  `--backward-start-share` of rollout episodes start at k and the rest from
  deployment; the level steps down by one once the level's OWN episodes
  succeed at `--backward-start-advance` over `--backward-start-window`
  rollouts; rows `curriculum/start_groups` and `curriculum/level_success`;
  provenance `backward_start*`.
- Tests `tests/test_backward_start.py` (placement, the reset option, the
  schedule, a short run); the one-squad augmentation's bit-identity tests
  untouched and passing.

## The arm

| arm | config | flags beyond the original recipe | budget | tag |
|---|---|---|---|---|
| **BS** | `a5_points.yaml` (unchanged) | `--backward-start 4 --backward-start-share 0.75 --backward-start-advance 0.8 --backward-start-window 8` | 122,880 rounds, the original recipe (128 per update, `--ent-coef 0.003`, `gamma` 0.9, `mean` credit, eval and checkpoint every 512, recording and video on) | `a5bs` |

Three seeds from scratch, on Wandb under `curriculum-a5`, launched from the
worktree so the live control and warm-start runs on the main tree are not
touched. Level 4 is the highest that leaves a squad to walk: four of six
squads on four of five objectives, the fifth objective and two squads free.
The comparator is the original A5-points from scratch (`uagmul66` /
`wedjjozy` / `ici5py46`), 0.030 / 0.060 / 0.330 at 122,880, 0.040 / 0.140 /
0.090 at 40,960, 0.220 / 0.220 / 0.110 at 81,920. Read greedy at n=100 on
700000+ **from deployment** (the rung's criterion; the schedule never
touches an evaluation) at 40,960 / 81,920 / 122,880 with the by-turn census,
sampled beside greedy at the end.

## Criteria

- **PASS (the half-step):** success ≥ 0.95 on 3/3 at 122,880 from
  deployment, no in-run dip below 0.80 after the first rolling pass.
- **MOVES:** ahead of the original 0.030 / 0.060 / 0.330 seed for seed by
  more than two binomial SE on 3/3 at 122,880.
- **FAIL:** neither.
- **The schedule's own readouts, reported whatever the verdict:** the
  level reached on each seed and the rounds at each step-down
  (`curriculum/start_groups`); the level's rolling success
  (`curriculum/level_success`); success from deployment at each read;
  held, the census, the panel (explained variance — the starts change the
  return distribution), sampled beside greedy.

**What each answer says.** PASS: the wall was search, and the backward
start is the lever for every conjunction rung (A5 gets the same arm).
MOVES with the level at 0: the walk back completed and the deployment
start is learnable but slower than the budget — extend once to 2× the cap
as A4x was. A seed whose level never leaves 4: the policy cannot learn the
LAST step even with four objectives given — the failure is not search but
something in what the policy can represent about the empty objective, and
the next lever is an observation that names it. A seed that walks the
level down to 0 and still fails from deployment: the levels were learned
and forgotten, and the share or the window is the defect (report it as
such; the schedule is a first cut).

## What I expect (a guess, written so it can be wrong)

Level 4 is learned within 10,240 rounds on every seed (one squad walks to
one empty objective, which A2 learned in two thousand rounds); level 3
within 30k; level 2 and 1 slower, the walk-off returning as the number of
free squads grows; level 0 reached on one or two seeds by 100k. Success
from deployment at 122,880 **0.5–0.8**, held 4.2–4.6, MOVES on 3/3, PASS on
one seed at most. If the level never leaves 4 on any seed I was wrong
about the mechanism in the useful direction.
