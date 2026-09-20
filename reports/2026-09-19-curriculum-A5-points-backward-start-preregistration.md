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

## Amendment 1 — written 2026-09-20 02:15, the arm read at 40,960 / 81,920 / 122,880

**FAIL as pre-registered, and the level never left four on any seed.**
Runs `inhpvik7` / `fhbs37ln` / `feg4n4uf`, launched 22:17 from the
worktree, exited 01:55–02:05. Greedy at n=100 on 700000+ from deployment:

| rounds | success | held of 5 | the original A5-points at the same rounds |
|---|---|---|---|
| 40,960 | 0.000 / 0.000 / 0.010 | 2.19 / 1.55 / 2.19 | 0.040 / 0.140 / 0.090 |
| 81,920 | 0.000 / 0.070 / 0.000 | 2.77 / 3.14 / 1.29 | 0.220 / 0.220 / 0.110 |
| **122,880** | **0.000 / 0.030 / 0.030** | 2.08 / 3.02 / 2.07 | 0.030 / 0.060 / 0.330 |

- **PASS:** no. **MOVES:** no seed ahead of the original by two SE at the
  end; behind it at 40,960 and 81,920 on every seed. **FAIL.**
- **The schedule's readouts.** `curriculum/start_groups` stayed at **4**
  for the whole run on all three seeds. The level's own success
  (`curriculum/level_success`, rolling over eight rollouts) by quarter:
  0.12 / 0.32 / 0.30 / 0.35 (s1), 0.18 / 0.30 / 0.34 / 0.25 (s2), 0.25 /
  0.26 / 0.22 / 0.27 (s3) — a plateau at a third, never near the 0.8 that
  steps the level down. The pre-registration's own reading for this
  outcome: **the policy cannot learn the LAST step even with four
  objectives given.**
- **What the last step fails on, measured** (a census of level-4 episodes
  under the final checkpoints, n=30 on 700000+, greedy; the bar on the
  same starts succeeds 1.00 in 6.1 turns with 3.9 of the 4 placed squads
  still on their objectives): success **0.53 / 0.33 / 0.50**; the placed
  squads still on objectives at turn 3 **2.93 / 3.38 / 3.93** and at the
  end **2.70 / 1.97 / 3.13** of 4; the free squads reaching the empty
  objective **0.83 / 1.00 / 0.63**; held at the end 3.53 / 2.97 / 3.77;
  7.6–9.2 turns. Both halves of the step are half-learned, and the
  larger loss is the first: **given an objective, the policy walks off
  it** — 1–2 of the 4 placed squads have left by the end. Nothing on this
  config pays a body per model for standing on an objective (the travel
  term is zero inside, coverage is a broadcast mean at the close, the
  bonus terminal), and a target switch is unpaid, so leaving is free.
- **From deployment** the census is the half-step's: s1 puts 9.6 of 18
  bodies on objectives by turn 5 (the highest arrival on this half-step)
  and then walks off to 4.0; s2 climbs to 6.8 and holds 3.0; s3 walks off
  from 6.4 to 2.7. Sampled play holds 3.1–3.2 where greedy holds 2.1–3.0
  (−2.5 / −3.2 / +0.6 vp greedy − sampled): diffuse.
- **The panel is healthy** — explained variance 0.55–0.68 by quarter,
  displacement entropy 3.1 → 2.2–2.4 nats, clip fraction 0.25–0.35 — so
  this is not the reward-shape arms' broken critic; the value function
  fits and the policy still does not keep what it is given.

**What the answer says, as written.** The wall is not only search: put
the policy in the state where the last objective pays and it does not
reliably take the step, and half of what it loses is the four it was
handed. The lever the pre-registration named for this outcome — an
observation that names the empty objective — addresses the arriving half;
the walk-off half wants a per-model term that pays a body for standing on
an objective, which the A3 speed screen measured as null under the mean
credit and A5e as null under actor credit on the six-objective rung. The
schedule itself is a first cut (share 0.75, advance 0.8 over eight
rollouts) and never got to be exercised; a lower advance bar would have
walked the level down onto a policy that holds a third of its starts.
