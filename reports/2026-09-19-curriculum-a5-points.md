# Curriculum — the A5-points half-step (six squads, five objectives)

Seven per-model arms and the whole-army control on the half-step between A4 and A5. Each pre-registration is followed by its result, word for word as first written.

## Contents

- [Pre-registration: A5-points at gamma 0.99 — is the conjunction wall the horizon?](#2026-09-19-curriculum-A5-points-discount-preregistration)
- [Curriculum A5-points at gamma 0.99: the horizon is not the wall — the critic sees the longer game and the policy does not move](#2026-09-19-curriculum-a5-points-discount-the-horizon-is-not-the-wall)
- [Pre-registration: the reward-shape arms on A5-points — is the conjunction unpaid, or unfindable?](#2026-09-19-curriculum-A5-points-reward-shape-preregistration)
- [Curriculum A5-points, the reward-shape arms: paying the criterion as a lump breaks the critic, and paying per point buys the near column](#2026-09-19-curriculum-a5-points-reward-shape-the-lump-breaks-the-critic)
- [Pre-registration: the whole-army control and the A4x warm start on A5-points — whose wall is it, and does a start help?](#2026-09-19-curriculum-A5-points-control-and-warm-start-preregistration)
- [Pre-registration: the backward start curriculum on A5-points — start with the objectives held and walk the start back to deployment](#2026-09-19-curriculum-A5-points-backward-start-preregistration)
- [Pre-registration: A5-points at the whole-army trainer's rounds per update — is the per-model trainer's wall the number of games each update sees?](#2026-09-20-curriculum-A5-points-rounds-per-update-preregistration)
- [Pre-registration: the staying arm on A5-points — pay a body for keeping an objective on its own step](#2026-09-20-curriculum-A5-points-staying-preregistration)
- [Curriculum A5-points, three arms in one night: the wall is the per-model trainer's, a warm start carries four objectives of five, and a start with four objectives given is walked off](#2026-09-20-curriculum-a5-points-whose-wall-it-is)

---

<a id="2026-09-19-curriculum-A5-points-discount-preregistration"></a>

## Pre-registration: A5-points at gamma 0.99 — is the conjunction wall the horizon?

Written 2026-09-19 12:05, **before any training round at this discount**,
on branch `feature/per-model-actor-credit` (PR #383). Parent question
#340; the first optimisation-side arm of the ladder's new rule (a rung
that fails at the cap gets one pre-registered optimiser arm before the
ladder moves). Set as a goal by Sash 2026-09-19 12:00: "run the discount
arm on A5-points from scratch".

### The question

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

### The one change

`--gamma 0.99` (the discount per close; `gae_lambda` stays 0.95).
Everything else is A5-points as pre-registered on 2026-09-18: from
scratch, `configs/experiments/curriculum/a5_points.yaml`, seeds 1 / 2 /
3, 122,880 rounds at 128 per update (`--rollout-rounds 32
--num-rollout-envs 4`), `--ent-coef 0.003`, the `mean` credit, in-run
eval every 512 rounds at n=30 on 500000+, Wandb `curriculum-a5`, tag
`a5pg`. No whole-army control (a diagnostic arm on a half-step, as
A5-points was). Read greedy at n=100 on 700000+ at 40,960 / 81,920 /
122,880 with the by-turn census, sampled beside greedy at the end.

### Criteria

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

### What I expect (a guess, written so it can be wrong)

MOVES on 3/3 and PASS on none: held rises from 2.5–3.8 toward 4.5 and
the walk-offs stop (the point once reached stays worth holding to the
end), success 0.4–0.8 with the fifth point still the residual; the
critic's explained variance falls from A5-points' 0.63–0.65 under the
longer horizon. If it passes I am wrong in the useful direction; if it
does not move, the horizon was not the wall.

### Amendment 1 — written 2026-09-19 14:10, the arm read at 40,960 / 81,920 / 122,880

**FAIL as pre-registered. The horizon is not the wall.** Runs
`meawm0gk` / `6lr588y7` / `0sjtnn0n`, launched 12:08, exited ~13:50
(about a thousand rounds a minute at 128 per update). Greedy at n=100 on
700000+ at 122,880 rounds (`last.pt`):

| row | success | turns | vs bar, paired | held of 5 | on points | coherent | stat |
|---|---|---|---|---|---|---|---|
| `squad_march_take` | 1.000 | 6.71 | — | 5.00 | 0.889 | 0.832 | — |
| s1 (γ 0.99) | **0.070** | 9.87 | +3.16 ± 0.08 | 3.04 | 0.264 | 0.183 | 0.07 |
| s2 (γ 0.99) | **0.180** | 9.65 | +2.94 ± 0.10 | 3.42 | 0.321 | 0.076 | 0.10 |
| s3 (γ 0.99) | **0.100** | 9.79 | +3.08 ± 0.09 | 3.11 | 0.327 | 0.236 | 0.06 |
| A5-points s1 (γ 0.9, 2026-09-18) | 0.030 | 9.94 | +3.23 ± 0.07 | 2.47 | 0.263 | 0.194 | 0.04 |
| A5-points s2 | 0.060 | 9.96 | +3.25 ± 0.06 | 2.92 | 0.321 | 0.195 | 0.04 |
| A5-points s3 | 0.330 | 9.56 | +2.85 ± 0.09 | 3.83 | 0.377 | 0.211 | 0.06 |

- **PASS:** no seed is near 0.95.
- **MOVES:** ahead of 0.030 / 0.060 / 0.330 seed for seed by more than
  two binomial SE on 3/3. s1 +0.04 (two SE ≈ 0.06): not ahead. s2 +0.12
  (two SE ≈ 0.07): ahead. s3 **−0.23**: behind. **One of three.**
- **FAIL:** both clauses missed. The written expectation (MOVES 3/3,
  held toward 4.5, the walk-offs stopping) was wrong in the direction
  the pre-registration named: it did not move.

**At matched rounds the arm was behind the original on every seed at
both earlier reads.** 40,960: 0.000 / 0.030 / 0.060 against the
original's 0.040 / 0.140 / 0.090 (held 1.73 / 1.15 / 2.48 against 1.90 /
2.87 / 2.76). 81,920: 0.100 / 0.080 / 0.040 against **0.220 / 0.220 /
0.110** (held 2.73 / 3.06 / 2.85 against 3.39 / 3.41 / 2.81). The
original's own 80k read is new: its s1 and s2 peaked near 0.22 at 80k
and fell to 0.03 / 0.06 by the end, the drift the census has shown on
this rung since A5b, while s3 climbed to 0.33. So the comparator's
final row is one seed that kept climbing and two that collapsed, and the
arm at 0.07 / 0.18 / 0.10 is inside that band, not above it.

**The census is the original's on every column** (n=100 on 700000+;
bodies on points of 18 and points held of 5 after turns 3 → 5 → 7 →
end; share of episodes with each point empty at the end; max stack):

| policy | success | turns | on points | held | empty by point | max stack |
|---|---|---|---|---|---|---|
| `squad_march_take` | 1.00 | 6.70 | 11.7 → 12.8 → 15.8 → 15.8 | 3.0 → 4.2 → 5.0 → 5.0 | 0 / 0 / 0 / 0 / 0 | 5.0 |
| s1 | 0.07 | 9.87 | 3.5 → 5.8 → 5.0 → 4.8 | 1.7 → 2.8 → 2.9 → 3.0 | 0.34 / **0.83** / 0.26 / 0.39 / 0.14 | 2.3 |
| s2 | 0.17 | 9.68 | 1.6 → 4.9 → 5.8 → 5.8 | 1.2 → 2.5 → 3.3 → 3.4 | 0.21 / 0.46 / 0.42 / 0.17 / 0.33 | 2.6 |
| s3 | 0.10 | 9.79 | 2.9 → 4.3 → 5.4 → 5.9 | 1.3 → 2.1 → 3.0 → 3.1 | 0.54 / **0.73** / 0.21 / 0.27 / 0.14 | 2.9 |

Five or six of eighteen bodies on points from turn 5 to the end against
the script's sixteen; one point empty in 46–83% of episodes on every
seed (s1 and s3 abandon point 1, s2 spreads its misses); s1's bodies on
points fall from 5.8 after turn 5 to 4.8 at the end, the walk-off. The
clock runs out on every seed (turns 9.65–9.87 of ten). Sampled play is
**6.0–7.4 vp worse than greedy** (paired, n=100, `just
measure-per-model-eval-mode`), where A5i's agreed within 1.2: the policy
is diffuse, as A5-points' was.

**The readouts the pre-registration asked for.** The critic sees the
longer horizon and the policy does not act on it: explained variance by
quarter **0.78 / 0.78 / 0.80 / 0.82** (s1), 0.79 / 0.80 / 0.77 / 0.83
(s2), 0.80 / 0.80 / 0.84 / 0.85 (s3) against the original's 0.63–0.73,
return std 0.61–0.70 (the original's 0.6–0.7 — the longer horizon did
not raise the value target's variance at this reward scale, because
`gamma` applies only across closes and an episode is ten of them).
In-run success by quarter (n=30 on 500000+, every 512 rounds): 0.7 /
2.5 / 6.9 / **15.7** (s1), 1.2 / 3.2 / 9.8 / 7.3 (s2), 0.9 / 8.6 /
13.5 / 11.5 (s3) against the original's 0.2 / 7.8 / 13.3 / 15.0, 0 /
7.0 / 13.7 / 19.6, 0.4 / 5.4 / 12.3 / 8.6 — the same curve. Clip
fraction 0.33 / 0.40 / 0.40 in the last quarter (the original 0.30 /
0.38 / 0.31), displacement entropy 1.85–2.07 nats (the original
1.60–1.88), declaration entropy 0.09–0.12, gradient clipped on 99.98%
of updates on both arms. The panel is not red; nothing on it separates
the arms.

**What the answer says, as written.** FAIL: the conjunction over five
points is hard for per-step credit at any horizon this trainer can
value. The bar's reward stream reaching the first decision at 4.19
instead of 2.79 changed the critic's fit and nothing the policy does.
The reward-shape arm the pre-registration named as next (a terminal
bonus not scaled by the rounds remaining, and coverage paid per flag to
its unique holder) is the next arm on this half-step; a second
discount, an entropy coefficient or a longer budget is not.

Every checkpoint of the arm has a recorded greedy episode beside it
(`<run_dir>/recordings/pm-<rounds>-seed700000.json` at 40,960 and
81,920, `last-seed700000.json` at the end, decision cadence), made post
hoc; from the next launch the trainer records in-run
(`--record-every-rounds`, merged at `f7539d2`).

---

<a id="2026-09-19-curriculum-a5-points-discount-the-horizon-is-not-the-wall"></a>

## Curriculum A5-points at gamma 0.99: the horizon is not the wall — the critic sees the longer game and the policy does not move

**Verdict first.** A5-points (six squads of three over five points, the
half-step between A4 and A5; the bar `squad_march_take` 1.000 in 6.71
turns) trained from scratch with one change to the optimiser — the
discount per turn close raised from 0.9 to **0.99**, so the terminal
bonus and the coverage stream reach the first decisions at 4.19 of the
bar's 4.38 instead of 2.79 — reads **0.070 / 0.180 / 0.100** at
122,880 rounds, greedy at n=100 on 700000+, against the original
A5-points' **0.030 / 0.060 / 0.330** on the same seeds:

| clause | needs | reads | verdict |
|---|---|---|---|
| PASS | ≥ 0.95 on 3/3, no in-run dip after the first pass | 0.07 / 0.18 / 0.10 | no |
| MOVES | ahead of 0.030 / 0.060 / 0.330 by > 2 binomial SE, 3/3 | +0.04 (two SE 0.06) / **+0.12** (two SE 0.07) / **−0.23** | 1 of 3, no |
| **FAIL** | neither | | **FAIL as pre-registered** |

At matched rounds the arm was **behind** the original on every seed at
40,960 (0.00 / 0.03 / 0.06 against 0.04 / 0.14 / 0.09) and at 81,920
(0.10 / 0.08 / 0.04 against 0.22 / 0.22 / 0.11). Its by-turn census is
the original's on every column — five or six of eighteen bodies on
points from turn 5 to the end against the script's sixteen, one point
empty in 46–83% of episodes on every seed, the walk-off on s1, the
clock run out on all three — and its sampled play is 6–7 vp worse than
its greedy play, the diffuse signature A5-points had. The one thing the
discount changed is the critic: explained variance **0.78–0.85** against
the original's 0.63–0.73, on the same in-run curve. **The conjunction
over five points is hard for per-step credit at any horizon this
trainer can value.** The pre-registration's own next step stands: the
reward-shape arm on this half-step (a terminal bonus not scaled by the
rounds remaining; coverage paid per flag to its unique holder), not a
second discount, an entropy coefficient or a longer budget.

### Provenance

| field | value |
|---|---|
| date | 2026-09-19: pre-registered 12:05, launched 12:08, trainers exited ~13:50 (about a thousand rounds a minute), final reads 13:54–14:00 |
| GPU / no-GPU | GPU (RTX 4090), three trainers, nothing else on the box |
| seeds | 1 / 2 / 3 from scratch — the same three seeds as the original A5-points of 2026-09-18, so the per-seed comparison is paired at initialisation |
| n | 100 at seed base 700000 (every read, the census, the greedy-against-sampled table); 30 at 500000 (in-run, every 512 rounds) |
| config | `configs/experiments/curriculum/a5_points.yaml`, unchanged: six squads of three, five points, no enemy, `terminate_on_success: true`, ten rounds; unrefereed by design |
| decode | none on the per-model facade |
| paired | per episode against the bar on identical seeds (turns); per seed against the original A5-points on the same seeds and layouts (success, held) |
| comparator | the original A5-points runs `uagmul66` / `wedjjozy` / `ici5py46` (gamma 0.9): 0.030 / 0.060 / 0.330 at 122,880, held 2.47 / 2.92 / 3.83; at 81,920 (read today) 0.220 / 0.220 / 0.110, held 3.39 / 3.41 / 2.81; at 40,960 0.040 / 0.140 / 0.090. The bar `squad_march_take` 1.000, 6.71 turns, held 5.00, coherent 0.832 |
| opponent | none |
| budget | 122,880 rounds at 128 per update (`--rollout-rounds 32 --num-rollout-envs 4`), `--ent-coef 0.003`, the default `mean` credit, **`--gamma 0.99`** (`gae_lambda` 0.95 unchanged); no extension |
| code revision | pre-registered at `194c95a`; runs on `194c95a`; amendment `8d0ff8d`; the branch `feature/per-model-actor-credit`, PR #383 |
| checkpoint | `last.pt` at 122,880 and every 512 rounds under `checkpoints/per_model/per-model-a5_points-2026-09-19-11-38-29-s{1,2,3}a5pg`; a recorded greedy episode (seed 700000, decision cadence) beside the 40,960, 81,920 and final checkpoints in `recordings/` |
| coherency | greedy at 700000+: 0.183 / 0.076 / 0.236 (the original 0.194 / 0.195 / 0.211; the bar 0.832); sampled 0.03–0.09 |
| Wandb | `curriculum-a5`: `meawm0gk` / `6lr588y7` / `0sjtnn0n` |
| pre-registration | `reports/2026-09-19-curriculum-a5-points.md#2026-09-19-curriculum-A5-points-discount-preregistration` (+1 amendment) |

### The read

#### The final table

| row | success | turns | vs bar, paired | held of 5 | on points | vp ± SE | coherent | stat |
|---|---|---|---|---|---|---|---|---|
| `squad_march_take` | 1.000 | 6.71 | — | 5.00 | 0.889 | 63.2 ± 0.9 | 0.832 | — |
| s1, γ 0.99 | **0.070** | 9.87 | +3.16 ± 0.08 | 3.04 | 0.264 | 72.7 ± 1.4 | 0.183 | 0.07 |
| s2, γ 0.99 | **0.180** | 9.65 | +2.94 ± 0.10 | 3.42 | 0.321 | 67.7 ± 1.1 | 0.076 | 0.10 |
| s3, γ 0.99 | **0.100** | 9.79 | +3.08 ± 0.09 | 3.11 | 0.327 | 63.8 ± 1.7 | 0.236 | 0.06 |
| s1, γ 0.9 (2026-09-18) | 0.030 | 9.94 | +3.23 ± 0.07 | 2.47 | 0.263 | | 0.194 | 0.04 |
| s2, γ 0.9 | 0.060 | 9.96 | +3.25 ± 0.06 | 2.92 | 0.321 | | 0.195 | 0.04 |
| s3, γ 0.9 | 0.330 | 9.56 | +2.85 ± 0.09 | 3.83 | 0.377 | | 0.211 | 0.06 |

The vp column reads above the bar on every seed because a policy that
never finishes plays all ten rounds and scores them, while the bar ends
its episodes at turn 6.71 — on a `terminate_on_success` rung vp is the
clock, not the play.

#### Three reads, and the original's own curve

| rounds | γ 0.99 success | γ 0.9 success | γ 0.99 held | γ 0.9 held |
|---|---|---|---|---|
| 40,960 | 0.000 / 0.030 / 0.060 | 0.040 / 0.140 / 0.090 | 1.73 / 1.15 / 2.48 | 1.90 / 2.87 / 2.76 |
| 81,920 | 0.100 / 0.080 / 0.040 | **0.220 / 0.220 / 0.110** | 2.73 / 3.06 / 2.85 | 3.39 / 3.41 / 2.81 |
| 122,880 | 0.070 / 0.180 / 0.100 | 0.030 / 0.060 / 0.330 | 3.04 / 3.42 / 3.11 | 2.47 / 2.92 / 3.83 |

The original's 80k row is new (read today to match the arm's). Its s1
and s2 peaked near 0.22 at 80k and collapsed to 0.03 / 0.06 by the end
— the walk-off drift the census has recorded on this rung since A5b —
while s3 climbed on to 0.33. The comparator's final row is therefore
one seed still rising and two that fell back, and the arm's 0.07 /
0.18 / 0.10 sits inside that band. Read at every checkpoint, the two
arms are one curve: in-run success by quarter (n=30 on 500000+) is 0.7
/ 2.5 / 6.9 / 15.7, 1.2 / 3.2 / 9.8 / 7.3, 0.9 / 8.6 / 13.5 / 11.5 for
the arm against 0.2 / 7.8 / 13.3 / 15.0, 0 / 7.0 / 13.7 / 19.6, 0.4 /
5.4 / 12.3 / 8.6 for the original.

**Paired per episode against the original on the same 100 seeds**
(the MOVES clause as written uses the binomial SE; this is the sharper
estimator, run after the verdict as a readout): s1 **+0.040 ± 0.032**
(t +1.27, 10 episodes differ), s2 **+0.100 ± 0.048** (t +2.07, 24
differ), s3 **−0.230 ± 0.058** (t −3.94, 39 differ). The same one-of-
three, and s3 is behind at four SE. (This script's own episode loop
reads the s2 pair at 0.070 / 0.170 where the batched read says 0.060 /
0.180 — one episode each way.)

#### The census is A5-points'

n=100 on 700000+; bodies on points of 18 and points held of 5 after
turns 3 → 5 → 7 → end; the share of episodes with each point empty at
the end by index; the largest stack on one point:

| policy | success | turns | on points | held | empty by point | max stack |
|---|---|---|---|---|---|---|
| `squad_march_take` | 1.00 | 6.70 | 11.7 → 12.8 → 15.8 → 15.8 | 3.0 → 4.2 → 5.0 → 5.0 | 0 / 0 / 0 / 0 / 0 | 5.0 |
| s1 | 0.07 | 9.87 | 3.5 → 5.8 → 5.0 → 4.8 | 1.7 → 2.8 → 2.9 → 3.0 | 0.34 / **0.83** / 0.26 / 0.39 / 0.14 | 2.3 |
| s2 | 0.17 | 9.68 | 1.6 → 4.9 → 5.8 → 5.8 | 1.2 → 2.5 → 3.3 → 3.4 | 0.21 / 0.46 / 0.42 / 0.17 / 0.33 | 2.6 |
| s3 | 0.10 | 9.79 | 2.9 → 4.3 → 5.4 → 5.9 | 1.3 → 2.1 → 3.0 → 3.1 | 0.54 / **0.73** / 0.21 / 0.27 / 0.14 | 2.9 |

A third of the bodies on points against the script's seven eighths; s1
and s3 abandon point 1 in 73–83% of episodes, s2 spreads its misses
over four points; s1's bodies on points fall from 5.8 after turn 5 to
4.8 at the end; max stack under three where the script stacks a squad
of three plus two; coherency 0.08–0.24. The original's row (amendment
3 of the half-steps pre-registration) reads the same on every column.
The horizon did not make the point once reached worth holding, and it
did not make the fifth point worth walking to.

#### Greedy against sampled

`just measure-per-model-eval-mode` at n=100 on 700000+, paired:

| seed | greedy vp | sampled vp | greedy − sampled | greedy held | sampled held |
|---|---|---|---|---|---|
| s1 | 71.5 ± 1.5 | 65.1 ± 1.4 | **+6.4 ± 1.6** | 2.71 | 3.56 |
| s2 | 70.0 ± 1.2 | 62.6 ± 1.1 | **+7.4 ± 1.5** | 3.58 | 3.37 |
| s3 | 66.5 ± 1.4 | 60.5 ± 1.2 | **+6.0 ± 1.6** | 2.99 | 3.45 |

Six to seven vp apart, where A5i's greedy and sampled agreed within
1.2: a diffuse policy, as A5-points' was and E1's per-model rows were.

#### The critic saw the horizon; the policy did not

The pre-registration asked for the panel to be read before the arm,
because a higher discount raises the return scale and the value
target's variance. By quarter of the run:

| metric | γ 0.99, s1 / s2 / s3 (last quarter) | γ 0.9, s1 / s2 / s3 (last quarter) |
|---|---|---|
| explained variance | **0.82 / 0.83 / 0.85** (0.78–0.80 in the first) | 0.65 / 0.63 / 0.70 |
| return std | 0.64 / 0.67 / 0.67 | 0.6–0.7 |
| clip fraction | 0.33 / 0.40 / 0.40 | 0.30 / 0.38 / 0.31 |
| displacement entropy (nats) | 1.85 / 2.07 / 2.04 | 1.60 / 1.82 / 1.88 |
| declaration entropy | 0.09 / 0.12 / 0.09 | 0.10 / 0.08 / 0.09 |
| gradient clipped | 99.98% | 99.98% |

The return scale did not grow with the discount, because on this facade
`gamma` applies only across turn closes and an episode is at most ten
of them: the same 4.38 of bar reward is worth 2.79 at the first
decision under 0.9 and 4.19 under 0.99, a 1.5× change in what the
first decisions see, not a 10× change in variance. The critic fit that
longer signal better — explained variance up 0.15 on every seed, the
ladder's highest on a spread rung — and the policy's curve, census and
diffuseness are unchanged. Nothing on the panel is red; nothing on it
separates the arms.

### What this says

- **The horizon is not the wall.** The D3 addendum measured that a late
  payoff reaches the first decisions at a quarter to a third of its
  value under gamma 0.9, and the pre-registration bet that the
  conjunction over five points fails because its payoff is discounted
  away. Paying it at 0.96 of face value instead changed nothing the
  policy does. On the ladder's spread rungs the failure at five and six
  points is not that the reward arrives too late; it is that per-step
  credit over eighteen bodies does not tell any one body that the
  empty point is its job.
- **Read a critic improvement beside the policy it serves.** Explained
  variance rose 0.63–0.73 → 0.78–0.85 with success unchanged: the value
  function can fit the longer-horizon return well and the policy
  gradient it feeds still cannot find the fifth point. A better critic
  is not a better policy on a search problem, which the critic probe of
  2026-08-23 already said of the whole-army trainer.
- **A comparator's final row is not its curve.** The original's s1 and
  s2 were at 0.22 at 80k and 0.03 / 0.06 at the end; a seed-for-seed
  MOVES clause against the final row would have read "ahead on s2"
  where the matched-rounds reads say "behind on every seed". Read the
  comparator at every checkpoint the arm is read at.
- **Not worth running next on this half-step:** another discount (0.95
  or 1.0 sit between two readings that agree), an entropy coefficient
  (the heads are not collapsed: displacement 1.9–2.1 nats), a longer
  budget (the curve is flat from 80k on both arms), or a clone start
  (D3). **Next:** the reward-shape arm the pre-registration named — a
  terminal bonus paid flat rather than scaled by the rounds remaining
  (`phase_manager.py` pays a first success at round ten 0.5 of 5), and
  coverage paid per flag to its unique holder so a body on the empty
  point is paid for the point and not for the army's mean — one change
  per arm, pre-registered, three seeds from scratch, read at 40k / 80k
  / 122,880 with this census.

### Where this lands

1. This report.
2. `reports/README.md` index row.
3. `CLAUDE.md`: a ladder row for the discount arm and a rule bullet
   (the horizon is not the wall; read a critic improvement beside the
   policy; read the comparator at matched rounds).
4. Live doc: n/a — no config added, no doctrine entry priced. The
   in-training recorder (`--record-every-rounds`, `f7539d2`) merged
   after these runs exited and is documented in
   `wargame_rl/wargame/model/CLAUDE.md`.

---

<a id="2026-09-19-curriculum-A5-points-reward-shape-preregistration"></a>

## Pre-registration: the reward-shape arms on A5-points — is the conjunction unpaid, or unfindable?

Written 2026-09-19 15:05, **before any training round on either config**,
on branch `feature/per-model-actor-credit` (PR #383). Parent question
#340; the second optimiser-side pass on the half-step after the discount
arm (FAIL, `c24aa10`: the horizon is not the wall). Set as a goal by Sash
2026-09-19 14:50: "run the reward-shape arm on A5-points".

### The question

A5-points (six squads of three over five points; the bar
`squad_march_take` 1.000 in 6.71 turns) reads 0.030 / 0.060 / 0.330 from
scratch at 122,880 rounds, and 0.070 / 0.180 / 0.100 with the discount
raised to 0.99. On both arms the census is the same: five or six of
eighteen bodies on points from turn 5, one point empty in half to four
fifths of episodes on every seed, the clock run out. Two things about
how the criterion is *paid* are on the record and untested on this
half-step:

1. **The success bonus shrinks with the rounds remaining.** With
   `terminate_on_success: true`, `phase_manager.terminal_bonuses` pays
   `5.0 × (max_turns − turn + 1) / max_turns`. A policy that first holds
   all five points in round nine of ten — which is where these runs are
   when they succeed at all (turns 9.6–9.9) — is paid **1.0**; in round
   ten, **0.5**. The criterion the rung is judged on is nearly unpaid
   exactly where a learning policy first meets it. The bar, succeeding
   in round seven, is paid 2.0.
2. **The conjunction pays nothing until the last point is taken.** Four
   of five points held at the clock is worth exactly what none are
   worth, at the terminal step. The dense coverage term (0.3 × fraction
   controlled, at the close) is the only per-point signal, and it is a
   broadcast mean. The whole-army record's critic probe and the
   per-model record's D3 addendum both say the policy has trouble
   *finding* the last point; nothing on the record says whether it would
   find it if the fourth were paid.

**The arms ask, one change at a time, whether the conjunction over five
points is learnable when a late success is paid in full (R1), and when
each point is paid at the end on its own (R2).**

### The changes (built today, default-off, goldens bit-identical)

- `RewardPhaseConfig.terminal_bonus_speed_scaling: bool = true` — false
  pays `terminal_success_bonus` in full whenever the criteria hold.
- `RewardPhaseConfig.terminal_objective_bonus: float = 0.0` — a second
  terminal term, `bonus × (objectives controlled / objectives)` on the
  final board under VP's control rule (the read `objective_coverage`
  makes every step), paid whether or not the phase succeeded, never
  scaled by the turns left.

Both flow through `RewardPhaseManager.terminal_bonuses`, which the
per-model retimer calls at the terminating close, so the two facades
cannot disagree. Tests: `tests/test_reward_phases.py`
(`test_terminal_bonus_speed_scaling_switch`,
`test_terminal_objective_bonus_pays_the_controlled_fraction`);
`tests/test_reward_golden.py` bit-identical; the bridge on R2's config
checked below.

| arm | config | the one change | tag |
|---|---|---|---|
| **R1** | `configs/experiments/curriculum/a5_points_flat.yaml` | `terminal_bonus_speed_scaling: false` | `a5r1` |
| **R2** | `configs/experiments/curriculum/a5_points_flat_points.yaml` | R1 **plus** `terminal_objective_bonus: 5.0` (each point worth 1.0 at the end) | `a5r2` |

R2 is one change on R1, so R2 − R1 prices the per-point payment and
R1 − original prices the flat bonus, each paired by seed and layout.

### The recipe

Everything else is A5-points as pre-registered on 2026-09-18: from
scratch, seeds 1 / 2 / 3, 122,880 rounds at 128 per update
(`--rollout-rounds 32 --num-rollout-envs 4`), `--ent-coef 0.003`,
**`gamma` 0.9** (the original's; the discount is a closed arm), the
`mean` credit, in-run eval every 512 rounds at n=30 on 500000+, Wandb
`curriculum-a5`, six trainers side by side. No whole-army control (a
diagnostic arm on a half-step). Every checkpoint carries a recorded
greedy episode (`--record-every-rounds`, on by default since
`f7539d2`). Read greedy at n=100 on 700000+ at 40,960 / 81,920 /
122,880 with the by-turn census; sampled beside greedy at the end; the
original A5-points and the discount arm read at the same rounds as the
comparators.

### Criteria (each arm on its own)

- **PASS (the half-step):** success ≥ 0.95 on all three seeds at
  122,880, no in-run dip below 0.80 after the first rolling pass.
- **MOVES:** ahead of the original A5-points' 0.030 / 0.060 / 0.330
  seed for seed at 122,880 by more than two binomial SE (0.06 at 0.1;
  0.09 at 0.33), 3/3; for R2 also ahead of R1 seed for seed on 2/3
  (else R2's movement is R1's).
- **FAIL:** neither.
- **Readouts:** held of 5; bodies on points by turn; points abandoned by
  seed; turns; `reward/components/terminal_success_bonus` and
  `terminal_objective_bonus` per episode against the travel and
  coverage terms (is the new pay a large share of the episode's
  reward?); explained variance and return std on the panel (a flat
  bonus raises the terminal return by up to 10×; R2's per-point term
  raises every episode's return); coherency; sampled beside greedy.

**What each answer says.** R1 PASS or MOVES: the criterion was unpaid
where it was first met — the speed scale is wrong for a rung judged on
success, and every `terminate_on_success` rung on the ladder should be
re-read with it off. R2 MOVES where R1 does not: the conjunction is
learnable when decomposed, and the ladder's A5 rung gets the same
change before any architecture arm. Both FAIL with the same census: the
fifth point is a search failure that no terminal payment reaches — the
policy never stands on four points long enough for the fifth to be
paid — and the next lever is exploration or representation, not reward.
A red panel (explained variance falling under the larger terminal
return, or return std doubling) with a moved policy is reported as a
move with a defect.

### What I expect (a guess, written so it can be wrong)

R1 moves on one or two seeds and passes on none: success 0.1–0.4, held
3.3–3.8, the walk-off reduced because a late hold is now worth
holding; the terminal bonus share of episode reward rises from ~2% to
10–30% on succeeding episodes. R2 moves on 3/3 and ahead of R1 on 2/3:
held 3.8–4.3 as the fourth point is paid, success 0.3–0.6, the fifth
point still the residual, PASS on none. If R2 passes I am wrong in the
useful direction; if neither moves, the terminal payment was never the
signal the policy was missing and the search account stands.

### The bar's reward stream on each config, and the bridge

`squad_march_take` on the per-model facade, n=40 on 700000+, `gamma`
0.9 per close (the travel and coverage terms are identical on all
three; the bar succeeds in round seven, so its scaled bonus was 2.16 of
5.0 on average):

| config | episode reward | discounted return at the first decision | turns 1–6 / 7–12 | terminal terms |
|---|---|---|---|---|
| `a5_points` (original) | 4.38 | 2.79 | 2.88 / 1.50 | success 2.16 |
| `a5_points_flat` (R1) | **7.21** | 4.35 | 3.82 / 3.40 | success 5.00 |
| `a5_points_flat_points` (R2) | **12.21** | 7.10 | 5.69 / 6.52 | success 5.00 + per-point 5.00 |

The value target roughly doubles on R1 and triples on R2 at the bar's
level of play — read the critic's explained variance and return std
before the arm, as on the discount arm. For a policy failing at the
clock the original pays nothing terminal, R1 pays nothing, and R2 pays
`5 × held / 5` — three or four of five points held at the end is
3.0–4.0, more than the whole of the original's episode reward.

`just measure-bridge` on R2's config (n=20, `squad_march_take`, seeds
700000+): **identical on every shared field** (vp 63.5 ± 2.3, held 5.00,
success 1.000, 6.70 turns) — the new terminal term is paid the same on
both facades.

### Amendment 1 — written 2026-09-19 19:35, both arms read at 40,960 / 81,920 / 122,880

**FAIL as pre-registered on both arms, and both are BEHIND the original at
every read.** Runs R1 `008ftjrj` / `pvafw3km` / `ih78ta51`, R2 `krhhq1v5` /
`d12n5luj` / `8jnrjqu5`, launched 15:15 (a first launch at 15:12 was stopped
at ~1,500 rounds and its directories removed because its eval and
checkpoint cadence was 256, not the recipe's 512; those six Wandb ids are
dead), exited 19:05–19:13 (six trainers at ~500–850 rounds a minute). Greedy
at n=100 on 700000+ at 122,880 (`last.pt`):

| row | success | turns | vs bar, paired | held of 5 | on points | coherent | stat |
|---|---|---|---|---|---|---|---|
| `squad_march_take` | 1.000 | 6.71 | — | 5.00 | 0.889 | 0.832 | — |
| R1 s1 | **0.000** | 10.00 | +3.29 ± 0.05 | **0.90** | 0.107 | 0.449 | 0.00 |
| R1 s2 | **0.030** | 9.91 | +3.20 ± 0.07 | 2.89 | 0.253 | 0.119 | 0.18 |
| R1 s3 | **0.010** | 9.98 | +3.27 ± 0.06 | 2.52 | 0.276 | 0.407 | 0.06 |
| R2 s1 | **0.020** | 9.99 | +3.28 ± 0.05 | 2.41 | 0.241 | 0.299 | 0.09 |
| R2 s2 | **0.030** | 9.95 | +3.24 ± 0.06 | 2.93 | 0.304 | 0.151 | 0.13 |
| R2 s3 | **0.000** | 10.00 | +3.29 ± 0.05 | 2.38 | 0.263 | 0.328 | 0.28 |
| original s1 / s2 / s3 (2026-09-18) | 0.030 / 0.060 / 0.330 | 9.56–9.96 | | 2.47 / 2.92 / 3.83 | 0.26–0.38 | 0.19–0.21 | 0.04–0.06 |

- **PASS:** no seed above 0.03.
- **MOVES:** ahead of 0.030 / 0.060 / 0.330 by two SE on 3/3 — **0 of 3 on
  either arm** (R1 −0.03 / −0.03 / −0.32; R2 −0.01 / −0.03 / −0.33). R2
  against R1: +0.02 / 0.00 / −0.01, nothing.
- **FAIL, both arms.** The written expectation (R1 moves a little, R2 moves
  on 3/3 toward held 4) was wrong in the other direction: both arms are
  behind the original at 40,960, 81,920 and 122,880.

**At every read** (success; held of 5):

| rounds | R1 | R2 | original | discount arm |
|---|---|---|---|---|
| 40,960 | 0.00 / 0.02 / 0.00; 1.9 / 1.7 / 1.3 | 0.00 ×3; 2.2 / 2.0 / 2.0 | 0.04 / 0.14 / 0.09; 1.9 / 2.9 / 2.8 | 0.00 / 0.03 / 0.06; 1.7 / 1.2 / 2.5 |
| 81,920 | 0.00 / 0.00 / 0.02; 2.1 / 2.6 / 2.1 | 0.00 ×3; 1.9 / 2.6 / 2.5 | **0.22 / 0.22 / 0.11**; 3.4 / 3.4 / 2.8 | 0.10 / 0.08 / 0.04; 2.7 / 3.1 / 2.9 |
| ~85–90k (`last.pt` read early by a chain fault; a readout, not a pre-registered read) | 0.01 / 0.03 / 0.00 | 0.01 / 0.05 / 0.00 | — | — |
| 122,880 | 0.00 / 0.03 / 0.01; 0.9 / 2.9 / 2.5 | 0.02 / 0.03 / 0.00; 2.4 / 2.9 / 2.4 | 0.03 / 0.06 / 0.33; 2.5 / 2.9 / 3.8 | 0.07 / 0.18 / 0.10; 3.0 / 3.4 / 3.1 |

**The census** (n=100 on 700000+; bodies on points of 18 and points held of
5 after turns 3 → 5 → 7 → end; share of episodes with each point empty at
the end by index, 0–2 the near column, 3–4 the far column; max stack):

| policy | success | on points | held | empty by point | max stack |
|---|---|---|---|---|---|
| `squad_march_take` | 1.00 | 11.7 → 12.8 → 15.8 → 15.8 | 3.0 → 4.2 → 5.0 → 5.0 | 0 / 0 / 0 / 0 / 0 | 5.0 |
| R1 s1 | 0.00 | **10.2 → 2.1** → 6.8 → **1.9** | 2.7 → 1.5 → 2.0 → 0.9 | 1.00 / 0.89 / 1.00 / 0.41 / 0.80 | 1.7 |
| R1 s2 | 0.03 | 3.3 → 5.0 → 4.6 → 4.5 | 1.6 → 2.2 → 2.8 → 2.9 | 0.20 / 0.43 / 0.25 / 0.54 / 0.69 | 2.2 |
| R1 s3 | 0.01 | 6.1 → 5.6 → 5.0 → 5.0 | 1.9 → 2.4 → 2.5 → 2.5 | 0.45 / 0.43 / 0.69 / 0.29 / 0.62 | 2.9 |
| R2 s1 | 0.02 | 6.6 → 4.7 → 4.3 → 4.3 | 2.0 → 2.2 → 2.1 → 2.4 | 0.92 / 0.30 / 0.51 / 0.32 / 0.54 | 2.4 |
| R2 s2 | 0.03 | 2.2 → 4.5 → 5.4 → 5.5 | 1.2 → 2.1 → 2.9 → 2.9 | 0.38 / 0.18 / 0.24 / 0.65 / 0.62 | 2.8 |
| R2 s3 | 0.00 | 6.2 → 4.2 → 4.6 → 4.7 | 1.9 → 2.1 → 2.3 → 2.4 | 0.68 / 0.36 / 0.40 / 0.33 / 0.85 | 3.0 |

Two things are new against the original's census. **R1 s1 is the sharpest
walk-off on the ladder**: ten bodies on points after turn 3, two after
turn 5, seven after turn 7, two at the end — held 0.90 of 5, every near
point empty in 89–100% of episodes; its sampled play holds **3.20** where
its greedy play holds 0.90 (−6.6 ± 1.1 vp greedy − sampled), so the greedy
argmax is what walks off a diffuse policy. **R2 at 81,920 had the far
column empty in 90–100% of episodes on every seed** (near column mostly
held: the per-point payment bought the three points reached by turn 5 and
not the walk to the far two); by 122,880 that has diffused to 0.33–0.85
across all five. Neither arm's max stack exceeds three; the failure is
under-arrival and abandonment, as on every read of this half-step.
Sampled against greedy: −6.6 / +3.6 / −0.6 (R1), +1.1 / +8.4 / +5.8 (R2)
vp — diffuse on four of six.

**The panel, and the mechanism.** The pre-registration asked for the
critic before the arm, and the critic is where the arms broke.
`train/explained_variance` by quarter: R1 −0.02 / 0.09 / −0.14 / −0.01,
0.11 / 0.41 / **0.73** / 0.53, 0.21 / 0.22 / 0.03 / −0.17; R2 **−0.32 /
−0.19 / −0.19 / −0.37**, 0.12 / −0.31 / −0.32 / −0.47, −0.24 / −0.23 /
−0.14 / −0.14 — against the original's 0.63–0.73 and the discount arm's
0.78–0.85 on the same rung. Return std 0.44–0.88 (the original 0.6–0.7).
In-run success in the last quarter (n=30 on 500000+) 1.9 / 2.3 / 2.2 (R1)
and 0.8 / 2.5 / 0.2 (R2) against the original's 15.0 / 19.6 / 8.6. Clip
fraction 0.18–0.31, displacement entropy 2.0–2.6 nats (the original
1.6–1.9), gradient clipped throughout. A red panel with a policy that did
not move is a FAIL with a named defect: **paying the criterion as a large
lump at the end — 5.0 on success (R1), or 1.0 per point at the clock on
every episode (R2) — made the value target a rare or turn-keyed jump the
critic could not fit, and the advantages PPO trains on became noise.** The
original's remaining-rounds scale, which this arm called a defect, was
also keeping the terminal lump small (0.5–2.0) beside the dense terms.

**What the answer says, as written.** FAIL on both: the terminal payment
was never the signal the policy was missing; and the search account
stands — the policy never holds four points long enough for a fifth to
be paid, and paying the fourth taught it to keep the near column. Next
on this half-step is not a reward-shape arm and not an optimiser
setting: the record now has the discount (critic better, policy the
same), the terminal shape (critic worse, policy the same) and the credit
(A5d/A5e, nothing). What is left is exploration or representation —
directed exploration toward the empty point, or an observation that
names it — and the ladder's rule that a rung failing at the cap gets
one optimiser arm has now been spent on this half-step twice.

**Recordings.** Every checkpoint of all six runs carries a recorded
greedy episode (`recordings/pm-<rounds>-seed500000.json`), the first
arms recorded in-run; thirteen of each run's recordings are rendered to
MP4 beside their logs and logged to the run's own Wandb record under
`episode_recording` (post hoc, `362518a`'s pipeline; sibling runs
`850oup4q` / `3yauw38m` / `ptja7rxk` / `078e0p6p` / `1p5qwba8` /
`clqr3exm` carry the same videos, logged while the trainers ran).

---

<a id="2026-09-19-curriculum-a5-points-reward-shape-the-lump-breaks-the-critic"></a>

## Curriculum A5-points, the reward-shape arms: paying the criterion as a lump breaks the critic, and paying per point buys the near column

**Verdict first.** Two one-change arms on the half-step A5-points (six
squads of three over five points; the bar `squad_march_take` 1.000 in
6.71 turns), trained from scratch under the original recipe, three seeds
each, 122,880 rounds, read greedy at n=100 on 700000+:

| arm | the change | verdict | success at 122,880 | held of 5 | the original (2026-09-18) |
|---|---|---|---|---|---|
| **R1** | the success bonus paid in full however late (`terminal_bonus_speed_scaling: false`) | **FAIL — behind the original at every read** | 0.000 / 0.030 / 0.010 | 0.90 / 2.89 / 2.52 | 0.030 / 0.060 / 0.330; held 2.47 / 2.92 / 3.83 |
| **R2** | R1 plus 1.0 per point held on the final board (`terminal_objective_bonus: 5.0`) | **FAIL — behind the original at every read** | 0.020 / 0.030 / 0.000 | 2.41 / 2.93 / 2.38 | |

Neither arm is ahead of the original on any seed at 40,960, 81,920 or
122,880 (the original at 81,920 reads 0.22 / 0.22 / 0.11; the arms 0.00 /
0.00 / 0.02 and 0.00 ×3). The census is under-arrival and abandonment as on
every read of this half-step, with two new signatures: R1's first seed
walks ten bodies onto points by turn 3 and ends with two (held 0.90,
sampled play holds 3.20), and R2 at two thirds of the budget had the far
column empty in 90–100% of episodes on every seed — the per-point payment
bought the three near points and not the walk to the far two. **The
mechanism is on the panel: explained variance sat between −0.5 and +0.2 on
five of six seeds for the whole run** (the original 0.63–0.73, the discount
arm 0.78–0.85), and in-run success in the last quarter was 0.2–2.5% against
the original's 9–20%. A large terminal lump — 5.0 on a rare success, or a
per-point sum keyed to the clock — is a value target this critic cannot fit,
and PPO's advantages became noise. The original's remaining-rounds scale,
which the pre-registration called a defect, was also what kept the terminal
lump small beside the dense terms. **Three optimiser-and-reward arms on this
half-step now read the same policy: the discount made the critic better,
the terminal shape made it worse, the credit changed nothing — and the
policy is the same on every one.** What is left for the spread wall is
exploration or representation, not the reward's timing or shape.

### Provenance

| field | value |
|---|---|
| date | 2026-09-19: build and pre-registration 14:50–15:05, launched 15:15 (a 15:12 launch at the wrong cadence stopped at ~1,500 rounds and removed), trainers exited 19:05–19:13, final reads 19:13–19:25 |
| GPU / no-GPU | GPU (RTX 4090), six trainers side by side at 500–850 rounds a minute |
| seeds | 1 / 2 / 3 per arm from scratch — the same seeds as the original A5-points and the discount arm, so per-seed comparisons are paired at initialisation |
| n | 100 at seed base 700000 (every read, the census, greedy against sampled); 30 at 500000 (in-run, every 512 rounds) |
| config | `configs/experiments/curriculum/a5_points_flat.yaml` (R1) and `a5_points_flat_points.yaml` (R2): `a5_points.yaml` with the one phase field each; unrefereed by design |
| decode | none on the per-model facade |
| paired | per episode against the bar on identical seeds (turns); per seed against the original and against each other on the same seeds and layouts |
| comparator | the original A5-points runs `uagmul66` / `wedjjozy` / `ici5py46`: 0.030 / 0.060 / 0.330 at 122,880 (0.040 / 0.140 / 0.090 at 40,960; 0.220 / 0.220 / 0.110 at 81,920); the discount arm `meawm0gk` / `6lr588y7` / `0sjtnn0n` 0.070 / 0.180 / 0.100; the bar `squad_march_take` 1.000, 6.71 turns, held 5.00, coherent 0.832 |
| opponent | none |
| budget | 122,880 rounds at 128 per update (`--rollout-rounds 32 --num-rollout-envs 4`), `--ent-coef 0.003`, `gamma` 0.9, the `mean` credit, eval and checkpoint every 512; no extension |
| code revision | build and pre-registration `509abb3`; runs on `509abb3`; amendment `9d817fb`; the branch `feature/per-model-actor-credit`, PR #383 |
| checkpoint | `last.pt` at 122,880 and every 512 rounds under `checkpoints/per_model/per-model-a5_points_flat{,_points}-2026-09-19-15-15-31-s{1,2,3}a5r{1,2}`; a recorded greedy episode beside every checkpoint (the first arms recorded in-run), thirteen per run rendered to MP4 and on the run's Wandb record |
| coherency | greedy at 700000+: R1 0.449 / 0.119 / 0.407, R2 0.299 / 0.151 / 0.328 (the original 0.19–0.21; the bar 0.832); sampled 0.05–0.11 |
| Wandb | `curriculum-a5`: R1 `008ftjrj` / `pvafw3km` / `ih78ta51`, R2 `krhhq1v5` / `d12n5luj` / `8jnrjqu5`; video sibling runs `850oup4q` / `3yauw38m` / `ptja7rxk` / `078e0p6p` / `1p5qwba8` / `clqr3exm` |
| pre-registration | `reports/2026-09-19-curriculum-a5-points.md#2026-09-19-curriculum-A5-points-reward-shape-preregistration` (+1 amendment, which carries every table) |

### The read

The final table, the reads at every checkpoint, the census by turn and
by point, the greedy-against-sampled table and the panel by quarter are
in the amendment; this section says what they mean.

#### The two changes did what they were built to do, and the policy did not follow

The bar's episode reward went 4.38 → 7.21 (R1) → 12.21 (R2), and both new
terms were earned in training: R1 seeds collected the flat bonus at
0.04–0.07 per close, R2 seeds the per-point bonus at ~0.27 per close —
more than the travel and coverage terms together. Nothing about the
payment failed to reach the policy. The policy's census is the
half-step's on every column, and worse: fewer bodies on points than the
original at the end (4–6 of 18 against 5–7), held 2.4–2.9 against 2.5–3.8,
the clock run out on every seed.

#### The critic could not fit the lump

Explained variance by quarter, the pre-registered readout:

| run | Q1 | Q2 | Q3 | Q4 |
|---|---|---|---|---|
| R1 s1 | −0.02 | 0.09 | −0.14 | −0.01 |
| R1 s2 | 0.11 | 0.41 | 0.73 | 0.53 |
| R1 s3 | 0.21 | 0.22 | 0.03 | −0.17 |
| R2 s1 | −0.32 | −0.19 | −0.19 | −0.37 |
| R2 s2 | 0.12 | −0.31 | −0.32 | −0.47 |
| R2 s3 | −0.24 | −0.23 | −0.14 | −0.14 |
| original (γ 0.9) | 0.65 / 0.67 / 0.68 | 0.65 / 0.73 / 0.71 | 0.63 / 0.67 / 0.69 | 0.65 / 0.63 / 0.70 |
| discount arm (γ 0.99) | 0.78–0.80 | 0.78–0.80 | 0.77–0.84 | 0.82–0.85 |

A value function worse than predicting the mean, on five of six seeds,
for a hundred and twenty thousand rounds. The original's return is
dominated by dense terms (travel 1.31, coverage 0.91 of the bar's 4.38)
with a terminal bonus that the remaining-rounds scale keeps at 0.5–2.0;
R1's return has a 5.0 jump on the rare episode that succeeds, and R2's a
3.0–5.0 jump at the clock on every episode whose size depends on the
board at turn ten. The critic sees the turn index, but the per-model
value head is asked to predict a lump paid to the army from a token set
that changes every decision; it did not learn to. With the advantages
noise, the policy gradient has no direction, and the displacement head
sits at 2.0–2.6 nats (the original 1.6–1.9) with clip fractions of
0.18–0.31.

#### Partial credit bought the partial behaviour

At 81,920 R2's far column (points 3 and 4, fifteen inches beyond the
near three) was empty in 90 / 100 / 100% of episodes on its three seeds
while the near column was mostly held — the cleanest measurement of a
shaping term buying exactly what it pays for. Three points at the clock
pays 3.0, more than the whole of the original's episode reward, and the
walk to the far pair is unpaid until it lands. By 122,880 that had
diffused (far column empty 0.33–0.85) without any point being taken more
often. The A5 record's rule — *before training a shaping term, ask
whether the behaviour it wants pays more in total than the behaviour it
replaces* — was satisfied here (five points pays more than three) and
was not enough: the increment for the fourth and fifth is paid only on
arrival, and arrival is the thing the policy does not find.

#### The sharpest walk-off on the ladder

R1's first seed puts 10.2 of 18 bodies on points after turn 3, 2.1 after
turn 5, 6.8 after turn 7 and 1.9 at the end — held 0.90 of 5, the near
column empty in 89–100% of episodes. Its sampled play holds 3.20, so the
greedy argmax of a diffuse displacement head is what walks off; the
discount arm's rows showed the same sign (sampled 6–7 vp *worse* there,
here −6.6 vp *better* on this seed). On this trainer a policy that has
not converged can read as a walk-off under greedy scoring and as a hold
under sampled scoring; the two rows together are the reading.

### What this says

- **A terminal lump is a critic problem before it is a policy problem.**
  On this trainer the remaining-rounds scale was doing two jobs — a speed
  incentive, and keeping the terminal payment small beside the dense
  stream — and removing it for the first job broke the second. Do not
  raise a terminal payment on a per-model rung without reading the
  critic's explained variance at 20k rounds; a red panel that early is
  the arm's answer.
- **Partial credit buys the partial behaviour.** A per-point terminal
  payment taught the army to keep the near column. On a conjunction
  rung, decomposing the criterion into per-part payments pays the parts
  that are easy and leaves the hard part exactly as unpaid as before.
- **Three arms, one policy.** Discount, terminal shape and credit each
  changed what the critic or the return looked like and none changed
  what the army does at five points. The spread wall on the per-model
  trainer is not the reward's timing, size, shape or attribution. The
  remaining levers are the ones the whole-army critic probe named in
  August: directed exploration and representation. Do not run a fourth
  reward arm on this half-step.
- **Gate a final read on the trainer exiting, not on `last.pt`
  existing.** `last.pt` exists from the first checkpoint; a chain gated on
  it read the arms at ~85k rounds and called it final. Caught by the
  timestamps; the real final read was re-queued on the process count.
- **The videos are on the runs.** Every per-model launch from `362518a`
  renders one recording in twenty to MP4 and logs it under
  `episode_recording`; these six runs carry theirs post hoc.

### Where this lands

1. This report.
2. `reports/README.md` index row.
3. `CLAUDE.md`: a ladder row for the reward-shape arms and a rule bullet
   (the lump breaks the critic; partial credit buys the partial
   behaviour; three arms, one policy; gate a final read on exit).
4. Live docs: `docs/reward-phases.md` carries the two new phase fields
   (`509abb3`); `configs/README.md` the two arm configs; the in-training
   video pipeline is documented in `wargame_rl/wargame/model/CLAUDE.md`.

---

<a id="2026-09-19-curriculum-A5-points-control-and-warm-start-preregistration"></a>

## Pre-registration: the whole-army control and the A4x warm start on A5-points — whose wall is it, and does a start help?

Written 2026-09-19 20:05, **before any training round on either arm**, on
branch `feature/per-model-actor-credit` (PR #383). Parent question #340.
Set as a goal by Sash 2026-09-19 20:00: "run the A5-points whole-army
control and the A4x warm-start beside it; then build and run the backward
start curriculum on A5-points".

### The question

A5-points (six squads of three over five objectives; the bar
`squad_march_take` 1.000 in 6.71 turns) has now been read from scratch
under four settings of the per-model trainer — the original recipe
(0.030 / 0.060 / 0.330), the discount at 0.99 (0.070 / 0.180 / 0.100),
the flat terminal bonus (0.000 / 0.030 / 0.010) and the per-objective
terminal bonus (0.020 / 0.030 / 0.000) — and the census is the same on
every one: a third of the bodies on objectives, one objective empty, the
clock run out. Two things the ladder's own contract asks for have not
been run on this half-step:

1. **The whole-army control.** Every A rung ran `train.py` on the same
   config as the pipeline control. On A5 (six objectives) the control
   read 0.80 / 0.94 / 0.98 at 120 epochs where the per-model arm read
   0.26 / 0.18 / 0.16. Five objectives has no control. If the whole-army
   trainer passes here, the wall is the per-model trainer's; if it fails,
   five objectives is hard for both trainers at this budget.
2. **The warm start from the objective rung below.** S3 (A3 from A2b)
   read AHEAD 3/3 and is the one lever that ever moved a spread rung;
   T1 (A5 from A4x) read ahead on every seed and passed on none. A4x —
   the same army's four-squad ancestor at three objectives, passed 3/3 at
   245,760 rounds — has not been tried as the start for five objectives.

### The arms

| arm | trainer | config | start | budget | tag |
|---|---|---|---|---|---|
| **CTL** | whole-army `train.py` | `a5_points.yaml` | scratch | 60 epochs (`--n-eval-episodes 30`, `--record-during-training`; defaults otherwise, `ent_coef` 0.03 as on every control), extended once to 120 if any seed is still rising at 60 | `a5pts-ctl` |
| **W** | per-model `train_per_model.py` | `a5_points.yaml` | `--warm-start-from` A4x seed-for-seed: `per-model-a4-2026-09-15-07-08-53-s{1,2,3}a4/pm-00245760.pt` (weights only; cold critic and optimiser as T1 ran) | 122,880 rounds, the original recipe (128 per update, `--ent-coef 0.003`, `gamma` 0.9, `mean` credit, eval and checkpoint every 512, recording and video on) | `a5pw` |

Three seeds each, on Wandb under `curriculum-a5`. The bridge on this
config was identical on 2026-09-18 (`squad_march_take` 1.000 on both
facades). The per-model comparator is the original A5-points from
scratch (`uagmul66` / `wedjjozy` / `ici5py46`) read at 40,960 / 81,920 /
122,880; the control is read against the script on its own facade.

### Criteria

**CTL** (the rung's letter, as on every A rung):
- **PASS:** success ≥ 0.95 on 3/3 at n=100 on 700000+ at 60 epochs (or at
  120 after the one extension), turns a readout.
- **FAIL:** any seed under 0.95 at the last read.
- **Readouts:** rounds-to-pass in-run (rolling 95% at n=30), turns, held,
  the by-turn census, which objective is left empty.

**W** (the transfer readouts T1 taught, beside the rounds bound):
- **PASS:** success ≥ 0.95 on 3/3 at 122,880, no in-run dip below 0.80
  after the first rolling pass.
- **MOVES:** ahead of the original 0.030 / 0.060 / 0.330 seed for seed by
  more than two binomial SE on 3/3 at 122,880, **and** ahead at 40,960
  (0.040 / 0.140 / 0.090) and 81,920 (0.220 / 0.220 / 0.110) on 3/3 — a
  start that is ahead at every read.
- **FAIL:** neither.
- **Readouts:** success at 20,480 (T1's read: where scratch ends, the warm
  start already was), the in-run peak, held, the census, the panel,
  sampled beside greedy.

**What the pair says.** CTL PASS and W FAIL: the wall is the per-model
trainer at five objectives, and #283's question has its first negative
row on the A rungs. CTL FAIL and W FAIL: five objectives is beyond both
trainers at this budget; the backward start curriculum (goal item 3) is
built on that. W MOVES or PASS: the three-objective policy carries to
five, and the C rungs' warm-start rule extends to the half-step. A W
seed that is ahead at 20k and behind at the end is T1's shape: transfer
onto a failure, reported as such.

### What I expect (a guess, written so it can be wrong)

CTL passes 2 of 3 at 60 epochs and 3 of 3 at 120, a round behind the
script, leaving the far column's second objective empty when it fails —
the A5 control's shape one objective smaller. W reads 0.3–0.5 at 20,480
(ahead of everything from scratch), plateaus at 0.4–0.6 with held
3.5–4.0, ahead of the original at every read on 3/3 (MOVES) and PASS on
none — T1's shape, the fifth objective still the residual. If CTL fails
too, the half-step is harder than the A5 control's numbers implied and
the backward start is the next thing on both trainers.

### Amendment 1 — written 2026-09-19 22:40, the control read at 60 and 120 epochs

(The header's "20:05" is wrong; the pre-registration was committed at
22:02, `c9f23d4`, and the arms launched at 22:03.)

**CTL: FAIL on the letter by two seeds at 120 epochs — and it is the best
read this half-step has had from any trainer.** Runs `pic9zjka` /
`ccheoveo` / `oct4luxd` (60 epochs, 22:03–22:14), extended once to 120
per the pre-registration as `ryk2bfuc` / `0wlaqa3l` / `xwsqlrtn`
(`--resume-ckpt-path`, 22:17–22:30). Greedy at n=100 on 700000+:

| row | success | turns | vs bar, paired | held of 5 | on objectives | coherent |
|---|---|---|---|---|---|---|
| `squad_march_take` | 1.000 | 6.71 | — | 5.00 | 0.889 | 0.832 |
| s1 at 60 | **0.980** | 7.22 | +0.51 ± 0.09 | 4.98 | 0.936 | 0.609 |
| s2 at 60 | 0.620 | 8.80 | +2.09 ± 0.11 | 4.61 | 0.948 | 0.664 |
| s3 at 60 | 0.870 | 8.27 | +1.56 ± 0.08 | 4.86 | 0.940 | 0.655 |
| s1 at 120 | 0.910 | 7.33 | +0.62 ± 0.10 | 4.91 | 0.965 | 0.645 |
| s2 at 120 | 0.810 | 7.73 | +1.02 ± 0.14 | 4.80 | 0.930 | 0.575 |
| s3 at 120 | **0.950** | 7.35 | +0.64 ± 0.10 | 4.95 | 0.955 | 0.570 |

- **PASS** needs ≥ 0.95 on 3/3: one seed at 120 (s3), one at 60 (s1).
  s1 fell 0.98 → 0.91 over the extension (a drift, as A2 and the
  whole-army A5 control showed); s2 rose 0.62 → 0.81, s3 0.87 → 0.95.
  **FAIL on the letter.**
- Against the per-model trainer on the same config and budget the
  control holds **4.8–4.95 of five objectives with 93–97% of its bodies on
  them**, half a round to a round behind the bar, where five per-model
  settings from scratch read 0.00–0.33 with a third of the bodies on
  objectives. The whole-army trainer solves the five-objective spread
  from scratch to within a seed of the letter; the per-model trainer
  does not approach it. **The wall on this half-step is the per-model
  trainer's**, on the reading the pre-registration wrote for this
  quadrant — pending W (a warm start that carries) and BS (a start
  curriculum that walks back), both in flight.
- Readouts still to take: the control's in-run curve (its Wandb file has
  no history rows readable by the local scanner; read after the runs
  exit through `just run-summary`), rounds-to-pass, and which objective
  the failing episodes leave empty (a whole-army census).

### Amendment 2 — written 2026-09-20 02:20, the warm start read at 20,480 / 40,960 / 81,920 / 122,880

**W: FAIL on the letter — and the best per-model result reward has
produced on this half-step, still climbing at the cap.** Runs
`8clgcy59` / `klv24tu2` / `8h4eq4lj`, from A4x seed-for-seed, exited
01:58–02:10. Greedy at n=100 on 700000+:

| rounds | W success | W held of 5 | the original at the same rounds |
|---|---|---|---|
| 20,480 | 0.030 / 0.120 / 0.180 | 2.94 / 3.37 / 3.31 | (0.03 / 0.06 / 0.33 is the original's END) |
| 40,960 | 0.150 / 0.110 / 0.250 | 3.23 / 3.40 / 3.75 | 0.040 / 0.140 / 0.090 |
| 81,920 | 0.100 / 0.250 / **0.520** | 2.98 / 3.53 / 4.22 | 0.220 / 0.220 / 0.110 |
| **122,880** | **0.460 / 0.260 / 0.350** | 3.96 / 3.53 / 3.62 | 0.030 / 0.060 / 0.330 |

- **PASS:** no seed near 0.95.
- **MOVES:** ahead of the original at the end by two SE on s1 (+0.43)
  and s2 (+0.20), not s3 (+0.02 against 0.330); at 81,920 s1 was behind
  (0.10 against 0.22). Two of three at the end, not three at every read.
  **FAIL on the letter.**
- Against every per-model read on this half-step it is the top row:
  0.46 with 3.96 held on s1, 0.52 with 4.22 held on s3 at 81,920, where
  five settings from scratch never passed 0.33 or 3.8 held. The in-run
  curve on s1 is still rising at the cap — success by quarter 4 / 13 /
  18 / **42%**, a rolling 50% with 4.45 held over the last eight
  evaluations — and s3 sits at 34–38%. The displacement head is
  concentrated (1.2–1.5 nats against 2.0–2.6 from scratch), explained
  variance 0.48–0.78, clip fraction 0.25–0.32.
- **T1's shape did not repeat.** T1 (A5 from A4x, six objectives) was
  ahead at 20k and flat after. Here the start carried the walk (six or
  seven bodies on objectives from turn 3, no walk-off at 20k) and reward
  then kept climbing on two seeds through the whole budget.
- **The ladder's symmetric cap rule applies.** The control on this
  half-step needed its once-only extension to 120 epochs (about 245k
  rounds) and still read 0.91 / 0.81 / 0.95, so the per-model arm under
  test gets the same rounds before the rung is called (A3x, A4x). The
  three runs are resumed in place to **245,760** for one read; this
  amendment records the 122,880 read as the pre-registered one, and the
  next records the extension.

### Amendment 3 — written 2026-09-20 04:40, the warm start read at 245,760 (the once-only extension under the cap rule)

Runs resumed in place (`xw154bst` / `8dcfq1xa` / `kwdxlh12`, 02:13–04:15).
Greedy at n=100 on 700000+:

| rounds | W success | W held of 5 | on objectives | turns | vs the original's end (0.030 / 0.060 / 0.330) |
|---|---|---|---|---|---|
| 122,880 | 0.460 / 0.260 / 0.350 | 3.96 / 3.53 / 3.62 | 0.40 / 0.36 / 0.38 | 9.04 / 9.44 / 9.05 | ahead on s1, s2 |
| **245,760** | **0.340 / 0.170 / 0.600** | 3.59 / 3.52 / 4.29 | 0.35 / 0.33 / 0.45 | 9.37 / 9.70 / **8.12** | ahead on 3/3 (+0.31 / +0.11 / +0.27, each past two SE) |

- **PASS:** no. **MOVES on the letter (at 122,880, at every read):** no —
  s3 level at 122,880, s1 behind at 81,920. At the extended read the arm
  is ahead of the original's final row on every seed, and s3's 0.600
  with 4.29 held and 8.0 of 18 bodies on objectives by turn 7 (the bar's
  15.8) is the highest read reward has produced on this half-step by a
  wide margin. s1 and s2 fell back (0.46 → 0.34, 0.26 → 0.17) over the
  second 122k rounds: the in-run curve on s2 was flat at 25–30% for the
  whole extension and s1 oscillated 33–52% by quarter, while s3 climbed
  40 → 58% and ended at a rolling 63%. **FAIL on the letter, ahead 3/3
  at 2× the cap, one seed still climbing.**
- The census at 245,760 has no walk-off on any seed (bodies on objectives
  4.2 → 6.3, 3.5 → 5.9, 7.0 → 8.0 across the turns) and the misses spread
  over all five objectives; max stack 2.5–3.1. Sampled play holds 3.5–4.1
  against greedy's 3.5–4.3, within ±6 vp: converged, not diffuse. Panel
  over the extension: explained variance 0.41–0.57, displacement entropy
  1.0–1.3 nats, clip fraction 0.25–0.32.
- **What the pair says, as written.** CTL FAIL by two seeds at 0.81–0.95
  while W reaches 0.17–0.60: the whole-army trainer is a factor of
  two to five ahead of the per-model trainer's best on this half-step
  at equal rounds, and the per-model trainer's best is a warm start
  from the objective rung below, not anything learned from scratch. The
  warm-start rule ("from the rung below") holds on the half-step where
  "from A4x onto six objectives" (T1) did not; the difference is one
  objective, and the reading is that the three-objective policy carries
  four of five here and the fifth is what reward adds, slowly and on one
  seed of three.

---

<a id="2026-09-19-curriculum-A5-points-backward-start-preregistration"></a>

## Pre-registration: the backward start curriculum on A5-points — start with the objectives held and walk the start back to deployment

Written 2026-09-19 22:14, **before any training round under the schedule**,
on branch `feature/per-model-backward-start` (built in a worktree off
`c9f23d4` while the control and the A4x warm start train; to be merged into
`feature/per-model-actor-credit`, PR #383, once those exit). Parent question
#340. Goal item 3 of Sash's 2026-09-19 22:00 goal.

### The question

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

### The build (default off, goldens and every evaluation bit-identical)

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

### The arm

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

### Criteria

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

### What I expect (a guess, written so it can be wrong)

Level 4 is learned within 10,240 rounds on every seed (one squad walks to
one empty objective, which A2 learned in two thousand rounds); level 3
within 30k; level 2 and 1 slower, the walk-off returning as the number of
free squads grows; level 0 reached on one or two seeds by 100k. Success
from deployment at 122,880 **0.5–0.8**, held 4.2–4.6, MOVES on 3/3, PASS on
one seed at most. If the level never leaves 4 on any seed I was wrong
about the mechanism in the useful direction.

### Amendment 1 — written 2026-09-20 02:15, the arm read at 40,960 / 81,920 / 122,880

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

---

<a id="2026-09-20-curriculum-A5-points-rounds-per-update-preregistration"></a>

## Pre-registration: A5-points at the whole-army trainer's rounds per update — is the per-model trainer's wall the number of games each update sees?

Written 2026-09-20 05:20, **before any training round**, on branch
`feature/per-model-actor-credit` (PR #383). Parent question #340; asked by
Sash 2026-09-20 05:00: "can you investigate what exactly makes it hard to
learn?" after the control on this half-step read 0.91 / 0.81 / 0.95 where
the per-model trainer from scratch reads 0.00–0.33 under five settings.

### The question

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

### The arm

| arm | config | the one change | budget | tag |
|---|---|---|---|---|
| **RR** | `a5_points.yaml` (unchanged) | `--rollout-rounds 512` with `--num-rollout-envs 4`: **2,048 rounds per update**, the whole-army trainer's; eval and checkpoint every 2,048 (the cadence must be a multiple of the rollout) | 122,880 rounds (60 updates instead of 960; the same transitions, `n_epochs` 5 and `batch_size` 128 so the same order of gradient steps), `--ent-coef 0.003`, `gamma` 0.9, `mean` credit, from scratch, recording and video on | `a5rr` |

Three seeds. Comparator: the original A5-points from scratch at 128 rounds
per update (`uagmul66` / `wedjjozy` / `ici5py46`), 0.030 / 0.060 / 0.330 at
122,880, 0.040 / 0.140 / 0.090 at 40,960, 0.220 / 0.220 / 0.110 at 81,920.
Read greedy at n=100 on 700000+ at 40,960 / 81,920 / 122,880 with the
by-turn census and the walk-off probe (stay share on an objective), sampled
beside greedy at the end.

### Criteria

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

### What I expect (a guess, written so it can be wrong)

MOVES on 2/3, PASS on none: success 0.2–0.5, held 3.5–4.2, the walk-off
reduced but not gone (the reward still pays nothing for staying). If it
passes, the September regime finding was under-scoped — 128 rounds per
update was enough for one, three and four objectives and not for five.

### Amendment 1 — written 2026-09-20 13:00, the arm read at 40,960 / 81,920 / 122,880

(The header's "05:20" is wrong; the pre-registration was committed at
10:44, `41f3291`, and the arm launched at 10:45.)

**FAIL as pre-registered — the update regime is not the wall, and the
bigger batch was worse.** Runs `hipu8crt` / `uc5e4png` / `i8n9lzp9`
(10:45–12:45, sixty updates of 2,048 rounds). Greedy at n=100 on 700000+:

| rounds | RR success | RR held of 5 | the original (128 rounds per update) |
|---|---|---|---|
| 40,960 | 0.000 / 0.000 / 0.000 | 1.76 / 0.00 / 1.18 | 0.040 / 0.140 / 0.090 |
| 81,920 | 0.000 / 0.010 / 0.000 | 0.02 / 1.52 / 1.93 | 0.220 / 0.220 / 0.110 |
| **122,880** | **0.000 / 0.010 / 0.000** | 0.12 / 1.49 / 2.04 | 0.030 / 0.060 / 0.330 |

- **PASS / MOVES:** no. Behind the original at every read on every seed.
- **The census is the walk-off at its largest.** s1 puts 11.3 of 18
  bodies on objectives after turn 3 and has 0.1 there at the end, in
  formation (coherent 0.83); s3 stacks 5.3 on one objective and leaves
  the far pair empty in 85–100% of episodes; s2 holds 1.5. The near
  column is empty in 78–100% of episodes on s1 and s2 — the opposite
  allocation to every 128-round run, and not a better one.
- **The walk-off probe** (n=20, displacement decisions of a body inside an
  objective): stays **0.00 / 0.02 / 0.01**, leaves the objective **0.81 /
  0.73 / 0.65** (the 128-round runs 0.60–0.62; the bar stays 0.47), step
  pay for leaving −0.003 to −0.005, for staying 0.
- **Greedy is far below sampled**: greedy − sampled **−25.4 / −5.7 /
  −10.4 vp**, sampled play holding 3.2 / 3.2 / 3.9 objectives against
  greedy's 0.1 / 1.5 / 2.0 — the displacement head stayed diffuse
  (2.7–4.0 nats on s1 and s3 against 1.0–2.6 at 128 rounds per update),
  so the argmax is a corner of a policy that, sampled, holds three to
  four objectives. The panel otherwise: explained variance 0.49–0.71,
  clip fraction 0.09–0.25, `ratio_p99` 1.3–1.9 (a different quantity at
  this rollout size), s1's episode reward falling 1.9 → 1.1 by quarter.

**What the answer says, as written.** FAIL with the same census: the
regime is not it. What the per-model trainer lacks on this half-step is
not outcomes per update, not the horizon, not the terminal payment's
size or shape, not the credit's attribution, and not perception. The
one measured defect that every arm shares is the walk-off, and the probe
puts a number under it: **a body standing on an objective is paid the
same for staying as for leaving, to within a few thousandths, and never
learns to stand still.** The next arm on this half-step pays a body for
standing on an objective on its own decision step — which the retimer's
`actor` credit path can land, and which must not be the mean-credited
hold term the A3 screen (S1) and A5e read as null.

---

<a id="2026-09-20-curriculum-A5-points-staying-preregistration"></a>

## Pre-registration: the staying arm on A5-points — pay a body for keeping an objective on its own step

Written 2026-09-20 17:16, **before any training round on either config**,
on branch `feature/per-model-actor-credit` (PR #383). Parent question #340.
Set as a goal by Sash 2026-09-20 17:12: "build and run the staying arm on
A5-points".

### The question

Every per-model arm on the half-step A5-points (six squads of three over
five objectives; the bar `squad_march_take` 1.000 in 6.71 turns) shares
one measured defect, isolated on 2026-09-20 after the horizon, the
terminal payment, the credit, the update regime and perception were each
ruled out: **a body standing on an objective is paid the same for staying
as for leaving, to within a few thousandths, and stands still on 0–2% of
its decisions** (the walk-off probe; the bar 47%). The travel term is zero
inside an objective, the coverage term is a broadcast mean at the close,
the bonus is terminal, and `objective_hold` — which prices the standing —
is a STATE term on this facade, paid at the close as the army mean, so the
body that stayed and the body that left are paid alike (S1 on A3 and A5e on
A5 read it as null for that reason). The backward start showed the
consequence: handed four objectives, the policy keeps two or three.

**This arm pays the mover, on its own decision step, for ending inside an
objective**, with the objective's pot split among its occupants so
spreading raises income and stacking does not multiply it.

### The build (default off; goldens bit-identical; 143 reward tests pass)

- `objective_stay` (`reward/calculators/objective_stay.py`,
  `ObjectiveStayCalculator`, `crowding_exponent` default 1.0): for a model
  inside an objective, `1 / occupants ** exponent`; outside, 0. Registered;
  classed `potential_action` in the retimer so it is computed for the ACTOR
  on its own step from the board after its move; documented in
  `docs/reward-phases.md`; tested in `tests/test_per_model_reward_timing.py`
  (the mover that ends inside is paid over the alive count, the one that
  leaves is not; the pot splits; the exponent is validated).
- Measured under the new config with the walk-off probe (n=10): a body on
  an objective is now paid **+0.005 to +0.026** per step for staying inside
  (stand still or move within) against **+0.0005 to +0.009** for walking
  out, a gap the size of the travel signal (+0.013 to +0.018 for a step
  toward an objective from outside); before the term the gap was under
  0.005 in the wrong direction.

### The arms

| arm | config | the one change | budget | tag |
|---|---|---|---|---|
| **S** | `configs/experiments/curriculum/a5_points_stay.yaml` | `objective_stay` weight **0.5** | 122,880 rounds, the original recipe (128 per update, `--ent-coef 0.003`, `gamma` 0.9, `mean` credit, eval and checkpoint every 512, recording and video on), from scratch | `a5st` |
| **S-low** | `configs/experiments/curriculum/a5_points_stay_low.yaml` | `objective_stay` weight **0.15** | the same | `a5stl` |

Three seeds each, six trainers. The bar's episode reward: original 4.38,
S 4.89 (the stay term 0.51 of it, all in the first six turns), S-low 4.53.
Comparator: the original A5-points from scratch (`uagmul66` / `wedjjozy` /
`ici5py46`), 0.030 / 0.060 / 0.330 at 122,880, 0.040 / 0.140 / 0.090 at
40,960, 0.220 / 0.220 / 0.110 at 81,920. Read greedy at n=100 on 700000+ at
40,960 / 81,920 / 122,880 with the by-turn census and the walk-off probe
(the stay share on an objective is the mechanism readout), sampled beside
greedy at the end.

### Criteria (each arm on its own)

- **PASS:** success ≥ 0.95 on 3/3 at 122,880, no in-run dip below 0.80
  after the first rolling pass.
- **MOVES:** ahead of the original seed for seed by more than two binomial
  SE on 3/3 at 122,880.
- **FAIL:** neither.
- **Mechanism readout, reported whatever the verdict:** the walk-off
  probe's stay share and leave share on an objective at each read (the
  original's 0.00 / 0.60–0.62; the bar 0.47 / 0.33); bodies on objectives
  by turn (does the arrival at turn 3–5 hold to the end); held; max stack
  (the pot is meant to stop stacking — a stack over 3.5 is the term paying
  for crowding); the panel; sampled beside greedy.

**What each answer says.** PASS: the wall was the unpaid stay, and A5 gets
the same term. MOVES with the stay share up and the walk-off gone from the
census but the fifth objective still empty: staying is fixed and arriving
at the last objective is the residual — the observation that names the
empty objective, or the warm start plus this term, is next. FAIL with the
stay share still near zero: the term is not reaching the decision (check
the breakdown: is `objective_stay` earned in training at the expected
~0.5 per episode?) or its magnitude is below what the head resolves — the
weight, not the mechanism, is the next arm. FAIL with the stay share up
and stacking up (max stack > 3.5): the pot split is not enough against
the travel term's fallback pull, and the crowding exponent is the lever.

### What I expect (a guess, written so it can be wrong)

S: the stay share on an objective rises from 0 to 0.2–0.5 by 40,960 and
the walk-off leaves the census; success 0.3–0.6 at 122,880 with held
3.8–4.4, MOVES on 3/3, PASS on none, the fifth objective the residual.
S-low: the same direction at half the size. If the stay share stays at
zero on both, the head is not resolving a per-step signal of this size
and the wall is in the policy's representation of "do nothing".

### Amendment 1 — written 2026-09-20 21:50, both arms read at 40,960 / 81,920 / 122,880

Six trainers launched 17:18, exited 21:24–21:33 at 122,880 rounds, no
errors (Wandb S `9d7o26ln` / `zsl9nuvs` / `am3xbsr2`, S-low `oxu59s8s` /
`3b8235au` / `iu61hiqg`). Greedy at n=100 on seeds 700000+; the census at
n=100 at the end (n=20 at the interim reads); the walk-off probe at n=10;
sampled beside greedy at the end, paired. Comparator: the original
A5-points from scratch at matched rounds.

#### Success, seed for seed

| rounds | S (0.5) | S-low (0.15) | original |
|---|---|---|---|
| 40,960 | 0.040 / 0.010 / 0.020 | 0.020 / 0.050 / 0.060 | 0.040 / 0.140 / 0.090 |
| 81,920 | 0.170 / 0.150 / 0.190 | 0.070 / 0.140 / 0.140 | 0.220 / 0.220 / 0.110 |
| **122,880** | **0.210 / 0.190 / 0.120** | **0.230 / 0.110 / 0.110** | 0.030 / 0.060 / 0.330 |

Held at the end: S 3.36 / 3.43 / 3.26, S-low 3.37 / 3.13 / 3.05 of 5;
turns 9.4–9.8 (bar 6.71); coherent 0.14–0.27; vp 65–78 against the bar's
63.

**PASS: no** — no seed of either arm reaches 0.95. **MOVES: no** — S is
ahead of the original by more than two binomial SE on s1 (+0.18, 4.1 SE)
and s2 (+0.13, 2.8 SE) and **behind on s3 (−0.21, 3.7 SE)**; S-low is
ahead on s1 (+0.20, 4.4 SE), level on s2 (+0.05, 1.3 SE) and behind on s3
(−0.22, 3.9 SE). At matched rounds both arms are behind the original on
two seeds at 40,960 and on two seeds at 81,920. Pooled over seeds the
arms read 0.17 and 0.15 against the original's 0.14 — inside the
original's own seed spread, which one seed (s3 at 0.33) carries.

**Verdict: FAIL as pre-registered, both arms.**

#### The mechanism readout

| read | S stay / leave | S-low stay / leave |
|---|---|---|
| 40,960 | 0.00 / 0.69 · 0.00 / 0.50 · 0.00 / 0.58 | 0.00 / 0.70 · 0.00 / 0.73 · 0.00 / 0.61 |
| 81,920 | 0.01 / 0.54 · 0.00 / 0.60 · 0.00 / 0.60 | 0.00 / 0.61 · 0.00 / 0.64 · 0.00 / 0.48 |
| 122,880 | 0.00 / 0.65 · 0.00 / 0.67 · 0.00 / 0.48 | 0.00 / 0.66 · 0.00 / 0.60 · 0.00 / 0.58 |

The original: 0.00 / 0.60–0.62; the bar 0.47 / 0.33. **The stay share
never left zero on any seed of either arm**, against the written
expectation of 0.2–0.5 by 40,960. The payment is there at play: under S a
body that ends its step inside an objective is paid +0.013 to +0.018 and
one that walks out −0.003 to +0.000 (S-low +0.005 to +0.007 against
−0.002 to −0.006), a gap the size of the travel signal (+0.010 for a step
toward an objective from outside). ⚠ The readout I named was the wrong
number: the term pays for ENDING inside whether the body stood still or
moved within, so the mechanism is the **leave share**, not the stationary
share. It moved from the original's 0.60–0.81 to 0.48–0.67 at the end —
one to two tenths, on the seeds that moved at all — and the stationary
action was never learned on any seed at either weight.

The by-turn census at the end (n=100), bodies on objectives after turns
3 → 5 → 7 → end, held the same way, max stack:

| seed | S | S-low |
|---|---|---|
| s1 | 3.9 → 5.4 → 5.5 → 5.5 · held 1.9 → 3.4 · stack 2.4 | 2.4 → 5.4 → 5.5 → 6.3 · held 1.1 → 3.4 · stack 3.2 |
| s2 | 4.8 → 5.3 → 5.9 → 6.0 · held 1.8 → 3.5 · stack 2.7 | 5.6 → 5.5 → 5.5 → 5.4 · held 2.1 → 3.1 · stack 2.6 |
| s3 | 4.1 → 5.6 → 6.6 → 6.1 · held 2.0 → 3.3 · stack 3.0 | 5.8 → 4.9 → 5.6 → 5.8 · held 1.9 → 3.0 · stack 3.0 |

No aggregate walk-off on S: bodies on objectives rise from turn 3 to the
end on every seed, and the stack stays under the 3.5 mark on all six
runs (the pot split is doing what it was built to do). But the count on
objectives is a third of the army and a body on an objective still leaves
it on half to two thirds of its decisions: the aggregate holds by churn —
bodies leaving and others arriving — not by keeping. The empty objective
is a different one each episode on every seed (the failures' empty share
is spread over all five columns; no column above 0.81). The panel is
normal: explained variance 0.66–0.75 (the original 0.63–0.73), clip
fraction 0.35–0.38 and `ratio_p99` 2.1–2.2 at the end (the original
0.30–0.40), displacement entropy falling to 1.7–1.9 nats; in-run success
in the last quarter 13 / 21 / 14% (S) and 15 / 12 / 19% (S-low) against
the original's 9–20%. Sampled beside greedy, paired: S −2.9 / +0.1 / +4.0
vp, S-low +6.3 / +5.9 / +5.3 (greedy ahead; not diffuse). The trainer's
component log does not carry `objective_stay`, so the in-training income
is unverified from the run; the probe verifies the payment at play.

#### What this says

The pre-registration's FAIL clause named two readings for "stay share
still near zero": the term not reaching the decision, or a magnitude
below what the head resolves, with the weight as the next arm. The two
weights were the dose-response inside this arm: **a 3.3× change in the
weight changed neither the leave share nor the success**, so the weight
is not the next arm. The per-decision signal is present, of the travel
signal's size, at two magnitudes, for the whole run, and the policy does
not resolve it into keeping an objective. That closes the sixth
reward-side lever on this half-step (horizon, terminal shape, credit,
regime, and now a per-decision staying payment at two weights; perception
was desk-checked). What every per-model policy on the half-step still
lacks is a commitment a body can hold across steps and a channel by which
one squad's intent reaches another: the design for that is #384 (a
persistent, visible per-unit commitment to a board token), and it is the
next arm. **Do not run a seventh reward arm on this half-step.**

Files: reads, censuses and probes in the session drafts
(`read-a5st{,l}-{40960,81920,final}.txt`, `census-…`, `walkoff-…`,
`evalmode-stay-final.txt`).

---

<a id="2026-09-20-curriculum-a5-points-whose-wall-it-is"></a>

## Curriculum A5-points, three arms in one night: the wall is the per-model trainer's, a warm start carries four objectives of five, and a start with four objectives given is walked off

**Verdict first.** Three arms on the half-step A5-points (six squads of
three over five objectives; the bar `squad_march_take` 1.000 in 6.71
turns), read greedy at n=100 on 700000+ from deployment:

| arm | what | verdict | success | held of 5 | report of the read |
|---|---|---|---|---|---|
| **CTL** | the whole-army trainer (`train.py`) from scratch, 60 epochs then the once-only extension to 120 (≈245k rounds) | **FAIL on the letter by two seeds — the best read this half-step has had from any trainer** | 0.980 / 0.620 / 0.870 at 60; **0.910 / 0.810 / 0.950** at 120 | 4.98 / 4.61 / 4.86; 4.91 / 4.80 / 4.95 | 93–97% of bodies on objectives, half a round to a round behind the bar |
| **W** | the per-model trainer warm-started from A4x seed-for-seed (the three-objective pass), 122,880 rounds then the same extension to 245,760 | **FAIL on the letter; ahead of the original on 3/3 at 2× the cap; s3 the best per-model read on the half-step** | 0.460 / 0.260 / 0.350 at 122,880; **0.340 / 0.170 / 0.600** at 245,760 | 3.96 / 3.53 / 3.62; 3.59 / 3.52 / 4.29 | no walk-off; s3 8 of 18 bodies on objectives by turn 7, 8.1 turns |
| **BS** | the per-model trainer from scratch under the backward start curriculum (episodes begin with four squads on four objectives; the level walks down as the level's episodes succeed), 122,880 rounds | **FAIL — the level never left four** | 0.000 / 0.030 / 0.030 | 2.08 / 3.02 / 2.07 | with four objectives given the policy keeps 2.0–3.1 of them and reaches the fifth 63–100% of the time |

The original per-model run from scratch reads 0.030 / 0.060 / 0.330 at
122,880; the four optimiser and reward arms before this night read
0.00–0.18. **The whole-army trainer solves the five-objective spread from
scratch to within a seed of the letter; the per-model trainer does not
approach it from scratch under any setting tried, and its best — 0.60 on
one seed — comes from carrying the three-objective policy in.** The
backward start isolated what the per-model policy does with an
objective it is standing on: it leaves. On level-four episodes the bar
keeps 3.9 of its 4 placed squads on their objectives and succeeds every
time; the arm keeps 2.0–3.1 and succeeds a third to a half of the time,
so the schedule never stepped down. That walk-off is the same event every
census on this half-step has recorded, in its smallest form, and it is a
property of the policy under this reward rather than of the start.

### Provenance

| field | value |
|---|---|
| date | 2026-09-19 22:03 – 2026-09-20 04:35: CTL 22:03–22:14 (60) and 22:17–22:30 (120); W 22:03–02:10 and resumed 02:13–04:15; BS 22:17–02:05 (from a worktree); reads as each exited |
| GPU / no-GPU | GPU (RTX 4090): three whole-army and up to six per-model trainers side by side |
| seeds | 1 / 2 / 3 per arm; W from `per-model-a4-2026-09-15-07-08-53-s{1,2,3}a4/pm-00245760.pt` seed-for-seed (weights only) |
| n | 100 at seed base 700000 (every read, the census, greedy against sampled); in-run 30 at 500000 every 512 rounds (per-model) and per epoch (whole-army); the level-4 census n=30 |
| config | `configs/experiments/curriculum/a5_points.yaml`, unchanged for all three; unrefereed by design; bridge identical 2026-09-18 |
| decode | none on the per-model facade; the whole-army control scored as every A-rung control (argmax, `measure-rung`) |
| paired | per episode against the bar on identical seeds (turns); per seed against the original A5-points on the same seeds |
| comparator | the original A5-points (`uagmul66` / `wedjjozy` / `ici5py46`): 0.030 / 0.060 / 0.330 at 122,880, 0.040 / 0.140 / 0.090 at 40,960, 0.220 / 0.220 / 0.110 at 81,920; the bar 1.000, 6.71 turns, held 5.00 |
| opponent | none |
| budget | CTL 60 epochs (`--n-eval-episodes 30 --record-during-training`, defaults otherwise, `ent_coef` 0.03) + once to 120 via `--resume-ckpt-path`; W and BS 122,880 rounds at 128 per update, `--ent-coef 0.003`, `gamma` 0.9, `mean` credit, eval and checkpoint every 512, recording and video on; W once to 245,760 via `--resume-from` under the cap rule; BS `--backward-start 4 --backward-start-share 0.75 --backward-start-advance 0.8 --backward-start-window 8` |
| code revision | pre-registrations `c9f23d4` (CTL, W) and `8572189` (BS, with the build); runs on `c9f23d4` (CTL, W) and `8572189` (BS); merged at `2cebc65`; amendments `a4c61cb` / `81f56a7` / `852e649` (CTL, W) and `1b3596c` (BS); branch `feature/per-model-actor-credit`, PR #383 |
| checkpoint | CTL `checkpoints/ppo-transformer-curriculum_a5_points-m18-opp0-obj5-b60x44-ph1-2026-09-19-22-03-41-s*a5pts-ctl{,-x}/last.ckpt`; W `checkpoints/per_model/per-model-a5_points-2026-09-19-22-03-42-s*a5pw/last.pt` (resumed in place; `pm-00122880.pt` for the cap read); BS `…-2026-09-19-22-17-{29,30}-s*a5bs/last.pt`; every per-model checkpoint with a recorded greedy episode and one MP4 per 10,240 rounds on the run |
| coherency | greedy at 700000+: CTL 0.57–0.66; W 0.24–0.29 (245,760); BS 0.08–0.47; the bar 0.832 |
| Wandb | `curriculum-a5`: CTL `pic9zjka` / `ccheoveo` / `oct4luxd` and `ryk2bfuc` / `0wlaqa3l` / `xwsqlrtn`; W `8clgcy59` / `klv24tu2` / `8h4eq4lj` and `xw154bst` / `8dcfq1xa` / `kwdxlh12`; BS `inhpvik7` / `fhbs37ln` / `feg4n4uf` |
| pre-registration | `reports/2026-09-19-curriculum-a5-points.md#2026-09-19-curriculum-A5-points-control-and-warm-start-preregistration` (+3 amendments) · `reports/2026-09-19-curriculum-a5-points.md#2026-09-19-curriculum-A5-points-backward-start-preregistration` (+1) — every table is in them |

### The read

#### The whole-army trainer is two to five times ahead at equal rounds

At 60 epochs (≈123k rounds, the per-model cap) the control reads 0.98 /
0.62 / 0.87 with 4.6–5.0 objectives held and 94–95% of its bodies on
objectives; the per-model trainer from scratch at the same rounds reads
0.03–0.33 with a third of its bodies on objectives, under five settings.
At 120 epochs the control reads 0.91 / 0.81 / 0.95 — one seed drifted
down over the extension, as the whole-army A2 and A5 controls did — and
the per-model arm's best at the same rounds is W's 0.60 on one seed. It
is a FAIL on the rung's letter (two seeds under 0.95) and a finding on
#283's question: on the five-objective spread the step-of-one-model
trainer is the one with the wall.

#### The warm start carries four objectives and reward adds the fifth slowly

W is where the original ends by 20,480 rounds (0.03 / 0.12 / 0.18, held
2.9–3.4, six or seven bodies on objectives from turn 3 with no walk-off),
climbs to 0.46 / 0.26 / 0.35 at the cap and, extended, to 0.34 / 0.17 /
0.60 — ahead of the original's final row on every seed, past two SE, and
s3 at 0.60 with held 4.29 the highest per-model read on the half-step.
Two seeds fell back over the extension while s3 kept climbing (in-run
40 → 58% by quarter, a rolling 63%). The census at the end is the first
on this half-step with no walk-off on any seed and misses spread across
all five objectives; greedy and sampled play agree within ±6 vp. T1's
shape (ahead at 20k, flat after) did not repeat; the difference between
T1 and W is one objective (six against five), and the reading is that
the A4x policy's three-objective spread carries to four here and reward
finds the fifth on one seed of three.

#### The backward start: given four, it keeps two or three

The level never stepped down on any seed: the level's own success
plateaued at 0.25–0.35 by quarter against the 0.8 the schedule needed.
Under the final checkpoints on level-four starts (n=30): success 0.53 /
0.33 / 0.50; the placed squads still on their objectives at turn 3 2.93 /
3.38 / 3.93 and at the end **2.70 / 1.97 / 3.13** of 4; the free squads
reaching the empty objective 0.83 / 1.00 / 0.63; the bar on the same
starts 1.00, keeping 3.9. Both halves of the last step are half-learned
and the larger loss is the first: **given an objective, the policy walks
off it.** Its critic was healthy (explained variance 0.55–0.68), so this
is not the reward-shape arms' failure. From deployment the arm reads
0.00 / 0.03 / 0.03, behind the original at every read, and its census
has the half-step's walk-off with a new variant on s1 — 9.6 of 18 bodies
on objectives by turn 5, the highest arrival on the half-step, stacked on
the near column. The schedule (share 0.75, advance 0.8 over eight
rollouts) was never exercised; a lower bar would have walked the level
onto a policy that holds a third of its starts, and whether that helps
is untested.

### What this says

- **On the five-objective spread the wall is the per-model trainer's.**
  Same config, same rounds: the whole-army trainer 0.81–0.95 with 4.8–4.95
  held; the per-model trainer from scratch 0.00–0.33 under five settings,
  and 0.60 at best from a warm start at twice the cap. #283's premise —
  that a step of one model beats a step of the whole army — has its first
  clear negative row on the A rungs.
- **A warm start from the objective rung below is the one per-model lever
  that moves a spread half-step**, as S3 said on A3 and T1 did not on A5.
  Read a transfer arm at matched rounds and at the extended budget; the
  per-seed spread (0.17–0.60) is wider than any from-scratch arm's and one
  seed of three carries the result.
- **The per-model policy under this reward does not keep an objective it
  is standing on.** Isolated by the backward start: four squads placed,
  one to two leave. The walk-off in every census on this half-step is
  this event, not a failure to arrive. Nothing per-model pays a body for
  standing (the travel term is zero inside, coverage is a broadcast mean,
  the bonus terminal); the A3 speed screen and A5e measured a hold term
  as null, under the mean and under actor credit, on other rungs, so the
  lever is not obviously a hold term either — but the next per-model arm
  on this half-step should be aimed at staying, not at arriving.
- **A start curriculum needs its advance bar calibrated on the policy's
  own level success.** 0.8 was never approached; the schedule never
  stepped and so measured only level four. Pre-register the bar against
  the bar's own success on the same starts (1.00 here) and the policy's
  first read at the top level.
- **Process.** Three seeds launched in one loop can land in run
  directories a second apart; a chain that derives every seed's directory
  from the first seed's waits forever (caught after an hour). Resolve each
  seed's directory separately. And gate a final read on the trainer
  exiting, never on `last.pt` existing.

### Where this lands

1. This report.
2. `reports/README.md` index row.
3. `CLAUDE.md`: three ladder rows (the control, the warm start, the
   backward start) and a rule bullet.
4. Live docs: the backward start's flags are documented in
   `wargame_rl/wargame/model/CLAUDE.md` and its placement in
   `wargame_rl/wargame/envs/CLAUDE.md` (`2cebc65`); no config added.

### Addendum — 2026-09-20 13:00: what exactly makes it hard, three checks and one more arm

Asked after the control's read: what in the per-model trainer's design
makes five objectives hard? Three candidates were checked
(`reports/2026-09-19-curriculum-a5-points.md#2026-09-20-curriculum-A5-points-rounds-per-update-preregistration`
and its amendment carry the tables):

1. **Perception — ruled out by inspection.** The objective token carries
   our count on it (`row[2]`) and the enemy's; every body reads its
   offset to every objective as a relation. An empty objective is
   observable to the body that should walk to it.
2. **Outcomes per update — ruled out by an arm.** The whole-army trainer
   updates every 2,048 rounds (~200 episodes on this one-phase config),
   the per-model trainer every 128 (~13). At 2,048 rounds per update the
   per-model trainer from scratch reads **0.000 / 0.010 / 0.000** at
   122,880, behind the original at every read, with the displacement head
   diffuse and the walk-off at its largest (11.3 of 18 bodies on
   objectives after turn 3 and 0.1 at the end on one seed).
3. **The reward's silence on staying — measured, and shared by every
   arm.** A probe over the displacement decisions of a body standing
   inside an objective, on six trained per-model policies (the warm
   start's, the original's, the regime arm's): it stands still on
   **0.00–0.02** of them and walks out of the objective on **0.60–0.81**;
   the scripted bar stands still on 0.47. The step reward for leaving is
   −0.002 to −0.005 and for staying 0.000, against +0.003 to +0.014 for a
   step toward an objective from outside. Standing on an objective is the
   one state in which every action is paid alike, and the policy never
   learns the stand-still action at all. The whole-army trainer trains on
   the same terms but pays each model its own travel term in full every
   step and broadcasts coverage in full; the per-model retimer divides
   the action term by the army and pays coverage once at the close, so
   the per-step magnitudes a body sees are an order of magnitude smaller
   and, on an objective, indistinguishable.

The wall on the five-objective half-step is therefore neither the
horizon, the terminal payment, the credit's attribution, the update
regime nor perception. It is that nothing pays a body for keeping an
objective on its own step, and the backward start showed what follows:
handed four objectives, the policy keeps two or three. The next arm pays
a body for standing on an objective on its own decision step, through
the `actor` credit path, with a mechanism that is not the mean-credited
hold term S1 and A5e read as null.

### Addendum 2 — 2026-09-20 21:50: the staying arm

The arm the first addendum named ran the same evening: `objective_stay`,
a per-decision pot for a body that ends its step inside an objective,
paid to the mover on its own step, at weight 0.5 (S) and 0.15 (S-low),
three seeds each, the original recipe otherwise
(`reports/2026-09-19-curriculum-a5-points.md#2026-09-20-curriculum-A5-points-staying-preregistration`
and its amendment carry the tables). **FAIL as pre-registered, both
arms**: S 0.210 / 0.190 / 0.120 and S-low 0.230 / 0.110 / 0.110 at
122,880 against the original's 0.030 / 0.060 / 0.330 — ahead on two
seeds and behind on the third for S, ahead on one for S-low, and behind
the original on two seeds at both interim reads. The mechanism readout
did not move: a body on an objective is now paid +0.013 to +0.018 for
ending inside against −0.003 for leaving (S), and it stands still on
0.00 of its decisions there and leaves on 0.48–0.67 (the original
0.60–0.81, the bar leaves on 0.33). The stack stays under 3.5 on every
run and the count on objectives rises to the end, so the pot split works
and the aggregate walk-off is gone — by churn, not by keeping. A 3.3×
change of weight changed nothing, so the magnitude is not the lever.
Six reward-side levers are now closed on this half-step. What is left,
and what the record's evidence points at, is that no body can commit to
anything across steps and no squad can tell another what it intends —
#384 designs that layer.
