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

## Amendment 1 — written 2026-09-19 14:10, the arm read at 40,960 / 81,920 / 122,880

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
