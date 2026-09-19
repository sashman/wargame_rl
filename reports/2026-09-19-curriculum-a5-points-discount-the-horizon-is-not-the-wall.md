# Curriculum A5-points at gamma 0.99: the horizon is not the wall — the critic sees the longer game and the policy does not move

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

## Provenance

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
| pre-registration | `reports/2026-09-19-curriculum-A5-points-discount-preregistration.md` (+1 amendment) |

## The read

### The final table

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

### Three reads, and the original's own curve

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

### The census is A5-points'

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

### Greedy against sampled

`just measure-per-model-eval-mode` at n=100 on 700000+, paired:

| seed | greedy vp | sampled vp | greedy − sampled | greedy held | sampled held |
|---|---|---|---|---|---|
| s1 | 71.5 ± 1.5 | 65.1 ± 1.4 | **+6.4 ± 1.6** | 2.71 | 3.56 |
| s2 | 70.0 ± 1.2 | 62.6 ± 1.1 | **+7.4 ± 1.5** | 3.58 | 3.37 |
| s3 | 66.5 ± 1.4 | 60.5 ± 1.2 | **+6.0 ± 1.6** | 2.99 | 3.45 |

Six to seven vp apart, where A5i's greedy and sampled agreed within
1.2: a diffuse policy, as A5-points' was and E1's per-model rows were.

### The critic saw the horizon; the policy did not

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

## What this says

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

## Where this lands

1. This report.
2. `reports/README.md` index row.
3. `CLAUDE.md`: a ladder row for the discount arm and a rule bullet
   (the horizon is not the wall; read a critic improvement beside the
   policy; read the comparator at matched rounds).
4. Live doc: n/a — no config added, no doctrine entry priced. The
   in-training recorder (`--record-every-rounds`, `f7539d2`) merged
   after these runs exited and is documented in
   `wargame_rl/wargame/model/CLAUDE.md`.
