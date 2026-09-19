# Pre-registration: the reward-shape arms on A5-points — is the conjunction unpaid, or unfindable?

Written 2026-09-19 15:05, **before any training round on either config**,
on branch `feature/per-model-actor-credit` (PR #383). Parent question
#340; the second optimiser-side pass on the half-step after the discount
arm (FAIL, `c24aa10`: the horizon is not the wall). Set as a goal by Sash
2026-09-19 14:50: "run the reward-shape arm on A5-points".

## The question

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

## The changes (built today, default-off, goldens bit-identical)

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

## The recipe

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

## Criteria (each arm on its own)

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

## What I expect (a guess, written so it can be wrong)

R1 moves on one or two seeds and passes on none: success 0.1–0.4, held
3.3–3.8, the walk-off reduced because a late hold is now worth
holding; the terminal bonus share of episode reward rises from ~2% to
10–30% on succeeding episodes. R2 moves on 3/3 and ahead of R1 on 2/3:
held 3.8–4.3 as the fourth point is paid, success 0.3–0.6, the fifth
point still the residual, PASS on none. If R2 passes I am wrong in the
useful direction; if neither moves, the terminal payment was never the
signal the policy was missing and the search account stands.

## The bar's reward stream on each config, and the bridge

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

## Amendment 1 — written 2026-09-19 19:35, both arms read at 40,960 / 81,920 / 122,880

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
