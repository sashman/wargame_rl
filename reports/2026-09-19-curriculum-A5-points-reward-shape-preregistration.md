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
