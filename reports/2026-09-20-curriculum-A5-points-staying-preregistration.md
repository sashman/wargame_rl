# Pre-registration: the staying arm on A5-points — pay a body for keeping an objective on its own step

Written 2026-09-20 17:16, **before any training round on either config**,
on branch `feature/per-model-actor-credit` (PR #383). Parent question #340.
Set as a goal by Sash 2026-09-20 17:12: "build and run the staying arm on
A5-points".

## The question

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

## The build (default off; goldens bit-identical; 143 reward tests pass)

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

## The arms

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

## Criteria (each arm on its own)

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

## What I expect (a guess, written so it can be wrong)

S: the stay share on an objective rises from 0 to 0.2–0.5 by 40,960 and
the walk-off leaves the census; success 0.3–0.6 at 122,880 with held
3.8–4.4, MOVES on 3/3, PASS on none, the fifth objective the residual.
S-low: the same direction at half the size. If the stay share stays at
zero on both, the head is not resolving a per-step signal of this size
and the wall is in the policy's representation of "do nothing".
