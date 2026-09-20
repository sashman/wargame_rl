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

## Amendment 1 — written 2026-09-20 21:50, both arms read at 40,960 / 81,920 / 122,880

Six trainers launched 17:18, exited 21:24–21:33 at 122,880 rounds, no
errors (Wandb S `9d7o26ln` / `zsl9nuvs` / `am3xbsr2`, S-low `oxu59s8s` /
`3b8235au` / `iu61hiqg`). Greedy at n=100 on seeds 700000+; the census at
n=100 at the end (n=20 at the interim reads); the walk-off probe at n=10;
sampled beside greedy at the end, paired. Comparator: the original
A5-points from scratch at matched rounds.

### Success, seed for seed

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

### The mechanism readout

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

### What this says

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
