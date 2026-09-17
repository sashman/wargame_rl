# Pre-registration: curriculum rung C3b — C3 with success read at the end of the game, so a dash has to hold the point

Written 2026-09-17 09:50, **before any training number on this arm
exists**, on branch `feature/curriculum-c3` (PR #371, beside C3's own
arm). Issue #372, parent question #340, rung **C3** — a second arm
on the rung, with one change to the scenario.

## The one change

`configs/experiments/curriculum/c3b.yaml` is `c3.yaml` with
**`terminate_on_success: false`**: the episode runs its twelve rounds
and `all_objectives_occupied` is judged on the final board, not the
instant every point first has a body on it. Everything else — the armed
squad at range 18 and four attacks, the blockers of four wounds and
eight attacks at range 12, the points, the reward — is C3's.

## Why

C3's per-model arm, warm-started from C2, read 0.660 / 0.920 / 0.830 at
245,760 with the first kill preceding the first unarmed arrival in only
0.67 / 0.71 / 0.62 of episodes: it found the **dash** — put a last body
onto the blockers' point the moment the other three are held and let
the success termination end the game before the blockers fire again,
with the blockers alive in four episodes of five and a third of its
bodies dead. The rules' own game punishes that: a body that dashes in
has to hold the point to the last round under eight attacks a blocker.
The from-scratch companion, with no rush-in habit to carry, learned the
order (0.95–1.00) and not the walk (0.74 / 0.74 / 0.09). The rung was
built to test sequencing, and instant termination let a policy skip it.

## The bar, measured first

Seeds 700000+ at n=100, **both facades identical on every shared
field**: `scripted_escort` success **0.990**, `held` 3.97, `alive`
0.963, `on_obj` 0.993, vp +63.5 ± 2.0 (the game now runs to the end, so
vp accrues for twelve rounds), coherent 0.858, turns 24 (no early end);
first kill at turn 6.0, blockers wiped at 11.0, first unarmed body on
their point at 12.9, the kill before the arrival in 100%.
`squad_march_take` (must fail): **0.280**, `held` 2.33, `alive` 0.507 —
the dash and the walk-in are both punished now.

## The arms

As C3's, on `c3b.yaml`: the whole-army control (3 seeds, 60 epochs,
extended once to 120 if still rising), the per-model arm warm-started
from **C2's** `last.pt` seed for seed (the C-rung rule; the C3 arm's
checkpoints carry the dash, so they are not the start), and the
from-scratch companion; 122,880 rounds at 128 per update, `--ent-coef
0.003`, in-run eval every 512 rounds at n=30 on 500000+, resumed in
place to 245,760 if the symmetric cap applies. Wandb `curriculum-c3b`,
tags `c3b-ctl` / `c3b` / `c3bs`.

## Criteria

Read greedy, no decode, n=100 on 700000+, from `last.pt` at the budget.
C3's, unchanged:

- **PASS:** success ≥ **0.90** on all three seeds of the warm-started
  arm **and** the first kill precedes the first unarmed arrival on the
  blockers' point in ≥ **0.90** of episodes on every seed, and no in-run
  dip below 0.75 after the first rolling pass.
- **FAIL:** any seed below 0.90, or the ordering clause missed on any
  seed; the census says which.
- **NULL (scenario):** only if the script fails its own criterion. It
  reads 0.990 / ordering 1.00.
- **Pass with a defect:** the bound holds but the health panel is red
  over the last quarter.
- **Readouts:** `alive` (bar 0.963); vp paired; `held`; `on_obj`;
  coherency greedy and sampled; rounds to rolling 50 / 80 / 90%; the
  panel; the control and the companion beside every row; C3's rows
  beside every row, since the two arms differ by one flag.

Power: binomial SE 0.030 at p=0.90, n=100, on both clauses.

## What I expect (a guess, written so it can be wrong)

The warm-started arm unlearns the dash, because a dashed body now dies
on the point, and finds the order slowly: under 0.90 on two seeds at
122,880, one or two seeds at the mark at 245,760, ordering above 0.90
where success is. The companion reads about where C3's did on success
(0.7) with the order intact. The control passes on no seed.
