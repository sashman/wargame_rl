# Curriculum rung D1: the escort cloned into the set network — the clone holds the plan and misses the mark; FAIL on the letter, with a defect in the match clause

**Verdict first.** On the supervised rung (#340, arm #373) — the
escort's decisions on C3b's scenario copied into the set network by
maximum likelihood, 300 demonstration episodes × 40 epochs, two clones
from identical data differing only in the fit's seed — the clones read
held-out `joint` match ****0.441 / 0.440** (mark 0.95)**, greedy success ****0.830 / 0.850**** at n=100
against the escort's 0.990, and the kill before the unarmed arrival in
****0.99 / 1.00**** of episodes against the escort's 1.00. ****FAIL as pre-registered** on the match clause and the success clause (bound 0.96); PASS on the ordering clause.**
The two clones agree to a thousandth on every number, so nothing here is the fit's seed. What the numbers say is two things the pre-registration did not separate. The joint match counts a factor no clone can learn: which unit the escort opens next is not a function of the observation — the clones sit at chance on held-out opening decisions (0.24 for four units) and at 0.60 on their own training episodes. And the displacement head over-fits 240 episodes (0.95 on the training episodes, 0.61 held-out), a data-size question, not a representation one: the clone tracks the escort's board turn for turn, fires first in 99% of episodes, keeps 90% of its bodies, and ends a body short on one point in one episode of six. Reward alone reached 0.20 on this scenario; imitation reaches 0.85 from 240 games. D1b (#375) runs the one change the reading names — four times the demonstrations — with the match clause corrected.

## Provenance

| field | value |
|---|---|
| date | 2026-09-17, clones launched 14:55 on the CPU, read 15:12–15:14 |
| GPU / no-GPU | no GPU |
| seeds | fit seeds 0 and 1 on identical demonstrations (800000+, 300 episodes, the last 60 held out) — paired on data, not on init |
| n | 100 at seed base 700000 (final, the ordering census); 100 at 900000 (greedy against sampled); 20 at 700000 (by-turn census); ~9,144 held-out decisions for the match |
| config | `configs/experiments/curriculum/c3b.yaml` — success on the final board after twelve rounds; unrefereed by design |
| decode | none on the per-model facade |
| paired | per episode against `scripted_escort` on identical seeds |
| comparator | `scripted_escort`, both facades identical: success 0.990, held 3.97, `alive` 0.963, `on_obj` 0.993, coherent 0.858; first kill 6.0, blockers wiped 11.0, first unarmed body on their point 12.9, kill before arrival 1.00. Reward-trained on the same scenario: the per-model arm from C2 0.200 / 0.130 / 0.230, from scratch 0.02–0.06, the control 0.280 / 0.000 / 0.450 |
| opponent | `scripted_baseline` wrapping `hold_and_shoot`: three models, one unit, on the point at (35, 27), four wounds, rifle range 12 with eight attacks |
| budget | 40 epochs of supervised fitting over ~36,700 decision steps, batches of 64, Adam 3e-4, grad-norm 1.0; no PPO |
| code revision | `245c48a` on `feature/curriculum-d1` (PR #374) |
| checkpoint | `checkpoints/per_model/clones/escort-c3b-s{0,1}.pt` (zero rounds) with `.clone.json` beside each |
| coherency | greedy 0.752 / 0.770 at 700000+; 0.78 / 0.76 greedy against 0.68 / 0.71 sampled at 900000+ |
| pre-registration | `reports/2026-09-17-curriculum-D1-preregistration.md` at `245c48a` (2026-09-17 14:50), before any clone at house fidelity |

## The read

| row | success | held of 4 | on obj | alive | vp | coherent | stat | held-out match: joint · selector · declaration · displacement · unit |
|---|---|---|---|---|---|---|---|---|
| `scripted_escort` (the teacher) | 0.990 | 3.97 | 0.993 | 0.963 | +63.5 ± 2.0 | 0.858 | — | — |
| **clone, fit seed 0** | **0.830** | 3.79 | 0.929 | 0.902 | +59.7 ± 2.4 | 0.752 | 0.38 | 0.441 · 0.631 · 0.935 · 0.606 · 1.000 |
| **clone, fit seed 1** | **0.850** | 3.79 | 0.938 | 0.903 | +58.8 ± 2.6 | 0.770 | 0.34 | 0.440 · 0.638 · 0.929 · 0.597 · 1.000 |
| the same, on their TRAINING episodes | — | — | — | — | — | — | — | 0.796 · 0.829 · 0.980 · **0.946** · 1.000 (seed 0); 0.774 · 0.812 · 0.973 · 0.939 · 1.000 (seed 1) |
| reward-trained on this scenario (C3b): arm from C2 · scratch · control | 0.20 / 0.13 / 0.23 · 0.02–0.06 · 0.28 / 0.00 / 0.45 | 2.4–2.7 · 1.5–1.8 · 2.2–3.1 | 0.70–0.79 · 0.43–0.53 · 0.83–0.85 | 0.57–0.60 · 0.49–0.61 · — | +60 to +71 · +25 to +33 · +29 to +40 | 0.54–0.66 | 0.47 | — |

**The match by decision kind** (train against held-out, seed 0; seed 1 within 0.02 everywhere):

| decisions | n held-out | selector, train → held-out | joint, train → held-out |
|---|---|---|---|
| `open` (which unit opens next) | 3,134 | 0.604 → **0.243** | 0.592 → 0.227 |
| `act` in movement (who moves, where) | 5,251 | 0.945 → 0.831 | 0.895 → **0.510** |
| `act` in shooting (who fires, at what) | 759 | 0.947 → 0.846 | 0.947 → 0.846 |

Chance on `open` is one in four (four units). The clone is at chance
held-out and at 0.60 on the episodes it was fitted on: the escort's
opening order is not in the observation. The movement selector
generalises (0.83); the displacement column does not (0.51 joint, 0.61
per head), from 0.95 on the training episodes.

**The ordering clause** (the sequencing census at n=100 on 700000+):

| policy | success (n=100) | first kill | blockers wiped (share) | first unarmed body on their point (share) | **kill precedes arrival** | alive | armed squad dead of 3 |
|---|---|---|---|---|---|---|---|
| `scripted_escort` | 0.990 | 6.0 | 11.0 (0.99) | 12.9 (0.90) | **1.00** | 0.963 | 0.05 |
| clone, seed 0 | 0.830 | 6.1 | 11.3 (0.98) | 13.3 (0.87) | **0.99** | 0.902 | 0.08 |
| clone, seed 1 | 0.850 | 6.1 | 11.5 (0.98) | 13.5 (0.86) | **1.00** | 0.903 | 0.10 |
| C3b's arm from C2 (reward) | 0.13–0.23 | 6.1–7.1 | 0.75–0.79 | 8.1–8.3 | 0.77–0.86 | 0.57–0.60 | 0.81–1.02 |

The clone fires first, wipes the blockers in 98% of episodes, keeps its
riflemen alive, and sends the unarmed squads in after the wipe, half a
turn later than the escort does.

**Greedy against sampled** (900000+, n=100, paired): vp +0.1 ± 1.5 / −0.4 ± 1.6; `held` 3.88 / 3.86 greedy against 3.90 /
3.87 sampled; win rate 97–99% either way; coherency 0.78 / 0.76 greedy
against 0.68 / 0.71 sampled; 151–156 decisions per episode. The sampled
clone plays as well as the greedy one — the fitted distribution is sharp
where it matters.

**The by-turn census** (n=20 on 700000+, points held of 4 and bodies on
points of 12 at turn 9 → 12 → 16 → 24; empty points at the end by index,
2 the blockers'):

| policy | success (n=20) | held / on points, 9 → 12 → 16 → 24 | empty points at the end | max stack |
|---|---|---|---|---|
| `scripted_escort` | 1.00 | 0.7/1.8 → 1.3/3.5 → 3.3/9.5 → 4.0/12.0 | 0 / 0 / 0 / 0 | 3.0 |
| clone, seed 0 | 0.85 | 0.7/1.8 → 1.3/3.5 → 3.2/9.0 → 3.8/11.5 | 0 / 0.05 / 0.10 / 0.05 | 3.0 |
| clone, seed 1 | 0.80 | 0.8/2.0 → 1.4/3.8 → 2.8/7.9 → 3.6/11.4 | 0 / 0.10 / 0.15 / 0.15 | 3.2 |

The clone's board is the escort's to the first decimal through turn 12
— it waits — and lags it by half a body at turn 16 and half a point at
the end: a squad arriving a turn late, or a body short, on one point.

## What it says

- **The set network can hold an ordered plan it was never paid for.**
  From 240 demonstration games the clone fires first in 99% of
  episodes, wipes the blockers in 98%, keeps 90% of its bodies and puts
  a body on every point in 83–85% — on the scenario where 245,760
  rounds of reward-driven training reached 0.20 from a C2 start and
  0.02–0.06 from scratch, and the whole-army control 0.28. #340's D1
  consequence — "cannot reach the match: the architecture cannot hold a
  sequenced plan; stop and redesign" — does not fire on this reading,
  because the match it names is not the thing that failed.
- **The match clause has a defect: it counts a factor that is not in
  the observation.** Which unit the escort opens next is the script's
  internal plan order; the clones are at chance on it held-out (0.24)
  and at 0.60 on their own training episodes, so no clone of this
  teacher can reach a joint match of 0.95. Per head given the teacher's
  model, the declaration head is at 0.93 and the unit-pointer head at
  1.00. **Score a clone by per-head match given the teacher's model and
  by the rung's own criterion, never by a joint match that includes
  whose turn it is.** (D1's letter is read as written and the arm is
  a FAIL; the corrected clause is D1b's.)
- **The displacement head over-fits, and that is where the success
  gap lives.** 0.95 on the training episodes against 0.61 held-out;
  the escort's move is a continuous vector quantised into a column, and
  240 games do not cover the geometry. The by-turn census puts the miss
  at a body short on one point at the end, not at the plan. Four times
  the demonstrations (D1b, #375) is the one change that reading names; more
  epochs on the same games is not.
- **The clones agree to a thousandth**, so the two-clone rule is met
  and nothing here is the fit's seed.
- **D2 can start from this clone**: at 0.85 with the order intact it
  is a better start than anything reward produced, and the question of
  whether per-model PPO improves or destroys it (the whole-army record:
  a cold critic destroyed a clone at every entropy setting) is the next
  rung's. The critic is unfitted here by design.

## What was not done

- No PPO from the clone: that is D2.
- No third clone; the two agree to within a thousandth (0.441 / 0.440 joint, 0.830 / 0.850 success).
