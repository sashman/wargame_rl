# Curriculum rung A5b: the per-model arm fails the allocation rung, and not the way the control did — FAIL as pre-registered

**Verdict first.** On the last movement-only rung of the ladder (#340,
arm #355) — eight squads of three, six objectives, ten rounds, success
only when every point has a body on it — the per-model arm from scratch
reads greedy success **0.260 / 0.180 / 0.160** at 245,760 rounds, n=100,
`held` 4.57 / 4.10 / 4.21 of 6, `on_obj` **0.37 / 0.32 / 0.33**, turns
9.70 / 9.51 / 9.70 of a ten-round game against the script's 6.77,
coherent 0.10 / 0.23 / 0.10. **FAIL** on the pre-registered clause, on
every seed, and the arm never reached a rolling 50% in-run (peaks 48 /
43 / 30%). The whole-army control read 0.800 / 0.940 / 0.980 at the same
rounds by putting twenty-two of twenty-four bodies on points and leaving
one point empty. The per-model arm holds four points with a third of its
bodies on any point at all: this is not the control's allocation error,
it is the walk-off the A3 speed screen saw at 20k rounds, at full scale
and never outgrown. The pre-registration's guess — a pass on two or
three seeds with `held` above the control's — is wrong on every count.

## Provenance

| field | value |
|---|---|
| date | launched 2026-09-15 10:07; killed by the box crash at 16:38 (90,624 / 90,624 / 90,112 rounds); resumed in place 2026-09-16 12:43; exited 23:10 / 23:06 / 23:15 |
| GPU / no-GPU | GPU (RTX 4090), beside the A3 speed screen, then T1, C1 and C2 — up to twelve trainers |
| seeds | 1 / 2 / 3, from scratch; rollout layouts at seed×100+ |
| n | 100 at seed base 700000 (final); 30 at 500000 (in-run, every 512 rounds); 100 at 900000 (greedy against sampled); 20 at 700000 (census) |
| config | `configs/experiments/curriculum/a5.yaml` — unrefereed by design; success `all_objectives_occupied` |
| decode | none on the per-model facade; K=1 on the control |
| paired | per episode against `squad_march_take` on identical seeds |
| comparator | `squad_march_take`, both facades: success 1.000, held 6.00, turns 6.77, `on_obj` 0.856, coherent 0.822 |
| opponent | none |
| budget | 245,760 rounds at 128 per update (`--rollout-rounds 32 --num-rollout-envs 4`), `--ent-coef 0.003`; resumed at ~90k with cadences explicit (#346); the control's 120 epochs is the same rounds |
| code revision | first leg `bd15db9` (PR #362's tip); resumed leg `ed791f1` (PR #366's tree) — docs and the travel term's off-by-default flag differ, nothing `a5.yaml` sets |
| checkpoint | `last.pt` at 245,760; `checkpoints/per_model/per-model-a5-2026-09-15-10-07-{48,49,51}-s{1,2,3}a5b` |
| coherency | greedy 0.099 / 0.226 / 0.096 at 700000+; 0.10 / 0.21 / 0.09 greedy against 0.06 / 0.06 / 0.06 sampled at 900000+ |
| Wandb | `curriculum-a5`: `b9zwoiib` / `2t2nj8tb` / `g9pnffwx` (to the crash), `gegkkxnl` / `ipjuvfu8` / `jdyb1j5a` (resumed); control `koti5sv5` / `jemeom83` / `49n4y1l5` and `5wh7plaj` / `gmonofrt` / `jlnshwd0` |
| pre-registration | `reports/2026-09-15-curriculum-A5b-preregistration.md` (2026-09-15, before launch); amendment 1 (245,760 rounds, `ent_coef` 0.003) at `80b1234`, before launch |

## The read

| row | success | turns | vs bar, paired | held | on obj | coherent | stat | in-run: rounds to rolling 50 / 80 / 95%, peak |
|---|---|---|---|---|---|---|---|---|
| `squad_march_take` | 1.000 | 6.77 | — | 6.00 | 0.856 | 0.822 | — | — |
| **s1 at 245,760** | **0.260** | 9.70 | +2.93 ± 0.09 | 4.57 | 0.371 | 0.099 | 0.02 | never / never / never; peak 48% at 174k |
| **s2 at 245,760** | **0.180** | 9.51 | +2.74 ± 0.13 | 4.10 | 0.316 | 0.226 | 0.11 | never / never / never; peak 43% at 133k |
| **s3 at 245,760** | **0.160** | 9.70 | +2.93 ± 0.10 | 4.21 | 0.327 | 0.096 | 0.01 | never / never / never; peak 30% at 211k |
| control at 60 epochs | 0.890 / 0.820 / 0.600 | 8.31–8.79 | +1.54 to +2.02 | 5.58–5.88 | 0.92–0.95 | 0.58–0.64 | — | — |
| control at 120 epochs | 0.800 / 0.940 / 0.980 | 8.32 / 7.96 / 7.21 | +1.55 / +1.19 / +0.44 | 5.76 / 5.93 / 5.98 | 0.93 / 0.93 / 0.96 | 0.62 / 0.57 / 0.47 | — | — |
| s1 / s2 / s3 at their in-run peak (174k / 133k / 211k, `pm-*.pt`) | 0.330 / 0.360 / 0.220 | 9.40 / 9.44 / 9.76 | +2.63 / +2.67 / +2.99 | 4.68 / 4.79 / 4.29 | 0.37 / 0.37 / 0.32 | 0.14 / 0.13 / 0.11 | 0.02–0.04 | — |

**Greedy against sampled** (900000+, n=100, paired): vp +1.7 ± 0.9 /
+5.7 ± 1.3 / +3.1 ± 1.3; `held` 4.70 / 4.02 / 4.58 greedy against 4.62 /
4.23 / 4.42 sampled; coherency 0.10 / 0.21 / 0.09 greedy against 0.06 ×3
sampled; 285–311 decisions per episode either way. The policy a score
reports and the policy the rollouts train are the same policy here —
both hold four and a half points, and neither keeps a squad together.

**Health panel, last quarter** (~480 updates per seed): explained
variance 0.69 / 0.62 / 0.75, clip fraction **0.37 / 0.40 / 0.42**, ratio
p99 **2.25 / 2.29 / 2.40**, displacement entropy 1.47 / 1.63 / 1.66,
declaration entropy 0.04–0.06, advantage std 0.25–0.33. The last forty
in-run evaluations average 23.9 / 21.9 / 12.5%. Red on the clip line
and the tail together (A2's drift read 0.26–0.37 and 1.7–1.8), with the
critic fine — the update is fighting a policy that is not settling.

**The by-turn census** (n=20 on 700000+, bodies on objectives of 24 and
points held of 6, turn 3 → 5 → 7 → 10; which points are empty at the
end, by index 0–5):

| policy | success (n=20) | turns | on objectives, 3 → 5 → 7 → 10 | held, 3 → 5 → 7 → 10 | episodes with each point empty at the end | max stack |
|---|---|---|---|---|---|---|
| `squad_march_take` | 1.00 | 6.55 | 3.7 → 16.5 → 20.1 → 20.1 | 2.3 → 3.0 → 6.0 → 6.0 | 0 / 0 / 0 / 0 / 0 / 0 | 5.8 |
| control at 120 (from #353) | 0.80 / 0.94 / 0.98 | 7.2–8.3 | 22 of 24 on points at the end | 5.8–6.0 | one point, in the failing episodes | — |
| s1 | 0.20 | 9.85 | 0.4 → 5.2 → 9.1 → **8.7** | 0.4 → 2.5 → 3.4 → 4.4 | 0.25 / 0.25 / 0.20 / 0.35 / 0.25 / 0.30 | 3.4 |
| s2 | 0.20 | 9.40 | 1.3 → 7.9 → 8.3 → **7.5** | 1.1 → 2.6 → 4.6 → 4.2 | 0.35 / 0.40 / 0.65 / 0.05 / 0.05 / 0.30 | 3.0 |
| s3 | 0.15 | 9.75 | 0.4 → 5.9 → 8.2 → **8.5** | 0.4 → 2.7 → 3.8 → 4.5 | 0.10 / 0.40 / 0.20 / 0.40 / 0.35 / 0.10 | 3.5 |

**Two thirds of the army never arrives.** The script has 16.5 bodies on
points at turn 5 and 20 by turn 7, on all six; the arm has 5–8 at turn
5 and never more than ten, spread over four and a half points at two
bodies each. The empty point is a different one each episode (every
index is empty in 5–65% of episodes, no favourite), the max stack is
3.0–3.5 against the script's 5.8, and the sampled and greedy policies
agree. So this is neither the control's failure (bodies on points, one
point short) nor a stacking failure: it is **under-arrival** — squads
broken up (coherency 0.10), bodies drifting toward the points slowly,
`on_obj` flat or falling over the last three rounds. The A3 speed
screen's walk-off at 20k rounds is the same signature, and here it was
never outgrown.

## What it says

- **FAIL on every seed, and the failure is the arm's own.** The
  pre-registration's FAIL clause asked which way: if the arm fails the
  control's way (bodies on points, one empty), neither trainer solves
  allocation at this shape; if differently, say how. It fails
  differently. The control puts twenty-two bodies on points and leaves
  one empty; the arm puts eight to ten on points and holds four and a
  half. The control has an allocation problem; the arm has an
  **arrival** problem, and on this rung it is the larger one by far.
- **This is the ladder's first rung where the per-model arm is worse
  than the whole-army control at equal rounds, and by a wide margin**:
  0.16–0.26 against 0.80–0.98. On A3 it trailed at 123k and matched at
  245k; here at 245k it has not reached a rolling 50% in-run on any
  seed, and its peak was at 133k–211k with no climb after. The peak
  checkpoints score 0.33 / 0.36 / 0.22 at n=100 with the same `on_obj`
  (0.32–0.37) — the best this arm ever was is the same policy, a little
  less of it. Never learned, not drifted; more rounds is not the reading.
- **What changed from A4 to A5 is the distance and the count, and on
  this facade the count dilutes the credit.** A4's bar walks 4.70 turns,
  A5's 6.77 — farther points, twice the bodies. `gamma` counts rounds
  here (0.9 per round spans the ten-round game), so the horizon is not
  it. What the record does say is that every per-model state term on
  this facade — the travel term included — is paid at the turn close as
  the **mean over alive models** (§ The per-model curriculum): the
  scalar a body sees for its own inch closed is one twenty-fourth of the
  army's, against one twelfth on A3–A4 and one third on A1. The A rungs
  passed as that dilution doubled, and this is the rung where it doubled
  again and the arm stopped arriving. A hypothesis, not a finding — the
  build that would test it (pay a state term to the model that produced
  it, `envs/per_model/reward_timing.py`) is the one the speed screen
  already named.
- **The health panel is red and says the same thing.** Clip fraction
  0.37–0.42 and ratio p99 2.25–2.40 over the last quarter, with the
  critic at 0.62–0.75 explained variance — A2's drift signature, at the
  entropy coefficient that fixed A2. The critic can value the states the
  policy reaches; the policy is not settling on how to reach better ones.
- **Coherency 0.10 greedy, 0.06 sampled — the lowest on the ladder.**
  Unpaid on every rung, and here the squads dissolve. Whether formation
  would help arrival is a question for a rung that pays for it (the E
  rungs' referee); the record says nothing pays for it here.
- **The control's pass is not a pass either.** Read beside each other:
  one trainer leaves a point empty, the other leaves most of the army in
  the open. Neither solves eight squads over six points from scratch,
  and the script does at 1.000 in 6.77 turns. #340's answer for this
  axis is that per-step credit did not buy allocation here; T1 says
  whether a warm start buys arrival.

## What was not done

- No second budget: the symmetric cap already put the arm at 245,760.
- No arm at `ent_coef` 0.03 on this rung (A2's lesson stands; the clip
  fraction here is already A2's failing range at 0.003).
- The control's in-run curve was read from Wandb after its runs exited
  and is on #353; it is not re-read here.
