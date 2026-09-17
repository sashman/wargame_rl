# Pre-registration: three arms at why the spread rung learns slowly (A3 speed screen)

Written 2026-09-15, **before any number from any arm exists**, on branch
`feature/curriculum-a3-speed` (stacked on PR #354). Parent question #340,
rung **A3**. One issue per arm; this file carries the three criteria and
the reads that motivate them, because the three share one comparator and
one launch.

The ladder is paused (the user's instruction, 2026-09-15). These arms are
the investigation, not the ladder: nothing here changes A3's verdict, and
a lever that pays here is a proposal for the rungs above, not a rewrite of
the rungs below.

## The reads that motivate the arms (no GPU, 2026-09-15, A3's own board)

All on `configs/experiments/curriculum/a3.yaml`, greedy, seeds 700000+,
the census scripts in the session scratchpad (a copy of the travel-gate
instrumentation from `scripts/measure_shaping_gates.py`, run without the
map tables it always overlays, plus a by-turn read of bodies on objectives
and points held).

**1. The rung is not the rung its config describes.** The comment says
each squad deploys nearest a distinct point. Deployment is uniform along a
36-inch band, and in **24 of 200** deployments (12%) are the four squads'
nearest points four distinct points. In the other 88% at least one squad
must cross to a point that is nobody's nearest. A3 is an assignment
problem from a random start, every episode.

**2. The travel term points half the army at the wrong thing.**
`closest_objective_v2` assigns each candidate objective to the group with
the closest model, so one group can own several points and the group left
without one drops through `fallback_to_nearest` onto a point somebody
else is already taking. The scripted bar, n=100: one unit owns two or more
objectives on **49.1%** of steps; **50.1%** of model-steps are paid toward
the nearest point by fallback rather than toward an assigned one; squads
are paid toward different targets on 11.3% of squad-steps. The learned
policy (A3x `last.pt`, three seeds) reads the same gate numbers (51–61%,
44–50%) and splits its squads on **20–32%** of squad-steps.

**3. The bottleneck along the path is bodies that arrive and stay, not
allocation between points.** Seed 1's periodic checkpoints, n=30, bodies
on objectives (of 12) and points held (of 4) at turn 5 then turn 8:

| rounds | success | on objectives, turn 5 → 8 | held, turn 5 → 8 |
|---|---|---|---|
| 20,480 | 0.13 | 5.0 → **2.3** | 1.9 → **1.3** |
| 61,440 | 0.10 | 3.2 → 4.4 | 1.8 → 2.1 |
| 122,880 | 0.77 | 8.2 → 8.0 | 3.5 → 3.7 |
| 184,320 | 0.90 | 8.5 → 8.6 | 3.5 → 3.8 |
| 245,760 (n=100) | 0.96 | 9.6 → 9.7 | 3.9 → 3.9 |
| `squad_march_take` | 1.00 | 9.9 → 10.7 | 3.7 → 4.0 |

At 20k rounds the policy reaches points and **walks off them**. Bodies per
held point stay at 2.2–2.4 throughout, the script's own ratio, so there is
no stack-then-peel signature; the prediction that there would be one was
made before this read and is refuted by it. What the travel term pays
saturates at the boundary (progress zero once inside), the coverage term
is 0.3 over four points broadcast to twelve bodies alike, and the terminal
bonus never fires until all four are held. Nothing pays a body for
standing on a point it reached.

**4. A3's own in-run curve, the speed comparator** (n=30 at 500000+ every
512 rounds, rolling mean of five): rounds to 50% — 38,400 / 23,040 /
36,864; rounds to 80% — never / 90,624 / 101,888; rounds to 95% — never
on any seed inside 122,880. Final read at 122,880, n=100: **0.700 / 0.800
/ 0.770**, held 3.49 / 3.60 / 3.65, turns 6.05 / 5.95 / 5.81.

## The three arms

Each is `a3.yaml` with **one** change, three seeds, **122,880 rounds** at
128 rounds per update (`--rollout-rounds 32 --num-rollout-envs 4`),
`--ent-coef 0.003`, in-run eval every 512 rounds at n=30 on 500000+,
checkpoints every 512, Wandb group `curriculum-a3-speed`. The comparator
for all three is **A3's own three runs at 122,880** (Wandb `curriculum-a3`:
`wzalbuic` / `4x0p1q4a` / `jzmpi6pl`), read on identical seeds.

| arm | tag | the one change | config | paired on init? |
|---|---|---|---|---|
| **S1 hold** | `a3h` | `objective_hold` weight 1.0, `crowding_exponent` 1.0, added beside the two terms A3 has | `curriculum/a3_hold.yaml` | yes — same seed, same head, so the init is A3's |
| **S2 match** | `a3m` | `closest_objective_v2` `one_objective_per_group: true` (new flag, default off, shipped with this branch) | `curriculum/a3_match.yaml` | yes |
| **S3 warm** | `a3w` | `--warm-start-from` the A2b checkpoint of the same seed (`per-model-a2-2026-09-15-02-44-23-s{1,2,3}a2b/last.pt`) | `curriculum/a3.yaml` unchanged | **no** — the init is A2b's; layouts and seeds shared |

Bar checks before launch: `squad_march_take` on `a3_hold.yaml` reads
1.000 / held 4.00 / 5.28 turns, identical to `a3.yaml` (the term does not
touch the script). On `a3_match.yaml` the gate census reads one unit
owning two or more objectives on **0.0%** of steps (was 49.1%), squads
split on 1.1% (was 11.3%), every squad given a point of its own on the
first paid step in 100 of 100 episodes (was 9 of 100). A warm start from
A2b's `last.pt` into `a3.yaml` loads and trains (smoke, 256 rounds).

## Criteria, per arm, identical

Read at 122,880 rounds from `last.pt`, greedy, no decode, n=100 on
700000+, beside A3's same-seed read.

- **FASTER:** success ≥ 0.95 on **all three** seeds — the arm passes the
  rung inside the cap where A3 from scratch did not.
- **AHEAD:** not FASTER, but every seed ≥ its A3 read + 0.10. Binomial SE
  at p≈0.75 is 0.043 per seed, so +0.10 is 2.3 SE and a null lever clears
  it on 3 of 3 with probability under 1%.
- **NULL:** otherwise.
- **HARMFUL:** any seed below its A3 read by more than 0.10.

Readouts, all of them, none decides: rounds to rolling-5 in-run 50 / 80 /
95% beside A3's; `held`, `on_obj`, turns; coherency greedy and sampled
(`just measure-per-model-eval-mode`); the by-turn census at 20,480 /
61,440 / 122,880 (the walk-off signature at 20k: bodies on objectives
turn 5 → 8); the gate census on `last.pt`; the health panel over the last
quarter, explained variance in particular (A3: 0.34–0.45 at the cap).

## Predictions, written so they can be wrong

- **S1 hold** is the one aimed at the mechanism read 3 found. Prediction:
  the turn-5-to-8 drop in bodies on objectives is gone at 20k rounds, and
  the arm is **FASTER** on at least two seeds, with rounds to 80% under
  60k on all three.
- **S2 match** fixes read 2. Prediction: **AHEAD** but not FASTER, and the
  larger effect is on coherency — squad-steps split under 5%, greedy
  coherency above 0.6 (A3: 0.38–0.52).
- **S3 warm** starts from a policy that walks everyone to one point.
  Prediction: faster to the first bodies on objectives, then **NULL** on
  success — an A2b policy has to unlearn stacking, and the transfer
  question (T1) is better asked from A4 to A5 as the ladder says.

## What this screen cannot say

- Nothing about the rungs above: a term that pays here is a proposal for
  A4/A5, to be pre-registered there.
- S1 changes the reward, so its episode reward is not comparable to A3's;
  success, held and turns are.
- S3 is unpaired on init. Its per-seed differences carry init variance the
  other two do not.
- Three seeds at n=100 is a screen. A lever that reads AHEAD on one seed
  and NULL on two is a null.
