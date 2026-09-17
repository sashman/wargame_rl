# Curriculum rung C3: one armed squad has to clear the blockers before the unarmed squads walk in — the arm found the dash, not the plan; the run with no head start found the plan and not the walk

**Verdict first.** On the first rung that asks for a plan with an order
to it (#340, arm #370) — C2's four squads and four points with squad 0
armed (range 18, four attacks) against three blockers of four wounds and
eight attacks at range 12 on the point at (35, 27), twelve rounds — the
per-model arm warm-started from C2 reads greedy success ****0.660 / 0.920 / 0.830**** at
245,760 rounds, n=100, with the first kill preceding the first unarmed
arrival on the blockers' point in ****0.67 / 0.71 / 0.62**** of episodes, `alive`
0.616 / 0.741 / 0.695 against the bar's 0.963, turns 14.76 / 10.66 / 12.51 phase-clock of 24 against the
bar's 15.25. ****FAIL as pre-registered on both clauses** — two seeds under 0.90 at the budget, the ordering clause missed on all three, and the drift clause tripped on all three (rolling dips under 75% after the first pass: 53 / 10 / 17).** The from-scratch companion reads
**0.740 / 0.740 / 0.090** with the kill before the arrival in **1.00 / 0.97 / 0.95** of episodes. The whole-army control fails its own criterion on every seed
at its extended budget (**0.870 / 0.830 / 0.840** at 120 epochs) on a
rung the escort passes at 0.990 and a plain `squad_march_take` fails at
0.590. The prediction on file — two seeds at the mark by 122,880 with the waiting the thing that is hard to learn — is wrong about what the arm learned: it did not learn to wait, it learned that a body dashed onto the blockers' point the instant the other three are held ends the game before the blockers fire again. The scenario rewarded that, and the second arm on this rung (C3b, #372, in flight) removes the reward by reading success at the end of the game.

## Provenance

| field | value |
|---|---|
| date | launched 2026-09-17 04:15 (control, arm and companion together); control extended 04:22–04:33; first per-model leg to 122,880 exited 06:50–06:54, resumed in place, exited 09:19–09:40 |
| GPU / no-GPU | GPU (RTX 4090), nine trainers at launch, six from 04:33 |
| seeds | 1 / 2 / 3; rollout layouts at seed×100+; the arm from C2's `last.pt` of the same seed (245,760 rounds on C2: 0.890 / 0.920 / 0.950), fresh optimizer; the companion from scratch — **not paired on init** with either |
| n | 100 at seed base 700000 (final, the 122,880 readout, the ordering census); 30 at 500000 (in-run, every 512 rounds; n=20 after the resume, #346); 100 at 900000 (greedy against sampled); 20 at 700000 (by-turn census) |
| config | `configs/experiments/curriculum/c3.yaml` — unrefereed by design; success `all_objectives_occupied` counts our bodies; movement and shooting stepped (two agent phases per round, 24 phase-clock turns) |
| decode | none on the per-model facade; K=1 on the control |
| paired | per episode against `scripted_escort` on identical seeds |
| comparator | `scripted_escort`, both facades identical: success 0.990, held 3.97, `alive` 0.963, `on_obj` 0.915, turns 15.25, vp +5.3 ± 1.1, coherent 0.855; first kill at turn 6.0, blockers wiped at 11.0, first unarmed body on their point at 12.9, the kill before the arrival in 100%. `squad_march_take` (must fail): 0.590, alive 0.615 |
| opponent | `scripted_baseline` wrapping `hold_and_shoot`: three models, one unit, on the point at (35, 27), four wounds, rifle range 12 with eight attacks |
| budget | per-model 245,760 rounds at 128 per update (`--rollout-rounds 32 --num-rollout-envs 4`), `--ent-coef 0.003`, resumed in place at 122,880 with cadences and `--n-eval-episodes` explicit; control 60 epochs extended once to 120 (245,760 rounds) — the symmetric cap |
| code revision | `a5e603e` on `feature/curriculum-c3` (PR #371) at launch; docs-only commits after |
| checkpoint | `last.pt` at 245,760; `checkpoints/per_model/per-model-c3-*-s{1,2,3}c3` (arm) and `-s{1,2,3}c3s` (companion); control `checkpoints/ppo-transformer-curriculum_c3-…-s{1,2,3}c3-ctl-x2/last.ckpt` |
| coherency | greedy 0.442 / 0.548 / 0.499 (arm), 0.417 / 0.399 / 0.385 (companion) at 700000+; against sampled 0.37 / 0.33 / 0.35 and 0.29 / 0.23 / 0.21 at 900000+ |
| Wandb | `curriculum-c3`: arm `gfb0z2og` / `myyyfmo5` / `dfg3hndb`, companion `of8yv8f7` / `rzzt1oqk` / `eei1nkse`, resumed halves arm `3jud60gm` / `03csrlxc` / `xe8isgoo`, companion `k7f1x99v` / `991fx1z2` / `24n2mi0y`; control `xt95et6x` / `b00ym3pi` / `a9ubzqa7` (to 60), `bpa07dn5` / `do7u35wa` / `ib9ymhgv` (60–120) |
| pre-registration | `reports/2026-09-17-curriculum-C3-preregistration.md` at `a5e603e` (2026-09-17 04:10); amendment 1 at `13d07e4` (the control's 60-epoch read and the extension, 04:25) and amendment 2 at `d0c6a09` (the control's 120-epoch read, 04:36), both before any per-model number |

## The read

| row | success | turns (24 max) | vs bar, paired | held of 3 | on obj | alive | coherent | in-run: rounds to rolling 50 / 80 / 95%; last 40 |
|---|---|---|---|---|---|---|---|---|
| `scripted_escort` (the bar) | 0.990 | 15.25 | — | 3.97 | 0.915 | 0.963 | 0.855 | — |
| `squad_march_take` (must fail) | 0.590 | 15.25 | — | 2.66 | 0.957 | 0.615 | 0.937 | — |
| **arm s1** (from C2) at 245,760 | **0.660** | 14.76 | −0.49 ± 0.71 | 2.77 | 0.699 | 0.616 | 0.442 | 8.2k / 24.6k / never; 75.6 |
| **arm s2** at 245,760 | **0.920** | 10.66 | −4.59 ± 0.42 | 3.35 | 0.777 | 0.741 | 0.548 | 2.6k / 2.6k / 106k; 83.8 |
| **arm s3** at 245,760 | **0.830** | 12.51 | −2.74 ± 0.58 | 3.14 | 0.716 | 0.695 | 0.499 | 2.6k / 5.6k / never; 84.2 |
| arm s1 / s2 / s3 at 122,880 (readout) | 0.740 / 0.930 / 0.810 | 13.78 / 10.93 / 12.23 | −1.47 / −4.32 / −3.02 | 2.77 / 3.33 / 2.97 | 0.71 / 0.85 / 0.73 | 0.60 / 0.74 / 0.70 | 0.53 / 0.69 / 0.48 | — |
| companion s1 (scratch) at 245,760 | 0.740 | 16.46 | +1.21 ± 0.52 | 3.34 | 0.778 | 0.724 | 0.417 | 92k / 118k / never; 70.7 |
| companion s2 at 245,760 | 0.740 | 15.76 | +0.51 ± 0.50 | 3.36 | 0.697 | 0.772 | 0.399 | 98k / never / never; 49.2 |
| companion s3 at 245,760 | 0.090 | 22.91 | +7.66 ± 0.39 | 1.48 | 0.376 | 0.581 | 0.385 | never; 19.2 |
| companion at 122,880 (readout) | 0.720 / 0.550 / 0.170 | 17.3 / 19.4 / 22.1 | +2.03 / +4.18 / +6.86 | 3.50 / 2.97 / 2.25 | 0.71 / 0.64 / 0.56 | 0.78 / 0.71 / 0.67 | 0.31 / 0.40 / 0.42 | — |
| control at 60 epochs | 0.790 / 0.560 / 0.870 | 18.2–19.3 | +2.9 to +4.1 | 2.66–3.61 | 0.77–0.96 | — | 0.73–0.80 | 80% at epoch 35 / never / 56 |
| **control at 120 epochs** | **0.870 / 0.830 / 0.840** | 18.27 / 17.31 / 17.66 | +2.06 to +3.02 | 3.29–3.61 | 0.94–0.95 | — | 0.74–0.79 | 80% at 64 / 96 / 64; 90% at 88 / never / never |

⚠ The arm is **faster than the bar** on two seeds (10.7 and 12.5 turns
against 15.25) with a third of its bodies dead: that is the dash, not a
better plan. The drift clause: after its first rolling pass the arm's
rolling mean fell under 75% on 53 / 10 / 17 evaluations (seed 1 read
0.740 at 122,880 and 0.660 at the end). ⚠ In-run rows after the resume
at 122,880 are n=20 (#346).

**The ordering clause** (the sequencing census at n=100 on 700000+: the
turn of the first kill, of the blockers' wipe, and of the first unarmed
body on their point; the share of episodes in which the kill came
first):

| policy | success (n=100) | first kill | blockers wiped (share) | first unarmed body on their point (share) | **kill precedes arrival** | armed squad dead of 3 |
|---|---|---|---|---|---|---|
| `scripted_escort` | 0.990 | 6.0 | 11.0 (0.99) | 12.9 (0.90) | **1.00** | 0.05 |
| `squad_march_take` | 0.590 | 6.7 | 12.5 (0.31) | 8.9 (0.50) | 0.77 | 0.72 |
| arm s1 | 0.660 | 8.1 | 13.1 (**0.22**) | 8.7 (0.85) | **0.67** | 1.09 |
| arm s2 | 0.920 | 6.5 | 10.6 (**0.16**) | 8.5 (0.88) | **0.71** | 1.01 |
| arm s3 | 0.830 | 7.1 | 12.5 (**0.30**) | 8.5 (0.90) | **0.62** | 0.56 |
| companion s1 | 0.740 | 7.0 | 11.4 (0.65) | 12.0 (0.50) | **1.00** | 0.93 |
| companion s2 | 0.740 | 6.8 | 11.8 (0.78) | 10.7 (0.77) | **0.97** | 0.23 |
| companion s3 | 0.090 | 8.1 | 12.4 (0.48) | 10.5 (0.24) | **0.95** | 1.48 |

The arm's unarmed bodies reach the blockers' point at turn 8.5–8.7 —
*before* the escort's armed squad has even finished killing (11.0) and
with the blockers alive in 70–84% of its episodes. Its own armed squad
walks in too (about one of three dead). The companion never sends an
unarmed body in before the first kill on two seeds and almost never on
the third.

**Greedy against sampled** (900000+, n=100, paired): vp +6.2 ± 3.6 / −7.0 ± 2.5 / +0.7 ± 3.1 (arm), +4.5 / −1.6 / +1.3
(companion); `held` within 0.4 either way; coherency 0.50 / 0.55 / 0.53
greedy against 0.37 / 0.33 / 0.35 sampled (arm), 0.39–0.42 against
0.21–0.29 (companion); 91–109 decisions per episode (arm, the dash ends
the game) against 129–145 (companion, the game runs on). The sampled
policy plays the same game as the greedy one on both.

**Health panel, last quarter** (~480 updates per seed): arm — explained variance 0.5–0.6, clip fraction 0.21 / 0.19 / 0.21,
ratio p99 1.78 / 1.71 / 1.74, displacement entropy 1.0–1.2, declaration
entropy 0.03; companion — clip **0.37 / 0.33 / 0.26**, ratio p99
**2.27 / 2.08 / 1.87**, displacement entropy 1.5–1.7, declaration
entropy 0.06–0.12. The arm's panel is green while it drifts (the clip
line does not see a policy alternating between two solutions of similar
value); the companion's is A5b's red, still moving.

**The by-turn census** (n=20 on 700000+, bodies on objectives of 12 at
turn 7 → 9 → 12 and at the end, points held at the end, empty points by
index with index 2 the blockers'):

| policy | success (n=20) | turns | on objectives 7 → 9 → 12 → end | held at the end | empty points at the end | max stack |
|---|---|---|---|---|---|---|
| `scripted_escort` | 1.00 | 15.80 | 0.7 → 1.8 → 3.5 → 10.0 | 3.95 | 0 / 0 / 0 / 0 | 3.0 |
| arm s1 | 0.70 | 14.10 | 0.9 → **6.5** → 7.0 → 7.9 | 2.85 | 0.05 / 0.20 / 0.20 / 0.25 | 2.2 |
| arm s2 | 0.85 | 11.65 | 1.0 → **8.5** → 9.2 → 9.3 | 3.50 | 0.05 / 0.15 / 0.05 / 0.05 | 2.5 |
| arm s3 | 0.95 | 11.35 | 1.3 → **7.3** → 8.9 → 8.7 | 3.40 | 0 / 0.05 / 0.05 / 0.05 | 2.7 |
| companion s1 | 0.65 | 17.25 | 0.5 → 5.1 → 7.6 → 8.3 | 3.00 | 0.10 / 0.15 / 0.35 / 0.25 | 2.9 |
| companion s2 | 0.60 | 17.40 | 0.6 → 2.9 → 6.1 → 7.7 | 3.10 | 0.05 / 0.25 / 0.35 / 0.25 | 2.6 |
| companion s3 | 0.05 | 23.35 | 1.3 → 5.6 → 6.3 → 5.4 | 1.75 | 0.35 / 0.65 / 0.85 / 0.90 | 1.6 |

The escort has 1.8 bodies on points at turn 9 because it is waiting; the
arm has 6.5–8.5 there — everyone walks in at once, the C2 approach
unchanged. The companion's seed 3 walks in and dies, stack 1.6, three
points empty at the end.

## What it says

- **FAIL as pre-registered on both clauses, and the census says which
  way.** Success 0.660 / 0.920 / 0.830 with the kill before the
  unarmed arrival in 0.67 / 0.71 / 0.62 of episodes. The FAIL clause
  offered three readings — dead unarmed squads, a live blocker, C1's
  spread residual — and the answer is the second, by design: the arm
  leaves the blockers alive in 70–84% of episodes and wins anyway,
  because `terminate_on_success` ends the game the instant a last body
  lands on their point. A dash is a legal move in the scenario as
  written; it is not the plan the rung was built to test.
- **The head start carried the wrong habit.** The C2 policy rushes
  everyone in and survives because C2's blockers barely kill. On C3
  that habit lands at the blockers' point at turn 8.5 with a third of
  the army dead, and is paid for it with an instant win. The arm never
  had to discover waiting; the from-scratch companion, with nothing to
  carry, learned the order (kill first in 95–100% of episodes, the
  blockers wiped in half to three quarters) and is slow at the walk
  that follows (0.74 / 0.74 / 0.09). The two runs found the two halves
  of the escort's plan, and neither found both. The C rungs' warm-start
  rule now has its first counter-example: a warm start helps when the
  rung below's skill is the rung's skill (C1, C2) and hurts when it is
  the thing to unlearn.
- **The arm drifts between the dash and something else.** Rolling dips
  under 75% after the first pass on every seed (53 / 10 / 17), seed 1
  down from 0.740 to 0.660 across the second half, with the health
  panel green throughout. Two solutions of similar value under this
  reward — dash and wait — and per-step credit does not settle it.
- **The control fails on all three seeds at 120 epochs (0.870 / 0.830
  / 0.840)** a round or more behind the escort, as it did on C2 with
  fewer guns. Read beside the arm it is the better of the two on
  success and the worse on speed; neither trainer learned the plan.
- **The rung's scenario has a defect, and it is measured.** With
  success read at the end of the game (`terminate_on_success: false`)
  the escort still reads 0.990 and plain `take` falls from 0.590 to
  0.280, because a dashed body has to hold the point under fire. That
  is C3b (#372), pre-registered and launched with this report:
  the same arms, one flag. **Rule for the ladder: a success criterion
  that ends the episode the instant it holds rewards whatever reaches
  it first; on any rung where reaching a state and keeping it are
  different skills, judge success on the final board.**
- **Coherency 0.44–0.55 greedy, 0.33–0.37 sampled** — unpaid, as on
  every rung.

## What was not done

- No census of the control's own failure (which of walking in early or
  never clearing the point) beyond its turns and `held`.
- No second budget; the symmetric cap already put the arm at 245,760.
- No arm warm-started from C3's own checkpoints into C3b: they carry the
  dash, so C3b starts from C2 as C3 did.
