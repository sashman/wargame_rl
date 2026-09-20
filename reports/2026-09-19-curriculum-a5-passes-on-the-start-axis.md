# Curriculum rung A5, the second pass: the credit is not the wall, the clone is the ceiling, and the anchor holds it over the mark

**Verdict first. A5 PASSES on the letter, on the start axis** — the bar
cloned from 2,000 games (0.960) and held by anchored PPO on A5's own
reward at **0.960 / 0.980 / 0.960** (A5i, 122,880 rounds, greedy at
n=100 on 700000+, no in-run dip, sampled play identical) — **and
nothing on this rung was learned from reward.** Ten per-model arms and
two half-step rungs on the allocation rung (#340; eight squads of three over six points, success =
every point occupied, the bar `squad_march_take` at 1.000 in 6.77
turns), run 2026-09-17 23:42 to 2026-09-19 00:46, read greedy at n=100
on 700000+:

| arm | change | verdict | success | where read |
|---|---|---|---|---|
| **A5c** | A5b with the actor credit, undivided | **FAIL (a PPO change)** — returns 13×, the pre-clip gradient norm 5×, clipped on every step | 0–1.3% in-run | stopped at ~75k |
| **A5d** | A5b with the actor credit divided by the model count | **FAIL, MOVES not shown** | 0.050 / 0.070 / 0.010, held 3.9 / 3.8 / 2.7 | ~110k of 245,760 |
| **A5e** | A5d + `objective_hold` (crowding 1.0) paid per model on the actor's step | **FAIL; a small, consistent edge over A5d, nowhere near a pass** | 0.170 / 0.060 / 0.060, held 4.0 / 3.7 / 2.8 | ~108k of 245,760 |
| **clone, 1,200 games** | the bar cloned into the set network | **FAIL by imitation, by 0.02** | **0.930**, held 5.93, turns 6.96 (bar 6.77) | — |
| **A5f** | anchored PPO from the clone (coef 10, target 0.03), actor credit | **HOLDS, IMPROVES not shown** | 0.930 / 0.910 / 0.960 at 80k (0.95 / 0.92 / 0.93 at 20k) | stopped at ~89k of 122,880 |
| **A5g** | A5f + the hold term (anchor 1, target 0.10) | **HOLDS, IMPROVES not shown** | 0.960 / 0.900 / 0.920 at 80k | stopped at ~89k |
| **A5h** | A5g with the anchor loosened tenfold (0.1, target 0.3) | **loses the clone slowly** | 0.950 / 0.840 / 0.820 at 40,960; in-run 84–89% against A5g's 90–96% | stopped at ~44k |
| **clone, 2,000 games** | 1.67× the games, 60 epochs | **PASS by imitation** | **0.960**, held 5.96, turns 6.97; displacement match 0.72 → 0.821 | — |
| **A5i** | anchored PPO from the 2,000-game clone (coef 10 / 0.03), actor credit | **HOLDS 3/3 — the rung's PASS; IMPROVES not shown** | **0.960 / 0.980 / 0.960**, held 5.96–5.98, turns 6.88 / 6.81 / 7.04 (clone 6.97, bar 6.77); 40k 0.970 ×3, 60k 1.000 / 0.970 / 0.970, 80k 1.000 / 0.980 / 0.960 | 122,880 |
| **A5-bodies** | eight squads over THREE points (the bodies axis alone) | **FAIL, one seed over the mark; a turn or more slow** | 0.960 / 0.930 / 0.860, turns 5.46 / 5.80 / 6.23 (bar 4.43), held 2.94 / 2.90 / 2.70 | 122,880 |
| **A5-points** | SIX squads over five points (the points axis alone) | **FAIL on every seed — A5's failure at six squads** | 0.030 / 0.060 / 0.330, held 2.47 / 2.92 / 3.83 of 5, turns 9.6–10.0 (bar 6.71) | 122,880 |

Four things the pass says, each against the record's own hypothesis:

1. **The credit is not the wall.** The hypothesis on file since the A3
   speed screen — a body's own credit diluted 1/24 by the mean over the
   army — was built (`--credit actor`) and run three ways. The first cut
   changed PPO, not the reward: paying the actor undivided scaled the
   returns thirteenfold and the value loss swamped the clipped policy
   gradient. Scaled back to a constant, the credit moved nothing
   (A5d 0.01–0.07 against A5b's 0.16–0.26 at less than half the rounds),
   and adding the state term it was built to carry moved arrival by a
   tenth (A5e). The per-model trainer's failure on A5 is not a
   credit-scale problem.
2. **The clone is the ceiling, and it is 0.93.** The set network holds
   the bar's plan from 1,200 games at 0.930 with 5.93 of six points and
   within a fifth of a turn of the script — and every one of its seven
   failures in a hundred is the same event: the script's assigned squad
   walks to a point another squad is already taking, two squads share
   it, one point stays empty. Never a late arrival. The teacher breaks
   that tie by its own assignment order, which is not in the
   observation; a bigger fit closes the gap only if the tie-break is a
   function of the board. **The 2,000-game clone reads 0.960** (held 5.96, turns 6.97, coherent
0.771; displacement match 0.821 held-out), the first policy of any kind
on the per-model facade to clear A5's mark. Four failures in a hundred,
the same event, spread over three points where the smaller clone's
seven were six on one — more games shrank the residual by three
hundredths without changing its kind.
3. **Anchored PPO sits on the clone and does not lift it.** Five reads
   over 20k–80k rounds on two arms put every seed in the clone's band
   (0.85–0.97, binomial SE 0.026 at 0.93); one seed at 0.95–0.96 on each
   read, never the set. The coefficient question the D2 report left
   open has its answer: 10 holds, 1 holds, 0.1 loses, and none improves.
   From the 2,000-game clone the same recipe (A5i) holds 0.960 at
   0.960–0.980 on every read from 40k to the end — which is the rung's
   mark on three seeds, and is the clone's band.
4. **The wall is the number of points to cover at once.** Two half-steps
   from A4 (0.93–0.97 at the cap): doubling the army at A4's three
   points reads 0.960 / 0.930 / 0.860, a turn slow; adding two points to
   A4's army reads **0.030 / 0.060 / 0.330** with A5's census. The
   spread rungs at the cap go 0.70–0.80 at four points, 0.03–0.33 at
   five, ~0 at six. Army size is a second-order cost.

So the ladder row reads PASS with its provenance in the same cell: the
per-model architecture plays A5 at the bar's level from imitation and
keeps it under reward; the per-model trainer does not learn it from
reward at twice this budget, with or without the credit build, and
fails one point down. Both halves are the finding.

## Provenance

| field | value |
|---|---|
| date | A5c launched 2026-09-17 23:42; A5d 2026-09-18 00:57; A5e 01:16; the 1,200-game clone read 04:05; A5f and A5g 03:46; A5h 08:22; every PPO arm stopped by 12:50; A5-bodies 12:56 (resumed 17:01 after a reboot at ~15:20), read 19:40; A5-points 17:05, read 20:35; the 2,000-game clone fitted 17:01–20:42 (a first fit lost to the reboot), read 20:45; A5i 20:46–2026-09-19 00:37, read 00:46 |
| GPU / no-GPU | GPU (RTX 4090), up to fifteen trainers and a clone fit on a 31 GB box — memory, not the GPU, set the stops |
| seeds | 1 / 2 / 3 on every PPO arm; the clone at fit seed 0 on games 800000+ with the last 240 (400 for the 2,000-game fit) held out; A5f–A5h are three seeds off one warm start, so their agreement is the clone's and not three samples |
| n | 100 at 700000+ (every final and mark read, greedy); 30 at 500000+ in-run every 512 rounds |
| config | `configs/experiments/curriculum/a5.yaml` (A5c, A5d, A5f, A5h's twin; the clone); `a5_hold.yaml` (A5e, A5g, A5h: a5 + `objective_hold` w1.0 crowding 1.0); `a5_bodies.yaml` (eight squads, three points at x=35); `a5_points.yaml` (six squads, five points); every bar 1.000 |
| decode | none on the per-model facade |
| paired | per seed by rounds against A5b (A5c, A5d) and A5d (A5e); A5f–A5h against the clone on identical seeds |
| comparator | `squad_march_take` 1.000 / held 6 / 6.77 turns on a5 (4.43 on bodies, 6.71 on points); A5b 0.260 / 0.180 / 0.160 at 245,760; the 1,200-game clone 0.930 / 5.93 / 6.96 |
| opponent | none |
| budget | 245,760 rounds at 128 per update for A5c–A5e (stopped at 75k / 110k / 108k for memory, see the amendments); 122,880 for A5f–A5h (stopped at 89k / 89k / 44k by decision), A5i (run to the end) and the half-steps; `--ent-coef 0.003` throughout; `--credit actor` on every A5 PPO arm after A5b |
| code revision | A5c on `14998f2` (the credit); A5d–A5h on `ba622a4` (the scale fix) and after; the clones on `e632084` (`compact_tokens`); branch `feature/per-model-actor-credit`, PR #383 stacked on #382 |
| checkpoint | every 512 rounds under `checkpoints/per_model/per-model-a5*-2026-09-1[78]-*-s{1,2,3}a5{c,d,e,f,g,h,i,bodies,points}` (A5i and the half-steps read at `last.pt` = 122,880; the stopped arms at their last `pm-*.pt`); clones `checkpoints/per_model/clones/take-a5-{1200,2000}-s0.pt` |
| coherency | greedy at 700000+: A5d 0.21–0.27, A5e (n/a), the clones 0.744 / 0.771, A5i 0.78–0.79 (sampled 0.775–0.791 at 900000+), A5-bodies 0.19–0.36, A5-points 0.19–0.21; the bar 0.822 on a5, 0.759 on bodies, 0.832 on points |
| Wandb | `curriculum-a5`: A5c `lly5o8n4` / `5hy0hl7s` / `orcdnor7`; A5d `hemrqmbj` / `x10xp0iw` / `wt4cz7xs`; A5e `jeqtuhnq` / `5mynmsc2` / `77rr4xyi`; A5f `j8u2qkep` / `qomreieo` / `3p9ydybu`; A5g `hsmeo4ut` / `auwr7xyy` / `kvjuhpiu`; A5h `sorye78j` / `kmkmyqxi` / `76c3y9lr`; A5-bodies `5qbyx99r` / `w15c7xih` / `3vb272ip` then `j1e0mpam` / `0ioij69h` / `nazcv56j` after the resume; A5-points `uagmul66` / `ici5py46` / `wedjjozy`; A5i `cb7pmoks` / `7hvtp7ju` / `m2ujugw1` |
| pre-registration | `reports/2026-09-17-curriculum-A5c-preregistration.md` (+2 amendments), `reports/2026-09-18-curriculum-A5e-A5f-preregistration.md` (+8), `reports/2026-09-18-curriculum-A5-half-steps-preregistration.md` (+3) |

## The read

### The actor credit: a PPO change first, then nothing

The build (`Credit.actor` in `envs/per_model/reward_timing.py`) was the
speed screen's named test: pay a model its own action term where the
mean had divided it by the alive count, and land each per-model state
term's value on the model's own step of the turn instead of paying the
army mean once at the close. A5's reward has no state term — the travel
term to the mover and a global coverage term — so A5c tested the first
half alone.

**A5c, the first cut, paid the actor undivided and was a PPO change.**
PPO here normalises advantages but not value targets, so a payment 24×
larger is a return 24× larger: at 20k rounds the returns read 13× A5b's,
the pre-clip gradient norm 5×, and the clip was binding on every update
— the value loss owned the gradient. In-run success 0–1.3% at 75k where
A5b was at 15–25%. The credit was rewritten the same night (`ba622a4`)
to divide every payment by the model count: the actor's action term
lands where the mean put it, the common payments shrink by the army
size, and the state credits are per model. A5c ran on as the control
until memory took it at ~75k.

**A5d, the scaled credit, moved nothing.** At ~110k rounds it reads
0.050 / 0.070 / 0.010 with 3.9 / 3.8 / 2.7 of six held, coherency
0.21–0.27, turns at the ten-round cap on every seed — it never finishes.
Its census is A5b's: whole points abandoned rather than crowded, seed 3
leaving the near column's point empty in 92–98% of episodes. In-run it
averaged 1–6% over 60k–110k where A5b's in-run evaluations averaged
23.9 / 21.9 / 12.5% over its run with peaks of 48 / 43 / 30% at
174k / 133k / 211k. MOVES — ahead of A5b on 3/3
— is not shown at any read; the pre-registered FAIL applies.

**A5e, the state half, is a tenth.** `objective_hold` at crowding 1.0,
paid per model on the actor's step, the term the speed screen's S1 read
NULL under the mean: 0.170 / 0.060 / 0.060 at ~108k with 4.0 / 3.7 / 2.8
held, against A5d's 0.050 / 0.070 / 0.010 and 3.9 / 3.8 / 2.7 at the same
rounds. In-run 1–14% over 60k–110k against A5d's 1–6%. Ahead of A5d on
held on every seed and on success on two — a consistent, small edge in
arrival, and the first upward reward signal on this rung from any
per-model arm. Nowhere near a pass, and MOVES asks for 245,760 rounds
neither arm reached.

### The clone: 0.93, and one failure mode

`squad_march_take` cloned into the set network from 1,200 games on
800000+ (the last 240 held out, 40 epochs, fit seed 0): **0.930**, held
5.93 of six, 6.96 turns against the script's 6.77, about twenty of
twenty-four bodies on points, max stack 5.7 (the script's 5.8 — it
stacks two squads where it has a spare). One point is short: index 4 in
6% of episodes, index 0 in 1%. The D-route's clone reached the rung's
bound on C3b (D1b, 0.96); on A5 it is two hundredths under.

**Every one of its seven failures in a hundred is the same event.** A
census of the failed episodes (per-squad targets by turn against the
script's assignment on the same seeds): in all seven the clone's
assigned squad went to a point another squad was already taking, so two
squads shared a point and the assigned one stayed empty. None is a late
arrival, a dissolved squad or a body stuck behind a friend. The
script's assignment is a greedy matching with a tie-break by squad
order — its own plan order, not a function of the board — and D1
already recorded the network at chance on a teacher's plan order
held-out. So the residual is an assignment tie the observation cannot
carry, and more games close it only where the tie is in fact decided by
something on the board. The 2,000-game fit (60 epochs, the demonstrations
recorded at half precision to fit the box) was that test: **0.960**,
held 5.96, four failures of the same kind spread over three points —
the residual narrowed by three hundredths without changing its event.

### Anchored PPO from the clone: five reads, one band

Anchored PPO from the 1,200-game clone, three seeds each, read greedy at
n=100 on 700000+ at every 20k checkpoint:

| rounds | A5f (a5, anchor 10 / 0.03) | A5g (a5_hold, anchor 1 / 0.10) |
|---|---|---|
| clone | 0.930 | 0.930 |
| 20,480 | 0.950 / 0.920 / 0.930 | 0.920 / 0.910 / 0.960 |
| 40,960 | 0.850 / 0.930 / 0.940 | 0.910 / 0.950 / 0.940 |
| 61,440 | 0.880 / 0.910 / 0.950 | 0.900 / 0.970 / 0.900 |
| 81,920 | 0.930 / 0.910 / 0.960 | 0.960 / 0.900 / 0.920 |

Twenty-four reads, every one in 0.85–0.97, the clone's binomial band at
n=100 (0.930 ± 0.026); one seed at or above 0.95 on each row, never the
set; no trend over 60k rounds on either arm. HOLDS on both; IMPROVES —
turns ahead of the clone's paired on 3/3 — on neither. **A5h**, A5g with
the anchor an order of magnitude looser (coef 0.1, target 0.3), drifted
the other way: rolling in-run 84 / 89 / 88% at 40k against A5g's 90–96%
at the same rounds with the same reward, greedy 0.950 / 0.840 / 0.820 at
40,960. The coefficient question D2 left open reads: 10 holds, 1 holds,
0.1 loses the clone slowly, and none of the three lifts it.

Two readings that go with the table. In-run success (n=30 on 500000+)
ran 5–10 points above the greedy held-out read on every arm and every
row, so an in-run curve at 95% is a held-out read near 0.90 — never call
a pass from the curve. And the three seeds of each arm are three seeds
off one warm start: their agreement is the clone's, and a fourth seed
would say nothing the census does not.

### A5i: the same recipe from the bigger clone, read to the end

| rounds | s1 | s2 | s3 | turns | held of 6 |
|---|---|---|---|---|---|
| the clone | 0.960 | — | — | 6.97 | 5.96 |
| 20,480 | 0.980 | 0.940 | 0.970 | 6.87 / 7.09 / 6.90 | 5.98 / 5.94 / 5.97 |
| 40,960 | 0.970 | 0.970 | 0.970 | 6.94 / 6.94 / 6.89 | 5.97 ×3 |
| 61,440 | 1.000 | 0.970 | 0.970 | 6.80 / 6.93 / 6.99 | 6.00 / 5.97 / 5.97 |
| 81,920 | 1.000 | 0.980 | 0.960 | 6.80 / 6.87 / 6.91 | 6.00 / 5.97 / 5.96 |
| **122,880** | **0.960** | **0.980** | **0.960** | 6.88 / 6.81 / 7.04 | 5.96 / 5.98 / 5.96 |

Every read from 40k on has all three seeds at or above 0.95, and the
final one does: **HOLDS 3/3**, the rung's PASS clause. The in-run
rolling success (n=30 on 500000+) reached 95% within two evaluations
on every seed and never fell below 94.0 / 95.8 / 95.0 after — the dip
clause holds. Sampled play at n=100 on 900000+, paired against greedy:
−0.1 / −1.2 / +0.1 ± 1.2 vp, held 5.96–5.98, no do-nothing fingerprint
(stationary 0.16 either way). Paired against the clone on identical
seeds the turns read −0.09 ± 0.10 / **−0.16 ± 0.10** / +0.07 ± 0.13 —
ahead on two, behind on one, so **IMPROVES is not shown** on its 3/3
clause, and the two that are ahead are ahead by a tenth of a turn. The
census (n=100) is the clone's: 20.1–20.4 of 24 bodies on points and all
six held by turn 7, max stack 5.7–5.8, 4 / 2 / 4 failures each a single
short point. What reward did from a 0.960 start over 122,880 rounds is
a tenth of a turn on two seeds and nothing to the success rate outside
the clone's band.

### The half-steps

**A5-bodies** (eight squads of three over three points at the middle
column; the bar 1.000 in 4.43 turns) reads **0.960 / 0.930 / 0.860** at
122,880 with 2.94 / 2.90 / 2.70 of three held, turns 5.46 / 5.80 / 6.23,
coherency 0.19–0.36 (bar 0.759). The by-turn census (n=20): three
points of radius 4 hold about eight bodies between them, so "on points"
saturates near eight for every policy; the rows differ on the clock.
The script has five bodies on points after turn 3 and every point held
after turn 5. The arm has under one body on a point after turn 3 on
every seed, then arrives — s1 and s2 hold all three by turn 7, a turn
to a turn and a half behind the bar; s3 holds 2.4 and stops, its five
failures in twenty having 2.6 of three points empty and 5.5 bodies on
points at the end, A5b's under-arrival on a fifth of the episodes. The
prediction was a pass by 60k; the axis alone is not learned to the mark
within the cap. It does reach the high nineties in-run on every seed
and 0.93–0.96 held-out on two where A5 never reached a rolling 50%.

**A5-points** (six squads of three over five points; the bar 1.000 in
6.71 turns) reads **0.030 / 0.060 / 0.330** at 122,880 with 2.47 / 2.92 /
3.83 of five held, the clock run out on every seed (9.6–10.0 of ten
turns), coherency 0.19–0.21. The census is A5b's: a third of the bodies
on points at the end against the script's seven eighths, whole points
abandoned by seed (s1 leaves three of the five empty in 60–85% of
episodes, s2 one point in 90%, s3 one in 55%), max stack under three,
and on s1 the bodies on points *fall* from 7.1 after turn 3 to 3.6 after
turn 5 — squads walk off points they reached. In-run it never reached a
rolling 10% on two seeds and peaked in the thirties on the third.

**Together the pair puts the wall on the point count.** At the same
rounds and with the same trainer: A4 (four squads, three points)
0.93–0.97; A5-bodies (twice the army, the same three points) 0.86–0.96,
a turn slower; A5-points (A4's army plus two squads, five points)
0.03–0.33 with A5's census; A5 itself never a rolling 50%. Doubling the
army costs a turn and one seed's pass; two more points cost everything.
The curve in the point count is steep — 0.70–0.80 at four points (A3 at
the cap), 0.03–0.33 at five, ~0 at six — and the army size is a real but
second-order cost. On the letter the quadrant is "both FAIL"; the
magnitudes are the finding.

## What it says

- **The credit hypothesis is closed.** Since the A3 speed screen the
  record carried "a body's own credit is diluted 1/24 by the mean" as
  the untested explanation for A5's under-arrival. Built and run three
  ways it moved nothing, and its first cut was a lesson of its own: on
  this trainer advantages are normalised and value targets are not, so
  a reward whose scale changes is a PPO change, and it has to be read
  as one (returns, gradient norm, clip fraction) before its arm is read
  at all. Do not fund another attribution build against A5's arrival.
- **A clone's residual is read by censusing its failures, not its
  match.** The 1,200-game clone matches the teacher's displacement head
  at 0.72 held-out and reads 0.930 by the rung's criterion; both numbers
  say "close" and neither says why. Twenty failed episodes say: one
  event, every time, a tie the observation does not carry. That is the
  reading that tells whether more games can help (only if the tie is
  decided by something on the board) — the D1 rule "score a clone by
  the rung's criterion and the ordering" gains "and census the
  failures". The 2,000-game fit narrowed it (0.930 → 0.960), so the tie is at least partly on the board.
- **Anchored PPO from a clone holds it and does not lift it, at every
  coefficient that holds.** D2 left the coefficient as the IMPROVES
  question; it is answered — 10 holds, 1 holds, 0.1 loses the clone
  slowly, none improves, twenty-four reads in one band. A policy that
  the anchor keeps within 0.03 nats of its start cannot move a success
  rate two hundredths; a policy free to move loses the plan. The lever
  that lifts a clone is not the anchor's strength and is not more
  rounds; it is unnamed, and until it is named the D-route's ceiling
  on a rung is the clone's read.
- **A5's wall is the number of points to cover at once, and the curve
  is steep.** With one trainer at one budget the spread rungs read
  0.70–0.80 at four points, 0.03–0.33 at five, ~0 at six, while
  doubling the army at three points costs a turn and a seed. A success
  that is a conjunction over N points is what the per-model critic has
  struggled to value since A3 (explained variance falling from 0.7–0.9
  to 0.3–0.4 there), and the census at five points shows the policy
  side of the same thing — points reached and then walked off. The
  next lever on A5 is about valuing or decomposing a conjunction, not
  about bodies, credit or the start.
- **Never call a pass from the in-run curve.** Across every arm here the
  n=30 in-run read on 500000+ ran 5–10 points above the greedy n=100
  read on 700000+; A5-bodies' seeds sat at 93–98% rolling and read
  0.86–0.96 held-out. The curve says when to read; the read decides.
- **Three seeds off one warm start agree because they share the start.**
  A5f–A5h's per-seed spread is the clone's binomial band, not seed
  variance; a fourth seed from the same clone would add nothing the
  census does not. Vary the clone before counting seeds.

## What was not done

- No whole-army control on any of these arms — A5's own control (0.80 / 0.94 / 0.98 at 245k) stands as the rung's.
- A5c–A5e were not run to budget: memory on a 31 GB box carrying up to fifteen trainers and a clone fit, then a reboot; each stop is an amendment with the read at the stop.
- A5f–A5h were stopped by decision at 89k / 89k / 44k; the chance of a 0.95 × 3 read from a true rate near 0.93 is about one in fifty.
- The 2,000-game fit ran twice: the first was killed by the reboot 4.5 hours in with nothing saved.
- A5's IMPROVES was never shown by any lever (A5i: ahead of the clone on two seeds by a tenth of a turn); the anchor's coefficient is answered, the lever that lifts a clone is not.
- No whole-army control was run for the imitation route; the D rungs established it on C3b and the phase facade has its own clone tooling, but a whole-army clone of the bar on A5 was not fitted.
- A5-points warm-started from A4x, at twice the cap, is the one reward-only arm the half-steps name and it was not run.
- One fit seed per clone. Two fit seeds agreed to a thousandth on D1; here the second clone differs from the first in games as well as seed, so the 0.930 → 0.960 step carries a fit-seed term nobody measured.
