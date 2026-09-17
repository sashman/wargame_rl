# Curriculum rung C3b: with success read at the end of the game, nobody holds the point — FAIL on every seed of every trainer, and the arm's failure is now the plan's second half

**Verdict first.** The second arm on the sequencing rung (#340, #372) —
C3 with one flag changed, `terminate_on_success: false`, so
`all_objectives_occupied` is judged on the final board after twelve
rounds instead of the instant it first holds — reads greedy success
****0.200 / 0.130 / 0.230**** on the per-model arm warm-started from C2 at 245,760 rounds,
n=100, and ****0.020 / 0.020 / 0.060**** on the from-scratch companion; the first kill
precedes the first unarmed arrival in **0.86 / 0.77 / 0.85** (mark 0.90) (arm) and 0.67 / 0.79 / 0.91
(companion). ****FAIL as pre-registered** — under 0.90 on every seed by a wide margin, the ordering clause missed on every seed of the arm by 0.04–0.13, no seed of any trainer past a rolling 51% in-run (peaks 37 / 42 / 51 for the arm, 22 / 14 / 12 for the companion).** The whole-army control fails at 120
epochs on every seed (**0.280 / 0.000 / 0.450**). The escort reads
0.990 with 96% alive on the same scenario; plain `squad_march_take`
0.280. With the instant win gone, the arm from C2 learned the first half of the escort's plan — it wipes the blockers in 75–79% of episodes, up from 16–30% on C3 — and stopped: from round five it holds about two and a half points with nine or ten bodies on them, stands still on 47% of its openings, and leaves the blockers' point and the far point empty in half the episodes at the end. Its vp margin (+60 to +71) is the escort's (+63.5); its success is not. The prediction on file — the dash unlearned, one or two seeds at the mark — is right about the unlearning and wrong about the mark.

## Provenance

| field | value |
|---|---|
| date | launched 2026-09-17 09:55 (control, arm and companion together); control extended 10:03–10:14; first per-model leg to 122,880 exited 12:10–12:23, resumed in place 12:25, exited 14:22–14:32 |
| GPU / no-GPU | GPU (RTX 4090), nine trainers at launch, six from 10:14 |
| seeds | 1 / 2 / 3; rollout layouts at seed×100+; the arm from C2's `last.pt` of the same seed (C3's checkpoints carry the dash and were not used), fresh optimizer; the companion from scratch — **not paired on init** |
| n | 100 at seed base 700000 (final, the 122,880 readout, the ordering census); 30 at 500000 (in-run, every 512 rounds; n=20 after the resume, #346); 100 at 900000 (greedy against sampled); 20 at 700000 (by-turn census) |
| config | `configs/experiments/curriculum/c3b.yaml` — `c3.yaml` with `terminate_on_success: false`; unrefereed by design; movement and shooting stepped, 24 phase-clock turns, every episode runs to the end |
| decode | none on the per-model facade; K=1 on the control |
| paired | per episode against `scripted_escort` on identical seeds |
| comparator | `scripted_escort`, both facades identical: success 0.990, held 3.97, `alive` 0.963, `on_obj` 0.993, vp +63.5 ± 2.0, coherent 0.858; first kill at turn 6.0, blockers wiped at 11.0, first unarmed body on their point at 12.9, the kill before the arrival in 100%. `squad_march_take` (must fail): 0.280, alive 0.507. C3's own rows beside every row (one flag apart) |
| opponent | `scripted_baseline` wrapping `hold_and_shoot`: three models, one unit, on the point at (35, 27), four wounds, rifle range 12 with eight attacks |
| budget | per-model 245,760 rounds at 128 per update (`--rollout-rounds 32 --num-rollout-envs 4`), `--ent-coef 0.003`, resumed in place at 122,880 with cadences and `--n-eval-episodes` explicit; control 60 epochs extended once to 120 |
| code revision | `8d456a8` on `feature/curriculum-c3` (PR #371) at launch; docs-only commits after |
| checkpoint | `last.pt` at 245,760; `checkpoints/per_model/per-model-c3b-*-s{1,2,3}c3b` (arm) and `-s{1,2,3}c3bs` (companion); control `checkpoints/ppo-transformer-curriculum_c3b-…-s{1,2,3}c3b-ctl-x2/last.ckpt` |
| coherency | greedy 0.660 / 0.543 / 0.641 (arm), 0.490 / 0.606 / 0.342 (companion) at 700000+; against sampled 0.45 / 0.36 / 0.37 and 0.23 / 0.37 / 0.20 at 900000+ |
| Wandb | `curriculum-c3b`: arm `u2fxw35y` / `7a709exu` / `7iz53zqn`, companion `zijgxv6m` / `720gc6a6` / `uf7bfm0x`, resumed halves arm `zr824rjk` / `ce82mijd` / `ydetbauh`, companion `cwir0s7g` / `ryk9o0lz` / `srw69nax`; control `wnle147m` / `qn1edzf8` / `uzsdlznj` (to 60), `bfibh5va` / `ugs0hohp` / `i8sh0mm1` (60–120) |
| pre-registration | `reports/2026-09-17-curriculum-C3b-preregistration.md` at `8d456a8` (2026-09-17 09:50); amendment 1 at `ac5e5ff` (the control's 60-epoch read and the extension, 10:05) and amendment 2 at `a5a1b79` (the control's 120-epoch read, 10:18), both before any per-model number |

## The read

| row | success | held of 4 | on obj | alive | vp | coherent | stat | in-run: peak rolling; last 40 |
|---|---|---|---|---|---|---|---|---|
| `scripted_escort` (the bar) | 0.990 | 3.97 | 0.993 | 0.963 | +63.5 ± 2.0 | 0.858 | — | — |
| `squad_march_take` (must fail) | 0.280 | 2.33 | 0.999 | 0.507 | +57.1 ± 4.5 | 0.939 | — | — |
| **arm s1** (from C2) at 245,760 | **0.200** | 2.50 | 0.793 | 0.586 | +62.2 ± 3.8 | 0.660 | 0.47 | 37% at 166k; 17.1 |
| **arm s2** at 245,760 | **0.130** | 2.36 | 0.695 | 0.566 | +60.2 ± 3.9 | 0.543 | 0.47 | 42% at 163k; 24.2 |
| **arm s3** at 245,760 | **0.230** | 2.68 | 0.768 | 0.601 | +71.2 ± 3.8 | 0.641 | 0.46 | 51% at 80k; 21.1 |
| arm at 122,880 (readout) | 0.080 / 0.220 / 0.220 | 1.80 / 2.65 / 2.62 | 0.46 / 0.62 / 0.75 | 0.56 / 0.58 / 0.56 | +32 / +62 / +65 | 0.37 / 0.48 / 0.50 | 0.45 / 0.18 / 0.47 | — |
| companion s1 (scratch) at 245,760 | 0.020 | 1.58 | 0.426 | 0.608 | +32.6 ± 5.1 | 0.490 | 0.41 | 22% at 225k; 11.2 |
| companion s2 at 245,760 | 0.020 | 1.55 | 0.528 | 0.522 | +25.2 ± 4.0 | 0.606 | 0.38 | 14% at 154k; 3.4 |
| companion s3 at 245,760 | 0.060 | 1.79 | 0.470 | 0.491 | +28.6 ± 4.0 | 0.342 | 0.34 | 12% at 208k; 5.4 |
| companion at 122,880 (readout) | 0.060 / 0.000 / 0.020 | 1.55 / 1.33 / 1.57 | 0.31 / 0.26 / 0.40 | 0.63 / 0.63 / 0.48 | +20 / +12 / +27 | 0.34 / 0.36 / 0.34 | 0.27 / 0.26 / 0.38 | — |
| control at 60 epochs | 0.280 / 0.120 / 0.200 | 3.09 / 2.35 / 2.18 | 0.94 / 0.81 / 0.77 | — | +38 / +32 / +26 | 0.72–0.79 | — | never 80% |
| **control at 120 epochs** | **0.280 / 0.000 / 0.450** | 2.77 / 2.15 / 3.14 | 0.83–0.85 | — | +35 / +29 / +40 | 0.62–0.67 | — | never 80% |
| C3's arm at 245,760 (one flag apart) | 0.660 / 0.920 / 0.830 | 2.77 / 3.35 / 3.14 | 0.70–0.78 | 0.62–0.74 | −0.5 to −4.6 | 0.44–0.55 | 0.02–0.06 | — |

Every game runs the full 24 turns now, so vp accrues for twelve rounds
and the paired vp column against the bar is meaningless (+0.00 on
every row: identical episode lengths); read the vp margin itself.
⚠ In-run rows after the resume at 122,880 are n=20 (#346).

**The ordering clause** (the sequencing census at n=100 on 700000+):

| policy | success (n=100) | first kill | blockers wiped (share) | first unarmed body on their point (share) | **kill precedes arrival** | alive | armed squad dead of 3 |
|---|---|---|---|---|---|---|---|
| `scripted_escort` | 0.990 | 6.0 | 11.0 (0.99) | 12.9 (0.90) | **1.00** | 0.963 | 0.05 |
| `squad_march_take` | 0.280 | 6.8 | 12.3 (0.60) | 9.1 (0.52) | 0.77 | 0.507 | 0.92 |
| arm s1 | 0.200 | 6.1 | 11.7 (**0.75**) | 8.1 (0.75) | **0.86** | 0.586 | 0.98 |
| arm s2 | 0.130 | 7.1 | 12.3 (**0.77**) | 8.1 (0.65) | **0.77** | 0.566 | 0.81 |
| arm s3 | 0.230 | 6.3 | 11.6 (**0.79**) | 8.3 (0.62) | **0.85** | 0.601 | 1.02 |
| companion s1 | 0.020 | 7.2 | 11.1 (0.54) | 8.0 (0.62) | 0.67 | 0.608 | 0.83 |
| companion s2 | 0.020 | 7.3 | 11.4 (0.41) | 7.1 (0.32) | 0.79 | 0.522 | 1.00 |
| companion s3 | 0.060 | 8.8 | 13.8 (0.68) | 10.4 (0.31) | 0.91 | 0.491 | 1.04 |
| C3's arm (instant win) | 0.66 / 0.92 / 0.83 | 6.5–8.1 | 0.16–0.30 | 8.5–8.7 (0.85–0.90) | 0.62–0.71 | 0.62–0.74 | 0.56–1.09 |

The arm now fires first and wipes the blockers in three quarters of
episodes — the half of the plan the instant win let it skip — and
still sends an unarmed body in at turn 8.1, before the wipe at 11.7,
and loses two of its three riflemen doing it. The escort's armed squad
loses 0.05.

**Greedy against sampled** (900000+, n=100, paired): vp −10.4 ± 4.3 / +2.8 ± 3.9 / +9.5 ± 4.5 (arm), +4.7 / +14.4 / +23.6
(companion); `held` 2.44–2.76 greedy against 2.44–2.68 sampled on the
arm; win rate 82–94% on the arm either way; coherency 0.59–0.68 greedy
against 0.36–0.45 sampled (arm); 116–121 decisions per episode. The
sampled arm plays the same game; the sampled companion is worse than
its greedy self by 5–24 vp, a diffuse policy.

**Health panel, last quarter** (~480 updates per seed): arm — explained variance **0.25 / 0.20 / 0.24** (the ladder's lowest),
clip fraction 0.13 / 0.16 / 0.13, ratio p99 1.57 / 1.65 / 1.55,
displacement entropy 0.87 / 0.91 / 1.09, declaration entropy 0.04,
advantage std 1.0–1.2; companion — explained variance 0.29 / 0.51 /
0.38, clip 0.17 / 0.21 / 0.25, ratio p99 1.62–1.86. The policy is
settled (clip and tail calm) on a critic that cannot value the final
board it rarely reaches.

**The by-turn census** (n=20 on 700000+, points held of 4 and bodies on
points of 12 at turn 9 → 12 → 16 → 24; which points are empty at the
end, by index, with 2 the blockers' and 3 the far one at y=36):

| policy | success (n=20) | held / on points, 9 → 12 → 16 → 24 | empty points at the end | max stack |
|---|---|---|---|---|
| `scripted_escort` | 1.00 | 0.7/1.8 → 1.3/3.5 → 3.3/9.5 → 4.0/12.0 | 0 / 0 / 0 / 0 | 3.0 |
| arm s1 | 0.20 | 3.0/9.3 → 2.6/9.4 → 2.6/9.9 → 2.5/10.1 | 0 / 0.35 / **0.45** / **0.70** | 3.2 |
| arm s2 | 0.15 | 2.3/8.9 → 2.7/8.8 → 2.5/8.9 → 2.4/8.6 | 0.10 / 0.25 / **0.60** / **0.65** | 2.9 |
| arm s3 | 0.10 | 2.5/9.6 → 2.6/8.7 → 2.6/8.6 → 2.6/8.7 | 0.10 / 0.55 / 0.50 / 0.30 | 2.3 |
| companion s1 | 0.00 | 2.2/8.4 → 1.9/6.3 → 1.5/4.6 → 1.5/4.5 | 0.35 / 0.50 / 0.75 / 0.95 | 1.8 |
| companion s2 | 0.00 | 1.9/9.3 → 1.6/7.0 → 1.4/5.7 → 1.4/6.0 | 0.15 / 0.50 / 0.95 / 1.00 | 2.3 |
| companion s3 | 0.10 | 1.7/3.9 → 2.1/5.3 → 2.0/6.0 → 1.9/5.8 | 0.35 / 0.20 / 0.90 / 0.65 | 2.1 |

**The arm's board is static from round five.** Nine or ten bodies on
points and two and a half points held at turn 9, the same at turn 24;
the escort has 1.8 bodies on points at turn 9 (waiting) and twelve at
the end. What the arm holds it keeps — the near points — and what it
does not reach by round five it never reaches: the blockers' point in
45–60% of episodes and the far point in 30–70%. The companion's board
*decays*: bodies on points fall from 8–9 at turn 9 to 4–6 at the end as
the blockers shoot them off.

## What it says

- **FAIL as pre-registered, every seed of every trainer, and the
  answer to the rung's question is no.** The per-model arm from C2:
  0.20 / 0.13 / 0.23, ordering 0.77–0.86. From scratch: 0.02–0.06. The
  whole-army control: 0.28 / 0.00 / 0.45. The escort: 0.990. Nothing
  trained here learned to shoot the blockers off the point *and then
  walk onto it and stay*.
- **The instant win was the whole of C3's success, and removing it
  moved the arm halfway to the plan.** On C3 the arm from C2 wiped the
  blockers in 16–30% of episodes and won by the dash; here it wipes
  them in 75–79% and fires first in 77–86% — the killing is learned,
  from a start that never held a gun. What it does not learn is what
  to do afterwards: it holds the two near points it walked onto by
  round four and parks (stationary 47% of openings, the ladder's
  highest) while the far point and the cleared point stay empty. The
  reward pays the same for standing on a held point as for walking to
  an empty one (the travel term is largely inert, § The travel reward
  audit), and the critic — explained variance 0.20–0.25 — cannot see the
  final board from round five.
- **The head start still helps here, and for the same reason it hurt
  on C3.** The companion from scratch reads 0.02–0.06 with a decaying
  board; the arm's C2 approach gets bodies onto the near points early
  and keeps them. The habit that was the dash on C3 is the near-point
  hold on C3b. The C-rung rule stands with the caveat C3 wrote: read
  the companion beside it.
- **The control fails harder than on any rung** (0.28 / 0.00 / 0.45):
  it walks in as `take` does and the blockers take the points back. On
  C2 it was speed; here it is the plan, and it has none.
- **Where the ladder stands.** #340's answer for this axis: neither
  trainer learns an ordered plan from reward alone at this budget, and
  the thing the per-model trainer lacks is not the order (it fires
  first) but the *continuation* — a policy that keeps going after the
  half of the plan it was paid for. That is the D rungs' question by
  design: D1 clones the escort into the set network (#331), D2 asks
  whether PPO can improve a policy that already carries the whole plan.
  The ladder goes there next. **Rule: on a rung whose solution has
  more than one phase, read the census by phase** — "killed the
  blockers", "arrived", "held to the end" — before saying which half
  a trainer learned; success alone reads 0.20 for a policy that has
  half the plan and 0.02 for one that has none.
- **Coherency 0.54–0.66 greedy, 0.36–0.45 sampled** — unpaid, as on
  every rung.

## What was not done

- No census of the control's failure beyond its `held` and `on_obj`.
- No arm from C3's checkpoints (they carry the dash) and no second
  budget; the symmetric cap already put both per-model runs at 245,760.
