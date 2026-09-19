# Curriculum rung E1: two armed armies on a flat board — the whole-army control learns the guns and loses the formation; the per-model arm learns neither, and A5i's walk does not survive contact

**Verdict first.** The join of the ladder's bodies and guns branches
(#340, #381): A5's army of twenty-four in squads of three, both sides
armed, a mirror `squad_march_take` opponent, twenty rounds on a flat
board. Three trainers, three seeds each, read against `squad_march_take`
on each trainer's own facade, paired per episode at n=180 on 700000+:

| row | vp v script, paired | held (clause) | coherent (clause) | alive (bar 0.27) | verdict |
|---|---|---|---|---|---|
| whole-army control, 120 epochs | −15.6 ± 9.2 / **+73.4 ± 10.3** / **+45.2 ± 10.5** | 2.47 / 3.05 / 1.98 (≥ 2.46) | 0.860 / 0.780 / 0.773 (≥ 0.82) | 0.58 / 0.50 / 0.47 | **FAIL on the letter on every seed, by a different clause each; beats the script by 45–73 vp on two** |
| per-model from scratch, 122,880 | **−158 / −152 / −160** (± 3–5) | 0.76 / 0.71 / 1.24 (≥ 2.68) | 0.58 / 0.38 / 0.33 (≥ 0.84) | 0.52 / 0.23 / 0.40 | **FAIL on every clause, every seed** |
| per-model from A5i, 122,880 | **−68 / −180 / −159** (± 5–6) | 1.26 / 0.43 / 0.58 | 0.37 / 0.74 / 0.47 | 0.33 / 0.14 / 0.18 | **FAIL on every clause, every seed** |

**FAIL as pre-registered, all nine runs.** The whole-army control found
the guns: on two seeds it kills its way to the largest margins over a
script on this ladder while dropping below the script's formation and,
on one, its points. The per-model arm found nothing: 150–160 vp behind
a script that walks and shoots on the way, a fifth of the bar's points,
a tenth to a fifth of its bodies on any point. **The companion from
A5i — the policy that puts twenty of twenty-four bodies on six points
by turn 7 on A5 — had that walk overwritten within 10,240 rounds** and
ended one seed at −68 and two at the scratch arm's level. The goal set
on 2026-09-19 asked whether the allocation survives guns: it does not,
and the census says which half broke — both. The E branch stops here
for the per-model trainer.

## Provenance

| field | value |
|---|---|
| date | 2026-09-19: control 01:15–01:35 (60 epochs) and 01:41–02:05 (extended once to 120), read 02:15; per-model scratch 01:15–09:40 and companion 01:17–09:05, read 09:45–10:20 |
| GPU / no-GPU | GPU (RTX 4090); nine E1 trainers, then six beside D3's six |
| seeds | 1 / 2 / 3 per trainer; the companion seed for seed from A5i's `last.pt` (three seeds off three A5i seeds, themselves off one clone of the bar) |
| n | 180 at 700000+ (final, paired per episode against the script on the same facade); 30 at 500000+ in-run every 512 rounds (per-model) and every epoch (control); 100 at 900000+ greedy against sampled; 60 at 700000+ by-turn census |
| config | `configs/experiments/curriculum/e1.yaml`: 24 v 24 in squads of three, both sides armed (range 12), flat board, fixed layout, six points, twenty rounds, unrefereed; opponent `scripted_baseline` wrapping `squad_march_take` |
| decode | none on the per-model facade; K=1 on the phase facade |
| paired | per episode against `squad_march_take` on the arm's own facade (`just measure-paired` for the control, `just measure-rung` for the per-model rows) |
| comparator | `squad_march_take` against itself, n=180 on 700000+: phase facade vp +20.2 (sd 102), per-model facade vp +20.2 ± 7.6, win 0.567, held 2.93, on_obj 0.946, alive 0.27–0.28, coherent 0.873 — the bridge diverges under a recorded rule (`shooting.targets_judged_after_casualties`), the two facades' rows within one SE |
| opponent | `squad_march_take`, the mirror |
| budget | control 60 epochs of 2048 steps extended once to 120 (the curve still rising at 60); per-model 122,880 rounds at 128 per update (`--rollout-rounds 32 --num-rollout-envs 4`), `--ent-coef 0.003`, the `mean` credit; the companion `--warm-start-from` A5i without an anchor |
| code revision | pre-registered `5566b7b` (+4 amendments to `bbb9df6`); runs on `10eeb1d`; branch `feature/per-model-actor-credit`, PR #383 |
| checkpoint | control `checkpoints/ppo-transformer-curriculum_e1-…-2026-09-19-01-41-2*-s{1,2,3}e1-ctl-x/last.ckpt` (epoch 120); per-model `checkpoints/per_model/per-model-e1-2026-09-19-01-15-10-s{1,2,3}e1` and `…-01-17-*-s{1,2,3}e1w`, `last.pt` at 122,880, every 512 rounds kept |
| coherency | greedy at 700000+: scratch 0.58 / 0.38 / 0.33, companion 0.37 / 0.74 / 0.47, control 0.86 / 0.78 / 0.77 (the script 0.87); sampled at 900000+ 0.27–0.50 |
| Wandb | `curriculum-e1`: control `iszzu87u` / `t7lm8pj3` / `a6zyzu3g` then `pp3ep94b` / `rkdln3xc` / `njr0mg5a`; scratch `n51yisrb` / `ronr0zl9` / `5jqvcdt5`; companion `qjpti8yr` / `5jaor2py` / `1d3rohef` |
| pre-registration | `reports/2026-09-17-curriculum-E1-preregistration.md` (+4 amendments); the 2026-09-17 23:13 runs were stopped by decision within the hour and never read |

## The read

### The whole-army control learned the guns and lost the formation

The control finished sixty epochs in twenty minutes with its in-run vp
still rising on every seed (by quarter −194 → −19, −196 → −21, −208 →
−69) and ran the pre-registered once-only extension to 120. Read on the
phase facade, K=1, n=180, paired against the script:

| seed | vp | script | paired | t | held | coherent | alive | on obj | win |
|---|---|---|---|---|---|---|---|---|---|
| s1 | 4.7 | 20.2 | −15.6 ± 9.2 | −1.68 | 2.47 | 0.860 | 0.577 | 0.877 | 0.43 |
| s2 | **93.6** | 20.2 | **+73.4 ± 10.3** | +7.10 | 3.05 | **0.780** | 0.502 | 0.781 | 0.79 |
| s3 | **65.4** | 20.2 | **+45.2 ± 10.5** | +4.31 | **1.98** | **0.773** | 0.466 | 0.808 | 0.76 |

Each seed fails a different clause: s1 vp (−15.6 against a −9.2
bound), s2 coherency (0.780 against 0.82), s3 held (1.98 against 2.46)
and coherency. Two seeds beat the script by 45–73 vp at t 4.3–7.1 —
the largest margins over a script on any rung of this ladder — and they
do it by killing: opponent VP 121–136 against the ~200 the script
concedes to itself. The E clause was a conjunction of three readouts
at the script's levels, and a policy that wins the fight and loses the
formation fails it; that is what the clause was for, and the row
reports both facts.

### The per-model arm from scratch plays a different game from every comparator

| seed | vp | win | held | on obj | coherent | alive | stat / hold |
|---|---|---|---|---|---|---|---|
| s1 | −158.1 ± 3.3 | 0.000 | 0.76 | 0.096 | 0.584 | 0.52 | 0.00 / 0.21 |
| s2 | −151.9 ± 4.8 | 0.028 | 0.71 | 0.169 | 0.383 | 0.23 | 0.49 / 0.09 |
| s3 | −159.8 ± 3.3 | 0.000 | 1.24 | 0.201 | 0.328 | 0.40 | 0.14 / 0.13 |

The script holds 2.93 with a tenth of a point of variance between the
facades; the scratch arm holds 0.7–1.2 with one or two bodies on any
point at any turn of the game (census: 1.5 → 1.6 → 0.8 → 1.3 bodies on
points on s1 at turns 3 → 5 → 7 → 20 against the script's 8.8 → 10.7 →
10.1 → 6.8). Its `alive` is at or above the bar's on two seeds — it is
not being shot off the points, it never reaches them — and its vp is
150–160 behind on every seed, more than the whole spread of any
comparator on this rung. On the health panel nothing is red: the
in-run curve rose from −250 to −150 and stopped.

### The companion from A5i: the walk did not survive contact

| seed | vp | win | held | on obj | coherent | alive | stat / hold |
|---|---|---|---|---|---|---|---|
| s1 | −67.9 ± 5.3 | 0.128 | 1.26 | 0.261 | 0.374 | 0.33 | 0.42 / 0.11 |
| s2 | −179.7 ± 5.1 | 0.006 | 0.43 | 0.128 | 0.735 | 0.14 | 0.78 / 0.05 |
| s3 | −159.1 ± 5.8 | 0.061 | 0.58 | 0.178 | 0.467 | 0.18 | 0.66 / 0.06 |

A5i puts 20 of 24 bodies on six points by turn 7 on A5. Loaded on E1
untrained it plays that walk into twenty-four rifles (−102 vp on three
episodes at the loading check). Trained on E1's reward from there, the
walk was gone within 10,240 rounds: the in-run stationary share read
0.5–0.9 from the first evaluations on every seed, and the census at the
end has 2.7–2.9 bodies on points at turn 3 against A5i's seventeen —
one seed climbs to 3.7 by turn 7, the other two never pass 3.8 and end
at 0.9, the points they reached lost (held 1.3–1.6 at turn 5, 0.4–1.2
at the end). What replaced the walk shoots no better than the scratch
arm (alive 0.14–0.33, below the bar's 0.27 on two seeds; vp 88–180
behind). One seed (s1, −68) is the per-model trainer's best row on E1
and is still a point and a half and 88 vp short.

**Sampled play is worse than greedy on five of six seeds**, by 13–32 vp
paired (n=100 on 900000+): the policies are diffuse, and the greedy
argmax a score reports is the better half of what training rolled out.
No do-nothing fingerprint on the scratch arm (stationary 0.00–0.49);
the companion's 0.42–0.78 is the residue of A5i's walk being unlearned
into standing still.

## What it says

- **A warm start does not survive a change of game.** T1 (A5 from A4x)
  carried a walk onto a bigger board and did not rescue it; C1 (from
  A3x) carried a walk past an unarmed enemy and it held; E1 (from A5i)
  carried the ladder's best walk into rifles and lost it in 10k rounds
  with nothing learned in its place. The C rungs' rule — warm-start
  from the rung below — was measured on rungs that added an enemy who
  does not shoot back; it does not extend to a rung that adds guns on
  both sides. On this trainer the walk is unlearned before the shooting
  is learned, and the reward's signal in the first 10k rounds is the
  casualties, not the points.
- **The per-model trainer has not learned to shoot from reward on any
  rung, and E1 is where that bites.** C2 gave it guns against it (it
  walked at the script's speed under fire); C3 gave it one armed squad
  (it found the dash, not the plan); C3b (the plan's first half, then
  stopped); E1 gives both sides guns and asks for walk and fire together
  — 150–200 vp behind. The whole-army control learns to kill at the
  cost of formation; the per-model arm learns neither. **Before another
  E rung, the per-model trainer needs a guns-only rung the way C2 was
  the enemy-only rung**: the bar's walk held under an anchor while the
  shooting head learns on top of it is the untested arm this record
  names, and it is a D-route arm, so D3's finding applies — the anchor
  will hold the walk and may add nothing.
- **Read a whole-army control's turns and coherency beside its vp on
  every guns rung** (C2's rule, confirmed): a control that beats the
  script by 73 vp and fails the rung is measuring what the clause
  measures, and the ladder row carries both.
- **Sampled play is a diagnostic on a failing per-model row.** Greedy
  ahead of sampled by 13–32 vp on five seeds says the policy is
  diffuse, not converged; on A5i the two agreed within 1.2 vp. Read the
  gap before spending rounds.

## What was not done

- No anchored companion (A5i held under the KL anchor while learning to
  shoot): named above as the next arm; D3 says the anchor will hold the
  walk and may add nothing, so it is a one-arm question with a likely
  answer.
- No guns-only per-model rung between C2 and E1 (both sides armed, the
  walk given, the shooting to learn); the ladder's E branch assumed the
  C branch had taught the guns, and C3b said it had taught half.
- The control's own 60-epoch reads were not scored at n=180 (its 120
  rows are the comparator; its 60 rows are in-run only).
- E2 and E3 are not reached; their old Stage 0 configs stay untouched.
