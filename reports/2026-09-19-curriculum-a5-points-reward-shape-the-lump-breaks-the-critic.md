# Curriculum A5-points, the reward-shape arms: paying the criterion as a lump breaks the critic, and paying per point buys the near column

**Verdict first.** Two one-change arms on the half-step A5-points (six
squads of three over five points; the bar `squad_march_take` 1.000 in
6.71 turns), trained from scratch under the original recipe, three seeds
each, 122,880 rounds, read greedy at n=100 on 700000+:

| arm | the change | verdict | success at 122,880 | held of 5 | the original (2026-09-18) |
|---|---|---|---|---|---|
| **R1** | the success bonus paid in full however late (`terminal_bonus_speed_scaling: false`) | **FAIL — behind the original at every read** | 0.000 / 0.030 / 0.010 | 0.90 / 2.89 / 2.52 | 0.030 / 0.060 / 0.330; held 2.47 / 2.92 / 3.83 |
| **R2** | R1 plus 1.0 per point held on the final board (`terminal_objective_bonus: 5.0`) | **FAIL — behind the original at every read** | 0.020 / 0.030 / 0.000 | 2.41 / 2.93 / 2.38 | |

Neither arm is ahead of the original on any seed at 40,960, 81,920 or
122,880 (the original at 81,920 reads 0.22 / 0.22 / 0.11; the arms 0.00 /
0.00 / 0.02 and 0.00 ×3). The census is under-arrival and abandonment as on
every read of this half-step, with two new signatures: R1's first seed
walks ten bodies onto points by turn 3 and ends with two (held 0.90,
sampled play holds 3.20), and R2 at two thirds of the budget had the far
column empty in 90–100% of episodes on every seed — the per-point payment
bought the three near points and not the walk to the far two. **The
mechanism is on the panel: explained variance sat between −0.5 and +0.2 on
five of six seeds for the whole run** (the original 0.63–0.73, the discount
arm 0.78–0.85), and in-run success in the last quarter was 0.2–2.5% against
the original's 9–20%. A large terminal lump — 5.0 on a rare success, or a
per-point sum keyed to the clock — is a value target this critic cannot fit,
and PPO's advantages became noise. The original's remaining-rounds scale,
which the pre-registration called a defect, was also what kept the terminal
lump small beside the dense terms. **Three optimiser-and-reward arms on this
half-step now read the same policy: the discount made the critic better,
the terminal shape made it worse, the credit changed nothing — and the
policy is the same on every one.** What is left for the spread wall is
exploration or representation, not the reward's timing or shape.

## Provenance

| field | value |
|---|---|
| date | 2026-09-19: build and pre-registration 14:50–15:05, launched 15:15 (a 15:12 launch at the wrong cadence stopped at ~1,500 rounds and removed), trainers exited 19:05–19:13, final reads 19:13–19:25 |
| GPU / no-GPU | GPU (RTX 4090), six trainers side by side at 500–850 rounds a minute |
| seeds | 1 / 2 / 3 per arm from scratch — the same seeds as the original A5-points and the discount arm, so per-seed comparisons are paired at initialisation |
| n | 100 at seed base 700000 (every read, the census, greedy against sampled); 30 at 500000 (in-run, every 512 rounds) |
| config | `configs/experiments/curriculum/a5_points_flat.yaml` (R1) and `a5_points_flat_points.yaml` (R2): `a5_points.yaml` with the one phase field each; unrefereed by design |
| decode | none on the per-model facade |
| paired | per episode against the bar on identical seeds (turns); per seed against the original and against each other on the same seeds and layouts |
| comparator | the original A5-points runs `uagmul66` / `wedjjozy` / `ici5py46`: 0.030 / 0.060 / 0.330 at 122,880 (0.040 / 0.140 / 0.090 at 40,960; 0.220 / 0.220 / 0.110 at 81,920); the discount arm `meawm0gk` / `6lr588y7` / `0sjtnn0n` 0.070 / 0.180 / 0.100; the bar `squad_march_take` 1.000, 6.71 turns, held 5.00, coherent 0.832 |
| opponent | none |
| budget | 122,880 rounds at 128 per update (`--rollout-rounds 32 --num-rollout-envs 4`), `--ent-coef 0.003`, `gamma` 0.9, the `mean` credit, eval and checkpoint every 512; no extension |
| code revision | build and pre-registration `509abb3`; runs on `509abb3`; amendment `9d817fb`; the branch `feature/per-model-actor-credit`, PR #383 |
| checkpoint | `last.pt` at 122,880 and every 512 rounds under `checkpoints/per_model/per-model-a5_points_flat{,_points}-2026-09-19-15-15-31-s{1,2,3}a5r{1,2}`; a recorded greedy episode beside every checkpoint (the first arms recorded in-run), thirteen per run rendered to MP4 and on the run's Wandb record |
| coherency | greedy at 700000+: R1 0.449 / 0.119 / 0.407, R2 0.299 / 0.151 / 0.328 (the original 0.19–0.21; the bar 0.832); sampled 0.05–0.11 |
| Wandb | `curriculum-a5`: R1 `008ftjrj` / `pvafw3km` / `ih78ta51`, R2 `krhhq1v5` / `d12n5luj` / `8jnrjqu5`; video sibling runs `850oup4q` / `3yauw38m` / `ptja7rxk` / `078e0p6p` / `1p5qwba8` / `clqr3exm` |
| pre-registration | `reports/2026-09-19-curriculum-A5-points-reward-shape-preregistration.md` (+1 amendment, which carries every table) |

## The read

The final table, the reads at every checkpoint, the census by turn and
by point, the greedy-against-sampled table and the panel by quarter are
in the amendment; this section says what they mean.

### The two changes did what they were built to do, and the policy did not follow

The bar's episode reward went 4.38 → 7.21 (R1) → 12.21 (R2), and both new
terms were earned in training: R1 seeds collected the flat bonus at
0.04–0.07 per close, R2 seeds the per-point bonus at ~0.27 per close —
more than the travel and coverage terms together. Nothing about the
payment failed to reach the policy. The policy's census is the
half-step's on every column, and worse: fewer bodies on points than the
original at the end (4–6 of 18 against 5–7), held 2.4–2.9 against 2.5–3.8,
the clock run out on every seed.

### The critic could not fit the lump

Explained variance by quarter, the pre-registered readout:

| run | Q1 | Q2 | Q3 | Q4 |
|---|---|---|---|---|
| R1 s1 | −0.02 | 0.09 | −0.14 | −0.01 |
| R1 s2 | 0.11 | 0.41 | 0.73 | 0.53 |
| R1 s3 | 0.21 | 0.22 | 0.03 | −0.17 |
| R2 s1 | −0.32 | −0.19 | −0.19 | −0.37 |
| R2 s2 | 0.12 | −0.31 | −0.32 | −0.47 |
| R2 s3 | −0.24 | −0.23 | −0.14 | −0.14 |
| original (γ 0.9) | 0.65 / 0.67 / 0.68 | 0.65 / 0.73 / 0.71 | 0.63 / 0.67 / 0.69 | 0.65 / 0.63 / 0.70 |
| discount arm (γ 0.99) | 0.78–0.80 | 0.78–0.80 | 0.77–0.84 | 0.82–0.85 |

A value function worse than predicting the mean, on five of six seeds,
for a hundred and twenty thousand rounds. The original's return is
dominated by dense terms (travel 1.31, coverage 0.91 of the bar's 4.38)
with a terminal bonus that the remaining-rounds scale keeps at 0.5–2.0;
R1's return has a 5.0 jump on the rare episode that succeeds, and R2's a
3.0–5.0 jump at the clock on every episode whose size depends on the
board at turn ten. The critic sees the turn index, but the per-model
value head is asked to predict a lump paid to the army from a token set
that changes every decision; it did not learn to. With the advantages
noise, the policy gradient has no direction, and the displacement head
sits at 2.0–2.6 nats (the original 1.6–1.9) with clip fractions of
0.18–0.31.

### Partial credit bought the partial behaviour

At 81,920 R2's far column (points 3 and 4, fifteen inches beyond the
near three) was empty in 90 / 100 / 100% of episodes on its three seeds
while the near column was mostly held — the cleanest measurement of a
shaping term buying exactly what it pays for. Three points at the clock
pays 3.0, more than the whole of the original's episode reward, and the
walk to the far pair is unpaid until it lands. By 122,880 that had
diffused (far column empty 0.33–0.85) without any point being taken more
often. The A5 record's rule — *before training a shaping term, ask
whether the behaviour it wants pays more in total than the behaviour it
replaces* — was satisfied here (five points pays more than three) and
was not enough: the increment for the fourth and fifth is paid only on
arrival, and arrival is the thing the policy does not find.

### The sharpest walk-off on the ladder

R1's first seed puts 10.2 of 18 bodies on points after turn 3, 2.1 after
turn 5, 6.8 after turn 7 and 1.9 at the end — held 0.90 of 5, the near
column empty in 89–100% of episodes. Its sampled play holds 3.20, so the
greedy argmax of a diffuse displacement head is what walks off; the
discount arm's rows showed the same sign (sampled 6–7 vp *worse* there,
here −6.6 vp *better* on this seed). On this trainer a policy that has
not converged can read as a walk-off under greedy scoring and as a hold
under sampled scoring; the two rows together are the reading.

## What this says

- **A terminal lump is a critic problem before it is a policy problem.**
  On this trainer the remaining-rounds scale was doing two jobs — a speed
  incentive, and keeping the terminal payment small beside the dense
  stream — and removing it for the first job broke the second. Do not
  raise a terminal payment on a per-model rung without reading the
  critic's explained variance at 20k rounds; a red panel that early is
  the arm's answer.
- **Partial credit buys the partial behaviour.** A per-point terminal
  payment taught the army to keep the near column. On a conjunction
  rung, decomposing the criterion into per-part payments pays the parts
  that are easy and leaves the hard part exactly as unpaid as before.
- **Three arms, one policy.** Discount, terminal shape and credit each
  changed what the critic or the return looked like and none changed
  what the army does at five points. The spread wall on the per-model
  trainer is not the reward's timing, size, shape or attribution. The
  remaining levers are the ones the whole-army critic probe named in
  August: directed exploration and representation. Do not run a fourth
  reward arm on this half-step.
- **Gate a final read on the trainer exiting, not on `last.pt`
  existing.** `last.pt` exists from the first checkpoint; a chain gated on
  it read the arms at ~85k rounds and called it final. Caught by the
  timestamps; the real final read was re-queued on the process count.
- **The videos are on the runs.** Every per-model launch from `362518a`
  renders one recording in twenty to MP4 and logs it under
  `episode_recording`; these six runs carry theirs post hoc.

## Where this lands

1. This report.
2. `reports/README.md` index row.
3. `CLAUDE.md`: a ladder row for the reward-shape arms and a rule bullet
   (the lump breaks the critic; partial credit buys the partial
   behaviour; three arms, one policy; gate a final read on exit).
4. Live docs: `docs/reward-phases.md` carries the two new phase fields
   (`509abb3`); `configs/README.md` the two arm configs; the in-training
   video pipeline is documented in `wargame_rl/wargame/model/CLAUDE.md`.
