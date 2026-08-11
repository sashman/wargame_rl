# Real terrain, and the abandonment gap

**2026-08-10** · `configs/experiments/25v25_polygon_terrain_objectives.yaml` ·
wandb group `phase03-real-terrain` · two seeds, 1000 epochs, `--no-tf32`

## The question

Phase 03 made the board continuous, gave models physical bases that collide, and
made terrain and objectives polygons. Does the agent still beat the scripted
shooting bar under those dynamics, as it did on the golden scenario (+30.8 /
+27.4 against +17.0)?

## The answer

**No.** Two seeds, n=100, identical layouts (seeds 700000-700099):

| policy | win | vp_margin | held | on_obj | alive | exposure | firepower |
|---|---|---|---|---|---|---|---|
| agent s1 (epoch 1000) | 0.84 | **+50.2** | 1.78 | 0.65 | 0.65 | 0.245 | 1.76 |
| agent s2 (epoch 1000) | 0.83 | **+51.3** | 1.75 | 0.72 | 0.62 | 0.257 | 1.68 |
| `squad_march_shoot` (the bar) | 0.87 | **+67.7** | 1.85 | 0.55 | 0.51 | 0.438 | 0.88 |
| `contest_and_spread` | 0.88 | +70.7 | 1.89 | 0.59 | 0.47 | 0.431 | 1.12 |
| `split_evenly` | 0.61 | +18.9 | 1.40 | — | 0.54 | — | — |
| `random` (the floor) | 0.00 | -143.2 | 0.23 | — | 0.78 | — | — |

The two seeds land within 1.1 vp of each other, ~17 short. That is a result, not
seed noise.

**It converged by epoch 680.** The same seeds scored +49.7 and +51.4 at epoch
680; the last 320 epochs moved them +0.5 and -0.1. The standing note that "1000
epochs is if anything too few" does not hold on this scenario — screening here
could stop at ~700.

## Where the 17 vp goes

Not to losing fights. Player VP is a dead heat — 152.9 and 155.2 against the
bar's 151.6. The entire gap is **conceded** VP: 102.8 and 103.9 against 83.9.

`just measure-objective-split`, n=100:

| | agent s1 | bar |
|---|---|---|
| **abandoned** | **0.380** | 0.157 |
| lost_close | 0.020 | 0.107 |
| lost_far | 0.007 | 0.120 |
| held | 0.593 | 0.617 |
| models alive at end | 16.3 | 12.7 |
| **surplus models on held objectives** | **8.58** | 3.43 |
| redistribution ceiling | 2.54 (+0.76) | 2.34 (+0.49) |

**The agent does not lose objectives — it never goes to them.** It is pushed off
an objective 0.7% of the time against the bar's 12%, and abandons 38% against the
bar's 16%. It simultaneously parks **8.6 surplus models** on the points it does
hold, against the bar's 3.4. With 16.3 alive and 1.78 held, roughly nine models
are contributing nothing.

This is the exact failure `objective_hold.crowding_exponent` exists to price: the
ninth model on a point should earn less than the first model on an empty one. On
this scenario it is not biting hard enough. Note the lever was tuned where models
were dimensionless points on a radius-3 disc; here an objective holds ~18 models.

## The agent is strictly better at fighting

Worth stating plainly, because it is the half that worked:

- **firepower_ratio 1.76 / 1.68 against the bar's 0.88** — it wins the firefight
  roughly two to one.
- **alive 0.65 / 0.62 against 0.51** — it preserves its force.
- **exposure 0.245 / 0.257 against 0.438** — it takes *half* the exposure.

That last number looked like it might overturn the standing finding that **the
agent does not use terrain for cover, it manages range**
([2026-08-05](2026-08-05-stochastic-terrain-and-cover.md)), which was established
on terrain giving 0.194 hidden — more than double the real game's 0.088.

**Re-tested on the corrected terrain, and the old finding stands.** Exposure has
three possible causes here — cover, range, or simply keeping more models alive —
so it was split per alive model per shooting phase, n=20 on identical layouts:

| | range_margin | threatened | **blocked_share** |
|---|---|---|---|
| agent | **+2.11"** | **0.394** | **0.738** |
| `squad_march_shoot` | -0.26" | 0.679 | **0.735** |
| `random` | +13.33" | 0.034 | 0.456 |

`blocked_share` is the cover term and nothing else: of the enemies *within weapon
range*, the fraction whose sightline terrain breaks. **It is identical for the
agent and the bar.** Conditional on being in range, the agent is no better
covered than a policy that marches in a straight line — it gets the incidental
cover of walking among ruins, no more.

What differs is `threatened`: the agent has an enemy in range 39% of the time
against the bar's 68%, sitting +2.11" *outside* weapon range where the bar sits
0.26" inside it. The exposure gap is almost exactly proportional to that —
0.245/0.438 = 0.56 against 0.394/0.679 = 0.58.

So the halved exposure is **range management, measured directly rather than
inferred**, and the 2026-08-05 conclusion now holds on terrain matching the real
game rather than on terrain with twice its cover.

Two things this does *not* say. `hugging ruins` (0.827 agent, 0.704 bar) is not
evidence either way, because objectives **are** terrain in this scenario, so
holding one means standing in a ruin. And `random`'s low 0.456 is drawn from the
3.4% of the time it is threatened at all, so it is noise, not a floor.

Answering this at all needed terrain in the recordings (schema 2.1, #160) —
before it, random-terrain episodes could not even be replayed.

## What had to be fixed to ask the question at all

The first attempt at this run was killed at ~epoch 470 because its terrain was
mis-scaled. The profile (37 pieces of 3-6) had been tuned against
`just measure-terrain`'s own hidden-fraction metric, and that metric rewards piece
*count*. Two errors compounded:

1. `measure_terrain._coverage` billed each piece's **bounding box**. An inscribed
   hexagon fills ~65% of its box, so the config's claimed coverage of 0.233 was
   really 0.159. Fixed to use polygon area.
2. Maximising hidden-fraction then drove 37 tiny pieces, giving **0.194 hidden
   against the real game's 0.088** — double the cover a real table offers.

Measured against the 45 real layouts in `configs/evaluation/maps/`:

| | pieces | area | aspect | coverage | hidden |
|---|---|---|---|---|---|
| real maps (45) | 15.5 | 40.0 | 1.52 | 0.235 | 0.088 |
| 15 x 3-11 (now) | 15 | 38.7 | 1.50 | 0.223 | 0.111 |
| 37 x 3-6 (was) | 37 | 11.3 | ~1.3 | 0.159 | 0.194 |

**The mission was broken, not just the cover.** Objectives *are* terrain
footprints and a model has a 1.26" base that cannot overlap another, so footprint
size is a hard capacity cap:

| | 37 x 3-6 | 15 x 3-11 |
|---|---|---|
| models per objective | 5.0 | 18.2 |
| objectives that cannot hold one 5-model squad | 48/90 | 4/90 |

At 5 models per objective, `crowding_exponent` prices a problem the geometry
already prevents. The corrected profile restores a regime where stacking is a
real choice — and the agent promptly demonstrated the failure the lever is meant
to catch.

No generator change was needed: `_sample_piece` already draws width and height
independently, so a **low floor with a high ceiling** (3-11) gives median aspect
1.50 against the real 1.52. Raising `min_size` instead pushes the mean box area
up and the packing validator rejects it, which is why every first attempt at
larger terrain failed to load.

The corrected profile is also **19% faster to train** — 21.3 s/epoch against
26.2 — since 15 pieces means proportionally fewer segment-shape pairs to trace.

## What this does not support

- **Not "the agent cannot learn this scenario".** It beats every movement-only
  baseline, wins the firefight two to one, and sits 17 vp short of a bar it fails
  on one specific axis. The failure is allocation, not competence.
- **Not a verdict on `crowding_exponent` as a lever.** Its weight was never
  retuned for objectives holding 18 models instead of unlimited point-models. The
  measurement says the current setting is too weak here, not that the mechanism
  is wrong.
- **Not a cover result, and now measured as such.** `blocked_share` is
  identical to the bar's, so the exposure gap is range management, not
  terrain use. See above.

## Next

1. **Retune `crowding_exponent` for this geometry.** The diagnosis is specific:
   8.6 surplus models and 38% abandonment, with a redistribution ceiling of 2.54
   against 1.78 held.
2. **Spread the objectives.** All three currently fit inside a ~16" circle on a
   60x44 board, and 47% of objective pairs are within one weapon range. The lever
   with headroom is where the generator places pieces in the central strip, not
   which three are selected — in 59% of layouts only 3 pieces are eligible and no
   choice exists. Real maps afford min separation ~13" and a widest pair of ~28".
3. Both change the scenario, so **re-measure all six baselines at n=100 on seeds
   700000+ before quoting any agent number against them**.
