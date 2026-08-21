# The eval tables are generated now, and the tracing they replaced was right

**2026-08-20**, measurement revised **2026-08-21**. `configs/evaluation/maps/`
is now generated from the public layout Data API by `just fetch-maps`. The 45
tables are the *same* 45 layouts they always were — what changed is that their
geometry is the source's rather than an approximation of it, and that they now
carry their own deployment zones.

⚠ **The title of this report used to read "and the tracing was the error".**
That is retracted — see § 3. The hand-traced objectives match the published
layout cards on 45 of 45; the error was mine, in treating the API's objective
markers as authoritative.

**This is a re-derivation with provenance, not a new distribution.** Establishing
that was the first job, because it decides whether the change is a tidy-up or a
scenario swap. It is closer to the first than expected, and further than is
comfortable.

---

## 1. They are the same 45 layouts

Matched by axis-aligned piece bounds, every one of our 45 tables has a distinct
near-exact counterpart among the API's 45 `strike-force` layouts — a bijection.
The handful of rows where the second-best match is close are pairs of genuinely
near-identical *source* layouts, the same terrain reused for two mission slots.

So the numbering could be preserved, and it was. `maps_heldout` is still the
same nine tables, which is the only reason any before/after comparison here
means anything.

## 2. Three conventions the API does not document, settled by measurement

**The local frame is centred, not corner-relative.** Outline points are
piece-local and centred on the footprint origin, so a piece rotates about its own
middle and is then translated. The test is decisive rather than aesthetic: under
the centred reading all **720 pieces land on the board with zero overhang**,
while the corner reading throws **70 of them off the edge by up to 3.75"**.

**Simplification is mandatory.** Every source outline carries **167 to 348
vertices** against `TERRAIN_VERTEX_BUDGET = 8`. That is not a limitation to work
around; it is the explanation for why the hand-traced tables were quads.

**Douglas-Peucker is the method, chosen on numbers.** Measured over all 720
pieces against their true silhouette area:

| method | median | p90 | worst |
|---|---|---|---|
| footprint rectangle (4v) | 0.964 | 1.592 | **1.592** |
| convex hull → 8v | 1.068 | 1.160 | 1.160 |
| **Douglas-Peucker → 8v** | **1.027** | 1.078 | **1.078** |

The rectangle's tail is the disqualifier: an angled or L-shaped ruin blocks half
again as much board as it should. The hull fills in every concave bay.

## 3. An objective is a ruin — and the API is not the source for them

The current rules have no free-standing objective markers to stand near; the
marker says *which ground* is fought over. So an objective is a ruin, and a
ruin is a group of terrain pieces sharing at least 1" of boundary — the layouts
build one structure from several kit pieces (a rectangle split along a diagonal
seam, two bars butted into an L), and the source's own render draws each group
as a single blob. Over all 45 tables the shared-boundary lengths are strikingly
discrete: 110 pairs at ~0.32" (a corner touch), 84 at ~2.33", 6 at ~6.33" and 18
at ~13.6". Nothing falls between 0.33" and 1.45", so the 1.0" threshold sits in
empty space.

**But the API's objective markers are wrong on six of the 45 tables**, by 12 to
18 inches — a different ruin entirely. Neither the layout's own copy of its
deployment nor the deployment's canonical markers is right everywhere; switching
between them just moves which six fail. Terrain is unaffected: the piece
geometry matched on 45 of 45.

### The correction that matters

**An earlier section of this report said the hand-traced objectives "were not
the layout's". That was wrong, and it is retracted.**

Checked against positions read off the published layout cards — one card per
layout, each drawing an icon on every objective — the hand-traced tables are
right on **45 of 45, worst error 1.5 inches**. The tracing was not approximate;
it was better than anything derived from the API. What misled me was comparing
the traced objectives to the API's markers and assuming the markers were
authoritative.

So the tables are now built from **both** sources, each where it is reliable:
terrain from the API, objectives resolved from the validated positions
(`scripts/objective_markers.json`). Every objective is checked against the
published cards in `tests/test_map_objective_counts.py`.

### How a position becomes a ruin

- **The biggest ruin in reach wins, not the nearest.** A marker often sits in
  the gap between a real ruin and a scrap of scatter terrain.
- **Reach is 4.5", not the rules' 3" control range.** It is an *authoring*
  distance: marker-to-ruin distances cluster below 4" and again from 5", the 4.5
  bin holds 2 against 18 and 28 either side, and resolution is identical
  anywhere in 4.0–5.0.
- **One ruin per marker**, most-constrained-first.
- **A tie designates both.** The boards are point-symmetric, so a marker
  routinely sits between a ruin and its own reflection — on `table_01`, 1.02"
  from each of two 58.5 sq in wedges with mirror-image centroids. That is what
  makes a table carry six, and the published cards draw it as **two Centre
  icons**.
- **No discs.** A position out of reach of every ruin takes the nearest anyway.

Result: **45 of 45 objectives within 3" of a published one**, worst 2.0" (icon
centre against outline centroid), counts **24 fives and 21 sixes** exactly as
published, and the zone split back to **82 / 82 / 82**.

## 3c. The tables bring their own deployment zones

`deployment_zone` is `(x0, y0, x1, y1)` — an axis-aligned band. **Only one of
the six real deployments is one.** Two are triangles split by a board diagonal,
two are stepped staircases, one is bounded by arcs. A map now carries its own
outlines, named for their shape (`diagonal_halves`, `long_edges`,
`opposed_quadrants`, `short_edges`, `stepped_bands`, `stepped_columns`), used by
10/9/8/7/6/5 of the 45 tables.

**Here the API is trustworthy, unlike its objective markers**, and that was
checked rather than assumed: rasterised against the published cards, the tinted
deployment region is **at least 98% inside the API's polygon on all 45 tables**,
attacker to red and defender to blue every time.

Sampling still happens in the rectangle — it becomes the outline's bounding box —
and the outline rejects anything outside it, so the existing "is this zone big
enough for the army" check still fires. **Every model deploys inside its own zone
on all 45 tables**; without the outline test an army spills out on five of the
six shapes.

Two routes had to be wired, not one: training draws a `MapLayout` from a pool,
while evaluation installs a map onto a scenario and has no layout to carry
anything. Wiring only the first would have trained a map under its own
deployment and scored it under the rectangle — the same silent mismatch the
three terrain modes once had.

**`long_edges` is the one to watch.** It puts the armies along the 60" edges,
**twenty inches apart across the short axis**, where the others separate them by
24 to 40. At a twelve inch weapon range that is a materially different game from
turn one, and it is a fifth of the pool.

## 4. Two independent checks nobody designed for

Both were computed after the fact, and both came out exactly right — which is the
strongest evidence the pipeline is faithful.

- **The zone split is exactly 82 / 82 / 82** across player zone, middle and
  opponent zone. The sides balancing to within one objective is the
  That exact balance is the check that matters, since the tables are
  point-symmetric, and it fell out rather than being aimed at. Every wrong
  resolution rule tried here broke it by two to nine objectives.
- **The tables are point-symmetric to the measurement floor**: a table sits a
  median of **0.00"** from its own 180-degree rotation, worst 0.71". The
  hand-traced ones sat 1.7" out and at worst 3.9". *That asymmetry was tracing
  error.*

The second has a consequence: `map_pool.mirror` is **exactly 2x, not "about
2x"**. If `rot180` is the identity and it commutes with the x reflection, then
`flip_y` IS `flip_x` and `flip_xy` IS the original. Four orientations, two
distinct boards.

## 5. What changed on the board

| | traced | generated |
|---|---|---|
| pieces per table | 15 or 16 | **16** |
| outline | quads | 8-vertex silhouettes |
| objectives | 5 or 6, chosen by eye | **5**, resolved from markers |
| terrain area | 23.5% of board | 24.8% |
| own-zone/middle/opponent | 82/82/82 | 75/75/75 |
| distance from own rot180 | 1.7" median | **0.00"** |

**82% of objectives moved less than 3"; 18% moved further, up to 13.7".** That is
the part that voids baselines.

**The observation width did not change.** `objective_budget` stays 6 and
`terrain_budget` 16, so every checkpoint still loads and paired arms still pair.
Lowering the objective budget to 5 would change the tensor width and orphan every
checkpoint in `checkpoints/` — it must not be "tidied".

## 6. What it cost, measured

Four scripted policies on the golden training config, **all 45 tables**, n=30
each, traced geometry against the tables **as they ship** — terrain, objectives
*and* deployment zones — paired by table. All 45 are admissible here for the
same reason the joint decode was measured on all of them: **a scripted policy
has no weights**, so the held-out split constrains nothing about this
comparison.

The traced column reproduces the earlier run **to the decimal** on all three
deterministic scripts (−1.56, −7.78, −7.42), so the harness is deterministic and
the difference is the maps. `random` moves about 1 vp between runs, because its
own seed is not pinned — do not read a `random` difference under ~2 vp.

Reported with both a t and a **sign count**, because the per-table differences
are heavy-tailed: a few tables move hugely in both directions while most move
slightly one way, and the two tests disagree often enough that quoting either
alone misleads.

| | traced | generated | diff | t | same sign |
|---|---|---|---|---|---|
| **`vp_margin`** | | | | | |
| `random` | −218.7 | −222.5 | −3.7 | −3.28 | 29/45 |
| `squad_march_take` | −1.6 | **+5.9** | +7.5 | 2.06 | 27/45 |
| `squad_march_shoot` | −7.8 | −5.9 | +1.8 | 0.55 | 24/45 |
| `squad_march_deny` | −7.4 | **+5.4** | +12.8 | 3.87 | 31/45 |
| **`held`** | | | | | |
| `random` | 0.10 | 0.15 | +0.05 | 3.51 | 41/45 |
| `squad_march_take` | 2.64 | 2.67 | +0.03 | 0.60 | 23/45 |
| `squad_march_shoot` | 2.62 | 2.63 | +0.01 | 0.14 | 21/45 |
| `squad_march_deny` | 2.36 | 2.45 | +0.09 | 2.36 | 27/45 |
| **`on_obj`** | | | | | |
| `squad_march_take` | 0.975 | 0.981 | +0.01 | 2.72 | 32/45 |
| `squad_march_shoot` | 0.970 | 0.973 | 0.00 | 0.96 | 30/45 |
| `squad_march_deny` | 0.970 | 0.969 | −0.00 | −0.32 | 24/45 |
| **`coherent`** | all four | | **≤ 0.02** | | |

**The tables are not harder. Three of the four policies score the same or
better, and `held` does not fall for any of them.** The three scripted `held`
diffs are +0.03, +0.01 and +0.09; `deny` and `take` gain 12.8 and 7.5 vp; the
bar, `squad_march_shoot`, is a null on every metric except a −0.02 in coherency.
Only `random` loses ground, by 3.7.

`vp_margin` is still the weaker of the two tests. `deny` is decisive on both
(t = 3.87, 31/45), `take` is suggestive on one and chance-adjacent on the other
(t = 2.06, 27/45), and `shoot` is a flat null. Coherency is untouched, as it
should be — this changed terrain, objectives and starting positions, not
formation.

### ⚠ The earlier version of this section said the opposite, and it was measuring a configuration that does not ship

It reported the tables as **harder to hold** — `held` down −0.11 to −0.29 for
every scripted policy at t = −2.7 to −6.1, and `random` down **−18.9** (t =
−6.58) as the clearest single effect. Every one of those figures came from
running the new terrain and objectives under the **old rectangular deployment
zone**, because the zones had not landed yet. That combination is not a board
anyone plays: the layouts place the armies, and moving the objectives without
moving the deployments puts them in the wrong relation to each other.

With the zones in, `random`'s −18.9 is −3.7 and its `held` *rises* on 41 of 45
tables. The reasoning offered for the −18.9 — that `random` had been scoring by
deploying onto home objectives and standing there — was a good explanation of a
measurement artefact.

**A second error in that section: its `on_obj` row was the `alive` column.** The
traced values quoted for `on_obj` (0.44, 0.44, 0.40) are exactly the survivor
fractions (0.439, 0.442, 0.401); true `on_obj` for those scripts is 0.97, as it
must be for policies that march onto an objective and stand on it. So the
"`on_obj` falls with `held`" finding was never about objective occupancy — it
said models were dying more. Occupancy is in fact flat or slightly up.

### What the ruin merge itself cost

Isolated by re-running everything against the pre-merge tables — same policies,
same seeds, and an old-tables column that reproduced **to the decimal** on all
four policies, so the harness is deterministic and the difference is the maps.

⚠ **Both arms of this one predate the deployment zones**, so the absolute scores
below are the superseded lineage. The *comparison* still holds: merged and
unmerged differ only in the objectives, and both sat under the same rectangle.

| | unmerged | merged | diff | t | same sign |
|---|---|---|---|---|---|
| `random` | −237.0 | −236.8 | +0.2 | 0.50 | 21/45 |
| `squad_march_take` | −7.5 | −10.2 | −2.8 | −0.99 | **35/45** |
| `squad_march_shoot` | −10.3 | −12.1 | −1.8 | −0.59 | **32/45** |
| `squad_march_deny` | −10.7 | −14.7 | −4.0 | −1.35 | **35/45** |

**Merging the ruins made the scripts slightly worse, not better.** That was the
opposite of the prediction — larger objectives ought to be easier to stand on.
Every t is inside 1.4, but the sign counts are 35, 32 and 35 of 45 (p ≈ 0.0006,
0.02, 0.0006) while `random` sits at chance, so the direction is real even
though the size is not resolved.

**The mechanism is not established.** The obvious candidate is refuted: merging
moves an objective's centroid, and a policy steers at the centroid, so an
L-shaped ruin could have sent everyone into its notch — but **0 of 30 merged
objectives have a centroid outside their own outline**. The VP split says the
loss is on our side of the ledger (`vp_for` down on 34–35 of 45 tables) rather
than the opponent's, and no further explanation here is supported by evidence.

Merging is kept regardless: it is what the objective *is*. A rule that is
correct and costs a couple of VP is not a trade.

### The pool has a resolution floor, and it is about 6 vp

Per-table `vp_margin` sd is **18.5 to 20.6**, so at n=45 the standard error is
**2.75 to 3.07**. Effects under ~6 vp are not resolvable on this pool *at all* —
and more episodes per table cannot help, because the variance is **across
tables** and there are only 45 tables in existence.

That floor is **better than the traced tables' ~8 vp**, and the deployment zones
are why: every table now starts its armies in a shape the layout specifies
rather than in one rectangle imposed on all 45, which removes a source of
across-table spread rather than adding one.

**Pairing does not rescue it either, and the reason is worth keeping.** Pairing
cancels variance when the unit is identical and only the treatment differs. Here
**the table is the treatment**: `table_05` traced and `table_05` generated are
different boards, so pairing removes only the part of the layout that did not
change. This is the opposite of pairing two training arms on one map, where
`seed_everything` gives identical initial weights and pairing is worth an order
of magnitude. The one place pairing *did* pay was the merge comparison above,
where the tables really are the same and only the objectives moved.

An interim read of this comparison on the **held-out nine** made `deny` look
like the biggest mover (+4.9 → −6.6). At n=45 its traced score is −7.4, not
+4.9 — the nine-table subset was not representative of that policy. Nine tables
is not enough to characterise a map change.

### The new bar

All 45, n=30, seeds 700000+: `random` **−222.5** · `take` **+5.9** ·
`shoot` **−5.9** · `deny` **+5.4**. Coherency 0.75–0.82 throughout.

Note the bar for this config is `squad_march_shoot` by convention, but on these
tables it is the *worst* of the three scripts — `take` and `deny` both finish
positive and are within 0.5 vp of each other, well inside the ~6 vp floor. Quote
the policy by name, not as "the bar".

## 6b. The agent, re-trained on these tables

Three seeds of the documented recipe (`configs/golden/25v25_maps_two_mode.yaml`,
`ent_coef` 0.003, 300 epochs, wandb group `new-maps-baseline`), scored on the
nine held-out tables at n=30 under the verified top-3 decode, on the **refereed**
eval configs, with the scripts re-measured against each opponent.

| opponent | agent | per seed | best script | gap | t | same sign |
|---|---|---|---|---|---|---|
| `squad_march_take` | **+22.6** | +24.4 / +14.8 / +28.8 | −1.1 (`deny`) | **+23.7** | 3.14 | 8/9 |
| `squad_march_shoot` | **+40.2** | +49.0 / +22.4 / +49.3 | +23.0 (`take`) | +17.2 | 1.78 | 7/9 |
| `squad_march_deny` | **+24.4** | +31.8 / +12.0 / +29.5 | −8.9 (`take`) | **+33.4** | 4.00 | 8/9 |
| `contest_and_spread` | +21.8 | +30.9 / +8.1 / +26.4 | **+30.2** (`take`) | **−8.4** | −1.11 | 3/9 |

Intended unit coherency **0.950–0.954** on every opponent, against a scripted
0.903–0.908. Formation holds in all four, including the matchup it loses.

**Two decisive leads, one unresolved lead, one loss.** The `contest_and_spread`
loss is **not** statistically established either (t = −1.11, 3 of 9), but it is
the only matchup without a lead, and it contradicts a claim that was live in
`CLAUDE.md` — that the loss "no longer exists" and "was a property of the weaker
lineage". Measured here on the generated tables with the lineage the documented
recipe produces, it exists. That claim was made on the hand-traced tables.

### The referee is not optional, and leaving it off flatters the scripts

The first pass at this scored the same three seeds on the **training** config,
which sets no `enforce_move` and no `attrition`. It read **+20.6 for the agent
against a best script of +13.7** — a lead of 6.8 rather than 23.7.

The referee cancels a unit's move when the move breaks coherency, so it taxes
each policy in proportion to how often it breaks it. The agent intends 0.95
coherency and pays almost nothing; the scripts intend 0.90 and pay about 16 vp
(`squad_march_take` goes +13.7 unrefereed to −2.4 refereed on the same matchup).
Turning the referee off therefore removes a penalty the scripts have earned and
the agent has not.

This is the same error as § 6's deployment zones, twice in two days: **measuring
a configuration that is not the one being played.** The training config is not
the eval config, and the difference is not a detail.

## 7. Three bugs this turned up, two of them mine

- **Double rounding in the generator.** `round(v, 2)` and `f"{v:.2f}"` disagree at
  33.455, so an objective's outline differed by a hundredth from the very piece it
  *is*. Caught by an existing test, not by inspection. Round once.
- **Two tests had gone vacuous.** Every table now has exactly 16 pieces against a
  budget of 16, so the terrain-padding assertions were checking an empty slice
  and passing. They now trim a layout to 15.
- **A claim of mine that did not survive its own evidence.** I wrote that markers
  merge, citing "5 on 39 layouts, 4 on four, 3 on two". That histogram was
  counting *open-ground* markers, not merges. Marker collapse never fires.
- **And then the reverse, which shipped.** Having established markers do not
  merge, I concluded pieces need not either — so an objective became the single
  piece a marker landed on, and 30 of 225 covered half a ruin. It took looking
  at a picture. **Every count, split and resolution check passed**, because all
  of them were counting objectives rather than asking what ground each one was.
  The first implementation then compounded it: two abutting polygons do not fuse
  into one outline, and the code took the largest part, discarding the rest in
  silence. It now raises.

The new ingest tests were sensitivity-checked by injecting three bugs — a shifted
frame, a doubled control range, a removed dedupe — and confirming the right test
caught each.

## 8. What shipped

`scripts/fetch_map_layouts.py` and `just fetch-maps`, the 45 regenerated tables
and their previews, the nine synced held-out copies, and twelve tests.

**No API response is written to disk.** The layout slugs, deployment names and
mission-pack ids are the commercial product's vocabulary, and
`tests/test_no_ip_references.py` scans tracked *and untracked* files; only
geometry crosses the boundary.

Deployment zones ship with it: `DeploymentConfig` on the map, polygon sampling
in `wargame_model_placement`, zones travelling with a mirrored layout, and the
outlines drawn in the previews. A map without a `deployment` block still deploys
under the scenario's rectangle, so every generated-terrain config is unchanged.
