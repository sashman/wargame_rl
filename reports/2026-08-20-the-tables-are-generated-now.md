# The eval tables were traced by hand, and the tracing was the error

**2026-08-20.** `configs/evaluation/maps/` is now generated from the public
layout Data API by `just fetch-maps`. The 45 tables are the *same* 45 layouts
they always were — the hand-tracing that produced them is what changed.

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

## 3. An objective is a ruin

The current rules have no free-standing objective markers to stand near; the
marker says *which ground* is fought over. Getting from "a marker" to "which
ground" took three corrections, and **every one of them was caught by eye on a
rendered table, after the structural checks had passed.**

**A ruin is not a terrain piece.** The layouts build one structure out of
several kit pieces — a rectangle split along a diagonal seam, two bars butted
into an L — and the source's own board render draws each group as a single
connected blob. Resolving a marker to the nearest *piece* left **30 of 225
objectives covering half a ruin**: on `table_02` the centre rectangle came out
green on one side of its diagonal and brown on the other.

Pieces are grouped by **shared boundary length**, and the pool makes the
threshold obvious. Over all 45 tables the contact lengths are strikingly
discrete — 110 pairs at ~0.32" (a corner touch), 84 at ~2.33" (a bar's end
against another's side), 6 at ~6.33" and 18 at ~13.6" (one rectangle split in
two). **Nothing at all falls between 0.33" and 1.45"**, so 1.0" sits in empty
space: 3.0x above the highest contact it separates, 1.45x below the lowest it
joins.

**The biggest ruin in reach wins, not the nearest.** A marker routinely sits in
the *gap* between a real ruin and a scrap of scatter terrain, and nearest-wins
handed the objective to the scrap — on `table_01` and `table_10`, two of five
objectives came out on **12.9 sq in** slivers while **82.5 sq in** ruins stood
two inches away.

**And "reach" is not the rules' control range.** Using the rules' 3" was the
obvious choice and it was wrong: the layouts routinely place a marker
**3.75–4.0" from the large ruin it plainly means**, with a scrap 2–3" away on
the other side, so a 3" cutoff excluded the real answer and took the scrap. That
is what `table_02` and `table_06` showed — every one of their five objectives
disagreed with the hand-traced ones.

The right cutoff is an **authoring** distance, and the pool hands it over.
Marker-to-ruin distances cluster below 4" and again from 5", with a trough at
4.5 holding **2** of them against 18 just below and 28 just above; the
resolution is *identical* anywhere in 4.0–5.0, so the constant sits in a gap
rather than on a fitted edge. Agreement with the hand-picked objectives:
**96%** at 4.5", against 91% at the rules' 3" and 84% for nearest-wins.

**One ruin per marker.** Sharing was the first rule here, on the reasoning that
one piece of ground is held once. The pool refuted it: the only collisions were
markers **twelve to seventeen inches apart** both reaching one long ruin, which
are plainly two objectives. Each marker now takes the largest *unclaimed* ruin
in range, most-constrained-first so a marker with one option takes it before a
marker spoilt for choice.

**A tie designates both, and that is what makes a table carry six.** These
boards are point-symmetric, so the centre marker routinely sits in the *gap
between a ruin and its own reflection*. On `table_01` it is **1.02 inches from
each of two 58.5 sq in wedges whose centroids are exact mirrors** — there is no
basis on which to prefer one, and picking by list order dropped an objective and
broke the table's symmetry. Taking both restores it. This is not a tuned
exception: it reproduces the hand-traced 5-or-6 split on **41 of 45 tables**,
against 24 when every marker took exactly one.

**And no discs.** Eight of 225 markers sit beyond reach of every ruin —
at most 5.16", four of them the exact board centre. They resolve to the nearest
ruin anyway. An objective that is not ground would be the previous edition's
free-standing marker under a new name. Six of the eight sit at *exactly* 3.00",
one control range, so that ruin is precisely the ground you would hold the
marker from; two (4.0" and 5.16") are a genuine approximation.

Tables carry **five or six** objectives (25 and 20, against the hand-traced 24
and 21), none is a disc, and the count matches the hand-traced one on **44 of
45** tables. Objectives land within 3" of a hand-picked one **96%** of the
time.

### What the tracing had actually done

The old files' own comment said each objective was "the outline of the ruin the
layout puts a marker on". It was not. On `table_01`, **two of the layout's five
markers land inside an objective the file declares**; the rest sit elsewhere,
and the file's own comments name them `home`, `no man's land` and `centre` —
chosen by eye for board symmetry. The tables also carried **six** objectives on
27 of 45, which is the previous edition's mission count.

## 4. Two independent checks nobody designed for

Both were computed after the fact, and both came out exactly right — which is the
strongest evidence the pipeline is faithful.

- **The zone split is 78 / 78 across the two deployment zones** — exactly
  balanced — with 89 in the middle. The sides balancing to within one objective is the
  That exact balance is the check that matters, since the tables are
  point-symmetric, and it fell out rather than being aimed at; the middle is
  heavier because a tie is usually a pair of wedges flanking the board centre.
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

> ⚠ **Stale — re-running.** The figures below were measured before objective
> resolution was finished (they predate largest-in-range and one-ruin-per-marker).
> They are kept because the *method* section under them stands; the numbers do
> not.

Four scripted policies on the golden training config, **all 45 tables**, n=30
each, traced geometry against generated, paired by table. All 45 are admissible
here for the same reason the joint decode was measured on all of them: **a
scripted policy has no weights**, so the held-out split constrains nothing about
this comparison.

Reported with both a t and a **sign count**, because the per-table differences
are heavy-tailed: a few tables move hugely in both directions while most move
slightly one way, and the two tests disagree often enough that quoting either
alone misleads.

| | traced | generated | diff | t | same sign |
|---|---|---|---|---|---|
| **`vp_margin`** | | | | | |
| `random` | −217.9 | −236.8 | **−18.9** | −6.58 | 33/45 |
| `squad_march_take` | −1.6 | −10.2 | −8.7 | −2.43 | 30/45 |
| `squad_march_shoot` | −7.8 | −12.1 | −4.3 | −1.02 | 28/45 |
| `squad_march_deny` | −7.4 | −14.7 | −7.3 | −1.93 | **24/45** |
| **`held`** | | | | | |
| `random` | 0.11 | 0.00 | **−0.11** | −5.15 | **44/45** |
| `squad_march_take` | 2.64 | 2.35 | **−0.29** | −5.95 | 36/45 |
| `squad_march_shoot` | 2.62 | 2.41 | **−0.21** | −3.78 | 32/45 |
| `squad_march_deny` | 2.36 | 2.26 | **−0.11** | −2.72 | 29/45 |
| **`on_obj`** | | | | | |
| `squad_march_take` | 0.44 | 0.39 | **−0.05** | −6.09 | 38/45 |
| `squad_march_shoot` | 0.44 | 0.40 | **−0.04** | −3.71 | 30/45 |
| `squad_march_deny` | 0.40 | 0.37 | **−0.03** | −3.46 | 31/45 |
| **`coherent`** | all four | | **≤ 0.01** | | |

**The tables are harder to hold, and that is what is established.** `held` falls
for every policy and `on_obj` with it, at t = −2.7 to −6.1 and consistent in
sign. Coherency is untouched, as it should be — this changed terrain and
objectives, not formation.

**`vp_margin` is weaker than it looks.** Only `random` is decisive. `deny`'s
−7.3 has t = −1.93 but a sign count of **24/45, which is chance** — the mean is
carried by a handful of tables, not a broad shift. `take` is the one script with
both tests agreeing (t = −2.43, 30/45), and `shoot` is unresolved on both.

`random` collapsing by 18.9 is the clearest single effect and has an obvious
cause: it scored by deploying onto home objectives and standing there, which is
exactly the behaviour moving the objectives punishes, and nothing on the other
side loses with it.

### What the ruin merge itself cost

Isolated by re-running everything against the pre-merge tables — same policies,
same seeds, and an old-tables column that reproduced **to the decimal** on all
four policies, so the harness is deterministic and the difference is the maps.

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

### The pool has a resolution floor, and it is about 8 vp

Per-table `vp_margin` sd is **24 to 28**, so at n=45 the standard error is
**3.6 to 4.2**. Effects under ~8 vp are not resolvable on this pool *at all* —
and more episodes per table cannot help, because the variance is **across
tables** and there are only 45 tables in existence.

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

All 45, n=30: `random` −236.8 · `take` −10.2 · `shoot` −12.1 · `deny` −14.7.
Coherency 0.76–0.80 throughout.

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

Deployment zones — the other half of the source, six real shapes per layout
including one that faces the armies across the *short* axis — are deliberately
**not** in this change. They are a second scenario change and are measured
separately, or neither would be attributable.
