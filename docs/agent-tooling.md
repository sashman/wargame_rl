# Agent tooling — the data a plan is made of

[docs/rules/](rules/README.md) is what is **legal**. [docs/play-doctrine.md](play-doctrine.md) is
what is **good**. This file is the third leg: the **numbers a judgement needs**, and the tools
that produce them.

Doctrine states claims. Almost every entry there names a "cheapest test", and most of those
tests did not exist — so a claim like *"rank denial targets by how many models it takes to reach
parity"* had no way to be read off a board. A tool here answers exactly one such question, in
data, cheaply enough to run before anything is built on it.

## 1. Three standing warnings

- **A tool is not evidence, and this file is not a record.** `CLAUDE.md` and
  [reports/](../reports/README.md) hold measured numbers. `play-doctrine.md` § 1 names
  "becoming a second record" as *the* failure mode of a catalogue; this is a third catalogue and
  a third chance at it. **No measured result belongs in this file** — only what a tool computes,
  what it gets wrong, and how to run it.
- ⚠ **EVERY FIELD HERE IS A NEXT-TURN QUANTITY.** The opponent **moves before it shoots**, so a
  tool reporting what bears *now* answers a question nobody asks while deciding where to move,
  and answers it **false-safe**. This is the single most important thing in this document and it
  was learned the hard way: the shipped `[R]` overlay had been the only threat visualisation for
  months, and it draws the wrong horizon for that purpose.
- **A tool that overstates is fine; a tool that hides which way it overstates is not.** Every
  entry carries a *What it gets wrong* field naming the direction of each bias. It is never
  omitted, and "none known" is not an acceptable value — it means the analysis was not done.

## 2. Macro and micro

The split is the reasoning, not the code.

**Macro** is the plan: read the table, the mission and the opposing list, then decide which
ground to take, which to deny, which fights to seek and which to refuse. It is made mostly
*before* a model moves and revised each round. Its questions are about ground and matchups.

**Micro** is execution inside a plan: which unit fires at what and in what order, and exactly
where each model stands so the count crosses the threshold. Its questions are about allocation.

The asymmetry worth stating up front: **macro has most of the value here, and micro has a
measured ceiling.** Only 3.6% of declared attacks are discarded, and the closest measured proxy
for smarter target selection recovered +1.7 ± 5.7 (`play-doctrine.md` D-25, `refused`). Perfect
fire discipline is worth less than the noise on a three-seed screen. Ground is not.

## 3. How to read an entry

Cite tools by ID (`T-03`), never by title.

| field | contents |
|---|---|
| **Produces** | The data, with units and shape. |
| **Serves** | The `D-NN` doctrine entries, and the measured failure it speaks to. |
| **Consumer ladder** | `report` → `scripted policy` → `observation`, and the rung it is funded to. |
| **Already exists** | The symbols already in the repo. **This is the field that stops rebuilds.** |
| **What it gets wrong** | Named biases, each with its direction. Never omitted. |
| **Cheapest test** | The literal command, or `do not build` and why. |

**The consumer ladder is the discipline this project already runs on.** A number is worth having
as a printed table long before it is worth having in a tensor:

| rung | cost | what it voids |
|---|---|---|
| `report` | one CLI run, usually no GPU | nothing |
| `scripted policy` | one inference run, no GPU | nothing — and it is the only way a claim becomes evidence here |
| `observation` | a training run per seed | **every checkpoint**, and it needs a 3-seed paired screen |

⚠ **Nothing in this file is funded past `scripted policy`.** Widening the observation is a
separate decision with its own pre-registration, and the record is against the obvious version:
`observe_threat_count` was added, measured null across both seeds, and removed.

## 4. The layer

`wargame_rl/wargame/envs/board/` — a **leaf** package that may import `envs/domain`, `envs/types`,
stdlib and numpy, and nothing else. `tests/test_board_layer_is_a_leaf.py` enforces it.

It lives inside the package rather than in `scripts/` for one reason: **scripted policies live in
`envs/baseline/` and cannot import `scripts/`**. Since the cheapest form of any claim here is a
scripted policy, maths that a policy cannot reach can never be priced the way this project
prices things.

| module | owns |
|---|---|
| `board/grid.py` | `BoardGrid` — the sampling grid every board-wide read shares |
| `board/threat.py` | `ThreatField`, `VisibilityCache` — where it is dangerous to stand |
| `board/matchup.py` | `UnitProfile`, `Matchup` — unit-versus-unit trades |
| `board/reach.py` | `ObjectiveReach` — who arrives first, and whose ground it was |

`renders/v2/control.py` imports `board/` and is the **only** module in `renders/` that may — the
single-seam rule that file's own docstring already states.

---

## 5. A — Macro: the ground

| ID | Produces | Rung | Status |
|---|---|---|---|
| T-01 | Unit-versus-unit expected casualties, reach margin and free rounds | `report` | built |
| T-02 | Next-turn fire threat as a scalar field over the board | `report` | built |
| T-02b | Next-turn charge threat, as `P(2D6 ≥ gap)` | — | deferred to the melee merge |
| T-03 | Earliest arrival on each objective, both sides | `report` | built |
| T-04 | Own-zone / contested / hostile classification per objective | `report` | built |

### T-01 — The matchup table

**Produces.** For every pair of units: expected casualties per round of fire in the open,
expected wounds before the clip, the share of wounds the destroyed-unit cap discards, rounds to
halve the target, the reach margin in inches, and `free_rounds` — rounds of unanswered fire
while the shorter-ranged unit closes. Plus an exchange ratio quoted at **two** distances.

**Serves.** D-20 (trade cheap units for expensive ones), D-23 (shoot what scores). It is the
pre-game half of the "favourable matchup" question: which of ours wants to meet which of theirs.

**Consumer ladder.** `report`. The per-model form is **already at the `observation` rung** — see
below — so the only unfunded rung is a *unit-level* observation feature, which nothing wants yet.

**Already exists.** Nearly all of it. `domain/shooting.py::expected_damage` and
`expected_damage_matrix` compute the per-model matrix, memoised per *distinct stat pair*, and it
is hstacked onto every player model token in `model/common/observation.py`. `board/matchup.py`
is a **reduction of that same matrix** — attacker axis sums, defender axis does not — so the
table a human reads and the number the network sees cannot disagree. The closed-form theory,
including the abilities nothing implements yet, is [docs/expected-damage.md](expected-damage.md).

**What it gets wrong.**
- **Cover is not applied**, so every entry is the open-ground expectation — the same choice the
  shipped matrix documents. Overstates damage wherever the target is in a ruin, which on the
  real tables is wherever the target is scoring.
- **No allocation.** The defender's own model-allocation rule is not simulated; a volley is
  priced against one representative model.
- ⚠ **On the config that trains, this table is 1×1.** Both armies in
  `configs/golden/25v25_maps_two_mode.yaml` are one profile, and the report says so. It is
  informative only where profiles differ.
- **Range is never in the damage number.** This tool has no positions. Reach appears only as
  `reach_margin` and `free_rounds`. A blended scalar would hide which distance chose it.

**Cheapest test.** `just measure-matchups configs/experiments/30v15_fast_horde_vs_elite.yaml`

### T-02 — The next-turn threat field

**Produces.** A `(Q,)` scalar over a 1" grid: **expected casualties to one reference model
standing on that cell, during the opponent's next shooting phase**. Plus the shooter count per
cell, the unclipped wound figure, and disjoint quantile bands for drawing.

⚠ **The definition is `next_turn`, and that is the entire point:**

```
threatened(c) ⟺ ∃ model m, ∃ position p : |p−m| ≤ move(m) ∧ |p−c| ≤ range(m) ∧ LOS(p, c)
```

Sight is traced **from the ground they can reach**, not from the ground they occupy. A cell
behind a ruin from where an enemy stands is shot from beside that ruin one move later.

**Serves.** D-14 (threat is move plus range, not range) — this is that entry's claim made
computable. D-15 (stage outside threat range), D-16 (route through blocked sight), D-22.

**Consumer ladder.** `report`, and a renderer overlay. ⚠ **Deliberately not funded to
`observation`**, and the record is why: the agent **does not use terrain for cover; it manages
range** (`settled`), D-18 refuses cover-seeking outright, and `observe_threat_count` was a
measured null. Its honest value here is diagnostic and, later, scripted staging.

**Already exists.** The *current-turn* half, as a renderer overlay:
`renders/v2/control.py::compute_threat_region` sweeps range ∩ LOS on the same grid and
rasterises it to rings. `board/threat.py` reproduces it exactly at `ThreatHorizon.current`,
which is the only reason that horizon is kept — it pins the new code against already-verified,
pixel-asserted behaviour.

**What it gets wrong.**
- ⚠ **Cover is not applied**, and unlike T-01 this is not a memoisation choice: the three-state
  `visibility_matrix` is **not on `BattleView`**, and a grid cell **has no base radius**, while
  the cover predicate is *defined* by offsetting rays by the endpoints' radii. Cover at a cell is
  undefined rather than merely expensive. ⚠ The bias runs **against objectives specifically** —
  every marker on the real tables sits inside a terrain piece — so this field paints the safest
  ground in the game as dangerous. Read it beside `just measure-hold-hazard`.
- **Coherency binds the opponent's move.** A free `move`-radius disc per model is not a legal set
  of destinations. Overstates reach.
- **Freezing.** Only ~92% of ordered inches are delivered. Overstates reach again.
- **An advance never extends threat**, because declaring one spends the unit's shooting. The
  origin set is dilated by `M` and never by an advance rung — getting this wrong would overstate
  reach by up to 6".
- **The reachable set is sampled at cell centres**, and this one runs the *other* way — it
  **understates**, by up to the grid spacing. It is the reason not to draw the field coarser than
  the shipped 1" without saying so.
- **`ThreatHorizon.current` is in the API and is not a planning answer.** Choosing where to stand
  by reading it is the error the module exists to remove.

**Cheapest test.** Look at a board first, which is D-14's own instruction — now with the horizon
that matters:
`just play configs/golden/25v25_maps_two_mode.yaml squad_march_take tabletop --threat-field`
(press `[T]` to toggle, `[R]` for the current-turn outline; the ground between the two is the
finding). Then
`just measure-threat-field squad_march_take configs/golden/25v25_maps_two_mode.yaml 5 configs/evaluation/maps_heldout`.

### T-02b — Next-turn charge threat *(deferred)*

**Produces.** Per cell, `P(2D6 ≥ gap)` — the probability an opponent unit that moves first can
then complete a charge onto a model standing there. A genuine continuous spectrum rather than a
ring, computed off the **same reachable-origin set** the fire layer already builds.

**Serves.** D-11, D-12, D-32/D-33 (currently parked on the absent mechanic).

**Already exists.** Nothing here yet — `melee.enabled` defaults off on `main` and the charge and
fight phases are `absent` in the [gap map](rules/implementation-status.md). The mechanic is being
implemented on `feature/melee-stage-0`, where declaration needs a unit-to-unit gap ≤
`melee.charge_range`, the unit not already engaged and not having advanced or fallen back, and
**2D6 caps the charge move**.

**What it gets wrong.** Not yet built, so nothing measured. Two things to write into it when it
is: the same coherency and freezing overstatements as T-02, and the fact that on that branch
**the charge's value is the shooting shield rather than the blade** — so a charge-threat layer is
really "they can lock my unit out of shooting", and reading it as expected damage would badly
misprice it.

**Cheapest test.** `do not build until melee merges.` Building against an unmerged, still-moving
shape buys a rewrite.

### T-03 — Earliest arrival

**Produces.** Per objective, per side: the earliest **whole round** that side's fastest unit
could be standing on it, which unit that is, and the margin between the two sides.

**Serves.** D-15 (staging), D-26 (contest before *their* scoring moment), D-07/D-08 (choose the
cheapest points). It is the missing half of D-02.

**Consumer ladder.** `report`, and it is the entry here most worth taking to `scripted policy` —
a timing-aware `squad_march_take` subclass is Arm 1 of the doctrine backlog and this supplies the
arrival deadline it needs.

**Already exists.** Nothing computed arrival order before this. `measure-objective-split` reports
a **redistribution ceiling** that `CLAUDE.md` explicitly flags as "deliberately optimistic — no
travel time, no return fire"; T-03 is the travel-time half of that bill.

**What it gets wrong.** Three overstatements, all in the same direction — **arrival is a lower
bound, never a schedule**:
- Coherency binds the unit, so the straight line from its centroid is a bound and not a route.
- Freezing eats ~8% of ordered inches.
- Nothing routes around a base or a terrain piece.

⚠ And a reading trap: **a margin of 0 is a tie, and a tie controls nothing** — control is
`player_count > opponent_count`, strictly.

**Cheapest test.**
`just measure-ground configs/golden/25v25_maps_two_mode.yaml configs/evaluation/maps_heldout`

### T-04 — Objective triage

**Produces.** Each objective classified `own_zone` / `contested` / `hostile` from the deployment
outlines, aggregated over a table set.

**Serves.** D-01 directly, which is `blind` and has never been priced as a policy.

**Consumer ladder.** `report`. As an `observation` it is an objective-token widening, and
[docs/missions-design.md](missions-design.md) Tier 1 warns to **buy one widening for every such
feature at once** rather than one per idea.

**Already exists.** The zone outlines are on `BattleView` (`deployment_outline`,
`opponent_deployment_outline`) and in config. Nothing classified objectives against them.

**What it gets wrong.** ⚠ **The rules define no board regions** — no "half", no "centre" — so
these three classes are derived from the zone outlines and nothing else. 34 of the 45 real
tables have non-rectangular zones and `long_edges` splits the **short** axis, so any board-half
rule would mean a different thing on every table. Where a config carries rectangles rather than
outlines, everything reads `contested`, which is honest rather than informative.

**Cheapest test.** Printed by `just measure-ground` (above).

---

## 6. B — Macro: reading a policy against the ground

| ID | Produces | Rung | Status |
|---|---|---|---|
| T-05 | Exposure census: threat at every model's position, split on-objective / in-transit | `report` | built |
| T-06 | Contested-arrival ledger: models needed to cross the threshold, by round | `report` | proposed |
| T-07 | Scoring-clock ledger: VP forgone per round per objective | `report` | proposed |

### T-05 — The exposure census

**Produces.** For every living model at each battle round, the next-turn threat at its position,
split by whether it is **on an objective** or **in transit**. Plus the calibration: the share of
board cells the current-turn map calls clear and the next-turn map calls dangerous.

**Serves.** The measured standing failure directly. The agent finishes with far more of its army
alive than the scripts while holding fewer objectives, and the standing explanation is that it is
avoiding danger — which predicts its exposure is **lower** than theirs. If it is not, hoarding is
a **search** failure, which is what `measure-critic-probe` already concluded from the other side.

**Consumer ladder.** `report`. It is a falsifier, not a feature.

**Already exists.** `env_components/exposure.py` computes `exposure_rate` — "at least one enemy
can see me", boolean and current-turn. This is the graded, next-turn, position-priced form.
⚠ `eval/exposure_rate` **changed definition** on 2026-08-13 and is not comparable across that
date at all.

**What it gets wrong.** Everything T-02 gets wrong, and one thing more: the on-objective column
is **specifically** overstated, because cover is not applied and every objective on the real
tables is a ruin. `just measure-hold-hazard` prices the same trade with the real predicate — read
them together, and where they disagree, the hazard measurement wins.

**Cheapest test.**
`just measure-threat-field squad_march_take configs/golden/25v25_maps_two_mode.yaml 5 configs/evaluation/maps_heldout`,
then the same for a checkpoint. **The comparison is the result; neither column alone is.**

### T-06 — The contested-arrival ledger *(proposed)*

**Produces.** Per objective: the minimum models that must be landed, and by which round, to cross
the strict `>` threshold — given what the opponent can land there in the same time.

**Serves.** D-02, D-04, D-05, D-24. Control is a headcount and a tie holds nothing, so the cost
of denying a point is exactly the opponent's count there while the payoff is a flat 5 VP —
cheapest-first is the whole rule, and nothing computes the cost.

**Consumer ladder.** `report` → `scripted policy`.

**Already exists.** T-03 supplies the arrival half and `measure-objective-split` supplies
per-objective counts. This is the join.

**What it gets wrong.** Inherits every T-03 overstatement, and adds one: it assumes the opponent
contests, which is the pessimistic assumption where T-03's is optimistic. Say which when quoting.

**Cheapest test.** `do not build yet` — read `just measure-objective-split` and
`just measure-ground` side by side first, and build this only if the join is what is missing.

### T-07 — The scoring-clock ledger *(proposed)*

**Produces.** VP forgone per round per objective, cumulative over an episode.

**Serves.** D-26, D-28. ⚠ `CLAUDE.md` states the gap plainly: `held` is an **end-state snapshot
with no notion of which points were paid**, while VP accrues at every scoring moment. This is the
per-round form.

**Consumer ladder.** `report`.

**Already exists.** The mission calculator already computes control at each scoring moment;
`state/` records the episode. This is an accumulation over data already produced.

**What it gets wrong.** Nothing structural — it is bookkeeping over the real scoring events. The
trap is in reading it: a policy that banks late and a policy that banks early can total the same,
and only the per-round curve distinguishes them.

**Cheapest test.** `just analyze <recording>` first — some of this may already be derivable from
an event log without new code.

---

## 7. C — Micro: the drills

⚠ **Designed, not built.** Both entries here are recorded so the mechanism is not rediscovered
and the ceiling is quotable.

### T-08 — The positioning drill *(design only)*

**Produces.** Not data — a **training-time start-state distribution**, plus a fork-and-price
probe to evaluate it.

**Serves.** The search diagnosis. `measure-critic-probe` found `corr(dV, dVP) ≈ 0`: reward and
critic both value spreading correctly and the policy simply never finds it. That prescribes
exploration and representation work, **not** reward attribution.

**The mechanism, and why the obvious answer is wrong.** ⚠ **A drill is not a reward phase.**
`reward/phase_manager.py::try_advance(success_rate, epoch)` varies **reward weights and success
criteria only** — never the scenario, the starting board, the action mask or the phase set. The
one shipped mechanism that changes the training *state distribution* is
`place_for_episode(..., augment_start)` with `config.start_on_objective_probability`. So: yes a
curriculum, but a **start-state curriculum**.

**And the design is already written down in the repo.**
`domain/placement.py::start_group_on_objective`'s own docstring says:

> *"The honest fix is to start the squad **part way** along the approach rather than on top of
> the objective, which would also give the value function a path to propagate along; that is not
> built."*

That is T-08: `start_group_en_route(..., fraction)`, annealed `1.0 → 0.0`. It needs **no reward
term** (it touches placement, not `reward/`), **no charge phase**, and **no observation
widening**, so it voids no checkpoint. It must keep the three disciplines the existing
augmentation established — training-time only under `reset(options={"augment_start": True})`;
**off by default and drawing nothing from `rng` when off**, so a control run stays bit-identical;
and best-effort, never failing an episode.

⚠ **The real build cost is a channel that does not exist:** nothing anneals a config scalar
across epochs. Scope that, or ship at a fixed `fraction` first.

**What it gets wrong.** A teleport is not a legal move, so the start states are off-distribution
by construction — which is the point, and also why it must be evaluated by playing out rather
than by inspecting value estimates.

**Cheapest test.** Reuse `measure_critic_probe`'s fork-and-price machinery with the
counterfactual replaced by the drill start state. ⚠ **Pre-register on the paired mean across
three seeds with a stated lower bound, not on per-seed signs** — `CLAUDE.md` records a per-seed
−8 bound that fails 56% of the time at sd 11.3. **Power-check the bound against the expected
spread before writing it down.**

### T-09 — The shooting drill *(design only, and capped)*

**Produces.** A start state where one objective's control label is **one casualty from flipping**,
so the decision under drill is whether to concentrate.

**Serves.** D-24 (concentrate until a count changes) and only D-24.

⚠ **The ceiling, stated first because it decides whether to build at all.** Only **3.6%** of
declared attacks are discarded — a whole-unit-destroyed condition — and the closest measured
proxy for smarter target selection recovered **+1.7 ± 5.7**. D-25 is `refused` on exactly that
arithmetic. **Any efficiency- or ordering-flavoured shooting drill is below that ceiling and is
already refused.**

What is *above* it is D-24, because its value is a **step function on control counts** rather
than an efficiency gain: taking a defender from 2 to 1 where you stand with 1 turns their 5 VP
into nobody's. That is a mission quantity, not a targeting one.

**Consumer ladder.** `scripted policy` first, and it may never need more:
`BaselinePolicy.select_shooting` is a one-file change.

**What it gets wrong.** A drill that constructs a one-casualty-from-flipping state samples a
situation far more often than play produces it, so anything learned is conditioned on a
distribution the game does not have.

**Cheapest test.** `do not build the drill.` Write the scripted policy and run
`just measure-paired <new> squad_march_shoot configs/golden/25v25_maps_two_mode.yaml 100` on
three seed bases. If a scripted concentrator cannot beat the bar, no drill will teach it.

---

## 8. What this document may never propose

- **A tool whose output is not reachable from a scripted policy.** That is the rung this project
  trusts, and a number nothing can act on is a number nothing can price.
- **A threat quantity at the current-turn horizon, presented as a planning input.** It reads
  false-safe. `ThreatHorizon.current` exists to be tested against and to be measured *against*,
  not to be read.
- **An observation widening funded by this file.** Each is a training run per seed and voids
  every checkpoint. `objective_budget` 6 and `terrain_budget` 16 do not change.
- **A shooting-side tool aimed at efficiency or firing order.** Refused with numbers: D-25.
- **A measured number recorded here rather than in `CLAUDE.md` and `reports/`.**

## 9. Related

- [docs/play-doctrine.md](play-doctrine.md) — the claims these tools price. Cited by `D-NN`.
- [docs/rules/](rules/README.md) — what is legal. Wins over this file on any conflict.
- [docs/expected-damage.md](expected-damage.md) — the closed form behind T-01.
- [docs/shooting.md](shooting.md) — the attack sequence, masks, and the observation block.
- [docs/metrics.md](metrics.md) — what each existing measurement means.
- [docs/ddd-envs.md](ddd-envs.md) — the dependency direction `board/` sits in.
- [CLAUDE.md](../CLAUDE.md) and [reports/](../reports/README.md) — **the record**. Where they
  disagree with this file, they win.
