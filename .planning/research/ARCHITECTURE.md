# Architecture Research — v2.0 Terrain & Line-of-Sight Blocking

**Domain:** Brownfield RL tabletop wargame (discrete grid, DDD env, transformer-over-entities observation)
**Researched:** 2026-06-19
**Confidence:** **HIGH** (grounded in a full read of the env source: `domain/los.py`, `battle.py`, `battle_factory.py`, `battle_view.py`, `config.py`, `observation_builder.py`, `model/common/observation.py`, `net.py`, `renders/human.py`, plus `docs/ddd-envs.md`)

> **Scope reminder.** This milestone adds: terrain piece = **footprint rectangle** (area marker, *no gameplay effect this milestone*) + **thin walls** (L-shaped segments inside the footprint) that **block LOS only** (movement unaffected), and **terrain encoded in the observation**. Everything else (cover, dense, difficult, impassable, elevation, procedural placement, variable bases) is deferred.

---

## 0. Headline recommendations (for the roadmapper)

1. **Wall representation: WHOLE-CELL blocking, not cell-edge.** Rasterise wall segments into the existing cell-based blocking grid and feed them through the *unchanged* `is_blocking(x, y)` seam in `domain/los.py`. This is the single highest-leverage decision — see §4 for the full comparison and rationale. Cell-edge is more geometrically precise but requires rewriting the LOS traversal (supercover / segment-intersection), breaks the tested "interior-cells-only" contract, and duplicates ray logic — all explicitly against the milestone goal of *"integrated into `domain/los.py`, no duplicate Bresenham."*
2. **Terrain state lives in the domain.** New `domain/terrain.py` value objects + a pure rasterisation service; the `Battle` aggregate holds the terrain list and a **precomputed static blocking grid**; `BattleView` gains a read-only `terrain` accessor. Config → factory → aggregate, exactly like models/objectives.
3. **Observation: a new entity-token stream appended LAST.** Mirror the existing `opponent_model_embedding is None` pattern with a `terrain_embedding`. Terrain tokens go *after* opponents so player/opponent token positions (and the per-model action heads) are byte-for-byte unchanged. **No terrain → no terrain tokens → no terrain embedding → identical network and identical behaviour to today.** Shape stays stable; backward compatibility is structural, not incidental.
4. **One blocking seam serves everyone.** `view.has_line_of_sight_between_cells(...)` already backs shooting masks, action masks, the human renderer, and the snapshot. Source the merged (config `blocking_mask` ∪ rasterised walls) grid behind that one method and **every consumer gets terrain for free with zero call-site changes.**
5. **Two phases.** Phase A = terrain in the simulation (config + domain + LOS + render). Phase B = terrain in the observation (obs type + tensor + both networks). A depends on nothing new; B depends on A.

---

## 1. Existing architecture (what to reuse vs. extend)

### 1.1 The LOS seam already exists and is single-source

```
domain/los.py
  has_line_of_sight(x0,y0,x1,y1, w,h, is_blocking)  # Bresenham, interior cells only
  iter_los_cells(...)                                # same trace, for render/debug
        ▲
        │ is_blocking: Callable[[int,int], bool]   ← THE injection point
        │
wargame.py
  _make_is_blocking()  → closure over config.blocking_mask
  has_line_of_sight_between_cells(x0,y0,x1,y1)  ← BattleView method
        ▲
        ├── env_components/shooting_masks.compute_shooting_masks(has_los_fn=...)
        ├── env_components/observation_builder (shooting validity mask)
        ├── renders/human._draw_debug_los_line (via iter_los_cells)
        └── state/snapshot (LOS queries)
```

The current blocking predicate is `lambda x, y: bool(mask[y][x])` built from `config.blocking_mask` (a validated `list[list[bool]]`, shape `board_height × board_width`, default `None` = no blocking). **Walls become a second source merged into the same predicate.** `los.py` itself does not change at all.

> Minor existing inefficiency worth fixing in passing: `_make_is_blocking()` rebuilds a closure on *every* LOS query. Precomputing a static grid on the `Battle` (terrain never moves within an episode) removes that and is "complexity at startup, simple at runtime."

### 1.2 The observation is a transformer over entity tokens

Token sequence built in `net.py::encode_state` (HIGH confidence — read directly):

```
tokens = [ game(1) | objectives(N_obj) | players(N_p) | opponents(N_o) ]
                     └─ n_prefix = 1 + N_obj ─┘
player p   → token  n_prefix + p
opponent o → token  n_prefix + N_p + o
```

- Each entity type has its **own `nn.Linear` embedding**. Opponents use the `opponent_model_embedding: nn.Linear | None` pattern — **`None` when `opponent_model_size == 0`**, in which case zero opponent tokens are emitted and the forward pass is identical to a no-opponent game. **This is the exact precedent for terrain.**
- Per-model **action heads** read player latents at fixed positions `n_prefix + p`. The critic reads the **game token (index 0)**. The dead-row / alive logic infers an `alive` column from the *player feature width* and a trailing `n_opponents` expected-damage block — it inspects feature *columns*, not token *count*.
- The tensor pipeline (`model/common/observation.py`) returns an **ordered list**: `[game, objectives, players, opponents, action_mask]`. `encode_state` reads `opp = xs[3]`, `mask = xs[4]`.

**Implication:** appending terrain tokens *after* opponents leaves `n_prefix`, player positions, opponent positions, action heads, the critic token, and the alive-column inference all untouched. The only mechanical edits are: add a `terrain_embedding`, concat terrain tokens last, and shift the action-mask to the end of the tensor list (`xs[5]`) with its readers updated.

### 1.3 Config → factory → aggregate → view (the DDD spine)

`WargameEnvConfig` (Pydantic, backward-compatible optional fields) → `battle_factory.from_config` (builds entities, zones) → `Battle` (aggregate holds state) → `WargameEnv` implements `BattleView` → reward/render/obs consume the view. Terrain slots into this spine identically to how objectives already do.

---

## 2. Where terrain state lives (component placement)

| Concern | Location | New / Modified | Notes |
|---|---|---|---|
| **Authoring schema** (footprint rect, wall segments) | `types/config.py` — `TerrainPieceConfig`, `WallConfig`; `WargameEnvConfig.terrain: list[TerrainPieceConfig] \| None = None` | **New types, modified config** | Default `None` ⇒ no-op (backward compat). Validators: footprint in-bounds, walls inside footprint. |
| **Domain value objects** | `domain/terrain.py` — `Footprint`, `Wall`, `TerrainPiece` (frozen dataclasses) | **New file** | Pure; imports `types/` only. Mirrors `value_objects.py`. |
| **Wall → blocking-cell rasterisation** | `domain/terrain.py` — `build_los_blocking_grid(terrain, base_mask, w, h) -> np.ndarray[bool]` | **New (pure) service** | OR of config `blocking_mask` and rasterised wall cells. **Footprints are NOT rasterised** (no LOS effect this milestone). No Gym, no Bresenham. |
| **Aggregate state** | `domain/battle.py` — `Battle.terrain`, precomputed `Battle.los_blocking_grid` (+ `is_los_blocked(x,y)`) | **Modified** | Terrain is static per episode ⇒ rasterise once at construction. `reset_for_episode` leaves it intact. |
| **Factory wiring** | `domain/battle_factory.py` — build `TerrainPiece`s, call rasteriser, pass grid + list to `Battle` | **Modified** | Same shape as `_build_objectives`. |
| **Read-only projection** | `domain/battle_view.py` — `terrain` property (+ optionally `is_los_blocked`) | **Modified** | So obs/render depend on the view, not the env. |
| **Blocking seam** | `wargame.py::_make_is_blocking` → return `self._battle.los_blocking_grid` predicate | **Modified** | Replaces the `config.blocking_mask`-only closure. Single change point benefits all LOS consumers. |
| **Env terrain accessor** | `wargame.py` — `terrain` property delegating to `_battle` | **Modified** | Implements the new `BattleView.terrain`. |

---

## 3. How walls feed into `los.py` (no Bresenham duplication)

```
config.terrain ─► battle_factory ─► domain/terrain.build_los_blocking_grid(
                                        terrain, config.blocking_mask, w, h)
                                     │  (rasterise wall segments → bool cells,
                                     │   OR with existing blocking_mask)
                                     ▼
                              Battle.los_blocking_grid  (static np.bool_ [h][w])
                                     ▼
              WargameEnv._make_is_blocking → lambda x,y: grid[y][x]
                                     ▼
       domain/los.has_line_of_sight(..., is_blocking)   ← UNCHANGED
                                     ▼
   shooting_masks │ action masks │ renderer │ snapshot   ← UNCHANGED call sites
```

**Wall rasterisation** = walk each wall segment's cells (reuse `domain/los._bresenham_line` / `iter_los_cells` over the segment endpoints, or a dedicated tiny rasteriser) and mark those cells `True`. Because the LOS check is already cell-granular and interior-only, a wall that occupies its cells correctly blocks any ray whose interior passes through them. **Movement is untouched** — the grid is consumed *only* by the LOS predicate, and nothing in movement reads it.

This satisfies the milestone constraint verbatim: *"Walls become a new input alongside the existing optional `blocking_mask`"* and *"integrated into `domain/los.py`"* with **zero changes to the Bresenham core and zero new ray code in renderers or reward.**

---

## 4. KEY DECISION — cell-edge segments vs. whole-cell blocking

**Recommendation: WHOLE-CELL blocking (rasterise walls into the cell grid).** Confidence: HIGH.

| Criterion | **Whole-cell (recommended)** | Cell-edge (true thin walls) |
|---|---|---|
| **Geometric accuracy** | Wall inflates to ≥1 cell; coarse but matches the existing cell-granular model | Sub-cell precise; a wall sits exactly between two cells |
| **`los.py` integration** | **Zero changes** — reuse `is_blocking(x,y)` seam | Requires new traversal: Bresenham steps diagonally across corners, so edge-crossings are *missed* → need supercover line or continuous segment–segment intersection. **New ray logic.** |
| **Breaks tested contract?** | No — preserves "interior cells only, endpoints excluded" (`test_los.py`) | Yes — endpoints/edges semantics change; existing golden-trace tests must be reinterpreted |
| **Bresenham duplication** | None | High risk — a second traversal path or a parallel intersection routine |
| **Observation encoding** | Orthogonal — terrain still encoded as tokens (see §5) | Orthogonal — same |
| **Renderer** | Fill blocked cells / draw wall cells | Draw lines on cell boundaries (slightly nicer visually) |
| **Backward compat** | Trivial (`None` ⇒ empty grid) | Trivial, but more surface area to get wrong |
| **Milestone fit** | Matches the logged Phase-3 decision *"v2 terrain maps onto mask"* and KISS | Fights the existing model; more scope |

**Why whole-cell wins here.** The entire LOS subsystem is discrete and cell-based: models occupy cells, rays are sequences of cells, blocking is evaluated per interior cell. Introducing sub-cell edge precision creates a *model mismatch* — you'd need a continuous geometric line and supercover/edge-intersection logic, which is precisely the "duplicate Bresenham" the milestone forbids, and it invalidates the interior-only contract the current tests pin down. Whole-cell keeps **one source of truth, one ray algorithm, one tested contract**, and is backward compatible by construction. The accuracy loss is acceptable because (a) movement is unaffected this milestone, (b) the grid resolution already bounds LOS precision, and (c) the *authoring* model can still be "thin L-shaped walls" — they simply **compile down** to blocking cells at construction time (complexity at startup).

**Authoring nuance to preserve thinness:** author walls as segments (endpoints inside the footprint). The rasteriser converts each segment to the thin line of cells it crosses (typically 1-cell-wide), so an L-wall becomes an L of blocking cells — visually and tactically "thin," without sub-cell math.

**Deferral note for the roadmap:** if a future milestone needs true thin walls (e.g. peeking around a 1-cell gap), revisit as a dedicated LOS-precision phase that adds a *supercover* option behind the same `is_blocking` seam — but do not pay that cost now.

---

## 5. Observation encoding (shape-stable, backward-compatible)

**Approach: a new variable-length "terrain" token stream, appended after opponents, with its own embedding (`None` when absent).** This reuses the proven objectives/opponents pattern and the `opponent_model_embedding is None` backward-compat mechanism.

### 5.1 What a terrain token carries

Encode the LOS-relevant geometry plus footprint context. Recommended: **one token per wall segment**, each carrying its parent footprint's bbox so the agent also "sees" the footprint area (for future cover and spatial reasoning):

```
terrain token (per wall) = [
  wall_x0, wall_y0, wall_x1, wall_y1,        # segment endpoints, normalised to [-1,1]
  fp_cx, fp_cy, fp_hw, fp_hh                 # parent footprint centre + half-extent, normalised
]   # fixed width (≈8)
```

- **Footprint-only pieces with no walls** contribute no tokens this milestone (footprint has no LOS effect yet). When cover lands, add footprint tokens or a per-token "is_footprint" flag — no shape break because the stream is already variable-length.
- *Alternative considered:* one fixed-width token per **piece** with a capped wall block. Rejected — variable wall counts force padding/capacity limits, whereas per-wall tokens fit the existing variable-length sequence model (objectives, opponents) cleanly and keep token semantics uniform.

### 5.2 Pipeline changes (the full chain)

| File | Change | New/Mod |
|---|---|---|
| `types/terrain_observation.py` | `WargameTerrainObservation` dataclass + `size` | **New** |
| `types/env_observation.py` | `terrain: list[WargameTerrainObservation] = field(default_factory=list)`; include in `size` | **Mod** |
| `env_components/observation_builder.py` | `_terrain_to_obs(view.terrain)`; populate `WargameEnvObservation.terrain` | **Mod** |
| `model/common/observation.py` | extract terrain features → new array; **add a terrain tensor to the list** (insert before mask: `[game, obj, players, opp, terrain, mask]`); update `_observation_to_numpy`, `observation_to_tensor`, `observations_to_tensor_batch` | **Mod** |
| `model/net.py` (`TransformerNetwork`, `MLPNetwork`) | `terrain_embedding: nn.Linear \| None` (None when `terrain_size==0`); in `encode_state` append terrain tokens **after opponents**; update `mask = xs[5]`, `opp = xs[3]`; `from_env` reads `terrain_size`; `share_backbone_with` shares the new embedding | **Mod** |
| `model/ppo/networks.py` | reflect the shared-backbone / embedding addition | **Mod** |

### 5.3 Why the shape stays stable & backward compatible

- **No terrain in config** ⇒ `view.terrain == []` ⇒ terrain tensor has **0 rows** ⇒ `terrain_embedding` is `None` (`terrain_size == 0`) ⇒ **zero terrain tokens** ⇒ token sequence is `[game | obj | players | opp]` exactly as today. Identical logits, identical value, **identical checkpoints loadable** for no-terrain configs.
- Terrain tokens are appended **after** opponents ⇒ `n_prefix`, player token indices (`n_prefix+p`), opponent indices (`n_prefix+N_p+o`), per-model action heads, and the critic's index-0 game token are **all unchanged**.
- The alive-column inference in `net.py` reads *player feature columns* (and trailing `n_opponents` cols), which terrain tokens (a separate tensor) do not touch ⇒ unaffected.
- **One contract to update:** adding a tensor to the observation list is a known, documented change point (`pytorch-dqn.mdc`), so the test unpacking edits (`test_state.py`, `test_dqn.py`, etc.) are anticipated, not surprises.

---

## 6. Data flow (explicit direction)

**Simulation path (LOS):**
```
config.terrain
  → battle_factory.from_config
      → domain/terrain: TerrainPiece[] + build_los_blocking_grid(terrain, blocking_mask)
  → Battle{ terrain, los_blocking_grid }            (static for the episode)
  → WargameEnv._make_is_blocking → grid predicate
  → domain/los.has_line_of_sight (UNCHANGED)
  → shooting masks / action masks / renderer LOS / snapshot LOS   (UNCHANGED call sites)
```

**Observation path:**
```
BattleView.terrain
  → observation_builder._terrain_to_obs
  → WargameEnvObservation.terrain
  → model/common/observation: terrain feature array → terrain tensor (xs[4])
  → net.encode_state: terrain_embedding → tokens appended LAST
  → transformer trunk (player/opponent tokens unmoved)
```

**Render path:**
```
BattleView.terrain → renders/human: draw footprint rect (translucent) + wall cells/segments
```

---

## 7. New vs. modified components (consolidated checklist)

**NEW**
- `domain/terrain.py` — `Footprint`, `Wall`, `TerrainPiece`, `build_los_blocking_grid(...)` (pure, no Gym).
- `types/config.py` additions — `TerrainPieceConfig`, `WallConfig` (Pydantic).
- `types/terrain_observation.py` — `WargameTerrainObservation`.

**MODIFIED**
- `types/config.py` — `WargameEnvConfig.terrain` field + validators (in-bounds, walls ⊂ footprint).
- `domain/battle.py` — hold `terrain` + precomputed `los_blocking_grid`; accessor `is_los_blocked`.
- `domain/battle_factory.py` — build terrain, rasterise grid, pass to `Battle`.
- `domain/battle_view.py` — `terrain` property (+ optional `is_los_blocked`).
- `wargame.py` — `_make_is_blocking` sources the merged grid; `terrain` property; (optional) observation_space dict entry.
- `types/env_observation.py` — `terrain` list + size.
- `env_components/observation_builder.py` — `_terrain_to_obs`, populate observation.
- `model/common/observation.py` — terrain tensor in the list; update both `*_to_tensor` fns.
- `model/net.py` + `model/ppo/networks.py` — `terrain_embedding`, terrain tokens appended last, mask index shift, `from_env`, `share_backbone_with`.
- `renders/human.py` — terrain overlay.
- `tests/` — terrain LOS-blocking tests, observation-shape/backward-compat tests, config validation tests.
- `docs/` — `docs/reward-phases.md`/`tabletop-rules-reference.md`/movement notes as needed; a new `docs/terrain.md` is reasonable.

---

## 8. Recommended build order (maps to phases)

Order respects DDD dependency direction (`domain → types`; `reward/render/obs → BattleView`) and the rule "no terrain = today's behaviour."

**Phase A — Terrain in the simulation (LOS-blocking).** *Unlocks everything; depends on nothing new.*
1. Config types + validators (`TerrainPieceConfig`, `WallConfig`, `WargameEnvConfig.terrain`), default `None`.
2. `domain/terrain.py` value objects + `build_los_blocking_grid` (pure, unit-tested without Gym).
3. Wire `battle_factory` → `Battle.terrain` + `los_blocking_grid`; expose `BattleView.terrain`.
4. Repoint `WargameEnv._make_is_blocking` to the merged grid.
5. Tests: a wall blocks a previously-clear ray; no-terrain config byte-identical to today; config validation.
6. **Renderer overlay** (footprint + walls) — BattleView-only, low risk; can ride in Phase A or trail as a tiny slice.
   - *Exit:* shooting/action masks and `has_line_of_sight_between_cells` respect walls; movement unchanged; existing tests green.

**Phase B — Terrain in the observation.** *Depends on Phase A (needs `BattleView.terrain`).*
1. `WargameTerrainObservation` + `env_observation.terrain` + builder `_terrain_to_obs`.
2. Terrain tensor in `model/common/observation.py` (insert before mask; fix both converters).
3. `terrain_embedding` (None-guarded) + append-last token in **both** networks; shift mask index; `from_env`; PPO backbone share.
4. Backward-compat tests: no-terrain ⇒ identical tensor shapes and a pre-terrain checkpoint still loads/infers; with-terrain ⇒ tokens present, player/opponent positions unchanged.
   - *Exit:* agent observes terrain; no-terrain runs reproduce prior behaviour.

> **Phase boundary rationale:** A is *mechanics* (does the rule work?), B is *perception* (can the agent see it?). They have a clean one-way dependency, separate test surfaces, and separate risk profiles (A touches domain + the LOS seam; B touches the tensor/network contract). Splitting them keeps each PR small and each backward-compat guarantee independently verifiable.

---

## 9. Anti-patterns to avoid (domain-specific)

| Anti-pattern | Why it's wrong | Do instead |
|---|---|---|
| Rasterising the **footprint** into blocking cells | Footprint has *no* LOS effect this milestone; would block sight through open terrain | Rasterise **walls only**; footprint is observation/authoring metadata |
| Adding edge/supercover ray code now | Duplicates Bresenham, breaks interior-only contract, scope creep | Whole-cell rasterisation behind the existing `is_blocking` seam |
| Computing LOS-with-terrain inside the renderer or a reward calculator | Divergent rules vs. actual shooting; train/eval mismatch | Single seam: `BattleView.has_line_of_sight_between_cells` |
| Inserting terrain tokens **before** players/opponents | Shifts player token indices → breaks per-model action heads & dead-row logic | Append terrain **last**; keep `action_mask` the final tensor |
| Encoding terrain as a 50×50 per-cell channel | Explodes the obs, alien to the entity-token model | Per-wall **tokens** with footprint context, variable length |
| Rebuilding the blocking closure per LOS query | Needless work every shot/mask | Precompute static grid on `Battle` at construction |
| Making `config.terrain` required / non-defaulted | Breaks every existing YAML | Default `None`; empty ⇒ no-op everywhere |

---

## 10. Integration points (one-line each, for planners)

- **LOS:** `wargame.py::_make_is_blocking` (the *only* behavioural edit for blocking) → `domain/los.has_line_of_sight` unchanged.
- **State:** `Battle.terrain` + `Battle.los_blocking_grid`; `BattleView.terrain`.
- **Build:** `battle_factory.from_config` (+ `domain/terrain.build_los_blocking_grid`).
- **Config:** `WargameEnvConfig.terrain` + `TerrainPieceConfig`/`WallConfig`.
- **Obs:** `observation_builder._terrain_to_obs` → `WargameEnvObservation.terrain` → `model/common/observation` terrain tensor → `net.encode_state` terrain tokens (`terrain_embedding`, append last).
- **Render:** `renders/human.py` terrain overlay (BattleView-only).
- **Consumers that change automatically (no edits):** shooting masks, action masks, snapshot LOS — all flow through `has_line_of_sight_between_cells`.

---

## 11. Confidence & sources

| Claim | Confidence | Basis |
|---|---|---|
| `is_blocking(x,y)` is the single LOS seam; all consumers route through `has_line_of_sight_between_cells` | HIGH | `domain/los.py`, `wargame.py:196-221`, `shooting_masks.py`, `observation_builder.py:141`, `renders/human.py:440` |
| Whole-cell reuses `los.py` unchanged; cell-edge needs new traversal | HIGH | `los.py` Bresenham steps diagonally across corners (`_bresenham_line`), interior-only contract in `test_los.py` |
| Terrain tokens appended last preserve player/opponent positions & heads | HIGH | `net.py::encode_state`/`policy_from_encoded` token math; `opponent_model_embedding is None` precedent |
| `terrain=None` default keeps existing YAML valid | HIGH | `WargameEnvConfig` optional-field convention; `blocking_mask` precedent |
| Tensor-list change ripples into tests | HIGH | `pytorch-dqn.mdc` "changing the observation tuple length" note; `observations_to_tensor_batch` unpacking |
| Whole-cell accuracy loss acceptable this milestone | MEDIUM | Design judgement: movement unaffected, grid already bounds LOS precision, cover deferred |

---
*Architecture research for: v2.0 Terrain & Line-of-Sight Blocking (brownfield DDD Gymnasium wargame)*
*Researched: 2026-06-19*
