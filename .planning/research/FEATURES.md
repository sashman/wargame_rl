# Feature Research — v2.0 Terrain & Line-of-Sight Blocking

**Domain:** Terrain that blocks line of sight (footprint + thin walls) in a discrete-grid RL wargame env
**Researched:** 2026-06-19
**Confidence:** HIGH for tabletop mapping and existing-system integration (verified against code + rules ref + Warhammer Community); MEDIUM for the wall-representation decision (open design question, see Dependency Notes)

> **Scope guard (read first).** This milestone is *line of sight only*. Terrain = a **footprint** (large rectangle area marker, no gameplay effect this milestone) + **thin walls** (segments inside the footprint, usually L-shaped) that **block LOS only**. Movement is unaffected. Everything in the "Anti-Features / Out of Scope This Milestone" section is explicitly deferred. The downstream requirements author should keep proposed scope to the Table Stakes set.

---

## Tabletop grounding (how "ruins / area terrain with walls" actually work)

Verified against the project rules reference (`docs/tabletop-rules-reference.md`) and Warhammer Community / Goonhammer (40k 10th edition, current rules):

- **Ruins are defined by a footprint.** The footprint is the piece's area on the table (the base outline, or the vertical projection of the upper floors when there is no base).
- **In real 40k 10e the footprint itself blocks LOS** ("Ruins completely block visibility of all models through their footprint, regardless of windows"). Models outside can shoot *in*, models wholly inside can shoot *out*, and `TOWERING`/`AIRCRAFT` ignore the block.
- **Walls** in tabletop are primarily a *movement* + *cover* construct (infantry walk through walls; non-infantry are stopped by walls >2"). They are not, by themselves, the canonical LOS blocker — the footprint is.

**Design divergence to flag for the requirements author (HIGH importance):**
This milestone deliberately models **walls (segments) as the LOS blocker** and treats the **footprint as a no-op marker**. That is a *simplification/inversion* of 40k 10e (where the footprint obscures). It is a reasonable, intentional choice — segment-level walls give finer tactical geometry (you can be adjacent to a wall and still shoot down a corridor) and defer the "wholly-within / shoot-in-shoot-out" complexity. But requirements should state plainly that v2.0 terrain is **"walls block the ray"**, *not* "footprint obscures the area," so nobody expects 40k-faithful ruin obscuring yet. The footprint exists now only to (a) author/group walls, (b) appear in the observation, and (c) be the future hook for cover/dense-visibility.

---

## Feature Landscape

### Table Stakes (Required for a credible "walls block LOS + terrain in observation" milestone)

| Feature | Why Expected | Complexity | Notes / Dependency |
|---------|--------------|------------|--------------------|
| **Terrain config schema: footprint rect + wall segments** | Without authored terrain there is nothing to block; YAML is the only authoring path | LOW–MED | New Pydantic models in `envs/types/config.py` (e.g. `TerrainPieceConfig` = footprint `(x_min,y_min,x_max,y_max)` + `walls: list[WallSegmentConfig]`). Mirror existing `ObjectiveConfig`/`ModelConfig` style; validate in-bounds + walls-inside-footprint at construction (KISS: validate at init). Defaults to `None`/`[]`. |
| **Walls compiled into the LOS blocking input** | The single job of the milestone: a wall on the ray blocks the shot | MED | Extends `domain/los.py`. Today `WargameEnv._make_is_blocking()` builds an `is_blocking(x,y)` predicate from `config.blocking_mask`. Walls must feed the same single LOS service so there is **one** Bresenham path (renderer + shooting masks already share it). |
| **Wall grid representation decision (whole-cell vs cell-edge)** | Determines the LOS algorithm and the obs encoding; it is the milestone's central open question | MED (whole-cell) / HIGH (cell-edge) | See Dependency Notes. **Recommend whole-cell for v2.0** (walls → blocking cells, reuse existing interior-cell predicate, near-zero new LOS code). Cell-edge is more faithful to "thin walls" but needs a new edge-crossing LOS test. |
| **Backward-compatible no-op default** | Project constraint: existing YAML configs must behave exactly as today | LOW | `terrain: None`/absent → empty wall set → `is_blocking` returns `False` everywhere (same as current `blocking_mask=None`). Add backward-compat tests asserting identical obs/behaviour. |
| **Terrain encoded in observation space** | Stated milestone goal — agent must see/reason about terrain (observation honesty: terrain is public) | MED | Cleanest fit for the entity-centric transformer: terrain as a **new token type** (footprints and/or wall segments as tokens with their own `nn.Linear` embedding), exactly like `objectives`/`opponents`. Requires the full obs pipeline touch: `WargameEnvObservation` field → `observation_builder.build_observation` → `_observation_to_numpy` → `observation_to_tensor`/`observations_to_tensor_batch` → **both** networks' `forward` (Transformer needs the new embedding + token-order update; MLP just flattens). Fixed token count per episode (no shape changes mid-episode). |
| **Renderer draws footprints + walls** | Debuggability — humans/video must see what blocks LOS, and the existing `L` LOS debug overlay must agree with the new blockers | LOW–MED | `renders/human.py` already has gridlines + `iter_los_cells` LOS overlay. Add footprint rectangles (translucent fill) + wall segments (thick lines). Reuse existing scale/canvas transforms. |
| **Tests: LOS-through-wall, no-terrain regression, obs shape stability** | Milestone is a correctness change to a shared service; needs regression coverage | LOW–MED | Extend `tests/test_los.py` (wall blocks ray; open lane does not) + obs-shape/backward-compat tests. Deterministic, no randomness. |

### Differentiators (Optional polish — include only if cheap; not required to ship the milestone)

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| **Cell-edge ("thin") wall segments** | Faithful "thin L-shaped wall" geometry; models can sit either side of a wall in adjacent cells and still see down open lanes | HIGH | The more "correct" rendering of walls, but requires a grid-edge LOS algorithm (supercover/edge-crossing) rather than the current interior-cell predicate. A differentiator, not table stakes — recommend deferring unless the open question resolves toward edges. |
| **Footprint membership in observation (e.g. "model is inside terrain" flag)** | Cheap precursor to future cover / dense-visibility; gives the agent a richer terrain signal | LOW | Footprints are already authored; adding an "inside-footprint" per-model bit is a small obs addition. Pure no-op on gameplay this milestone. |
| **Terrain in the canonical `GameStateSnapshot`** | v9.0 shipped structured state; terrain in the snapshot keeps LLM/replay representations complete | MED | Extends `envs/state/snapshot.py` + JSON schema. Nice for cross-consumer consistency, but not needed for RL training to work. |
| **Wall LOS debug overlay toggle** | Show exactly which cells/edges block on the `L` debug view | LOW | Small renderer add; aids verification. |
| **Multiple terrain pieces per board** | Several ruins for richer geometry | LOW | Falls out naturally if the schema is `list[TerrainPieceConfig]`; not a separate feature, just don't cap it at one. |

### Anti-Features (Out of scope this milestone — defer, do NOT pull in)

These are all explicitly **DEFERRED** per `PROJECT.md`. Listed so the requirements author keeps them out of v2.0 scope.

| Feature | Why It Gets Requested | Why Out of Scope Now | Defer To |
|---------|----------------------|----------------------|----------|
| **Cover save bonus (+1 Sv vs ranged)** | "Terrain should protect units" | Touches shooting *resolution* (`domain/shooting.py`), not LOS; new reward/obs surface | Later terrain milestone |
| **Dense terrain visibility (woods / wholly-inside never fully visible)** | "Real ruins hide units inside" | Requires footprint-membership + partial-visibility model; this milestone makes footprint a no-op | Later terrain milestone |
| **Footprint-obscures-LOS (40k-faithful ruin)** | "That's how real ruins work" | Milestone intentionally models *walls* as the blocker; area-obscuring is a different LOS model + shoot-in/shoot-out rules | Later terrain milestone |
| **Difficult ground (movement penalty)** | "Terrain should slow you down" | Movement is explicitly unaffected this milestone | Later terrain milestone |
| **Impassable / blocking movement** | "Walls should stop models" | Movement unaffected; pathing/placement collision is a large change | Later terrain milestone |
| **Elevation / height / plunging fire** | "Ruins have floors" | 3D adds a whole dimension to grid + obs; far out of scope | Later terrain milestone |
| **Board templates / procedural terrain placement** | "Need variety for generalisation" | Authoring/generation system; orthogonal to the LOS mechanic | Later terrain milestone |
| **Variable base sizes** | "Tanks are bigger than infantry" | Flagged as its own research spike; affects movement/coherency/LOS broadly | Later terrain milestone |
| **Per-cell full-grid terrain channel in obs** | "Just feed a 2D map to a CNN" | Conflicts with the entity-centric transformer (variable board sizes, token model); blows up input dim | Use terrain *tokens* instead |

---

## Feature Dependencies

```
Terrain config schema (footprint + walls)
    └──requires──> WargameEnvConfig validation (envs/types/config.py)

Walls block LOS
    └──requires──> Terrain config schema
    └──requires──> Wall representation decision (whole-cell vs cell-edge)
    └──requires──> domain/los.py is_blocking predicate  [EXISTS]
                       └──shared by──> shooting masks + renderer LOS overlay  [EXISTS]

Terrain in observation
    └──requires──> Terrain config schema
    └──enhances──> agent's tactical reasoning (the milestone goal)
    └──touches──> full obs pipeline + BOTH networks (Transformer embedding + MLP flatten)

Renderer footprints/walls ──enhances──> debuggability (agrees with `L` LOS overlay)

No-terrain default ──conflicts-if-broken──> every existing YAML config (must stay no-op)
```

### Dependency Notes

- **Walls block LOS requires the wall-representation decision.** This is the milestone's pivotal choice and should be resolved in planning/discussion before implementation:
  - **Whole-cell blocking (RECOMMENDED for v2.0):** rasterise each wall segment into a set of blocking grid cells and OR them into the existing `is_blocking(x,y)` predicate (same mechanism as `config.blocking_mask`). **Pros:** ~zero new LOS code, reuses the proven interior-cell Bresenham, trivially consistent with renderer + shooting masks, easiest obs encoding (cells or segment tokens). **Cons:** a "thin" wall occupies whole cells, so two models in cells on opposite sides of a 1-cell wall cannot see each other even point-blank; corners are chunky.
  - **Cell-edge segments (DEFER / differentiator):** walls live on edges between cells; LOS is blocked only when the Bresenham/supercover ray *crosses* a blocked edge. **Pros:** faithful thin walls, better corner/corridor behaviour. **Cons:** requires a new edge-aware LOS routine (the current predicate is cell-keyed, endpoints excluded), more complex obs encoding, and more test surface. Recommend only if the open question resolves toward edges and there's appetite for the extra LOS work.
- **Terrain in observation requires the config schema and touches both networks.** Per the pytorch rule, adding an entity type means: obs dataclass field → builder → numpy tuple → `observation_to_tensor` (+ batch) → **both** `DQN`/`Transformer` forwards (Transformer needs a dedicated terrain embedding and updated token ordering; player tokens must remain extractable for per-model action heads) → fix unpacking in `test_state.py`/`test_dqn.py`. Encode terrain as **tokens**, not a dense grid, to fit the variable-size entity transformer.
- **No-terrain default conflicts with everything if broken.** The single hard backward-compat rule: absent/empty terrain must reproduce today's observation tensor shapes and LOS behaviour. If terrain adds an always-present token slot, that changes obs shape for *all* configs — acceptable only if it's a fixed, documented change covered by updated tests (preferred: a fixed minimum terrain token count, zero-padded when no terrain, mirroring how opponents are handled).

---

## MVP Definition

### Launch With (v2.0 — table stakes)

- [ ] **Terrain config schema** — `TerrainPieceConfig` (footprint rect + `walls`) in `WargameEnvConfig`, validated at construction, default no-op
- [ ] **Walls compiled into LOS** — walls feed the single `domain/los.py` blocking predicate (recommend whole-cell rasterisation OR-ed with existing `blocking_mask`)
- [ ] **Terrain in observation** — terrain tokens through the full obs pipeline + both networks, fixed token count, no mid-episode shape change
- [ ] **Renderer footprints + walls** — drawn on the Pygame canvas; `L` LOS overlay agrees with the new blockers
- [ ] **Backward-compatible default** — no terrain ⇒ identical obs + LOS to today (regression test)
- [ ] **Tests** — wall-blocks-ray, open-lane-passes, no-terrain regression, obs-shape stability (deterministic)

### Add After Validation (later terrain milestone)

- [ ] **Cell-edge thin walls** — trigger: whole-cell corners/adjacency prove too coarse for desired tactics
- [ ] **Footprint-membership obs flag** — trigger: starting cover or dense-visibility work
- [ ] **Terrain in `GameStateSnapshot`** — trigger: LLM/replay consumers need terrain

### Future Consideration (deferred terrain features)

- [ ] Cover save bonus · dense visibility · difficult ground · impassable terrain · elevation/plunging fire · board templates/procedural placement · variable base sizes — all per `PROJECT.md` deferral list

## Feature Prioritization Matrix

| Feature | User Value | Implementation Cost | Priority |
|---------|------------|---------------------|----------|
| Terrain config schema (footprint + walls) | HIGH | LOW–MED | P1 |
| Walls block LOS (whole-cell) | HIGH | MED | P1 |
| Terrain in observation (tokens) | HIGH | MED | P1 |
| Backward-compatible no-op default | HIGH | LOW | P1 |
| Renderer footprints + walls | MED | LOW–MED | P1 |
| Regression + LOS tests | HIGH | LOW–MED | P1 |
| Cell-edge thin walls | MED | HIGH | P3 |
| Footprint-membership obs flag | LOW–MED | LOW | P2 |
| Terrain in GameStateSnapshot | LOW | MED | P3 |

**Priority key:** P1 must-have for the milestone · P2 add if cheap · P3 defer.

## Existing-System Integration Map (for the requirements author)

| Existing system | File | What v2.0 touches |
|-----------------|------|-------------------|
| LOS service | `wargame_rl/wargame/envs/domain/los.py` | Walls feed `is_blocking`; keep single Bresenham path |
| LOS wiring | `wargame_rl/wargame/envs/wargame.py` (`_make_is_blocking`, `has_line_of_sight_between_cells`) | Combine walls with existing `blocking_mask` |
| Config | `wargame_rl/wargame/envs/types/config.py` | New `TerrainPieceConfig`/`WallSegmentConfig`; validate at init; default no-op |
| Observation types | `wargame_rl/wargame/envs/types/env_observation.py` | New terrain field |
| Obs builder | `wargame_rl/wargame/envs/env_components/observation_builder.py` | Emit terrain into observation |
| Tensor pipeline | `wargame_rl/wargame/model/common/observation.py` | New terrain tensor in `_observation_to_numpy` + `observation_to_tensor`(+batch) |
| Networks | `wargame_rl/wargame/model/net.py` | Terrain embedding + token order (Transformer); flatten (MLP) |
| Renderer | `wargame_rl/wargame/envs/renders/human.py` | Draw footprints + walls; LOS debug agreement |
| Shooting masks | `wargame_rl/wargame/envs/env_components/shooting_masks.py` | No change — already consumes `has_line_of_sight_between_cells` |
| Snapshot (optional) | `wargame_rl/wargame/envs/state/snapshot.py` | Differentiator only |

## Sources

- `/home/sash/Workspace/wargame_rl/.planning/PROJECT.md` — milestone scope, deferral list, constraints — **HIGH**
- `/home/sash/Workspace/wargame_rl/docs/tabletop-rules-reference.md` — terrain table (Ruins block LOS; footprint; infantry through walls) — **HIGH**
- Existing code: `domain/los.py`, `types/config.py` (`blocking_mask`), `env_components/observation_builder.py`, `model/common/observation.py`, `model/net.py`, `renders/human.py` — **HIGH** (read directly)
- `.cursor/rules/pytorch-dqn.mdc`, `gymnasium-env.mdc` — obs/entity extension checklist — **HIGH**
- Warhammer Community "Simple Terrain", Goonhammer Ruleshammer Terrain Guide, tuser.tv 10e ruins/visibility — current 40k 10e ruin/footprint/wall LOS behaviour — **MEDIUM–HIGH** (multiple sources agree; used to flag the footprint-vs-wall design divergence)
- `.planning/research/v1-FEATURES.md` — prior terrain categorisation reused — **HIGH**

---
*Feature research for: v2.0 Terrain & Line-of-Sight Blocking (footprint + thin walls)*
*Researched: 2026-06-19*
