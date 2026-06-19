# Pitfalls Research — v2.0 Terrain & Line-of-Sight Blocking

**Domain:** Brownfield discrete-grid RL wargame — adding wall-based LOS blocking + terrain observation tokens to a working PPO/DQN env
**Researched:** 2026-06-19
**Confidence:** **HIGH** for codebase-specific pitfalls (grounded in `domain/los.py`, `tests/test_los.py`, `wargame.py::_make_is_blocking`, `shooting_masks.py`, `renders/human.py`); **MEDIUM** for RL training/exploitation patterns (literature + community consensus).

> **Decided design (fixed, from sibling ARCHITECTURE.md):** walls authored as thin L-shaped segments but **rasterised to WHOLE blocking cells** OR'd into the existing `is_blocking(x,y)` seam (no Bresenham changes); footprint is a **no-op area marker** (not rasterised); terrain encoded as a **new entity-token stream appended LAST** in the observation (zero tokens when no terrain → byte-identical behaviour).
>
> **Phases referenced:** **Phase A** = terrain in the simulation (config + domain + LOS seam + render). **Phase B** = terrain in the observation (obs type + tensor + both networks). B depends on A.

This document is consumed by the roadmapper, planners, and test authors. Every pitfall lists concrete warning signs, an actionable + testable prevention strategy, and the owning phase.

---

## Critical Pitfalls

### Pitfall 1: Rasterising L-walls that accidentally seal a gap or leave a 1-cell leak

**What goes wrong:**
An L-shaped wall authored as two thin segments is rasterised into blocking cells that either (a) **over-block** — the two arms meet at the corner and the rasteriser fills the diagonal-adjacent corner cell so a ray that should slip past is sealed, or (b) **under-block** — the arms don't quite touch at the corner, leaving a 1-cell diagonal "leak" the policy will fire through. Whole-cell rasterisation of a thin segment is exactly where these corner/diagonal cases bite.

**Why it happens:**
A segment from `(x0,y0)` to `(x1,y1)` rasterised with the same Bresenham (`_bresenham_line`) produces a diagonally-connected chain of cells. Diagonal chains are **not 4-connected** — a ray can pass between two diagonally-touching blocking cells because Bresenham LOS steps through the corner without entering either cell. So an L of Bresenham cells looks solid on screen but has diagonal pinholes; conversely, "thickening" to close them can seal an intended firing lane.

**How to avoid:**
- Decide and **document the wall-connectivity contract**: walls should be **4-connected** (no diagonal-only links) so they actually block. Use a rasteriser that guarantees 4-connectivity for wall cells (e.g. supercover/"thick" line, or add the orthogonal filler cell at each diagonal step) — this is a *rasterisation* choice in `domain/terrain.build_los_blocking_grid`, **not** a change to the LOS Bresenham trace.
- Author the L's corner cell explicitly as blocking (the segments share an endpoint) so the elbow is never a leak.
- Golden-board tests: a known L-wall on a small board, assert LOS is blocked along every ray that crosses the wall *and* clear along the intended gap, including the **diagonal rays through the elbow** and through any deliberate 1-cell gap.

**Warning signs:**
- Policy learns to shoot from a specific diagonal offset that "shouldn't" have LOS through a wall.
- Renderer shows a solid L but `has_line_of_sight_between_cells` returns `True` across it for diagonal endpoints.
- A wall that visually has a gap blocks shots anyway (over-block).

**Phase to address:** **Phase A** (rasterisation lives in `domain/terrain`).

---

### Pitfall 2: LOS symmetry breaks — `A→B` clear but `B→A` blocked

**What goes wrong:**
Shooting masks and resolution query LOS in one direction (shooter → target). If `has_line_of_sight(A,B)` ≠ `has_line_of_sight(B,A)`, an agent gets asymmetric firing lanes — it can be shot from a cell it cannot shoot back at. RL **will** find and exploit this; it produces bizarre, non-reciprocal positioning.

**Why it happens:**
Bresenham is **not order-independent**: `_bresenham_line(A,B)` and `_bresenham_line(B,A)` can traverse *different* interior cells when the line passes near a cell corner (the tie-break in the error term differs by direction). With sparse walls this rarely shows; with a wall sitting on exactly one of the two traces, the result flips. The existing core never guaranteed symmetry — it just wasn't observable without blockers on the divergent cells.

**How to avoid:**
- **Property test (must-have):** for a randomised set of endpoint pairs and a randomised blocking grid, assert `has_line_of_sight(A,B) == has_line_of_sight(B,A)`. Use Hypothesis. This is the single highest-value test for the whole milestone.
- If the property fails (it likely will for some configs), pick a **canonicalisation rule** and apply it in the env seam (`has_line_of_sight_between_cells`): sort the two endpoints to a canonical order before tracing, so both directions hit the identical cell set. Document it as a decision.
- Keep this in the **env/domain seam**, not in each caller, so masks, resolution, renderer, and snapshot all inherit the symmetric result.

**Warning signs:**
- Agent positions to shoot from "one-way" cells near wall edges.
- Combat log shows opponent hitting the player from a tile the player's mask marks as no-LOS to that same tile.
- Hypothesis symmetry test fails on seeded boards.

**Phase to address:** **Phase A** (symmetry is a property of the LOS seam, independent of observation).

---

### Pitfall 3: Off-by-one / interior-only contract violated by terrain

**What goes wrong:**
The current contract is "**only strictly-interior cells** (endpoints excluded) are checked for blocking" (`los.py` lines 68–77; pinned by `test_los_interior_only_blocking_ignores_endpoint_blocker`). A wall rasterised onto the **same cell as a model** (shooter or target) must NOT block that model's own shot — but if terrain is naively merged or if a future "wall on the target's tile" case is mishandled, you either silently block legitimate shots or change the endpoint semantics.

**Why it happens:**
Terrain authoring may place a wall cell on a tile a model later stands on (movement passes through walls this milestone). The interior-only rule means a wall on the endpoint is ignored by design — but a developer "fixing" a perceived bug ("the wall is right there!") may switch to inclusive endpoints, breaking every existing golden trace and the shooting semantics.

**How to avoid:**
- **Do not touch `los.py`'s interior-only loop.** Re-affirm the contract in a comment and a regression test: a wall on the shooter's or target's exact cell does not block (extend `test_los_interior_only_blocking_ignores_endpoint_blocker` with a terrain-derived grid).
- Keep terrain merging at the **grid level** (OR into the bool grid); the interior-only consumption stays in `los.py` unchanged.
- Golden traces (`test_los_golden_trace_zero_three_one`) must remain **byte-identical** after terrain code lands with no terrain configured.

**Warning signs:**
- Existing `test_los.py` golden-trace / interior-only tests start failing after terrain merge.
- Models standing in a wall tile can't shoot out (over-block at endpoint).

**Phase to address:** **Phase A**.

---

### Pitfall 4: Action mask and shooting resolution use different LOS sources of truth

**What goes wrong:**
The shooting **mask** (what the agent is allowed to select, `shooting_masks.compute_shooting_masks(has_los_fn=...)`) and the shooting **resolution** (what actually happens) must use the *same* LOS function with the *same* merged terrain grid. If one path picks up walls and the other doesn't (e.g. resolution recomputes LOS with a stale or `blocking_mask`-only predicate), the agent is offered shots it can't make, or makes shots that were masked out — a classic train/serve skew the policy will exploit or be confused by.

**Why it happens:**
`_make_is_blocking()` currently rebuilds a closure over `config.blocking_mask` on **every** query (`wargame.py:196`). When walls are added, every call site that builds its own predicate (mask builder, resolution, snapshot, renderer) must be repointed to the **single merged grid** on `Battle`. Missing one site = divergence. The architecture flags this: precompute one static `los_blocking_grid` and route all consumers through `has_line_of_sight_between_cells`.

**How to avoid:**
- Centralise: one precomputed `Battle.los_blocking_grid`; `_make_is_blocking` returns a predicate over **that** grid; all consumers call `view.has_line_of_sight_between_cells(...)`. No call site builds its own blocking predicate.
- **Consistency test:** for a board with walls, assert that every `(shooter, target)` pair where the mask permits a shot also returns LOS-clear in the resolution path, and vice-versa (the mask is exactly `{pairs : has_line_of_sight_between_cells is True} ∩ range ∩ alive`). This is the v1 pitfall-#2 "every masked action is illegal, every illegal action is masked," specialised to terrain LOS.
- Grep for direct `has_line_of_sight(` / `blocking_mask` uses outside the seam after the change; there should be none in consumers.

**Warning signs:**
- Agent's selected shoot action is rejected/no-ops at resolution.
- Mask vs resolution consistency test fails when walls present.
- Snapshot LOS disagrees with mask LOS.

**Phase to address:** **Phase A** (single seam); verification spans masks + resolution + snapshot.

---

### Pitfall 5: Observation shape drift breaks pre-terrain checkpoint loading / backward compat

**What goes wrong:**
Terrain tokens are appended as a new tensor in the observation list and a new `terrain_embedding` in both networks. If the "no terrain → zero tokens → identical behaviour" guarantee is only *incidental* (not enforced), then: (a) a no-terrain config emits a non-empty/padded terrain tensor and shifts shapes; (b) player/opponent token indices move because terrain wasn't appended **last**; (c) a pre-terrain checkpoint fails to load because `terrain_embedding` weights are missing or the state-dict keys changed even for `terrain_size == 0`.

**Why it happens:**
The transformer reads entity tokens at fixed indices (`n_prefix + p`, `n_prefix + N_p + o`) and the tensor list is unpacked positionally (`opp = xs[3]`, `mask = xs[4]`). Inserting terrain mid-list, or instantiating `terrain_embedding` when `terrain_size == 0`, perturbs either the index math or the state-dict, silently breaking the no-op promise.

**How to avoid:**
- Mirror the proven `opponent_model_embedding is None` pattern exactly: `terrain_embedding` is `None` when `terrain_size == 0` ⇒ **zero terrain tokens emitted** ⇒ token sequence is `[game | obj | players | opp]` byte-identical to today.
- Append terrain **after** opponents; keep `action_mask` the **last** tensor; update unpackers to `opp = xs[3]`, `terrain = xs[4]`, `mask = xs[5]` in one place.
- **Backward-compat tests (must-have):**
  1. No-terrain config → observation tensor shapes and dtypes identical to a captured pre-terrain golden.
  2. A **pre-terrain checkpoint loads and infers** on a no-terrain config and produces identical (or numerically-equal) logits/values to before.
  3. With-terrain config → terrain tokens present; player/opponent token indices and per-model action-head outputs unchanged for the same model positions.
- Verify **both** networks (`TransformerNetwork` and `MLPNetwork`) and the PPO networks; the MLP path flattens differently and is easy to forget.

**Warning signs:**
- `test_state.py` / `test_dqn.py` tensor-unpacking tests fail unexpectedly.
- Loading an old checkpoint raises a state-dict key mismatch on a no-terrain run.
- No-terrain training run diverges from a pre-merge baseline (reward curve shifts).

**Phase to address:** **Phase B** (owns the tensor/network contract and checkpoint compat).

---

### Pitfall 6: Reward exploitation via terrain ("hide-and-plink" / wall camping)

**What goes wrong:**
Once walls block LOS, the agent discovers it can sit in a spot where it has LOS to the enemy but the enemy (especially a scripted opponent that doesn't reason about LOS) does not, or it parks behind a wall and never advances on objectives. Behaviour optimises the shaped combat signal while abandoning the real mission — the v1 pitfall #1 ("reward shaping hijacks the objective") sharpened by the new LOS asymmetry terrain creates.

**Why it happens:**
Terrain introduces strong, easily-found local optima (degenerate safe cells). Scripted opponents may have no LOS-aware behaviour, so "one-way" firing positions are pure free reward. Dense combat shaping + a passive opponent = wall-camping basin.

**How to avoid:**
- Keep a **sparse anchor** on true outcomes (objective control / VP / terminal win) in any curriculum phase where terrain is enabled, so camping that ignores objectives is penalised by opportunity cost.
- Evaluate against opponents that **pressure** the camper (scripted advance-to-objective at minimum); add a LOS-aware scripted shooter as a stretch.
- Log decomposed metrics (objective-control time, distance-to-objective, damage dealt vs taken) so camping shows up as high damage / low objective progress.
- Terrain is **observation + LOS only this milestone** — resist adding a "cover/terrain reward" calculator now (deferred). If a terrain curriculum phase is added later, tie it back to objectives (v1 pitfall #10).

**Warning signs:**
- Agent clusters on the same near-wall cells every episode; objective control drops.
- Combat reward up, win rate / VP flat or down once terrain is enabled.
- Behaviour looks degenerate in renderings (never crosses the wall line).

**Phase to address:** Awareness in **Phase A/B**; mitigation is a **curriculum/eval** concern (out of strict milestone scope — flag for the combat-reward phase / roadmap).

---

### Pitfall 7: Distribution shift when terrain is switched on for an existing policy

**What goes wrong:**
Enabling terrain mid-project (new config field, new tokens) on a policy trained without terrain produces a sudden input-distribution and dynamics shift: previously-clear firing lanes vanish, observation gains tokens the network has never seen, and pre-terrain tactics (clump in open ground) become bad. Training appears to "regress."

**Why it happens:**
The network's terrain embedding is freshly initialised; the policy has no learned response to walls; the value function is mis-calibrated for blocked lanes. Turning terrain on as a config flag mid-run, rather than as a deliberate curriculum step, conflates two changes.

**How to avoid:**
- Treat terrain-enabled as a **distinct training configuration / curriculum stage**, not an in-place flag flip on a running checkpoint. Either train fresh or fine-tune with terrain as an explicit phase.
- Provide at least one **terrain example config** in `examples/env_config/` and one no-terrain config; keep them as separate runs/Wandb groups so the shift is visible, not silent.
- Confirm the **no-terrain path is byte-identical** (Pitfall 5) so any regression is attributable to terrain itself, not to the merge.

**Warning signs:**
- Reward/winrate drop exactly when a terrain config is introduced.
- High value-function loss / entropy spike at terrain enablement.

**Phase to address:** **Phase B** (provide configs + note curriculum implications); roadmap owns the curriculum phase.

---

### Pitfall 8: Renderer / debug LOS diverges from domain truth

**What goes wrong:**
The Pygame debug overlay (`_draw_debug_los_line` uses `iter_los_cells`, which traces but does **not** apply blocking) draws the ray polyline without indicating *which* cell blocks, and the wall overlay is drawn separately. If the renderer draws walls from a different source than `Battle.los_blocking_grid` (e.g. straight from `config.terrain` segments, or only footprints), the human sees walls in one place but shots are blocked by cells elsewhere — debugging LOS becomes misleading and masks real geometry bugs.

**Why it happens:**
`iter_los_cells` and `has_line_of_sight` are separate calls; the renderer already only draws the trace, not the blocking verdict. Adding a wall overlay that renders authored segments (thin lines) while LOS uses rasterised whole cells creates a **visual vs. logical mismatch** — the screen shows a thin wall but a whole cell blocks.

**How to avoid:**
- Render the wall overlay from the **same rasterised `los_blocking_grid`** the LOS uses (draw blocked cells), not (only) from the authored thin segments. Optionally also show the thin authored segment for authoring clarity, but the blocking truth must be the rasterised cells.
- Colour the debug LOS polyline by the **actual** `has_line_of_sight_between_cells` verdict (e.g. red when blocked, green when clear) and mark the first interior blocking cell, so the renderer can never claim clear while resolution blocks.
- All render data flows through `BattleView` (`view.terrain` + the grid), never a duplicate ray/rasterisation in Pygame.

**Warning signs:**
- A shot resolves as blocked but the debug line is drawn as clear (or vice-versa).
- Walls visible on screen don't line up with the cells that actually block shots.

**Phase to address:** **Phase A** (renderer overlay rides with the simulation slice).

---

### Pitfall 9: Terrain observation dishonesty (perfect-information leak or stale tokens)

**What goes wrong:**
Terrain is static and fully known, so it's tempting to dump everything — but two honesty traps exist: (a) encoding terrain in a way that **leaks** more than a player would have (e.g. precomputed per-opponent LOS-visibility flags that reveal hidden enemy positions through a terrain side-channel — violates `PROJECT.md` observation honesty); (b) terrain tokens whose **normalisation/coordinates drift** from the actual grid the LOS uses, so the agent "sees" a wall in a place that doesn't match where it blocks.

**Why it happens:**
Convenience features (e.g. "is this enemy visible" booleans) bundled into terrain tokens cross the line from terrain-geometry into privileged combat info. Separately, encoding authored segment endpoints (thin) while LOS blocks rasterised cells (whole) means the observed geometry ≠ effective geometry.

**How to avoid:**
- Terrain tokens carry **geometry only** (wall segment endpoints + footprint bbox, normalised), no derived per-enemy visibility. Static terrain is legitimately fully observable; LOS-derived enemy visibility is not part of terrain encoding.
- Make the observed wall geometry **consistent with the effective blocking** — either encode in the same normalised cell space used for blocking, or document the segment→cell mapping so the agent's representation matches reality.
- Test: observed terrain token count == number of walls; coordinates round-trip to the blocking cells they represent.

**Warning signs:**
- Agent reacts to enemies it shouldn't be able to see when terrain present.
- Observation encodes a wall at coordinates that don't correspond to the blocking cells.

**Phase to address:** **Phase B** (observation honesty), with geometry consistency depending on **Phase A** rasterisation.

---

## Technical Debt Patterns

| Shortcut | Immediate Benefit | Long-term Cost | When Acceptable |
|----------|-------------------|----------------|-----------------|
| Rasterise walls with plain Bresenham (allow diagonal-only connectivity) | Trivial to implement; reuse `_bresenham_line` | Diagonal pinholes the policy exploits (Pitfall 1); silent LOS leaks | **Never** — must guarantee 4-connectivity or document gaps as intentional |
| Rebuild `is_blocking` closure per query (keep current code) | No refactor | Per-shot allocation; multiple call sites drift (Pitfall 4) | MVP spike only; precompute `los_blocking_grid` before merge |
| Render walls from authored segments, not rasterised grid | Prettier thin walls | Visual/logical mismatch (Pitfall 8); hides geometry bugs | Only if also overlaying the rasterised blocking truth |
| Encode terrain as a full per-cell channel grid | "Complete" info | Obs explosion, alien to entity-token model, shape fragility | Never this milestone — use per-wall tokens |
| Flip a `terrain: true` flag on a running checkpoint | Fast experiment | Conflates distribution shift with bugs (Pitfall 7) | Throwaway experiments only |

## Integration Gotchas

| Integration | Common Mistake | Correct Approach |
|-------------|----------------|------------------|
| `domain/los.py` Bresenham core | Editing the trace/endpoint loop to "support walls" | Leave it untouched; merge terrain at the grid level behind `is_blocking` |
| `shooting_masks` ↔ resolution | Each recomputes LOS with its own predicate | Both call `has_line_of_sight_between_cells` over the one merged grid (Pitfall 4) |
| `GameStateSnapshot` LOS queries (v9.0) | Snapshot uses a path that doesn't see walls | Route snapshot LOS through the same seam; add a snapshot-with-terrain test |
| `MLPNetwork` (legacy) | Only updating `TransformerNetwork` for terrain tokens | Update both nets + PPO networks; flatten path differs |
| Tensor list unpack (`xs[3]`, `xs[4]`) | Inserting terrain mid-list | Append terrain at `xs[4]`, keep mask last (`xs[5]`); fix all unpackers once |
| Wandb run comparison | No-terrain and terrain runs in one group, indistinguishable | Separate `--wandb-group` / suffix per terrain config |

## Performance Traps

| Trap | Symptoms | Prevention | When It Breaks |
|------|----------|------------|----------------|
| Rebuilding blocking closure / re-rasterising per LOS query | Step time grows with shots-per-turn | Precompute static `los_blocking_grid` once at `Battle` construction | Many models × many targets per step |
| One terrain token **per cell** instead of per wall | Sequence length explodes; quadratic attention | Per-wall tokens (variable length) | Large boards / dense terrain |
| Many wall pieces → long token stream | Attention cost up, throughput down | Cap/document max walls per config; per-wall tokens stay small | Future dense-terrain milestone |

## Security Mistakes

| Mistake | Risk | Prevention |
|---------|------|------------|
| Observation leaks LOS-derived enemy visibility via terrain tokens | Perfect-information cheat; violates `PROJECT.md` honesty | Terrain tokens carry geometry only (Pitfall 9) |
| (N/A web security) | — | This is a local training pipeline; no network attack surface |

## UX Pitfalls

(Operator/researcher experience, not end-user.)

| Pitfall | User Impact | Better Approach |
|---------|-------------|-----------------|
| Debug LOS line drawn without blocking verdict | Researcher can't tell why a shot was blocked | Colour line by `has_line_of_sight_between_cells`; mark first blocking cell (Pitfall 8) |
| No example terrain config shipped | Hard to reproduce / demo terrain | Add a documented `examples/env_config/*terrain*.yaml` |
| Footprint silently has no effect | "Why doesn't my footprint block?" confusion | Document footprint = no-op area marker this milestone (it's authoring/observation only) |

## "Looks Done But Isn't" Checklist

- [ ] **Wall blocks LOS:** Verify it blocks **both directions** (A→B and B→A) — symmetry test, not just one ray (Pitfall 2).
- [ ] **L-wall corner:** Verify no **diagonal pinhole** at the elbow and no accidental sealing of an intended gap (Pitfall 1).
- [ ] **No-terrain config:** Verify observation tensors are **byte-identical** to a pre-merge golden and a **pre-terrain checkpoint still loads & infers** (Pitfall 5).
- [ ] **Mask == resolution:** Verify shooting mask and resolution agree on every pair with walls present (Pitfall 4).
- [ ] **Interior-only preserved:** Verify a wall on shooter/target cell does **not** block; existing golden traces unchanged (Pitfall 3).
- [ ] **Renderer truth:** Verify walls drawn from the rasterised grid match where shots actually block (Pitfall 8).
- [ ] **Movement unaffected:** Verify a model can still move **through** a wall cell (LOS-only this milestone).
- [ ] **Both networks + PPO:** Verify Transformer **and** MLP **and** PPO networks handle terrain tokens and `terrain_size == 0`.
- [ ] **Snapshot LOS:** Verify `to_snapshot()` LOS reflects walls (v9.0 integration).

## Recovery Strategies

| Pitfall | Recovery Cost | Recovery Steps |
|---------|---------------|----------------|
| LOS asymmetry shipped (Pitfall 2) | LOW | Add canonical endpoint ordering in the seam; re-run; no retrain needed if caught before training |
| Diagonal pinhole / over-seal (Pitfall 1) | LOW–MEDIUM | Fix rasteriser connectivity; re-rasterise (startup-only); retrain if a policy already exploited it |
| Obs shape drift / checkpoint break (Pitfall 5) | MEDIUM | Restore append-last + `None`-embedding invariant; old checkpoints loadable again; retrain only terrain runs |
| Policy learned to exploit LOS bug (Pitfall 1/2) | HIGH | Checkpoints invalid for affected behaviour; fix geometry, add regression test, retrain |
| Wall-camping policy (Pitfall 6) | MEDIUM | Add sparse objective anchor + pressuring opponent in curriculum; retrain affected phase |

## Pitfall-to-Phase Mapping

| Pitfall | Prevention Phase | Verification |
|---------|------------------|--------------|
| 1. L-wall corner pinhole / over-seal | **Phase A** | Golden-board test: L-wall blocks all crossing rays incl. diagonal elbow; intended gap stays clear |
| 2. LOS A→B vs B→A asymmetry | **Phase A** | Hypothesis property test: `los(A,B)==los(B,A)` over random grids/endpoints |
| 3. Interior-only contract violated | **Phase A** | Regression: wall on endpoint doesn't block; existing golden traces byte-identical |
| 4. Mask vs resolution LOS divergence | **Phase A** | Consistency test: mask-permitted pairs == LOS-clear∩range∩alive with walls |
| 5. Obs shape drift / checkpoint break | **Phase B** | No-terrain tensor golden identical; pre-terrain checkpoint loads & infers; indices unchanged |
| 6. Reward exploitation (camping) | Curriculum/eval (flag) | Decomposed metrics; eval vs pressuring opponent; objective-control not regressed |
| 7. Distribution shift on enable | **Phase B** + roadmap | Terrain as explicit config/curriculum stage; no-terrain baseline identical |
| 8. Renderer / debug LOS divergence | **Phase A** | Debug line colour == LOS verdict; walls drawn from rasterised grid |
| 9. Terrain observation dishonesty | **Phase B** | Tokens geometry-only; token count == wall count; coords round-trip to blocking cells |

## Sources

- Codebase (HIGH): `wargame_rl/wargame/envs/domain/los.py`, `tests/test_los.py`, `wargame.py::_make_is_blocking` / `has_line_of_sight_between_cells`, `env_components/shooting_masks.py`, `renders/human.py::_draw_debug_los_line`, `.planning/research/ARCHITECTURE.md`.
- Prior research (HIGH for repo): `.planning/research/v1-PITFALLS.md` (reward hijack #1, action/mask consistency #2, LOS exploits #3, curriculum #4, obs-as-noise #14).
- Grid LOS geometry (MEDIUM): Bresenham diagonal-connectivity / supercover behaviour and direction-dependence are well-documented properties of integer line rasterisation (Wikipedia "Bresenham's line algorithm"; standard grid-LOS / FOV references).
- RL reward-shaping & distribution-shift patterns (MEDIUM): strategy-game RL community consensus, consistent with v1-PITFALLS sources.

---
*Pitfalls research for: v2.0 Terrain & Line-of-Sight Blocking (wall LOS rasterisation + terrain observation tokens)*
*Researched: 2026-06-19*
