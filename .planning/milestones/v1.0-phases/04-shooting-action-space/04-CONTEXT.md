# Phase 4: Shooting Action Space - Context

**Gathered:** 2026-04-05
**Status:** Ready for planning

<domain>
## Phase Boundary

Extend the flat action space with a `"shooting"` slice in `ActionRegistry`, phase-gated to `BattlePhase.shooting`. Each player model selects one opponent model target (or stay) per shooting step. Action masks combine phase validity, LOS, weapon range, and target alive status. Shooting resolution (hit/wound/save/damage) is **Phase 5** — this phase only enables target selection and validity masking.

Phase 4 does **not** introduce damage, weapon stat resolution, multi-weapon targeting, or precomputed probability matrices. It adds the minimal config (stub weapon profiles with range) and action infrastructure needed for the agent to make targeting decisions.

</domain>

<decisions>
## Implementation Decisions

### Step Semantics
- **D-01:** One `env.step()` per phase. Enabling shooting means the agent receives a separate step during `BattlePhase.shooting` (movement slice masked, shooting slice valid) and a separate step during `BattlePhase.movement` (shooting masked, movement valid). Two agent decisions per game turn when shooting is active. Reuses the existing `GameClock` / `ActionRegistry` / phase-gated mask infrastructure unchanged.
- **D-02:** Shooting phase activation is **explicit YAML opt-in**. The user must configure `skip_phases` to exclude `BattlePhase.shooting`. Existing configs that don't mention shooting keep current movement-only behaviour. No automatic detection.

### Shooting Target Encoding
- **D-03:** Add a `"shooting"` slice to `ActionRegistry` with `number_of_opponent_models` indices, valid only in `frozenset({BattlePhase.shooting})`. Fixed at init, never resized mid-episode. Dead/out-of-range/no-LOS targets are masked to False at each step.
- **D-04:** Shooting target index K in the action slice corresponds to opponent model slot K in the observation. **Positional alignment between observation order and action index must be explicitly maintained** — if observation order ever changes, the shooting slice must follow. This alignment is what allows the transformer to learn the correspondence between "opponent features at position K" and "action to shoot opponent K."
- **D-05:** `STAY_ACTION` (index 0) remains valid in all phases including shooting. A model with no valid targets (all dead, out of range, no LOS) or that chooses not to shoot selects stay.

### Weapon Configuration
- **D-06:** Introduce `WeaponProfile` (Pydantic model) with a `range` field (integer, grid cells). Other weapon stats (`attacks`, `ballistic_skill`, `strength`, `ap`, `damage`) are **not** populated in Phase 4 — Phase 5 adds them.
- **D-07:** Add `weapons: list[WeaponProfile]` to `ModelConfig`. Defaults to empty list for backward compatibility. Models with no weapons have all shooting targets masked out — they effectively cannot shoot.
- **D-08:** For Phase 4 masking, a target is "in range" if **any** of the model's weapons can reach it (max range across all weapons ≥ distance to target). This extends naturally to per-weapon range checks when multi-weapon targeting is added.

### Action Masking
- **D-09:** Shooting mask for model M targeting opponent K is the AND of: (a) current phase is `BattlePhase.shooting`, (b) model M is alive, (c) opponent K is alive, (d) `has_line_of_sight` from M's position to K's position, (e) distance(M, K) ≤ max weapon range of M.
- **D-10:** Dead player models get `STAY_ACTION` only (existing Phase 2 behaviour, unchanged).

### Action Dispatch
- **D-11:** Stateless dispatch. When a model selects a shooting target during the shooting step, the env decodes the action int to identify the target index but does **not** mutate domain state. Phase 5 re-derives the target from the action passed to `step()` and applies resolution. No `model.shooting_target` or similar domain field in Phase 4.
- **D-12:** `ActionHandler.apply` (or a parallel shooting handler) must be phase-aware: during shooting phase, the action int maps to a target index rather than a movement displacement. The handler recognises shooting-slice indices and handles them appropriately (no-op in Phase 4, resolution in Phase 5).

### Claude's Discretion
- Whether to extend `ActionHandler` with phase-aware dispatch or introduce a separate `ShootingHandler` alongside it
- Exact distance metric for range check (Euclidean vs Chebyshev vs Manhattan on the grid) — should match existing `DistanceCache` convention
- Whether `WeaponProfile` lives in `types/config.py` alongside `ModelConfig` or in a new `types/weapons.py`
- Internal structure of the shooting mask computation (vectorized numpy vs per-model loop)

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Requirements and roadmap
- `.planning/REQUIREMENTS.md` — ACT-01, ACT-02, ACT-03, ACT-04, LOS-03, SHOT-03
- `.planning/ROADMAP.md` — Phase 4 goal, success criteria, dependencies (Phases 1-3)

### Prior phase context
- `.planning/phases/01-wounds-elimination/01-CONTEXT.md` — Flag-based elimination, `is_alive`, alive guards on all iteration (D-03 through D-05)
- `.planning/phases/02-alive-aware-observation/02-CONTEXT.md` — Dead model masking (`STAY_ACTION` only), alive/wound features in observation tensor (D-04, D-05)
- `.planning/phases/03-line-of-sight-service/03-CONTEXT.md` — `domain/los.py` API, `has_line_of_sight`, injectable `is_blocking`, interior-cell-only blocking (D-04 through D-12)

### Architecture and domain patterns
- `docs/ddd-envs.md` — DDD structure, `BattleView`, extension points for new actions/phases, dependency direction
- `docs/tabletop-rules-reference.md` — Shooting phase rules, eligible targets, range, LOS requirements, attack sequence overview

### Implementation touchpoints
- `wargame_rl/wargame/envs/env_components/actions.py` — `ActionRegistry`, `ActionHandler`, `STAY_ACTION`, `ActionSlice`, `get_model_action_masks`
- `wargame_rl/wargame/envs/wargame.py` — `WargameEnv.step()`, `_apply_player_action`, `skip_phases`, `run_after_player_action`, LOS helper wiring
- `wargame_rl/wargame/envs/domain/los.py` — `has_line_of_sight`, `iter_los_cells`
- `wargame_rl/wargame/envs/domain/entities.py` — `WargameModel`, `is_alive`, `alive_mask_for`, `stats`
- `wargame_rl/wargame/envs/env_components/observation_builder.py` — `build_observation`, `battle_phase_index`, action mask wiring from `ActionRegistry`
- `wargame_rl/wargame/envs/types/config.py` — `ModelConfig`, `WargameEnvConfig`, `skip_phases`, `blocking_mask`
- `wargame_rl/wargame/envs/types/env_action.py` — `WargameEnvAction`
- `wargame_rl/wargame/envs/types/game_timing.py` — `BattlePhase`, `BATTLE_PHASE_ORDER`
- `wargame_rl/wargame/envs/domain/game_clock.py` — `GameClock`, phase advancement
- `wargame_rl/wargame/model/net.py` — Transformer network, action output shape
- `wargame_rl/wargame/model/common/observation.py` — Tensor pipeline, `apply_action_mask`

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `ActionRegistry` with slice-based registration and `valid_phases` — shooting slice follows the identical pattern as `"stay"` and `"movement"` slices
- `get_model_action_masks(phase, n_models, alive_mask)` — already handles dead-model masking and phase gating; shooting mask extends this
- `has_line_of_sight` in `domain/los.py` — ready for target validity checks, takes `(x0, y0, x1, y1, width, height, is_blocking)`
- `alive_mask_for` in `domain/entities.py` — reusable for opponent alive filtering
- `BattlePhase.shooting` enum value already exists in `game_timing.py`
- `GameClock.advance_phase` already walks through `BATTLE_PHASE_ORDER` including shooting
- `DistanceCache` — existing distance computation infrastructure for range checks

### Established Patterns
- Action slices are contiguous flat indices: `[0=stay][1..N=movement][N+1..M=shooting]`
- Phase gating via `valid_phases` on `ActionSlice` — movement valid in movement phase only, stay valid in all phases
- Dead models get `STAY_ACTION` only via `alive_mask` parameter
- Config fields default to no-op values for backward compatibility
- Domain stays pure (no Gym imports); env components wire domain → Gym

### Integration Points
- `skip_phases` config controls which phases the agent steps through — removing `BattlePhase.shooting` from this list enables shooting steps
- `build_observation` reads `view.game_clock_state.phase` to determine which mask to build — shooting phase will automatically produce shooting-valid masks once the slice is registered
- Transformer network output shape `(n_models, n_actions)` grows when `n_actions` increases from the new shooting slice — network architecture adapts automatically via `observation_space`
- `WargameEnvAction.actions` is `list[int]` per model — shooting target indices are just ints in the enlarged Discrete space

</code_context>

<specifics>
## Specific Ideas

- Positional alignment between shooting target indices and opponent observation slots is critical for transformer learning. This is not optional — the correspondence must be maintained as an invariant.
- The "all weapons fire at one target" semantic in Phase 4 is deliberately simple. Multi-weapon targeting (sub-steps per weapon) is the planned expansion path.
- Precomputed attacker × defender probability matrices (hit chance, wound chance, expected damage) will serve as both observation features for the transformer and explainability tools. These are perfect information in the real game — players can calculate them exhaustively. Deferred to Phase 5+ when weapon profiles are complete.

</specifics>

<deferred>
## Deferred Ideas

### Multi-Weapon Targeting (Phase 5+)
- Models with multiple weapons should independently assign targets per weapon. Planned expansion: **sub-steps within shooting phase**, one per weapon firing opportunity. Keeps action space `Discrete(n_targets)`, transformer architecture unchanged, just more steps. Models with 1 weapon get 1 step, models with 5 get 5.
- Non-uniform weapon counts across models (mostly 1-2, up to 7) are the expected norm.
- "All eligible weapons fire" is the Phase 4 default; making fire/hold configurable per weapon is future work.

### Pointer-Network Attention
- If the transformer struggles to learn the implicit mapping between opponent observation position and shooting action index, a **pointer-network style cross-attention** mechanism can produce shooting logits directly from attention scores between acting model tokens and opponent tokens. This is a network architecture enhancement, not an action space change. Flag for investigation if shooting learning plateaus.

### Decision / Event Log Infrastructure
- Explicit recording of targeting decisions with observation context (what the transformer saw, precomputed probabilities, tactical positioning) for explainability. Resolution (dice rolls) as separate observable events. Targeting decision and resolution outcome should both be explicit, ordered, inspectable records.
- Feeds into v9.0 (structured game state & event streaming) — the event log architecture serves both explainability and API/replay needs.

### Precomputed Probability Matrices
- Attacker × defender expected damage tables computed from weapon profiles and target stats. Dual purpose: **observation feature** so the transformer has explicit efficiency information (perfect information system — mirrors real player capability), and **explainability tool** for post-hoc analysis of targeting decisions.
- Requires full weapon profiles (Phase 5) to compute. Could be computed at episode init and cached, or recomputed when state changes (wounds reduce targets, eliminations).

</deferred>

---

*Phase: 04-shooting-action-space*
*Context gathered: 2026-04-05*
