# Phase 2: Alive-Aware Observation - Context

**Gathered:** 2026-04-04
**Status:** Ready for planning

<domain>
## Phase Boundary

The RL agent’s observation (structured types, Gym-facing space, and flat tensors used by DQN/PPO) exposes **wound status** and an explicit **alive signal** for **every player and opponent model slot**, with **fixed slot count and tensor widths** across the episode. Eliminated models remain at their list indices (Phase 1); observation must not rely on removing or padding rows.

This phase does **not** add weapon stats (OBS-02) or LOS features — those belong to later roadmap phases.

</domain>

<decisions>
## Implementation Decisions

### Structured observation (`WargameModelObservation`)

- **D-01:** Add three fields per model, **player and opponent**: `alive: float` (`1.0` if `is_alive`, else `0.0`), `current_wounds: int`, `max_wounds: int` (from `WargameModel.stats`). Float for `alive` matches the existing float32 tensor pipeline.
- **D-02:** **No sentinel locations** for dead models — keep true grid location and objective distances as produced when building from the step’s `DistanceCache` (same cache as `step()` so dead models already see `inf` distances to objectives when `alive_mask` was applied in `compute_distances`). The policy disambiguates using `alive` and wound fields.
- **D-03:** Update `_models_to_obs` in `observation_builder.py` to populate these fields from each `WargameModel`.

### Action mask in observation

- **D-04:** In `build_observation`, pass `alive_mask_for(view.player_models)` into `action_registry.get_model_action_masks(..., alive_mask=...)` so the **observation’s** `action_mask` matches runtime behavior: dead models **only** valid on `STAY_ACTION` (index 0), same as `ActionHandler`.

### Tensor encoding (`observation.py`)

- **D-05:** Extend per-model feature vector by **3** floats (in order): `alive`; `current_wounds / max(max_wounds, 1)` clipped to `[0, 1]`; `max_wounds / 100.0` clipped to `[0, 1]` (denominator aligns with existing `WargameModel.to_space` stats upper bound). Append after existing blocks (normalized location, normalized distances, group one-hot, same-group distance).
- **D-06:** Recompute `feature_dim` from observation metadata (`n_objectives`, `max_groups`, **+3**). Keep `opponent_features` at **0 rows** when no opponents; width unchanged.
- **D-07:** Update `observation_to_tensor` / `observations_to_tensor_batch` docstrings and any hard-coded feature size logic in `net.py` (and tests) that assumes the old per-model width.

### Gym `observation_space`

- **D-08:** Extend `WargameModel.to_space()` so the Dict space for each model includes the same wound/alive information as structured observations (consistent with Gymnasium contract).

### Checkpoint compatibility

- **D-09:** Treat wider model features as a **breaking change** for saved policy/value weights. **No** checkpoint adapter in this phase — document that training runs need **new** runs after merge; old checkpoints are not load-compatible without manual architecture surgery (out of scope).

### Testing and success criteria

- **D-10:** **Required:** tests that (1) after `take_damage` elimination, `alive == 0.0` and wound fields match stats; (2) `action_mask` for dead player models only allows stay; (3) `observation_to_tensor` per-model row width matches the new `feature_dim` and is stable across steps with eliminations.
- **D-11:** **Optional / manual:** short training smoke after implementation; rely on existing CI training smoke (`test_z_e2e_training` or equivalent) once observation dimensions are updated — do not add a second heavy training job solely for this phase unless the existing smoke fails.

### Claude's Discretion

- Minor tweaks to normalization (e.g. shared wound cap constant vs inline `100.0`) if tests or dtype edge cases warrant it.
- Whether to duplicate `alive` as redundant with `current_wounds == 0` — **keep explicit `alive`** per OBS-03 and for clarity when wound semantics evolve in later phases.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Requirements and roadmap

- `.planning/REQUIREMENTS.md` — WOUND-04, OBS-01, OBS-03
- `.planning/ROADMAP.md` — Phase 2 goal and success criteria (tensor includes wounds; alive flag; training without shape mismatch)

### Phase 1 decisions (slot stability, wounds)

- `.planning/phases/01-wounds-elimination/01-CONTEXT.md` — fixed indices, `is_alive`, no row removal
- `.planning/phases/01-wounds-elimination/01-02-SUMMARY.md` — alive_mask usage in env loop (observation mask deferred explicitly to Phase 2)

### Architecture and env patterns

- `docs/ddd-envs.md` — BattleView, observation builder, dependency direction

### Implementation touchpoints

- `wargame_rl/wargame/envs/types/model_observation.py` — extend dataclass + `size` property
- `wargame_rl/wargame/envs/types/env_observation.py` — aggregate `size` if needed
- `wargame_rl/wargame/envs/env_components/observation_builder.py` — `_models_to_obs`, `build_observation` mask wiring
- `wargame_rl/wargame/envs/domain/entities.py` — `WargameModel.to_space()`, stats keys
- `wargame_rl/wargame/model/common/observation.py` — `_models_to_features`, `_observation_to_numpy`, batching
- `wargame_rl/wargame/model/net.py` — dynamic obs sizing if hard-coded
- `wargame_rl/wargame/envs/wargame.py` — `observation_space` composition

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable assets

- `alive_mask_for` in `domain/entities.py` — reuse for `get_model_action_masks` in `build_observation`.
- `WargameModel.stats["current_wounds"]` / `["max_wounds"]` — single source of truth for structured obs and tensors.
- `_models_to_features` / `_observation_to_numpy` — central place to widen per-model features; batching already stacks fixed-width rows.

### Established patterns

- Per-model features are **horizontal** `hstack` of normalized blocks ending with group one-hot and same-group distance.
- Action mask is bool tensor `(n_models, n_actions)` aligned with player model count only.

### Integration points

- `WargameEnv._get_obs` → `build_observation(self, cache)` — ensure the same `DistanceCache` instance from `step()` is passed so dead-model distances match reward/termination semantics.
- DQN `apply_action_mask` and PPO batch paths consume the fifth tensor list element — shape must stay `(n_models, n_actions)` with consistent `n_models`.

</code_context>

<specifics>
## Specific Ideas

User selected **all** discuss-phase gray areas; decisions above lock defaults without extra product references.

</specifics>

<deferred>
## Deferred Ideas

- **OBS-02** (weapon profiles in observation) — later combat phases.
- **Checkpoint migration / import** — out of scope; breaking change accepted.
- **Alternative encodings** (e.g. omit redundant `alive` float) — rejected in favor of explicit OBS-03 flag.

</deferred>

---

*Phase: 02-alive-aware-observation*
*Context gathered: 2026-04-04*
