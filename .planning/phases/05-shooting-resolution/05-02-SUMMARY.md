---
phase: 05-shooting-resolution
plan: 02
subsystem: environment
tags: [shooting, combat-rng, observation, expected-damage, step-context, masks]

requires:
  - phase: 05-shooting-resolution-01
    provides: Pure domain shooting resolution (resolve_shooting, expected_damage, ShootingResult, ENGAGEMENT_RANGE)
  - phase: 04-shooting-action-space
    provides: ActionRegistry shooting slice, shooting masks, WeaponProfile with range
provides:
  - "Combat RNG seeded per episode for deterministic shooting resolution"
  - "_resolve_shooting_action shared resolution path for player and opponent"
  - "Shooting masks extended with advanced_this_turn and engagement_range checks"
  - "Observation tensor with 7 combat stat columns per model"
  - "Expected damage per (attacker, target) pair in player observation tensor"
  - "StepContext combat outcome fields (damage dealt, models killed per side)"
  - "Network from_env uses actual tensor shapes (not observation.size) for input sizing"
affects: [06-combat-reward, observation-pipeline, transformer-network, mlp-network]

tech-stack:
  added: []
  patterns:
    - "Env-level shooting resolution: env routes shooting-phase actions to _resolve_shooting_action instead of ActionHandler.apply"
    - "Combat RNG: per-episode np.random.Generator seeded from env.np_random for deterministic replays"
    - "Expected damage tensor: closed-form analytical formula appended as per-target columns in player features"
    - "Network input sizing from actual tensors via observation_to_tensor instead of observation.size"

key-files:
  created: []
  modified:
    - wargame_rl/wargame/envs/wargame.py
    - wargame_rl/wargame/envs/env_components/shooting_masks.py
    - wargame_rl/wargame/envs/env_components/observation_builder.py
    - wargame_rl/wargame/envs/types/model_observation.py
    - wargame_rl/wargame/model/common/observation.py
    - wargame_rl/wargame/envs/reward/step_context.py
    - wargame_rl/wargame/model/net.py
    - tests/test_shooting_resolution.py
    - tests/test_dqn.py
    - tests/test_state.py
    - tests/test_wounds.py

key-decisions:
  - "Network from_env derives input sizes from actual observation_to_tensor output, not observation.size — prevents dim mismatch when tensor pipeline adds expected damage columns"
  - "Expected damage columns only in player model features; opponent features zero-padded to match feature_dim"
  - "Combat RNG uses self.np_random.integers(0, 2**31) to seed per-episode Generator, keeping env seed-deterministic"
  - "Shooting resolution called at env level before ActionHandler.apply — env owns combat flow, ActionHandler stays movement-only"

patterns-established:
  - "Env-level phase routing: _apply_player_action checks phase and routes to resolution or movement handler"
  - "StepContext as extensible data carrier: combat fields added without changing any existing reward calculator signatures"
  - "Feature dim = base_feature_dim + n_opponent: expected damage columns scale with opponent count"

requirements-completed: [SHOT-01, SHOT-02, SHOT-04, SHOT-05, OBS-02]

duration: 14min
completed: 2026-04-06
---

# Phase 05 Plan 02: Env Shooting Resolution Wiring Summary

**Combat RNG + shooting resolution wired into env step, observation tensor extended with 7 combat stats and expected damage per target, 17 new integration tests**

## Performance

- **Duration:** 14 min
- **Started:** 2026-04-06T14:33:37Z
- **Completed:** 2026-04-06T14:47:26Z
- **Tasks:** 3
- **Files modified:** 11

## Accomplishments
- Wired shooting resolution into env step: player and opponent shooting actions resolve damage via hit→wound→save→damage during shooting phase
- Extended observation tensor with 7 normalized combat stats per model (weapon attacks, BS, strength, AP, damage, toughness, save) plus per-target expected damage columns
- Extended shooting masks with advanced_this_turn and engagement_range filtering (structural prep for v3.0 advance mechanics)
- Added combat outcome fields to StepContext (player/opponent damage dealt, models killed) for Phase 6 reward calculators
- Fixed network from_env to derive input sizes from actual tensor shapes, preventing dimension mismatches

## Task Commits

Each task was committed atomically:

1. **Task 1: Wire combat RNG and shooting resolution into env + extend masks** - `2a99e1e` (feat)
2. **Task 2: Extend observation pipeline with combat features** - `21a33f2` (feat)
3. **Task 3: Integration tests for resolution wiring, masks, and observation** - `4288e01` (test)

## Files Created/Modified
- `wargame_rl/wargame/envs/wargame.py` - Combat RNG, _resolve_shooting_action, phase-routed _apply_player/opponent_action, alive tracking for kill counts
- `wargame_rl/wargame/envs/env_components/shooting_masks.py` - player_advanced and engagement_range keyword params
- `wargame_rl/wargame/envs/env_components/observation_builder.py` - Pass model_configs to _models_to_obs, pass player_advanced and ENGAGEMENT_RANGE to masks
- `wargame_rl/wargame/envs/types/model_observation.py` - 7 combat stat fields (weapon_attacks through save_stat)
- `wargame_rl/wargame/model/common/observation.py` - 7 combat stat tensor columns, expected damage matrix, updated feature_dim
- `wargame_rl/wargame/envs/reward/step_context.py` - player_damage_dealt, opponent_damage_dealt, player_models_killed, opponent_models_killed
- `wargame_rl/wargame/model/net.py` - MLPNetwork and TransformerNetwork from_env use observation_to_tensor for input sizing
- `tests/test_shooting_resolution.py` - 17 new tests (6 classes: integration, masks, observation, compat, RNG, StepContext)
- `tests/test_dqn.py` - Updated dim_model for +7 combat stats
- `tests/test_state.py` - Updated dim_model for +7 combat stats
- `tests/test_wounds.py` - Updated expected_dim for +7 combat stats

## Decisions Made
- Network from_env derives input sizes from actual observation_to_tensor output instead of observation.size — prevents dim mismatch when tensor pipeline adds columns beyond what WargameModelObservation.size knows about
- Expected damage columns only added to player model features; opponent features zero-padded to match feature_dim — agent doesn't need opponents' offensive potential in its own observation
- Combat RNG seeded from env.np_random for full determinism chain: env seed → np_random → combat_seed → combat_rng
- Shooting resolution at env level, not in ActionHandler — keeps ActionHandler focused on movement/space encoding, env owns combat flow

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed network from_env dimension mismatch**
- **Found during:** Task 3 (full test suite regression check)
- **Issue:** MLPNetwork and TransformerNetwork from_env used observation.size/size_wargame_models which don't account for expected damage columns added by the tensor pipeline
- **Fix:** Changed from_env to compute input sizes from actual observation_to_tensor output
- **Files modified:** wargame_rl/wargame/model/net.py
- **Verification:** All 432 tests pass including DQN/PPO network tests with opponents
- **Committed in:** `4288e01` (Task 3 commit)

**2. [Rule 1 - Bug] Updated hardcoded dim_model in test assertions**
- **Found during:** Task 3 (full test suite regression check)
- **Issue:** test_dqn, test_state, test_wounds hardcoded feature dimension without +7 combat stats
- **Fix:** Added +7 to dim_model calculations in all affected test files
- **Files modified:** tests/test_dqn.py, tests/test_state.py, tests/test_wounds.py
- **Verification:** All tests pass
- **Committed in:** `4288e01` (Task 3 commit)

---

**Total deviations:** 2 auto-fixed (2 bugs)
**Impact on plan:** Both fixes necessary for correctness — observation dimension change ripples through network construction and test assertions. No scope creep.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Full shooting pipeline operational end-to-end: action selection → resolution → damage → observation
- StepContext carries combat outcomes ready for Phase 6 reward calculators (damage_dealt, models_killed)
- All 432 tests pass with no regressions
- Expected damage in observation gives agent analytical information for target prioritization

## Self-Check: PASSED

- [x] `wargame_rl/wargame/envs/wargame.py` contains `_combat_rng` and `_resolve_shooting_action`
- [x] `wargame_rl/wargame/envs/env_components/shooting_masks.py` has `player_advanced` and `engagement_range` params
- [x] `wargame_rl/wargame/envs/types/model_observation.py` has 7 combat stat fields
- [x] `wargame_rl/wargame/model/common/observation.py` has expected damage computation
- [x] `wargame_rl/wargame/envs/reward/step_context.py` has combat outcome fields
- [x] Commit `2a99e1e` exists (Task 1)
- [x] Commit `21a33f2` exists (Task 2)
- [x] Commit `4288e01` exists (Task 3)

---
*Phase: 05-shooting-resolution*
*Completed: 2026-04-06*
