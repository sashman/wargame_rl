---
phase: 04-shooting-action-space
plan: 02
subsystem: environment
tags: [shooting-masks, action-masking, observation-pipeline, LOS, range-filtering]

requires:
  - phase: 04-shooting-action-space
    provides: "WeaponProfile config, shooting slice in ActionRegistry, phase-aware apply, BattleView LOS method"
  - phase: 03-line-of-sight-service
    provides: "has_line_of_sight domain service and WargameEnv.has_line_of_sight_between_cells"
provides:
  - "compute_shooting_masks pure function (alive + range + LOS + weapon filtering)"
  - "max_weapon_ranges helper to extract per-model max range from config"
  - "Shooting mask overlay in observation builder (per-target validity during shooting phase)"
  - "WargameEnv wiring: n_shoot_targets passed to both player and opponent ActionHandlers"
  - "Phase-aware apply calls from both _apply_player_action and _apply_opponent_action"
  - "ActionRegistry.has_slice convenience method"
affects: [05-shooting-resolution, 06-combat-reward-curriculum]

tech-stack:
  added: []
  patterns: ["Pure function shooting masks with callback-based LOS injection", "Observation builder overlay pattern for per-target filtering"]

key-files:
  created:
    - wargame_rl/wargame/envs/env_components/shooting_masks.py
  modified:
    - wargame_rl/wargame/envs/env_components/__init__.py
    - wargame_rl/wargame/envs/env_components/actions.py
    - wargame_rl/wargame/envs/env_components/observation_builder.py
    - wargame_rl/wargame/envs/wargame.py
    - tests/test_shooting_action.py

key-decisions:
  - "Shooting mask overlay uses bitwise AND on base registry mask — registry handles phase gating, overlay adds per-target filtering"
  - "has_los_fn callback injected into compute_shooting_masks rather than coupling to BattleView directly"
  - "max_weapon_ranges returns 0.0 for models with no weapons — row masked to all-False automatically"

patterns-established:
  - "Observation builder overlay: base mask from registry, then per-target refinement via compute_shooting_masks AND"
  - "Pure function with callback injection: shooting_masks.py receives has_los_fn rather than importing domain"

requirements-completed: [ACT-01, ACT-03, ACT-04, LOS-03, SHOT-03]

duration: 3min
completed: 2026-04-05
---

# Phase 04 Plan 02: Shooting Env Wiring & Mask Pipeline Summary

**Pure shooting mask function with LOS/range/alive filtering wired into observation builder, WargameEnv passing n_shoot_targets and phase to ActionHandlers**

## Performance

- **Duration:** 3 min
- **Started:** 2026-04-05T16:48:57Z
- **Completed:** 2026-04-05T16:55:00Z
- **Tasks:** 2
- **Files modified:** 6

## Accomplishments
- Created compute_shooting_masks pure function that filters targets by alive status, weapon range, and LOS via callback
- Wired shooting masks into observation builder as overlay on base registry mask during shooting phase
- Connected WargameEnv to pass n_shoot_targets and current phase to both player and opponent ActionHandlers
- Added 18 new tests (unit + integration + backward compat) bringing total shooting tests to 37

## Task Commits

Each task was committed atomically:

1. **Task 1: Create compute_shooting_masks pure function** - `f2201bf` (feat)
2. **Task 2: Wire shooting into env and observation pipeline** - `1234f68` (feat)

## Files Created/Modified
- `wargame_rl/wargame/envs/env_components/shooting_masks.py` - Pure compute_shooting_masks and max_weapon_ranges functions
- `wargame_rl/wargame/envs/env_components/__init__.py` - Export new functions
- `wargame_rl/wargame/envs/env_components/actions.py` - Added has_slice method to ActionRegistry
- `wargame_rl/wargame/envs/env_components/observation_builder.py` - Shooting mask overlay during shooting phase
- `wargame_rl/wargame/envs/wargame.py` - n_shoot_targets wiring, phase-aware apply calls
- `tests/test_shooting_action.py` - 18 new tests across 5 new test classes

## Decisions Made
- Used bitwise AND overlay rather than replacing the mask — registry handles phase gating, overlay adds LOS/range/alive filtering
- Kept compute_shooting_masks as a pure function with callback injection rather than coupling to BattleView

## Deviations from Plan
None - plan executed exactly as written.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Phase 4 complete — all 6 requirements (ACT-01 through ACT-04, LOS-03, SHOT-03) satisfied end-to-end
- Phase 5 (shooting resolution) can add hit→wound→save→damage logic triggered by shooting-slice actions
- Full test suite (373 tests) passes with no regressions

## Self-Check: PASSED

All files verified:
- `wargame_rl/wargame/envs/env_components/shooting_masks.py` — FOUND
- `wargame_rl/wargame/envs/env_components/__init__.py` — FOUND
- `wargame_rl/wargame/envs/env_components/actions.py` — FOUND
- `wargame_rl/wargame/envs/env_components/observation_builder.py` — FOUND
- `wargame_rl/wargame/envs/wargame.py` — FOUND
- `tests/test_shooting_action.py` — FOUND
- Commit `f2201bf` — FOUND
- Commit `1234f68` — FOUND

---
*Phase: 04-shooting-action-space*
*Completed: 2026-04-05*
