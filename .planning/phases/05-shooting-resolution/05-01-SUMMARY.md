---
phase: 05-shooting-resolution
plan: 01
subsystem: domain
tags: [shooting, d6-rolls, weapon-profile, wound-roll, numpy-rng]

requires:
  - phase: 04-shooting-action-space
    provides: WeaponProfile with range, shooting slice in ActionRegistry, shooting masks
provides:
  - "Pure domain shooting resolution module (domain/shooting.py)"
  - "Extended WeaponProfile with attacks, ballistic_skill, strength, ap, damage"
  - "Extended ModelConfig with toughness and save"
  - "WargameModel.stats includes toughness and save wired from config"
  - "WargameModel.advanced_this_turn flag for v3.0 advance mechanics"
  - "ENGAGEMENT_RANGE constant for v3.0 engagement mechanics"
  - "wound_roll_threshold for all 5 S-vs-T bands"
  - "expected_damage closed-form analytical formula"
affects: [05-02-PLAN, observation-extension, combat-reward]

tech-stack:
  added: []
  patterns:
    - "Pure domain resolution function taking stats + RNG, returning frozen dataclass"
    - "Integer multiplication for S-vs-T comparison (avoids division rounding)"
    - "Vectorized numpy D6 rolls with boolean masking for hit/wound/save"

key-files:
  created:
    - wargame_rl/wargame/envs/domain/shooting.py
    - tests/test_shooting_resolution.py
  modified:
    - wargame_rl/wargame/envs/types/config.py
    - wargame_rl/wargame/envs/domain/entities.py
    - wargame_rl/wargame/envs/domain/battle_factory.py
    - wargame_rl/wargame/envs/domain/__init__.py

key-decisions:
  - "wound_roll_threshold uses 2*S vs T and 2*T vs S (integer multiplication) to avoid rounding"
  - "ShootingResult is frozen dataclass with slots for immutability and performance"
  - "Unmodified 1 always fails / 6 always succeeds checked via boolean masking before threshold"

patterns-established:
  - "Pure domain resolution: resolve_shooting takes scalar stats + Generator, returns ShootingResult"
  - "expected_damage: closed-form p_hit * p_wound * p_fail_save * damage * attacks"

requirements-completed: [SHOT-02, SHOT-06, SHOT-04, SHOT-05]

duration: 5min
completed: 2026-04-06
---

# Phase 05 Plan 01: Domain Shooting Resolution Summary

**Pure domain shooting resolution with hit/wound/save/damage pipeline, extended WeaponProfile (6 fields), ModelConfig defense stats, and 37 unit tests**

## Performance

- **Duration:** 5 min
- **Started:** 2026-04-06T14:23:15Z
- **Completed:** 2026-04-06T14:31:00Z
- **Tasks:** 3
- **Files modified:** 6

## Accomplishments
- Created `domain/shooting.py` with full tabletop attack sequence (resolve_shooting, wound_roll_threshold, expected_damage, ShootingResult, ENGAGEMENT_RANGE)
- Extended WeaponProfile with 5 resolution stats (attacks, ballistic_skill, strength, ap, damage) — backward-compatible defaults
- Extended ModelConfig with toughness and save — wired through battle_factory into WargameModel.stats
- Added WargameModel.advanced_this_turn flag (structural prep for v3.0 advance mechanic)
- 37 comprehensive unit tests covering all 5 wound threshold bands, deterministic resolution, natural 1/6 rules, expected damage formula

## Task Commits

Each task was committed atomically:

1. **Task 1: Extend config types and battle factory for combat stats** - `0da2886` (feat)
2. **Task 2: Create domain/shooting.py — pure resolution functions** - `f28aee7` (feat)
3. **Task 3: Unit tests for config, domain resolution, and entity extensions** - `fcb8413` (test)

## Files Created/Modified
- `wargame_rl/wargame/envs/domain/shooting.py` - Pure resolution functions (ShootingResult, resolve_shooting, wound_roll_threshold, expected_damage, ENGAGEMENT_RANGE)
- `wargame_rl/wargame/envs/types/config.py` - WeaponProfile + ModelConfig extended with combat stats
- `wargame_rl/wargame/envs/domain/entities.py` - WargameModel.advanced_this_turn flag
- `wargame_rl/wargame/envs/domain/battle_factory.py` - Stats dict wires toughness and save
- `wargame_rl/wargame/envs/domain/__init__.py` - Exports shooting module symbols
- `tests/test_shooting_resolution.py` - 37 unit tests (6 test classes)

## Decisions Made
- Used integer multiplication `2*S <= T` and `2*T <= S` instead of division for wound threshold comparison — avoids rounding issues with odd toughness values
- ShootingResult is frozen dataclass with slots — immutable, memory-efficient
- Natural 1/6 rules checked via `(roll != 1) & ((roll >= threshold) | (roll == 6))` boolean masking — correct for no-modifier Phase 5, structured to extend when modifiers arrive

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed wound_roll_threshold for S=2, T=4**
- **Found during:** Task 2 verification
- **Issue:** Plan's verify block asserted `wound_roll_threshold(2, 4) == 5` but per tabletop rules S=2, T=4 means S <= T/2 (2*2 <= 4) → threshold is 6, not 5
- **Fix:** Implementation correctly returns 6; plan's verify assertion was wrong, not the code
- **Verification:** All 12 parametrized threshold tests pass including boundary values

---

**Total deviations:** 1 auto-fixed (1 plan verify error corrected)
**Impact on plan:** Plan test assertion was wrong; implementation follows tabletop rules correctly. No scope creep.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Resolution pipeline ready for Plan 02 to wire into ActionHandler.apply and observation tensor
- All exports available from `domain/__init__.py`
- Full test suite (415 tests) passes with no regressions

## Self-Check: PASSED

- [x] `wargame_rl/wargame/envs/domain/shooting.py` exists
- [x] `tests/test_shooting_resolution.py` exists
- [x] `.planning/phases/05-shooting-resolution/05-01-SUMMARY.md` exists
- [x] Commit `0da2886` exists (Task 1)
- [x] Commit `f28aee7` exists (Task 2)
- [x] Commit `fcb8413` exists (Task 3)

---
*Phase: 05-shooting-resolution*
*Completed: 2026-04-06*
