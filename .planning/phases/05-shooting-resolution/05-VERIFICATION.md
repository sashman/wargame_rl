---
phase: 05-shooting-resolution
verified: 2026-04-06T15:10:00Z
status: passed
score: 16/16 must-haves verified
re_verification: false
---

# Phase 5: Shooting Resolution Verification Report

**Phase Goal:** Shooting actions resolve damage through the tabletop attack sequence with configurable weapons
**Verified:** 2026-04-06T15:10:00Z
**Status:** PASSED
**Re-verification:** No — initial verification

## Goal Achievement

### Success Criteria (from ROADMAP.md)

| # | Criterion | Status | Evidence |
|---|-----------|--------|----------|
| 1 | A shoot action resolves via hit roll → wound roll → save → damage, applying wounds to the target | ✓ VERIFIED | `resolve_shooting` in `domain/shooting.py` implements full pipeline; `_resolve_shooting_action` in `wargame.py` calls it and applies `take_damage`; integration test `TestShootingIntegration::test_player_shooting_deals_damage` confirms opponent wounds decrease |
| 2 | Weapon profiles (range, attacks, BS, strength, AP, damage) are configurable per model in YAML | ✓ VERIFIED | `WeaponProfile` in `config.py` has 6 fields (range + attacks, ballistic_skill, strength, ap, damage) with backward-compatible defaults; `TestConfigExtensions::test_weapon_profile_defaults` confirms |
| 3 | Models that advanced cannot shoot; models in engagement range cannot shoot | ✓ VERIFIED | `compute_shooting_masks` accepts `player_advanced` and `engagement_range` kwargs; `observation_builder.py` passes both; `TestShootingMaskExtensions` confirms both masks work |
| 4 | Weapon-relevant stats appear in the agent's observation for informed targeting decisions | ✓ VERIFIED | 7 combat stat columns + expected damage per target in tensor pipeline; `TestObservationExtension::test_weapon_stats_nonzero_for_armed` and `test_expected_damage_nonzero` confirm |

**Score:** 4/4 success criteria verified

### Observable Truths (from Plan 01 must_haves)

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | WeaponProfile accepts attacks, ballistic_skill, strength, ap, damage with defaults matching D-06 | ✓ VERIFIED | `config.py:96-112` — all 5 fields with defaults (2, 3, 4, 1, 1); test `test_weapon_profile_defaults` |
| 2 | ModelConfig accepts toughness and save with defaults matching D-09 | ✓ VERIFIED | `config.py:134-142` — toughness=3, save=4; test `test_model_config_defense_defaults` |
| 3 | WargameModel.stats dict includes toughness and save wired from config | ✓ VERIFIED | `battle_factory.py:30-31` reads `mc.toughness`/`mc.save`; `battle_factory.py:41-45` puts them in stats dict; test `test_custom_stats` and `test_stats_keys` |
| 4 | resolve_shooting returns correct hits/wounds/unsaved/damage for a fixed seed | ✓ VERIFIED | `shooting.py:40-83` full implementation; test `test_deterministic_seed_42` asserts exact values |
| 5 | wound_roll_threshold returns correct D6 threshold for all 5 S-vs-T bands | ✓ VERIFIED | `shooting.py:23-37` integer-multiplication checks; 12 parametrized tests covering all bands |
| 6 | expected_damage analytical formula matches hand-calculated values | ✓ VERIFIED | `shooting.py:86-101` closed-form; test `test_default_profile` verifies `2*(4/6)*(4/6)*(4/6)` |
| 7 | WargameModel.advanced_this_turn flag exists, defaults False, resets per episode | ✓ VERIFIED | `entities.py:44` init; `entities.py:59` reset; tests `test_default_false` and `test_reset_clears_flag` |
| 8 | ENGAGEMENT_RANGE constant is defined in domain/shooting.py | ✓ VERIFIED | `shooting.py:9` — `ENGAGEMENT_RANGE = 1`; test `test_engagement_range_constant` |

### Observable Truths (from Plan 02 must_haves)

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 9 | A shoot action during shooting phase resolves damage via hit→wound→save→damage and applies wounds to the target | ✓ VERIFIED | `wargame.py:373-380` routes shooting phase to `_resolve_shooting_action`; `_resolve_shooting_action:322-369` calls `resolve_shooting` + `take_damage`; integration test confirms |
| 10 | Both player and opponent shooting actions resolve through same resolution path | ✓ VERIFIED | `_apply_player_action` (line 374) and `_apply_opponent_action` (line 403) both call `_resolve_shooting_action`; test `test_both_sides_shoot` |
| 11 | Shooting masks filter out models with advanced_this_turn=True | ✓ VERIFIED | `shooting_masks.py:50-51` checks `player_advanced`; `observation_builder.py:132-133` passes it; test `test_advanced_model_masked` |
| 12 | Shooting masks filter out models within ENGAGEMENT_RANGE of an enemy | ✓ VERIFIED | `shooting_masks.py:52-53` checks `engagement_range`; `observation_builder.py:143` passes `ENGAGEMENT_RANGE`; test `test_engagement_range_masks_model` |
| 13 | Agent's observation tensor includes per-player-model weapon stats and per-opponent-model defense stats | ✓ VERIFIED | `model_observation.py:17-23` has 7 combat fields; `observation_builder.py:68-76` populates from config; `observation.py:97-132` normalizes into tensor; test `test_weapon_stats_nonzero_for_armed` |
| 14 | Agent's observation includes precomputed expected damage per (attacker, target) pair | ✓ VERIFIED | `observation.py:179-201` computes `ed_matrix` from `expected_damage`; zero-pads opponent features; test `test_expected_damage_nonzero` and `test_opponent_ed_columns_zero` |
| 15 | A combat RNG seeded per episode produces deterministic rolls when env seed is fixed | ✓ VERIFIED | `wargame.py:120` init; `wargame.py:283-284` re-seeded in reset from `np_random`; tests `test_same_seed_same_results` and `test_different_seeds_differ` |
| 16 | Existing YAML configs without weapon/defense stats produce working environments | ✓ VERIFIED | All WeaponProfile/ModelConfig fields have defaults; tests `test_backward_compat_weapon_range_only`, `test_no_model_configs`, `test_zero_opponents_no_ed_columns`, `test_full_step_loop` |

**Score:** 16/16 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `wargame_rl/wargame/envs/domain/shooting.py` | Pure domain resolution functions | ✓ VERIFIED | 102 lines; exports ShootingResult, resolve_shooting, wound_roll_threshold, expected_damage, ENGAGEMENT_RANGE. No Gym imports. |
| `wargame_rl/wargame/envs/types/config.py` | Extended WeaponProfile and ModelConfig | ✓ VERIFIED | WeaponProfile has 6 fields; ModelConfig has toughness/save; all backward-compatible |
| `wargame_rl/wargame/envs/domain/entities.py` | WargameModel with advanced_this_turn | ✓ VERIFIED | Flag at line 44; reset at line 59 |
| `wargame_rl/wargame/envs/domain/battle_factory.py` | Stats dict wires toughness/save | ✓ VERIFIED | Lines 30-31 read from config; lines 41-45 put in stats dict |
| `wargame_rl/wargame/envs/domain/__init__.py` | Exports shooting symbols | ✓ VERIFIED | Lines 15-21 import; lines 52-56 in __all__ |
| `wargame_rl/wargame/envs/wargame.py` | Combat RNG, resolution wiring | ✓ VERIFIED | `_combat_rng` init/reset, `_resolve_shooting_action`, phase-routed apply methods |
| `wargame_rl/wargame/envs/env_components/shooting_masks.py` | Extended masks with advance/engagement | ✓ VERIFIED | `player_advanced` and `engagement_range` keyword params |
| `wargame_rl/wargame/envs/types/model_observation.py` | 7 combat observation fields | ✓ VERIFIED | weapon_attacks through save_stat; size includes +7 |
| `wargame_rl/wargame/model/common/observation.py` | Combat features in tensor pipeline | ✓ VERIFIED | 7 normalized columns + expected_damage matrix; feature_dim = base + n_opponent |
| `wargame_rl/wargame/envs/reward/step_context.py` | Combat outcome fields | ✓ VERIFIED | player/opponent_damage_dealt and player/opponent_models_killed |
| `tests/test_shooting_resolution.py` | Unit and integration tests | ✓ VERIFIED | 54 tests across 12 classes; all pass |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `battle_factory.py` | `WargameModel.stats` | toughness and save wired from ModelConfig | ✓ WIRED | Lines 30-31 read `mc.toughness`/`mc.save`; lines 41-45 pass into stats dict |
| `domain/shooting.py` | `numpy.random.Generator` | `rng.integers(1, 7, size=n)` | ✓ WIRED | Lines 55, 64, 73 use `rng.integers(1, 7, size=...)` |
| `wargame.py` | `domain/shooting.py` | `resolve_shooting` call in `_resolve_shooting_action` | ✓ WIRED | Line 21 imports; line 356 calls `resolve_shooting(...)` |
| `wargame.py` | `WargameModel.take_damage` | `_resolve_shooting_action` applies damage_dealt | ✓ WIRED | Line 367: `targets[target_idx].take_damage(result.damage_dealt)` |
| `observation.py` | `domain/shooting.py` | `expected_damage` called in `_observation_to_numpy` | ✓ WIRED | Line 5 imports; line 189 calls `expected_damage(...)` |
| `observation_builder.py` | `model_observation.py` | `_models_to_obs` passes weapon/defense fields | ✓ WIRED | Lines 73-76 read weapon stats from config; lines 80-95 pass to WargameModelObservation |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Shooting resolution test suite | `uv run pytest tests/test_shooting_resolution.py -x` | 54 passed in 0.17s | ✓ PASS |
| Full test suite regression | `uv run pytest -x` | 432 passed in 103s | ✓ PASS |
| Domain module imports | verified via test imports | All 5 symbols importable from `domain.shooting` | ✓ PASS |
| WeaponProfile backward compat | `WeaponProfile(range=12)` constructs with defaults | attacks=2, bs=3, etc. | ✓ PASS |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| SHOT-01 | 05-02 | Models can select a shoot action targeting an enemy model within weapon range | ✓ SATISFIED | Shooting resolution wired into env step; action routing at `wargame.py:373`; masks enforce range/LOS |
| SHOT-02 | 05-01, 05-02 | Shooting resolves via tabletop attack sequence: hit→wound→save→damage | ✓ SATISFIED | `resolve_shooting` in `domain/shooting.py` implements full pipeline; `_resolve_shooting_action` wires it into env |
| SHOT-04 | 05-01, 05-02 | Models that advanced or fell back cannot shoot | ✓ SATISFIED | `advanced_this_turn` flag on entity; `shooting_masks.py:50-51` filters; `observation_builder.py:132-133` passes flag |
| SHOT-05 | 05-01, 05-02 | Models in engagement range cannot shoot | ✓ SATISFIED | `ENGAGEMENT_RANGE=1` constant; `shooting_masks.py:52-53` filters; `observation_builder.py:143` passes range |
| SHOT-06 | 05-01 | Weapon profiles configurable per model (range, attacks, BS, strength, AP, damage) | ✓ SATISFIED | `WeaponProfile` has 6 configurable fields with defaults; wired through config → resolution |
| OBS-02 | 05-02 | Agent observation includes weapon profiles or combat-relevant stats | ✓ SATISFIED | 7 combat stat columns + expected damage per target in tensor pipeline |

**All 6 requirements satisfied. No orphaned requirements.**

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `observation.py` | 254 | "placeholder" in docstring | ℹ️ Info | Describes a zero-valued game feature position; not a code stub |

No blockers or warnings found.

### Human Verification Required

No items require human verification. All truths are verifiable programmatically and confirmed by automated tests.

### Gaps Summary

No gaps found. All 16 must-haves verified, all 6 requirements satisfied, all artifacts exist and are substantive and wired, all tests pass with no regressions.

---

_Verified: 2026-04-06T15:10:00Z_
_Verifier: Claude (gsd-verifier)_
