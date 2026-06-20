---
phase: 01-wounds-elimination
plan: 02
subsystem: envs
tags: [wounds, alive_mask, distance_cache, actions, reward, vp, opponent, render]

requires: [01-01-SUMMARY.md]
provides:
  - "compute_distances(..., alive_mask=) — dead rows/cols inf in norms"
  - "ActionHandler skips dead models; masks restrict dead to STAY_ACTION"
  - "Reward averaged over alive player models only; terminal OC uses alive_mask"
  - "DefaultVPCalculator uses alive_mask_for for player and opponent caches"
  - "ScriptedAdvanceToObjectivePolicy: alive centroid, STAY for dead opponents"
  - "Human renderer: grey + X for dead agents and opponents"
  - "WargameEnv.step: player alive cache, all_eliminated wiring, opponent masks"
  - "Success criteria all_at_objectives / all_models_grouped use alive_mask"
  - "GroupCohesionCalculator min_distances_to_same_group with alive_mask"
affects: [02-alive-aware-observation]

tech-stack:
  added: []
  patterns:
    - "Pass alive_mask_for(player_models) into distance cache queries"
    - "terminate_on_player_elimination (default False) for tabletop-accurate wipe"

key-files:
  created: []
  modified:
    - wargame_rl/wargame/envs/env_components/distance_cache.py
    - wargame_rl/wargame/envs/env_components/actions.py
    - wargame_rl/wargame/envs/reward/phase_manager.py
    - wargame_rl/wargame/envs/mission/vp_calculator.py
    - wargame_rl/wargame/envs/opponent/scripted_advance_to_objective_policy.py
    - wargame_rl/wargame/envs/renders/human.py
    - wargame_rl/wargame/envs/wargame.py
    - wargame_rl/wargame/envs/reward/criteria/all_at_objectives.py
    - wargame_rl/wargame/envs/reward/criteria/all_models_grouped.py
    - wargame_rl/wargame/envs/reward/calculators/group_cohesion.py
    - tests/test_wounds.py

key-decisions:
  - "Player full-wipe termination is opt-in via terminate_on_player_elimination (default False) so opponent can keep scoring after wiping the agent"
  - "Opponent full-wipe always contributes to all_eliminated when opponents exist"
  - "Observation action masks unchanged in Phase 1 (Phase 2 / WOUND-04)"

patterns-established:
  - "Criteria and calculators that read DistanceCache must pass player alive_mask when semantics are 'alive models only'"

requirements-completed: [WOUND-03, WOUND-05]

duration: session
completed: 2026-04-04
---

# Phase 01 Plan 02: Alive-guards & elimination wiring — Summary

**Thread `is_alive` through env iteration sites, VP, opponent policy, renderer, and `all_eliminated` termination; reward and success criteria ignore dead models where appropriate.**

## Performance

- **Completed:** 2026-04-04
- **Verification:** `just validate` — 316 passed

## Accomplishments

- Distance cache and action handler already exposed `alive_mask`; **success criteria** (`AllAtObjectivesCriteria`, `AllModelsGroupedCriteria`) and **group cohesion** now pass `alive_mask_for(view.player_models)` so dead models do not block phase success or distort same-group distances.
- **Human renderer:** dead opponents get an X overlay (matching dead agent styling).
- **Tests:** renamed `test_termination_all_player_eliminated` (was `test_player_elimination_terminates_when_flag_set`) to match plan naming; integration coverage for movement, OC, opponent wipe, zero-opponent non-vacuity, and alive-only objectives remains in `tests/test_wounds.py`.

## Product note

- **`terminate_on_player_elimination`** (default `False`): wiping all player models does **not** end the episode unless the flag is set — documented on `WargameEnvConfig` for tabletop-style continuation. Opponent elimination still terminates when `number_of_opponent_models > 0`.

## Self-check

- Full suite green after changes.
- Plan 02 acceptance criteria satisfied for implemented paths; observation masking for dead agents is explicitly deferred to Phase 2 per plan.
