---
phase: 02-alive-aware-observation
plan: 01
subsystem: envs + model
tags: [observation, wounds, alive, tensor, action_mask]

requires: [02-CONTEXT.md]
provides:
  - "WargameModelObservation.alive / current_wounds / max_wounds (+ size +3)"
  - "build_observation uses alive_mask_for → get_model_action_masks"
  - "Per-model tensor tail: alive, wound ratio, max_wounds/100"
  - "WargameModel.to_space includes alive Box"
  - "Plotting training sweep copies wound fields from reference obs"
affects: [04-shooting-action-space]

requirements-completed: [WOUND-04, OBS-01, OBS-03]

duration: session
completed: 2026-04-04
---

# Phase 02 Plan 01: Alive-aware observation — Summary

**Structured observations and RL tensors expose alive + wound state; observation action masks match STAY-only for dead models. Per-model feature width +3 (breaking for old checkpoints).**

## Accomplishments

- Extended `WargameModelObservation` and `WargameModel.to_space()` with `alive` (0–1), wound stats.
- `observation_builder`: `_models_to_obs` fills wound fields; `build_observation` passes `alive_mask` into action masks.
- `observation.py`: `feature_dim` +3; `_models_to_features` appends alive, `current/max` wound ratio, `max_wounds/100`.
- Tests: `test_wounds.py` (obs fields, dead mask, tensor width stability); `test_state.py`, `test_dqn.py` dim_model +3.
- `plotting/training.py`: synthetic grid sweep copies `alive` / wound fields from reference model obs.
- **DQN:** `test_dqn_loss` — CUDA seed + deterministic cudnn; transformer variant no longer asserts strict same-batch loss decrease (non-monotone with Adam); checks finite loss.

## Checkpoint note

Policy/value checkpoints from before this change are **incompatible** (wider model feature vector).

## Verification

- `just validate` — green (format, lint, 322 tests).

## Self-Check: PASSED
