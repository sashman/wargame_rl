# Phase 2: Alive-Aware Observation — Research

## RESEARCH COMPLETE

**Date:** 2026-04-04

## Summary

Phase 2 extends the **same** observation pipeline used by DQN/PPO: structured `WargameModelObservation` → `build_observation` / `build_info` → `_observation_to_numpy` → tensors. Networks already size from `observation.size` and `observation.size_wargame_models[0]` (MLP and Transformer), so **no fixed magic widths** in `net.py` beyond what the env reports.

## Key files

| Area | Path | Role |
|------|------|------|
| Model obs row | `wargame_rl/wargame/envs/types/model_observation.py` | `size` property drives flat obs size |
| Builder | `wargame_rl/wargame/envs/env_components/observation_builder.py` | `_models_to_obs`, `build_observation`, `build_info` |
| Gym space | `wargame_rl/wargame/envs/domain/entities.py` | `WargameModel.to_space()` Dict schema |
| Tensors | `wargame_rl/wargame/model/common/observation.py` | `_models_to_features`, `feature_dim`, batch stack |
| Env facade | `wargame_rl/wargame/envs/wargame.py` | `observation_space` Tuple of `to_space()` |
| Nets | `wargame_rl/wargame/model/net.py` | `from_env` reads `observation.size*` |

## Findings

1. **`feature_dim`** is computed once in `_observation_to_numpy` as `2 + n_objectives*2 + max_groups + 1`; add **+3** for alive/wound channels (per `02-CONTEXT.md` D-05).
2. **`_same_group_closest_distance`** uses all `locs` in the observation tensor path; dead models still have finite locations. Cohesion signal for dead slots is **non-authoritative**; alive/wound channels carry elimination signal. Changing cohesion calc to alive-mask would be extra scope — **not required** if CONTEXT keeps “append 3 floats” only; executor should **not** change `_same_group_closest_distance` unless tests show bad gradients (defer).
3. **`build_observation`** must receive the **same** `DistanceCache` as `step()` when updating `distances_to_objectives` so dead models get `inf` rows consistent with D-02 (already how `_get_obs` works if cache passed through).
4. **Tests** likely to need updates: any assert on `model.size`, `observation.size`, tensor channel counts, or snapshot of `WargameModelObservation` fields — search `size_wargame_models`, `feature_dim`, `WargameModelObservation(`.

## Risks

- **Breaking checkpoint compatibility** (accepted in CONTEXT D-09).
- **Gym Dict space** must stay consistent with what `reset()`/`step()` return if any code validates against `observation_space`.

## Recommended plan shape

Single execute plan: types + `to_space` → builder + mask → tensor pipeline → tests + grep for regressions.
