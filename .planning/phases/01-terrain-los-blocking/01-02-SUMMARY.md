# 01-02 Execution Summary

## DDD Wiring

Terrain flows through the full DDD spine:

```
battle_factory.from_config()
  → builds Footprint list from config.terrain
  → constructs Terrain(footprints)
  → passes terrain= to Battle()

Battle._terrain  →  Battle.terrain property
BattleView Protocol  →  terrain: Terrain property
WargameEnv.terrain  →  delegates to self._battle.terrain
```

`Terrain` is static per episode — `reset_for_episode` does not touch it.

## Endpoint-Aware LOS Seam

`_make_is_blocking(x0, y0, x1, y1)` builds a per-query blocking predicate:

1. Asks `Terrain.blocking_footprints_for_endpoints(x0, y0, x1, y1)` for the
   "active" footprints — those containing **neither** endpoint (10e see-out /
   see-into rule, evaluated per ruin independently).
2. A cell is blocking if `config.blocking_mask[y][x]` is True **OR** the cell
   is contained by any active footprint.
3. `has_line_of_sight_between_cells` canonicalises endpoints via
   `sorted([(x0,y0),(x1,y1)])` before calling `has_line_of_sight`, guaranteeing
   symmetry: `has_los(A,B) == has_los(B,A)`.

`domain/los.py` is **completely untouched** — all terrain logic lives in the
seam layer (`wargame.py`) and the domain model (`terrain.py`).

## BattleView.terrain Contract

`BattleView` Protocol now exposes `terrain: Terrain`. Consumers (renderers,
reward calculators, Phase 2 observation pipeline) can read footprints without
depending on `WargameEnv` or `Battle` directly.

## Renderer Helper: `los_line_color`

Module-level pure function in `renders/human.py`:

```python
def los_line_color(view: BattleView, x0, y0, x1, y1) -> tuple[int,int,int]:
    # green (80,200,80) if clear, red (255,80,80) if blocked
```

`_draw_debug_los_line` now uses this helper, so the debug LOS line reflects
terrain blocking (green = clear, red = blocked).

`_draw_terrain` renders each footprint as a translucent brown rectangle with
outline and "Ruin" label, drawn after deployment zones and before models.

## Demo Config

`examples/env_config/terrain_los_demo.yaml` — 60×44 board, 4v4, 3 objectives
along the center line, two ruin footprints flanking the middle objective. Good
for visual validation of LOS blocking and ruin rendering.

## Tests Added

### `tests/test_los.py` (8 new tests)
- `test_terrain_los_blocked_between_outside_models` — both outside → blocked
- `test_terrain_los_see_into_target_inside` — target inside → clear
- `test_terrain_los_see_out_observer_inside` — observer inside → clear
- `test_terrain_los_per_ruin_other_ruin_still_blocks` — inside A, B between → blocked
- `test_terrain_los_off_line_footprint_unaffected` — footprint off-line → clear
- `test_terrain_los_interior_only_endpoint_footprint_does_not_block` — endpoint cell → clear
- `test_terrain_los_blocking_mask_and_footprint_coexist` — OR semantics
- `test_terrain_los_symmetry` — Hypothesis property test (200 examples)

### `tests/test_shooting_resolution.py` (1 new test)
- `test_terrain_shooting_mask_blocks_through_footprint` — mask forbids through footprint

### `tests/test_env.py` (1 new test)
- `test_terrain_movement_through_footprint` — TERR-05: movement not blocked

### `tests/test_terrain_render.py` (3 new tests)
- `test_los_line_color_blocked` — red when blocked
- `test_los_line_color_clear` — green when clear
- `test_terrain_los_demo_config_loads` — demo YAML → env with 2 footprints

## Deviations

None. All tasks completed as specified.
