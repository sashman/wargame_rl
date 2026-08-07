# 01-01 Execution Summary

## Config Schema

### `TerrainPieceConfig` (in `wargame_rl/wargame/envs/types/config.py`)

| Field | Type | Description |
|-------|------|-------------|
| `footprint` | `tuple[int, int, int, int]` | Bounding rectangle `(x0, y0, x1, y1)` in grid cells |

### `WargameEnvConfig.terrain`

- Type: `list[TerrainPieceConfig] | None`
- Default: `None` (no terrain)
- Validation (via `validate_terrain` model_validator):
  - Normalises each footprint so `x0<=x1`, `y0<=y1`
  - Rejects footprints extending beyond board dimensions
  - Rejects overlapping pairs of footprints
  - Explicitly allows overlap with deployment zones and objectives

### Helper functions (module-level, private)

- `_normalise_rect(r) -> tuple[int,int,int,int]` — canonical min/max ordering
- `_rects_overlap(a, b) -> bool` — axis-aligned overlap test

## Pure Domain API (`wargame_rl/wargame/envs/domain/terrain.py`)

### `Footprint` (frozen dataclass, slots=True)

| Method | Signature | Description |
|--------|-----------|-------------|
| `contains` | `(x: int, y: int) -> bool` | Corner-inclusive containment test |
| `from_corners` | `classmethod(x0, y0, x1, y1) -> Footprint` | Factory with normalisation |

### `Terrain`

| Method | Signature | Description |
|--------|-----------|-------------|
| `__init__` | `(footprints: list[Footprint])` | Wrap a list of footprints |
| `footprints` | `@property -> list[Footprint]` | All terrain footprints |
| `blocking_footprints_for_endpoints` | `(x0, y0, x1, y1) -> list[Footprint]` | Footprints containing NEITHER endpoint (see-out filter) |

## Tests Added

### `tests/test_terrain.py` (4 tests)

- `test_footprint_contains_inclusive_of_corners`
- `test_footprint_normalises_unordered_corners`
- `test_blocking_footprints_for_endpoints_excludes_footprint_containing_endpoint`
- `test_terrain_empty_returns_no_blockers`

### `tests/test_los.py` (5 new tests)

- `test_terrain_config_parses_footprint`
- `test_terrain_config_default_none`
- `test_terrain_validation_off_board_corner_raises`
- `test_terrain_validation_overlapping_footprints_raises`
- `test_terrain_validation_overlap_with_zone_or_objective_allowed`

## Deviations from Plan

- Task 1 and Task 3 were committed together because the test file imports the domain module — keeping both GREEN from the start rather than having a RED intermediate state. The commit message reflects the combined scope.
- `hypothesis` was added as a dev dep (as planned) but not directly used in these tests; it's available for future property-based terrain tests in plan 01-02.
