# Terrain & Line-of-Sight Blocking

## Overview

Terrain (Ruins) blocks **line of sight only**, following the Warhammer 40k 10th Edition Ruins abstraction: a terrain piece is a **footprint rectangle** and the footprint itself is the LOS blocker. Movement is unaffected — models move through and occupy footprint cells freely.

A ruin blocks the line between two models only when its footprint lies between them and **both** models are outside it. A model inside a ruin can see out and be seen into (the 10e see-out / see-into exceptions, evaluated per ruin independently).

## Configuration

Terrain is configured in YAML via the `terrain` key on `WargameEnvConfig`. Each terrain piece is a footprint rectangle given as two opposite corners `[x0, y0, x1, y1]` in absolute board coordinates.

```yaml
board_width: 60
board_height: 44

terrain:
  - { footprint: [27, 8, 33, 16] }
  - { footprint: [27, 28, 33, 36] }
```

### Validation

Footprint corners are normalised (so `x0 ≤ x1`, `y0 ≤ y1`) at config load. The following are rejected with a clear error:

- A corner that falls outside the board dimensions
- Two footprints that overlap each other

Overlap with deployment zones and objectives is explicitly allowed.

### No-Terrain Default

When `terrain` is omitted or `null`, the environment behaves exactly as before — no terrain objects are created, no observation tokens are added, and pre-terrain checkpoints continue to load and infer correctly.

## Domain Model

The pure domain lives in `domain/terrain.py`:

| Class | Description |
|-------|-------------|
| `Footprint` | Frozen dataclass: `x0, y0, x1, y1` with `contains(x, y)` and `from_corners()` factory |
| `Terrain` | Collection of footprints; `blocking_footprints_for_endpoints(x0, y0, x1, y1)` returns footprints containing **neither** endpoint (the 10e filter) |

`Terrain` is static per episode — it is constructed from config during `battle_factory.from_config()` and does not change during play.

## DDD Wiring

```
WargameEnvConfig.terrain  →  battle_factory  →  Battle.terrain
Battle.terrain  →  BattleView.terrain (Protocol property)
WargameEnv  →  delegates to Battle (implements BattleView)
```

`BattleView.terrain` is the read-only interface that renderers, reward calculators, and the observation pipeline use to access terrain without depending on the full env.

## Line-of-Sight Blocking

LOS blocking is **endpoint-aware**: the blocking predicate is evaluated per query, not pre-computed as a static grid.

### Algorithm

For a query from cell `(x0, y0)` to cell `(x1, y1)`:

1. Ask `Terrain.blocking_footprints_for_endpoints(x0, y0, x1, y1)` for "active" footprints — those containing **neither** endpoint.
2. A cell along the Bresenham ray is blocking if:
   - `config.blocking_mask[y][x]` is True (legacy static blocking), **OR**
   - The cell is contained by any active footprint.
3. Endpoint order is canonicalised (`sorted([(x0,y0),(x1,y1)])`) before the call to `has_line_of_sight`, guaranteeing **symmetry**: `has_los(A, B) == has_los(B, A)`.

The Bresenham core in `domain/los.py` is **untouched** — all terrain logic lives in the seam layer (`wargame.py`) and the domain model (`terrain.py`).

### See-Into / See-Out Rules

The 10e Ruins rules state that a model inside a ruin can see out, and models outside can see into a ruin. This is implemented by the "neither endpoint" filter: a footprint only participates in blocking when both the observer and the target are outside it.

| Observer | Target | Footprint between them? | LOS |
|----------|--------|------------------------|-----|
| Outside | Outside | Yes | **Blocked** |
| Inside | Outside | N/A (observer inside → footprint excluded) | Clear |
| Outside | Inside | N/A (target inside → footprint excluded) | Clear |
| Inside | Inside | N/A (both inside → footprint excluded) | Clear |

Each footprint is evaluated independently. A model inside ruin A can still have LOS blocked by ruin B if both endpoints are outside B and B lies on the ray.

### Integration

Because all LOS queries route through the single `has_line_of_sight_between_cells` seam on `BattleView`, the following all agree on the same terrain blocking:

- Shooting masks (action masking)
- Shooting resolution (damage)
- Renderer debug LOS overlay
- Any future LOS consumer

## Movement

Terrain does **not** affect movement. Models can move through and occupy footprint cells freely. This is verified by `test_terrain_movement_through_footprint` in `tests/test_env.py`.

## Observation

Terrain footprints are encoded in the agent's observation as **entity tokens** — one token per footprint, appended after opponent models.

### Token Layout

Each terrain token carries normalised footprint geometry:

| Feature | Description |
|---------|-------------|
| `x0_norm` | Left edge, normalised to [-1, 1] |
| `y0_norm` | Top edge, normalised to [-1, 1] |
| `x1_norm` | Right edge, normalised to [-1, 1] |
| `y1_norm` | Bottom edge, normalised to [-1, 1] |

Normalisation uses `(corner - half_board) / half_board`, consistent with model and objective location normalisation.

### Tensor Pipeline

The observation tensor pipeline returns 6 tensors:

| Index | Tensor | Shape |
|-------|--------|-------|
| 0 | Game features | `(6,)` |
| 1 | Objectives | `(n_objectives, 2)` |
| 2 | Player models | `(n_models, feature_dim)` |
| 3 | Opponent models | `(n_opponents, feature_dim)` |
| 4 | **Terrain** | `(n_terrain, 4)` |
| 5 | Action mask | `(n_models, n_actions)` |

With no terrain configured, the terrain tensor has shape `(0, 4)` — zero rows, fixed width. This ensures no mid-episode observation shape change.

### Network Integration

**TransformerNetwork**: A `terrain_embedding` linear layer projects terrain tokens into the transformer embedding space. Terrain tokens are appended **last** in the token sequence (after opponents) and are always attendable (no alive/dead masking). When `terrain_size == 0` (no terrain), `terrain_embedding` is `None` and no tokens are appended — the network state dict is identical to a pre-terrain checkpoint.

Token sequence: `[game, objectives..., players..., opponents..., terrain...]`

Player and opponent token positions are unchanged by the presence of terrain, so per-model action heads and the critic token are unaffected.

**MLPNetwork**: Terrain features are included in the flat concatenation of all state tensors. With no terrain the terrain tensor contributes zero elements.

## Rendering

The renderer draws terrain footprints as translucent brown rectangles with an outline and "Ruin" label, drawn after deployment zones and before models. The debug LOS overlay line is coloured by the actual blocked/clear verdict (green = clear, red = blocked).

## YAML Example

A complete config with terrain (`examples/env_config/terrain_los_demo.yaml`):

```yaml
config_name: terrain_los_demo
board_width: 60
board_height: 44
number_of_wargame_models: 4
number_of_opponent_models: 4
number_of_objectives: 3
number_of_battle_rounds: 5

deployment_zone: [0, 0, 20, 44]
opponent_deployment_zone: [40, 0, 60, 44]

opponent_policy:
  type: scripted_advance_to_objective

terrain:
  - { footprint: [27, 8, 33, 16] }
  - { footprint: [27, 28, 33, 36] }

objectives:
  - { x: 30, y: 6 }
  - { x: 30, y: 22 }
  - { x: 30, y: 38 }
```

Two ruin footprints flank the middle objective along the centre line. Models must navigate around or through the ruins to establish line of sight to opponents on the other side.

## Future Extensions

The following terrain features are deferred to later milestones:

| Feature | Description |
|---------|-------------|
| Walls | Thin L-shaped segments within a footprint for finer-grained LOS and rendering |
| Dense/Woods | "Not fully visible" partial obscuring distinct from the Ruins block |
| Cover bonus | +1 armour save vs ranged when not fully visible due to terrain |
| Difficult terrain | Movement speed penalty |
| Impassable terrain | Models cannot move through |
| Elevation | Height advantage (+AP from elevated positions) |
| Procedural placement | Board templates and random terrain for training variety |
