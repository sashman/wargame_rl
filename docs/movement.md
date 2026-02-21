# Movement System

Wargame models use **polar coordinate movement** on a discrete integer grid. Each step, every model independently selects a direction and speed, or chooses to stay in place.

## Action Encoding

Each model's action is a single integer from `0` to `n_movement_angles × n_speed_bins`:

| Action | Meaning |
|--------|---------|
| `0` | Stay (no movement) |
| `1 .. N×S` | Move with a specific (angle, speed) pair |

For movement actions, the angle and speed indices are decoded as:

```
angle_idx = (action - 1) // n_speed_bins
speed_idx = (action - 1) %  n_speed_bins
```

With the default configuration (`n_movement_angles=16`, `n_speed_bins=6`), there are **97 discrete actions** per model (1 stay + 16 × 6 movement combinations).

## Direction

Angles are evenly spaced around the full circle starting at 0 radians (east / +x) and going counter-clockwise:

```
angle = 2π × angle_idx / n_movement_angles
```

With 16 angular bins, each bin is 22.5° apart:

| Index | Angle | Direction |
|-------|-------|-----------|
| 0 | 0° | East |
| 1 | 22.5° | ENE |
| 2 | 45° | NE |
| 3 | 67.5° | NNE |
| 4 | 90° | North |
| 5 | 112.5° | NNW |
| 6 | 135° | NW |
| 7 | 157.5° | WNW |
| 8 | 180° | West |
| 9 | 202.5° | WSW |
| 10 | 225° | SW |
| 11 | 247.5° | SSW |
| 12 | 270° | South |
| 13 | 292.5° | SSE |
| 14 | 315° | SE |
| 15 | 337.5° | ESE |

## Speed

Speed bins are linearly spaced from `max_move_speed / n_speed_bins` up to `max_move_speed`:

```
speed = max_move_speed × (speed_idx + 1) / n_speed_bins
```

With the defaults (`max_move_speed=6`, `n_speed_bins=6`), the available speeds are 1, 2, 3, 4, 5, 6 cells per step.

## Displacement Calculation

The continuous displacement is computed from the angle and speed, then **rounded to the nearest integer** so that model locations remain on the discrete grid:

```
dx = round(speed × cos(angle))
dy = round(speed × sin(angle))
```

All displacements are pre-computed at environment initialization for efficiency. After adding the displacement to the model's current location, the result is **clamped** to the board boundaries `[0, 0]` to `[board_width - 1, board_height - 1]`.

## Configuration

Movement parameters are set via `WargameEnvConfig`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_movement_angles` | `16` | Number of angular bins (22.5° apart) |
| `n_speed_bins` | `6` | Number of discrete speed levels |
| `max_move_speed` | `6.0` | Maximum cells a model can move per step |

These can be overridden in YAML environment config files:

```yaml
n_movement_angles: 16
n_speed_bins: 6
max_move_speed: 6.0
```

## Future: Per-Model Speed

The system is designed so that `max_move_speed` can become a per-model attribute. In that case, speed bins would represent **fractions** of each model's individual max speed rather than absolute values. The action space stays uniform across all models — "speed bin 3 of 6" means "move at 50% of my max speed" regardless of the model's actual maximum. This keeps the DQN architecture unchanged while allowing heterogeneous unit speeds.
