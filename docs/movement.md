# Movement & Action Space

## Union Action Space

Every model shares a single flat discrete action space. The space is partitioned into contiguous **slices**, each owned by an action type and tagged with the battle phases where it is valid. An `ActionRegistry` manages these slices and produces phase-aware boolean masks so the agent (and opponent policies) never select illegal actions.

Current slices:

| Slice | Indices | Valid phases |
|-------|---------|--------------|
| `stay` | `0` | All phases |
| `movement` | `1 .. N×S` | Movement phase only |

With the defaults (`n_movement_angles=16`, `n_speed_bins=6`), the total action space is **97** (1 stay + 96 movement).

### Action Masking

During each step, the environment generates a `(n_models, n_actions)` boolean mask based on the current `BattlePhase`. The mask is:

- Attached to the observation (`WargameEnvObservation.action_mask`).
- Threaded through the DQN tensor pipeline as a `torch.bool` tensor.
- Applied during **greedy** action selection (invalid Q-values set to `-inf` before argmax).
- Applied during **random** exploration (sampling restricted to valid indices).
- Applied in the **DQN loss** to mask target Q-values for next-state value estimation.

### Extending with New Phases

To add actions for a new phase (e.g. shooting, charging), register a new slice in `ActionHandler.__init__`:

```python
self._registry.register(
    "shooting",
    n_shooting_actions,
    frozenset({BattlePhase.shooting}),
)
```

This appends the new actions after the existing slices. The mask generation, observation pipeline, and DQN output layer automatically account for the larger `n_actions` — no other wiring changes are needed beyond implementing the action application logic itself.

An action can be valid in multiple phases by including them in the `valid_phases` frozenset (e.g. `stay` is valid in all phases).

## Movement Encoding

Each model's movement action is a single integer from `1` to `n_movement_angles × n_speed_bins` (index `0` is the phase-universal stay action):

| Action | Meaning |
|--------|---------|
| `0` | Stay (no movement) |
| `1 .. N×S` | Move with a specific (angle, speed) pair |

For movement actions, the angle and speed indices are decoded as:

```
angle_idx = (action - 1) // n_speed_bins
speed_idx = (action - 1) %  n_speed_bins
```

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
