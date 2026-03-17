# Opponent Policies

Opponent policies control how opponent models (units on the opposing side) select their actions each turn. The policy system is pluggable: a YAML config specifies **which** policy to use and its parameters, while the runtime resolves and instantiates the corresponding class.

## Configuration

Opponents are enabled by setting `number_of_opponent_models` to a value greater than 0 and providing an `opponent_policy` block. When there are no opponents, the environment behaves identically to before.

### Minimal example

```yaml
number_of_wargame_models: 4
number_of_opponent_models: 4
number_of_objectives: 3
objective_radius_size: 3
board_width: 60
board_height: 44

opponent_policy:
  type: random
```

### Full example with army composition

```yaml
number_of_wargame_models: 4
number_of_opponent_models: 4
number_of_objectives: 3
objective_radius_size: 3
board_width: 60
board_height: 44
turn_order: random

deployment_zone: [0, 0, 20, 44]
opponent_deployment_zone: [40, 0, 60, 44]

opponent_policy:
  type: scripted_advance_to_objective

models:
  - { group_id: 0 }
  - { group_id: 0 }
  - { group_id: 1 }
  - { group_id: 1 }

opponent_models:
  - { group_id: 0, max_wounds: 120 }
  - { group_id: 0, max_wounds: 120 }
  - { group_id: 1 }
  - { group_id: 1 }
```

### Config fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `number_of_opponent_models` | int | `0` | Number of opponent models. 0 disables opponents entirely. |
| `opponent_models` | list | `null` | Per-model config (reuses `ModelConfig`). Optional -- when absent, models get auto-assigned groups and default stats. Length must match `number_of_opponent_models`. |
| `turn_order` | string | `"player"` | Who moves first: `"player"`, `"opponent"`, or `"random"`. |
| `opponent_policy` | object | `null` | Policy engine config. **Required** when `number_of_opponent_models > 0`. |
| `opponent_deployment_zone` | list | right third of board | Deployment zone `[x_min, y_min, x_max, y_max]` for opponent placement. |

### Policy config fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `type` | string | *required* | Registry key identifying the policy class (see table below). |
| `params` | dict | `{}` | Keyword arguments forwarded to the policy constructor. |

### Opponent model config fields

Each entry in `opponent_models` uses `ModelConfig`, the same schema as player models:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `x` | int | `null` | X coordinate. If `null`, placed randomly in opponent deployment zone. |
| `y` | int | `null` | Y coordinate. Must be set together with `x` or both omitted. |
| `group_id` | int | `0` | Group this model belongs to. |
| `max_wounds` | int | `100` | Maximum wound pool for this model. |

## Turn Order

Each `env.step()` call advances the player through **one battle phase** (command, movement, shooting, charge, or fight). After the player completes their turn, the opponent's entire turn (all five phases) is auto-executed before the observation is returned. By default, non-movement phases are skipped (`skip_phases` config), so the player takes 1 step per round. Set `skip_phases: []` for full per-phase stepping (5 steps per round).

The `turn_order` field controls which side takes the first turn each round:

| Value | Behaviour |
|-------|-----------|
| `player` | The RL agent takes the first turn each round (agent is `player_1` on the game clock). This is the default. |
| `opponent` | The opponent takes the first turn each round. On `reset()` and after each player turn, the opponent's turn is auto-executed before the agent acts. |
| `random` | A coin flip at each `reset()` determines which side goes first for the episode. Reproducible with a fixed seed. |

## Available Policies

### `random`

Each opponent model selects a uniformly random action from the action space (stay, or any angle/speed combination).

```yaml
opponent_policy:
  type: random
```

**Parameters:** none.

**Use case:** Baseline opponent for initial training. Provides unpredictable but non-strategic opposition, useful for verifying the environment works before introducing smarter opponents.

### `scripted_advance_to_objective`

Each opponent model moves toward the nearest objective. The policy computes the angle from each model to its closest objective and selects the polar-coordinate action with the best matching direction. Step length is capped by distance to the objective boundary so models reduce speed when close and do not overshoot.

```yaml
opponent_policy:
  type: scripted_advance_to_objective
```

**Parameters:** none.

**Use case:** Provides goal-directed opposition that competes for the same objectives as the player. Good for training agents that need to learn to reach objectives before the opponent does.

## Planned Policies (Not Yet Implemented)

The following policies are designed in the architecture but raise `NotImplementedError` if used:

| Type key | Description |
|----------|-------------|
| `human` | Read actions from the renderer (keyboard/mouse input). Enables human-vs-agent play. |
| `model` | Load a pre-trained DQN checkpoint and use it as the opponent. Enables self-play and agent-vs-agent evaluation. |

## Adding a New Policy

To add a new opponent policy:

1. Create a file in `wargame_rl/wargame/envs/opponent/` (e.g. `scripted_hold_position_policy.py`).
2. Define a class extending `OpponentPolicy` and implement `select_action()`:

```python
from wargame_rl.wargame.envs.opponent.policy import OpponentPolicy
from wargame_rl.wargame.envs.opponent.registry import register_policy
from wargame_rl.wargame.envs.types import WargameEnvAction


class ScriptedHoldPositionPolicy(OpponentPolicy):
    def __init__(self, env, **kwargs):
        self._env = env

    def select_action(self, opponent_models, env, action_mask=None):
        # Every model stays in place
        return WargameEnvAction(actions=[0] * len(opponent_models))


register_policy("scripted_hold_position", ScriptedHoldPositionPolicy)
```

3. Import the module in `registry.py`'s `_auto_register()` so it registers on startup:

```python
def _auto_register():
    import importlib
    for mod in (
        "wargame_rl.wargame.envs.opponent.random_policy",
        "wargame_rl.wargame.envs.opponent.scripted_advance_to_objective_policy",
        "wargame_rl.wargame.envs.opponent.scripted_hold_position_policy",  # new
    ):
        importlib.import_module(mod)
```

4. Use it in a YAML config:

```yaml
opponent_policy:
  type: scripted_hold_position
```

The `select_action` method receives the list of opponent `WargameModel` instances, the full `WargameEnv`, and optionally `action_mask` (phase-aware valid actions), giving access to objectives, board dimensions, the action handler's `best_action_toward()` helper, and any other env state needed to compute actions.

### Naming convention

Scripted policies are prefixed with `Scripted` in the class name and `scripted_` in the registry key (e.g. `ScriptedFlankPolicy` / `"scripted_flank"`). This distinguishes hand-coded behaviour from learned or external policies.

## Observation Impact

When opponents are present, the player agent's observation includes opponent model positions as a separate list (`opponent_models`). This is converted to 5 tensors in the DQN observation pipeline:

| Tensor index | Content | Shape |
|--------------|---------|-------|
| 0 | Game state | `(6,)` — placeholder, normalized_round, phase_index, player_vp, opponent_vp, player_vp_delta (see `envs/types/env_observation.py`, `model/common/observation.py`) |
| 1 | Objectives | `(n_objectives, 2)` |
| 2 | Player models | `(n_player_models, features)` |
| 3 | Opponent models | `(n_opponent_models, features)` |
| 4 | Action mask | `(n_models, n_actions)` — bool, valid actions per model |

When there are no opponents, tensor 3 has shape `(0, features)` and the network handles it gracefully (empty sequence for the Transformer, zero-width concatenation for the MLP).

## File Layout

```
wargame_rl/wargame/envs/opponent/
  __init__.py                                   # Module exports
  policy.py                                     # OpponentPolicy ABC
  registry.py                                   # Type-string -> class registry + factory
  random_policy.py                              # RandomPolicy
  scripted_advance_to_objective_policy.py        # ScriptedAdvanceToObjectivePolicy

examples/env_config/with_opponents/
  4v4_random_opponent.yaml                      # 4v4 with random opponent
  4v4_scripted_opponent.yaml                    # 4v4 with scripted opponent, random turn order
  2v2_fixed_positions.yaml                      # 2v2 with fixed positions for all entities
```
