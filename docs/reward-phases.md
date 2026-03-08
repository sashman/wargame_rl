# Reward Phases

Reward phases implement **curriculum learning** for the training loop. Instead of a fixed reward function for the entire training run, reward phases let you define an ordered sequence of reward configurations that the agent progresses through as it learns. Each phase specifies its own reward calculators, success criteria, and advancement threshold.

## Motivation

Teaching an agent to play a full wargame in one shot is hard. The reward signal for complex goals (capture objectives while maintaining group cohesion while shooting opponents) is too sparse for an untrained agent to learn from. Reward phases break this into stages:

1. **Group up** -- learn to keep units together
2. **Move to objectives** -- learn to navigate toward goals while staying grouped
3. **Engage opponents** -- learn to shoot while doing everything above
4. **Win the game** -- optimise for Victory Points

Each phase uses a simpler reward that the agent can learn quickly, then advances to a harder phase once it has mastered the current one.

## Configuration

Reward phases are configured via the `reward_phases` field in the environment YAML config. When this field is absent (or `null`), the environment uses the original legacy `Reward` class -- no behaviour change.

### Minimal example

```yaml
reward_phases:
  - name: group_up
    reward_calculators:
      - type: group_cohesion
        weight: 1.0
        params: { group_max_distance: 5.0, violation_penalty: -1.0 }
    success_criteria:
      type: all_models_grouped
      params: { max_distance: 5.0 }
    success_threshold: 0.8
    min_epochs: 10

  - name: reach_objectives
    reward_calculators:
      - type: closest_objective
        weight: 1.0
      - type: group_cohesion
        weight: 0.3
        params: { group_max_distance: 5.0, violation_penalty: -1.0 }
    success_criteria:
      type: all_at_objectives
    success_threshold: 0.8
    min_epochs: 20
```

### Phase fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | string | *required* | Human-readable phase name, logged to wandb |
| `reward_calculators` | list | *required* | One or more reward calculators active during this phase |
| `success_criteria` | object | *required* | Criteria that determines whether an episode counts as successful |
| `success_threshold` | float | `0.8` | Fraction of evaluation episodes (0--1) that must succeed to advance |
| `min_epochs` | int | `0` | Minimum epochs spent in this phase before advancement is eligible |

### Reward calculator fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `type` | string | *required* | Registry key identifying the calculator class |
| `weight` | float | `1.0` | Multiplier applied to this calculator's output |
| `params` | dict | `{}` | Keyword arguments forwarded to the calculator constructor |

### Success criteria fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `type` | string | *required* | Registry key identifying the criteria class |
| `params` | dict | `{}` | Keyword arguments forwarded to the criteria constructor |

## Available Reward Calculators

| Type key | Scope | Parameters | Description |
|----------|-------|------------|-------------|
| `closest_objective` | per-model | *(none)* | +1.0 at objective, +0.5 closer, -0.05 no change, -0.5 farther. Distance normalised by board diagonal. |
| `group_cohesion` | per-model | `group_max_distance` (float, default 10.0), `violation_penalty` (float, default -10.0) | Negative reward proportional to excess distance beyond `group_max_distance` from the closest same-group model. 0 when within range or alone in group. |
| `objective_control` | global | *(none)* | Reward equal to VP earned this step from controlling objectives (5 VP per objective controlled, cap 15 per turn). Only non-zero at end of Command phase from round 2 onwards, matching the primary mission. Requires `objective_control_range` and env VP state. |

## Available Success Criteria

| Type key | Parameters | Description |
|----------|------------|-------------|
| `all_at_objectives` | *(none)* | Succeeds when every model is within the radius of at least one objective. |
| `all_models_grouped` | `max_distance` (float, default 10.0) | Succeeds when every model is within `max_distance` of at least one same-group member. Models alone in their group are considered grouped. |

## How Advancement Works

At the end of each training epoch, the training loop runs evaluation episodes (controlled by `n_episodes` in `DQNConfig`). For each episode, the active phase's success criteria is checked. The resulting success rate is compared against the phase's `success_threshold`:

```
advance if:
    success_rate >= success_threshold
    AND epochs_in_current_phase >= min_epochs
    AND current phase is not the final phase
```

When advancement triggers, the `RewardPhaseManager` moves to the next phase and logs the transition. The new phase's reward calculators take effect immediately for subsequent episodes.

**Success rate for advancement:** Both PPO and DQN use the current phase's success criteria (e.g. `all_at_objectives`) evaluated on the **last step** of each evaluation episode, then take the mean over episodes. Using "episode ended early" (steps &lt; max_turns) alone would be wrong when the game can also end by round limit, so the phase's `check_success(env, last_step_context)` is used.

The `reward_phase` metric (phase index, 0-based) is logged to wandb every epoch, making phase transitions visible in the training dashboard.

## Reward Aggregation

Each step, the phase manager computes the reward as follows:

1. **Per-model calculators**: For each model, every per-model calculator's output is multiplied by its `weight` and summed. The per-model totals are then **averaged across all models**.
2. **Global calculators**: Each global calculator's output is multiplied by its `weight` and summed.
3. **Final reward** = averaged per-model reward + global reward total.

## Backward Compatibility

The reward phases system is fully opt-in:

- When `reward_phases` is absent from the YAML config, the environment uses the original `Reward` class and `get_termination()` function with zero code path changes.
- All existing YAML configs continue to work without modification.
- The original `reward.py` and `termination.py` files are unmodified and remain the canonical legacy path.

To replicate the original reward behaviour using phases, set `group_cohesion_enabled: false` (to disable the legacy path) and configure a single phase with both calculators:

```yaml
group_cohesion_enabled: false

reward_phases:
  - name: move_and_group
    reward_calculators:
      - type: closest_objective
        weight: 1.0
      - type: group_cohesion
        weight: 1.0
        params: { group_max_distance: 5.0, violation_penalty: -0.1 }
    success_criteria:
      type: all_at_objectives
    success_threshold: 0.8
    min_epochs: 0
```

## Adding New Calculators and Criteria

To add a new reward calculator:

1. Create a class in `wargame_rl/wargame/envs/reward/calculators/` that extends `PerModelRewardCalculator` or `GlobalRewardCalculator`.
2. Implement the `calculate()` method. Constructor parameters become the `params` dict values in YAML.
3. Register it in `calculators/registry.py` by adding an entry to `CALCULATOR_REGISTRY`.

To add a new success criteria:

1. Create a class in `wargame_rl/wargame/envs/reward/criteria/` that extends `SuccessCriteria`.
2. Implement the `is_successful()` method.
3. Register it in `criteria/registry.py` by adding an entry to `CRITERIA_REGISTRY`.

Both calculators and criteria receive a `StepContext` object containing the distance cache, turn info, and board dimensions. As new game mechanics are added (combat, terrain, VP), additional fields will be added to `StepContext` without changing existing calculator signatures.

### StepContext fields

| Field | Type | Description |
|-------|------|-------------|
| `distance_cache` | `DistanceCache` | Pre-computed distances between models and objectives |
| `current_turn` | `int` | Step counter (increments each `env.step()` call; with default `skip_phases`, each step is one active phase, currently movement only) |
| `max_turns` | `int` | Maximum agent steps per episode (`n_rounds × (5 - len(skip_phases))`; default `n_rounds` since non-movement phases are skipped) |
| `board_width` | `int` | Board width in cells |
| `board_height` | `int` | Board height in cells |
| `current_round` | `int` | Current battle round (1-based) |
| `battle_phase` | `BattlePhase` | Current battle phase (`command`, `movement`, `shooting`, `charge`, or `fight`) |

## File Layout

```
wargame_rl/wargame/envs/reward/
  reward.py                        # Legacy reward (untouched)
  step_context.py                  # StepContext dataclass
  phase.py                         # Pydantic config models
  phase_manager.py                 # RewardPhaseManager
  calculators/
    base.py                        # PerModelRewardCalculator, GlobalRewardCalculator ABCs
    closest_objective.py           # Closest-objective reward
    group_cohesion.py              # Group cohesion penalty
    registry.py                    # Type-string -> class mapping
  criteria/
    base.py                        # SuccessCriteria ABC
    all_at_objectives.py           # All models at objectives
    all_models_grouped.py          # All models within group distance
    registry.py                    # Type-string -> class mapping
  types/
    model_rewards.py               # Legacy ModelRewards (untouched)
```
