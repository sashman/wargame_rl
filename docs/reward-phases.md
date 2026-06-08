# Reward Phases

Reward phases implement **curriculum learning** for the training loop. Instead of a fixed reward function for the entire training run, reward phases let you define an ordered sequence of reward configurations that the agent progresses through as it learns. Each phase specifies its own reward calculators, success criteria, and advancement threshold.

## Motivation

Teaching an agent to play a full wargame in one shot is hard. The reward signal for complex goals (capture objectives while maintaining group cohesion while shooting opponents) is too sparse for an untrained agent to learn from. Reward phases break this into stages:

1. **Group up** -- learn to keep units together
2. **Move to objectives** -- learn to navigate toward goals while staying grouped
3. **Engage opponents** -- learn to shoot while doing everything above
4. **Win the game** -- optimise for Victory Points

Each phase uses a simpler reward that the agent can learn quickly, then advances to a harder phase once it has mastered the current one.

**Phase order:** Put the "Win the game" (VP) phase **after** a phase where the agent learns to reach objectives (e.g. `move_and_group` with `all_at_objectives`, `closest_objective`, or `closest_objective_v2`). That way the agent already knows how to get to objectives before you ask it to score VP; the VP phase then focuses on holding control at scoring time.

## Configuration

Reward is always computed via reward phases. Configure the `reward_phases` field in the environment YAML; if omitted, a single default phase is used (reach objectives with `closest_objective`).

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
| `min_epochs_above_threshold` | int | `5` | Success rate must be ≥ success_threshold for this many consecutive epochs before advancing |
| `terminal_success_bonus` | float | `0.0` | Bonus added at episode end **when the phase's `success_criteria` is met** (previously hardcoded to all-models-at-objectives). Scaled by remaining-turn fraction so faster success gets higher reward. |
| `terminal_vp_bonus` | float | `0.0` | Bonus added at episode end when player VP meets the phase's VP threshold (for VP-based success criteria). |
| `terminate_on_success` | bool | `true` | Whether to end the episode as soon as all models are at objectives. Set `false` in VP phases when you want to keep scoring. |

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
| `closest_objective` | per-model | `progress_scale` (float, default 0.0), `best_distance_bonus_scale` (float, optional) | Shaping toward the nearest objective, normalised by board diagonal. **`progress_scale > 0` (recommended)** switches to a smooth linear potential pull (reward = `progress_scale × distance_closed`, positive when closing, negative when receding); speed-to-objective then comes from discounting + the remaining-fraction terminal bonus, not a convex term. Legacy mode (default): 0 when getting closer, a flat penalty when distance stays the same or increases, plus an optional cubic best-distance bonus. |
| `closest_objective_v2` | per-model | `progress_scale` (float, default 0.0), `fallback_to_nearest` (bool, default false), `best_distance_bonus_scale` (float, optional), `overstack_penalty_per_extra` (float, default 0.05), `non_improvement_penalty_slope` (float, default 2.0), `non_improvement_penalty_base` (float, default 0.3) | Targets the nearest objective where this model's arrival improves the player's control using the **same OC/count rule as VP scoring** (neutral→player, opponent→contested, contested→player); one objective assignment per group per step (de-stacks across models). **`progress_scale > 0` (recommended)** switches to a smooth linear potential pull toward the assigned objective (reward = `progress_scale × distance_closed`, positive when closing, negative when receding), replacing the legacy flat penalty + cubic bonus. **`fallback_to_nearest`** sends a model with no assigned target to its nearest objective (e.g. extra models when models > objectives). With `progress_scale = 0` it uses the legacy penalty/cubic shaping. Over-stacking an already-controlled objective incurs a small negative penalty. |
| `group_cohesion` | per-model | `group_max_distance` (float, default 10.0), `violation_penalty` (float, default -10.0) | Negative reward proportional to excess distance beyond `group_max_distance` from the closest same-group model. 0 when within range or alone in group. |
| `vp_gain` | global | *(none)* | Reward = weight × ((player_vp_delta - opponent_vp_delta) / cap_per_turn). `cap_per_turn` is read from mission config (default 15), so net VP swings are normalized to turn cap scale. Use in a "Win the game" phase. |
| `objective_flip_bonus` | global | `bonus_capture_first` (float, default 5.0), `bonus_flip_to_contested` (float, default 3.0), `bonus_contested_to_controlled` (float, default 5.0), `loss_penalty_scale` (float, default 1.0) | Symmetric control-state potential under the same OC/count rule as VP scoring. Gaining control adds value (neutral→player = `bonus_capture_first`; opponent→contested = `bonus_flip_to_contested`; contested→player = `bonus_contested_to_controlled`); losing control subtracts the mirror value × `loss_penalty_scale`. At `loss_penalty_scale=1.0` it is a pure (farming-proof) potential. Returns unweighted. |
| `objective_coverage` | global | *(none)* | Dense reward = (number of player-controlled objectives) / (number of objectives), using the same OC/count rule as VP scoring. Paid every step, so it rewards spreading models to hold *multiple distinct* objectives rather than over-stacking one. Returns unweighted. |

## Available Success Criteria

| Type key | Parameters | Description |
|----------|------------|-------------|
| `all_at_objectives` | *(none)* | Succeeds when every model is within the radius of at least one objective. |
| `all_models_grouped` | `max_distance` (float, default 10.0) | Succeeds when every model is within `max_distance` of at least one same-group member. Models alone in their group are considered grouped. |
| `player_vp_min` | `fraction_of_max` (float, e.g. 0.33), `min_vp` (int, default 0) | Succeeds when player VP at episode end ≥ threshold. Threshold = max(min_vp, round(fraction_of_max × theoretical_max)). Theoretical max depends on `number_of_battle_rounds`, objectives, and mission params, so the same fraction gives a higher VP bar when episodes have more rounds. |
| `player_ahead_on_vp` | *(none)* | Succeeds when `player_vp > opponent_vp` at evaluation time (a win-rate signal; use with `terminate_on_success: false`). |

## How Advancement Works

At the end of each training epoch, the training loop runs evaluation episodes (controlled by `n_episodes` in the active algorithm config — `PPOConfig` for the default PPO, or `DQNConfig` for DQN; overridable via `--n-eval-episodes`). For each episode, the active phase's success criteria is checked. The resulting success rate is compared against the phase's `success_threshold`:

```
advance if:
    success_rate >= success_threshold
    AND epochs_in_current_phase >= min_epochs
    AND success_rate has been >= success_threshold for the last min_epochs_above_threshold consecutive epochs
    AND current phase is not the final phase
```

When advancement triggers, the `RewardPhaseManager` moves to the next phase and logs the transition. The new phase's reward calculators take effect immediately for subsequent episodes.

The `reward_phase` metric (phase index, 0-based) is logged to wandb every epoch, making phase transitions visible in the training dashboard.

## Reward Aggregation

Each step, the phase manager computes the reward as follows:

1. **Per-model calculators**: For each **alive** model, every per-model calculator's output is multiplied by its `weight` and summed. The per-model totals are then **averaged across the alive models** (dead models are excluded; if no models are alive the per-model contribution is 0).
2. **Global calculators**: Each global calculator's output is multiplied by its `weight` and summed.
3. **Final reward** = averaged per-model reward + global reward total.

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
| `is_terminated` | `bool` | Whether this step terminates the episode (used to gate terminal bonuses). Default `False` |
| `current_round` | `int` | Current battle round (1-based) |
| `battle_phase` | `BattlePhase` | Current battle phase (`command`, `movement`, `shooting`, `charge`, or `fight`) |
| `player_damage_dealt` | `int` | Damage the player dealt this step. Default `0` |
| `opponent_damage_dealt` | `int` | Damage the opponent dealt this step. Default `0` |
| `player_models_killed` | `int` | Opponent models the player eliminated this step. Default `0` |
| `opponent_models_killed` | `int` | Player models the opponent eliminated this step. Default `0` |

## File Layout

```
wargame_rl/wargame/envs/reward/
  step_context.py                  # StepContext dataclass
  phase.py                         # Pydantic config models
  phase_manager.py                 # RewardPhaseManager
  calculators/
    base.py                        # PerModelRewardCalculator, GlobalRewardCalculator ABCs
    closest_objective.py           # Closest-objective reward
    closest_objective_v2.py        # OC/count-margin closest-objective reward
    group_cohesion.py              # Group cohesion penalty
    objective_coverage.py          # Dense fraction-of-objectives-controlled reward
    objective_flip_bonus.py        # Symmetric objective control-state potential
    vp_gain.py                     # VP gain reward (global)
    registry.py                    # Type-string -> class mapping
  criteria/
    base.py                        # SuccessCriteria ABC
    all_at_objectives.py           # All models at objectives
    all_models_grouped.py          # All models within group distance
    player_ahead_on_vp.py          # Player ahead on VP (win-rate) criteria
    player_vp_min.py               # Player VP min success criteria
    registry.py                    # Type-string -> class mapping
  types/
    model_rewards.py               # ModelRewards (per-model reward breakdown)
```
