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
| `terminal_success_bonus` | float | `0.0` | Bonus added at episode end **when the phase's `success_criteria` is met** (previously hardcoded to all-models-at-objectives). Scaled by remaining-turn fraction **only when `terminate_on_success` is true** — see below. |
| `terminal_vp_bonus` | float | `0.0` | Bonus added at episode end when player VP meets the phase's VP threshold (for VP-based success criteria). |
| `terminate_on_success` | bool | `true` | Whether to end the episode as soon as **this phase's `success_criteria`** is met. Set `false` in VP phases when you want to keep scoring — and read the warning below, because this flag changes what `success_rate` measures. |

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
| `models_at_objectives` | global | *(none)* | Dense reward = (alive models within some objective radius) / (alive models). The step-wise counterpart of the `fraction_at_objectives` criteria — unlike `objective_coverage` it does **not** saturate once a point is controlled, so it keeps paying as more models arrive. Dead models leave both numerator and denominator. Returns unweighted. **Not recommended:** every competent scripted baseline saturates it at 1.000, so it scores a 0.53-win policy and a 0.78-win policy identically. Prefer `objective_hold`. |
| `objective_hold` | **per-model** | `player_value` (float, default 1.0), `contested_value` (float, default 0.5), `opponent_value` (float, default 0.25) | Value of the objective this model occupies, keyed on control state under the same OC/count rule as VP scoring; 0 when off every objective, and 0 on a neutral one since holding a point nobody contests is worth nothing. The **only calculator that pays a model while it is stationary** — `closest_objective`'s progress is a potential that exhausts on arrival and is exactly 0 on shooting steps, and `group_cohesion` returns 0 inside its limit, so without this all models share an identical reward for most of an episode. Pays for *controlling* rather than standing, and supplies a gradient across `neutral → contested → player`, which `objective_coverage` and `vp_gain` are both flat across because control is a strict count comparison. |
| `model_kills` | **per-model** | `bonus_per_kill` (float, default 2.0) | Bonus for each opponent **this model** killed this step. The per-model counterpart of `killing`: under that global term every model is paid the same whether it fired, missed, or stood still, leaving the shooting head — half the agent's decisions — with no credit path. Note the scale difference: `killing` broadcasts its bonus to all N models per kill, so its default 5.0 makes a 25-model wipe worth 125 per model. |
| `killing` | global | `bonus_killing_opponent` (float, default 5.0) | Bonus × opponent models killed this step. **Prefer `model_kills`:** this spreads kill credit flat across the army, and at the default bonus it is easily the largest term in a phase. |

### Choosing calculators

Two rules, both learned by measurement rather than taste:

- **Every phase needs at least one per-model calculator.** Only `closest_objective`, `closest_objective_v2`, `group_cohesion`, `objective_hold` and `model_kills` are per-model; the rest are broadcast bit-identically to every model. A phase built purely from global terms gives all models the same reward and undoes per-model credit assignment by configuration.
- **Every phase should keep `vp_gain`.** A rung that drops the goal signal trains away from it: win rate fell 62% → 47% when a curriculum advanced into a phase that rewarded occupancy and had no VP term, while `success_rate` held at ~80%.

`tests/test_curriculum_configs.py` enforces both over the two shipped training configs — `examples/env_config/25v25_single_phase.yaml` (the single-phase control) and `25v25_curriculum.yaml` (the two-rung arm, whose final phase is identical to the control's). It also pins three narrower invariants read off past measurements: no phase may use `models_at_objectives`, a `player_vp_min` gate must sit above the 95/285 fraction a one-objective stack reaches unaided, and any phase setting `terminal_success_bonus` must keep `terminate_on_success: false` so the bonus is not scaled away.

## Available Success Criteria

| Type key | Parameters | Description |
|----------|------------|-------------|
| `all_at_objectives` | *(none)* | Succeeds when every model is within the radius of at least one objective. Dead models count as satisfied. **Scales badly with army size** — see the note below; prefer `fraction_at_objectives` beyond a handful of models. |
| `fraction_at_objectives` | `min_fraction` (float in (0, 1], default 0.5) | Succeeds when at least `min_fraction` of **alive** models are within the radius of an objective. Dead models are excluded from numerator and denominator alike, unlike `all_at_objectives`. |
| `all_models_grouped` | `max_distance` (float, default 10.0) | Succeeds when every model is within `max_distance` of at least one same-group member. Models alone in their group are considered grouped. |
| `player_vp_min` | `fraction_of_max` (float, e.g. 0.33), `min_vp` (int, default 0) | Succeeds when player VP at episode end ≥ threshold. Threshold = max(min_vp, round(fraction_of_max × theoretical_max)). Theoretical max depends on `number_of_battle_rounds`, objectives, and mission params, so the same fraction gives a higher VP bar when episodes have more rounds. |
| `player_ahead_on_vp` | *(none)* | Succeeds when `player_vp > opponent_vp` at evaluation time (a win-rate signal; use with `terminate_on_success: false`). |

### Choosing an at-objectives criteria

`all_at_objectives` requires **every** alive model inside a radius on the same
step. If each model independently has probability `p` of being on an objective,
success needs `p**n_models` — which collapses as the army grows:

| per-model accuracy | 4 models | 25 models |
|---|---|---|
| 0.90 | 0.66 | 0.07 |
| 0.95 | 0.81 | 0.28 |
| 0.99 | 0.96 | 0.78 |

A 25v25 run held `success_rate` at exactly 0 for 330 epochs while every other
metric improved. Lowering `success_threshold` does not help: that tunes how many
*episodes* must succeed, not how many *models* must arrive. Use
`fraction_at_objectives` and raise `min_fraction` across phases instead.

**Set `min_fraction` from measurement, not intuition.** Success is evaluated on
the episode's final step, so measure the final-step fraction your current policy
reaches and pick a bar just above it.

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

A training run holds several environments — one per rollout worker plus the evaluation envs — and each builds its own `RewardPhaseManager`, because calculators carry per-episode state (`closest_objective`'s previous distance, `objective_flip_bonus`'s potential) that one env resetting would corrupt for the others. What they share is a single `CurriculumPosition` (`reward/phase_manager.py`), passed in as `WargameEnv(..., phase_position=...)`, so one advance moves every env at once. Sharing the position rather than propagating an index means there is no synchronisation step to forget — which is how the rollout envs came to train on phase 0 for every run to date while `reward_phase` reported otherwise.

The `reward_phase` metric (phase index, 0-based) is logged to wandb every epoch, making phase transitions visible in the training dashboard.

### Size `min_epochs_above_threshold` against `n_episodes`

Each epoch's `success_rate` is an `n_episodes`-sample binomial, not a measurement of the policy's true rate. Requiring *consecutive* epochs above the bar multiplies that noise, so the effective gate sits well above the nominal `success_threshold`:

| True rate | P(one epoch ≥ 0.7) at `n_episodes=10` | P(10 consecutive) |
|---|---|---|
| 0.6 | 0.38 | 0.0001 |
| 0.7 | 0.65 | 0.013 |
| 0.8 | 0.88 | 0.28 |
| 0.9 | 0.99 | 0.88 |

At `n_episodes=10`, `min_epochs_above_threshold: 10` turns a nominal 0.7 threshold into an effective bar near 0.85. Prefer **more evaluation episodes and a shorter run** — `--n-eval-episodes 30` with `min_epochs_above_threshold: 3` keeps the effective bar close to the nominal one for roughly the same eval cost.

### `terminate_on_success` silently changes what `success_rate` means

The flag consults the phase's configured `success_criteria` (it was previously hardcoded to all-models-at-objectives, so fraction- and VP-gated phases could never end early). Switching it on is not a neutral speed optimisation — it redefines the metric every gate is calibrated against:

| `terminate_on_success` | `success_rate` measures |
|---|---|
| `false` | the criteria **holds at the final step** |
| `true` | the criteria was **ever met** — episodes that achieve it stop there, so it is true at the last step by construction |

Recorded matches show peak occupancy running ~3x final occupancy, so flipping this raises `success_rate` by roughly that factor without the policy improving at all. Every threshold in a ladder calibrated under one setting is wrong under the other.

For VP phases it is also semantically wrong: `win_at_the_end` must mean ahead *at the end*, not ahead at some point. Keep it `false` there.

### `terminal_success_bonus` and the remaining-turn scale

The bonus is multiplied by the fraction of turns left when the episode ends — a speed incentive that presumes success *ends* the episode. It is therefore applied **only when `terminate_on_success` is true**.

With `terminate_on_success: false` every episode runs to `max_turns`, which would leave a scale of `1/max_turns`. Note `max_turns = number_of_battle_rounds × (5 - len(skip_phases))`, so a 20-round config with three skipped phases has `max_turns = 40` and would shrink the bonus 40-fold. The bonus is delivered at full value in that case instead.

Two consequences worth checking when a phase will not advance:

- **Weigh the terminal bonus against the dense calculators, per episode.** A dense calculator emitting 0.08/step over 40 steps contributes 3.2, so a terminal bonus of 5.0 is comparable — but one of 0.5 is noise. `reward/components/*` are per-step means (see [metrics.md](metrics.md)), so multiply by `mean_episode_steps` before comparing.
- **Check `gamma` covers the episode.** The bonus lands on the last step, so its value at t=0 is `bonus × gamma^max_turns`. The `PPOConfig` default `gamma=0.9` discounts a 40-step episode by 0.015, making any terminal signal invisible to early actions; use `--gamma 0.99` for episodes this long.

## Reward Aggregation

Each step, the phase manager computes the reward as follows:

1. **Per-model calculators**: For each **alive** model, every per-model calculator's output is multiplied by its `weight` and summed. The per-model totals are then **averaged across the alive models** (dead models are excluded; if no models are alive the per-model contribution is 0).
2. **Global calculators**: Each global calculator's output is multiplied by its `weight` and summed.
3. **Final reward** = averaged per-model reward + global reward total.

The averaged scalar is what `env.step()` returns and what the episode-reward metrics sum. Alongside it the manager records `last_per_model_reward`, an array holding each alive model's own weighted total **plus** the global total broadcast whole (global terms are the part of the outcome not attributable to one model, so every model sees the same value). PPO trains on that vector — rewards, values, advantages and ratios all carry a per-model axis — because averaging 25 models into one number leaves each model's action explaining ~4% of the signal it is credited with. Terminal bonuses are added to both the scalar and the broadcast share.

This is why calculator scope is a design decision rather than an implementation detail: a phase built only from global calculators hands every model a bit-identical reward and reduces the per-model vector back to a broadcast scalar.

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
| `player_kills_by_model` | `np.ndarray \| None` | Kills made by each player model this step, shape `(n_player_models,)`; `player_models_killed` is its sum. `None` when no shooting was resolved this step. Read by `model_kills` so a kill is credited to the model that fired |

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
    model_kills.py                 # Per-model credit for this model's kills
    models_at_objectives.py        # Dense fraction-of-models-on-objectives reward
    objective_coverage.py          # Dense fraction-of-objectives-controlled reward
    objective_flip_bonus.py        # Symmetric objective control-state potential
    objective_hold.py              # Per-model control-scaled occupancy reward
    killing.py                     # Global kill bonus (prefer model_kills)
    vp_gain.py                     # VP gain reward (global)
    registry.py                    # Type-string -> class mapping
  criteria/
    base.py                        # SuccessCriteria ABC
    all_at_objectives.py           # All models at objectives
    fraction_at_objectives.py      # A fraction of alive models at objectives
    all_models_grouped.py          # All models within group distance
    player_ahead_on_vp.py          # Player ahead on VP (win-rate) criteria
    player_vp_min.py               # Player VP min success criteria
    registry.py                    # Type-string -> class mapping
  types/
    model_rewards.py               # ModelRewards (legacy two-field breakdown; the
                                   #   live per-model vector is the phase manager's
                                   #   last_per_model_reward)
```
