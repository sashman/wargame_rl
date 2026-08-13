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
| `min_epochs_above_threshold` | int | `5` | Success rate must be ≥ success_threshold for this many consecutive epochs before advancing. **This is why `--eval-every-n-epochs` is rejected on a multi-phase config**: "consecutive" counts epochs that were *evaluated*, so a coarser cadence moves the epoch a phase advances on and changes what the run trains |
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

> **A misspelled field is an error, not a default.** Every config model sets
> `extra="forbid"`, so `wieght: 1.0` fails to load rather than quietly leaving the
> weight at 1.0 — which is how an arm ends up not measuring what its config says.
> The strictness stops at `params`: its contents are free-form by design, because
> each registry entry defines its own keys, and they are validated by the
> calculator that receives them.

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
| `unit_coherency` | **per-model** | `value` (float, default 0.05), `straggler_penalty` (float, default 0.0, must be <= 0) | Pays a model for standing inside its unit's largest chain component, under the same 2"/9"/one-connected-group predicate `domain/coherency.py` evaluates. **The only coherency mechanism that is not a constraint** — deployment, the end-of-move revert and attrition all *make* the game legal, and none gives a policy a reason to stay in formation; measured under coherent deployment alone, the agent drifts to `eval/coherency_rate` 0.53-0.62 with ~3 of 25 models adrift. Keyed on *this* model's membership, so the straggler earns nothing while its squadmates earn — a flat payment to a coherent unit is the flat-`objective_hold` mistake and creates no gradient. A model alone in its unit is paid, matching the rule that a one-model unit is coherent by definition; fining it would price a casualty it did not cause. Small and positive by default (~a fifth of `objective_hold`): magnitude is the risk, not sign, and a negative term could make advancing on an objective a net loss. Move `straggler_penalty` before raising `value`, which inflates total income and changes what the policy trades objectives against. **Requires `observe_coherency`** — the config is rejected at construction otherwise. |
| `objective_hold` | **per-model** | `player_value` (float, default 1.0), `contested_value` (float, default 0.5), `opponent_value` (float, default 0.25), `crowding_exponent` (float, default 0.0) | Value of the objective this model occupies, keyed on control state under the same OC/count rule as VP scoring; 0 when off every objective, and 0 on a neutral one since holding a point nobody contests is worth nothing. The **only calculator that pays a model while it is stationary** — `closest_objective`'s progress is a potential that exhausts on arrival and is exactly 0 on shooting steps, and `group_cohesion` returns 0 inside its limit, so without this all models share an identical reward for most of an episode. Pays for *controlling* rather than standing, and supplies a gradient across `neutral → contested → player`, which `objective_coverage` and `vp_gain` are both flat across because control is a strict count comparison. **`surplus_value` (removed 2026-08-08, superseded by `crowding_exponent`) tried to price over-stacking**: control needs `opponent_count + 1` models and every model past that changes nothing about who scores the point, so at the default 1.0 the calculator is indifferent between 15 models on one objective and 8/7 across two — both pay `15 x player_value` — while VP is worth double for the second. Below 1.0 the models nearest each centre were paid in full up to the quota and the rest scaled down. It measured null-to-negative (occupancy 0.784 → 0.284) and was removed: the discount lowers *total* objective income, and it keys on a distance-to-centre rank no model can observe about itself. **`crowding_exponent` pays the point a pot instead of every occupant a wage** (prototype, 2026-08-07): the value is divided by `occupants ** a`, so `a = 0` is the flat default and `a = 1` conserves the pot — a point pays its value once, split among whoever stands there. This is the *third* attempt at the same over-stacking defect and the first whose gradient points at the behaviour rather than away from objectives: `surplus_value` (since removed) and the overstack penalty both strictly lower total objective income, so the policy reads either as "objectives pay less" and does fewer of them, which is what both rounds measured. Under pot conservation, spreading onto a second point strictly *raises* total income. It is also keyed only on the occupant count, which `observe_objective_control` puts directly in the observation, where the discount's cliff was keyed on a rank no model can observe about itself. Calibrate the weight so the occupancy that actually holds the contested point pays what the flat term paid, or the arm changes the price of correct play as well as the price of crowding. |
| `model_kills` | **per-model** | `bonus_per_kill` (float, default 2.0) | Bonus for each opponent **this model** killed this step. The per-model counterpart of `killing`: under that global term every model is paid the same whether it fired, missed, or stood still, leaving the shooting head — half the agent's decisions — with no credit path. Note the scale difference: `killing` broadcasts its bonus to all N models per kill, so its default 5.0 makes a 25-model wipe worth 125 per model. |
| `killing` | global | `bonus_killing_opponent` (float, default 5.0) | Bonus × opponent models killed this step. **Prefer `model_kills`:** this spreads kill credit flat across the army, and at the default bonus it is easily the largest term in a phase. |
| `models_lost` | global | `penalty_per_loss` (float, default 1.0) | Negative reward × **player** models lost this step (reads `ctx.opponent_models_killed`, where `opponent_` means "by the opponent"). The cost side of the shooting trade: `model_kills` pays for kills made and nothing charged for the models they cost, so exposing a model was free and declining a bad exchange could never look better than charging. **Global on purpose** — `RewardPhaseManager` runs per-model calculators over alive models only, and with `max_wounds: 1` every damaged model dies, so a per-model damage penalty would be identically zero. Calibrate against `model_kills` (see below). |

**Pricing the trade.** `model_kills` is per-model and divided by the alive count;
`models_lost` is global and broadcast whole. Comparing raw weights is therefore wrong — the
two reach the step reward through different arithmetic. At `model_kills` weight 1.0 / bonus
2.0 with ~20 models alive, a kill contributes ≈ 0.10, so `models_lost` weight 0.1 / penalty
1.0 matches it. Measured on `25v25_cover_full` with `squad_march_shoot` over 8 episodes, that
gives +1.44 per episode from kills against −1.60 from losses: an even exchange nets to
roughly zero and only *favourable* exchanges pay. Verify any change the same way, by summing
`just measure-income-share <policy|ckpt> <config>` rather than by guessing. Two traps it exists to avoid: weights are not shares, and a term's share of *mean* income bounds nothing about its influence on a *choice* — what moves a gradient is the term's variation across the actions being compared.

### Choosing calculators

Five rules, all learned by measurement rather than taste:

- **Before training a new reward lever, check the agent can *observe* the quantity it keys on.** This is a desk check that costs seconds and has already cost two full experiments (~10 GPU-hours) by being skipped. Round 1's `overstack_penalty_per_extra` and round 2's `objective_hold.surplus_value` are mechanically opposite — a penalty and a discount, the latter designed specifically so occupancy could never become unprofitable — and both halved objective occupancy on both seeds. The reason they failed identically is that both are keyed on *per-objective model counts*, and an objective reached the network as nothing but an `(x, y)` location. A reward the policy cannot attribute is one it can only experience as "this behaviour pays less on average", so it does less of it — regardless of which models the designer intended to charge. Ask: *if two states differ only in the quantity this term keys on, do they differ in the observation?* If not, add the input first (see `observe_objective_control`) or do not run the experiment. See [the 2026-08-06 report](../reports/2026-08-06-beat-the-shooting-opponent.md) § Part 8.
- **Every phase needs at least one per-model calculator.** Only `closest_objective`, `closest_objective_v2`, `group_cohesion`, `objective_hold`, `model_kills` and `unit_coherency` are per-model; the rest are broadcast bit-identically to every model. PPO consumes the per-model reward *vector* (`phase_manager.last_per_model_reward` → `ppo/lightning.py`), so this is real credit assignment and not bookkeeping — but global terms are broadcast whole to every alive model, where a per-model value baseline largely absorbs them. A phase built purely from global terms gives all models the same reward and undoes per-model credit assignment by configuration. This is why `model_kills` replaced the global `killing` term, under which every model was paid the same whether it fired, missed, or stood still.
- **Per-model is necessary and nowhere near sufficient — the number must vary across the choice the model is making, and should track its marginal contribution.** This is the rule the three failed anti-stacking rounds cost, and it is easy to satisfy the letter of the rule above while breaking it. Flat `objective_hold` *is* per-model, but it is **constant** over "which objective do I stand on": the thirteenth model on a point earns the same 0.25/step as the first, so no model ever has a private reason to leave. Per-model is not the same as per-model-*differentiated*, and only the second produces a gradient.

  Ask what the model's presence actually changes. Control needs `opponent_count + 1` models, so on a point defended by a quarter of a model the thirteenth occupant contributes **nothing** — and a reward that pays it in full is telling it otherwise. `crowding_exponent` prices it at `value / 13`, a crude approximation of marginal contribution, and crude was enough: it moved the agent from +3.25 to +28.4 vp_margin, past the scripted bar. (This is the *difference rewards* idea from multi-agent RL; the approximation does not need to be principled to work.)

  **The corollary, and the more useful half:** a lever against over-concentration must **redistribute** reward, not destroy it. `overstack_penalty_per_extra` (occupancy 0.925 → 0.520) and `surplus_value` (0.784 → 0.284) both lower *total* objective income, so the policy correctly learns the activity is worth less and does less of it. At `crowding_exponent = 1` the pot is conserved — `k` models on one point earn its value once, `k/2` on each of two earn it twice — so spreading strictly raises total pay. Before training a shaping term, run `just measure-income-share <policy|ckpt> <config>` (which sums `env.last_reward_breakdown` over whole episodes and splits per-model from global) under both settings and check the behaviour you want pays **more in total** than the behaviour it replaces. If it does not, it is a tax on the whole activity.
- **The sign of a term is not what makes it work — do not reach for "positive rewards over penalties".** This specific hypothesis was tested and refuted at the cost of a round. `surplus_value` *is* the positive version of the overstack penalty: it only pays surplus models less, never below zero, and that property was the whole reason to expect a different result. It failed identically — occupancy 0.784 → 0.284 against the penalty's 0.925 → 0.520. Two levers on opposite sides of the sign axis, the same collapse.

  Note also that `crowding_exponent` **reduces** what a crowded model earns (0.25 → 0.096 at thirteen occupants), so in sign it is indistinguishable from `surplus_value`. And the arm that beats the bar still carries two negative terms: `group_cohesion`'s `violation_penalty: -0.05` and `closest_objective_v2`'s `overstack_penalty_per_extra: 0.01`. What separates the winner is where the value *goes* — a discount deletes it, pot-sharing hands it to the models on another objective — which is the total-income rule above, not the sign.

  The fair residue of the intuition: a penalty is *more likely* to destroy total income by accident, because "charge for X" is the obvious way to say "do less of X" and it does not prompt you to ask where the value went. Sign correlates with the failure mode without causing it, so if you reach for a penalty, run the total-income check.

  Two things that *do* matter more than sign. **Magnitude:** `group_cohesion` at `violation_penalty: -0.2` costs a strung-out model ~26 per episode, enough to rank `split_evenly` (148 VP) below `greedy_nearest` (126 VP); at -0.05 it is a useful term in the winning config. Keep any term's episode integral near ~10 — check by summing `env.last_reward_breakdown` over a baseline's episodes. **Potential-based shaping is safe by construction:** `closest_objective_v2`'s progress term is a potential, so it is policy-invariant and cannot bias the optimum whatever its sign or weight, which is why it needs no decay.
- **Some quantities are genuinely not per-model, and forcing them there makes them zero.** `models_lost` must be global: `phase_manager.calculate_reward` iterates *alive* models only, so a model killed this step earns nothing that step, and with `max_wounds: 1` every damaged model dies — a per-model damage-taken penalty is identically zero on every config here. Global is the right home for outcomes no survivor can be charged with; it is the wrong home for anything a single model's action determines.
- **Every phase should keep `vp_gain`.** A rung that drops the goal signal trains away from it: win rate fell 62% → 47% when a curriculum advanced into a phase that rewarded occupancy and had no VP term, while `success_rate` held at ~80%.

`tests/test_curriculum_configs.py` enforces the two mechanical ones — a per-model calculator in
every phase, and `vp_gain` in every phase — over the two shipped training configs — `configs/golden/25v25_single_phase.yaml` (the single-phase control) and `25v25_curriculum.yaml` (the two-rung arm, whose final phase is identical to the control's). It also pins three narrower invariants read off past measurements: no phase may use `models_at_objectives`, a `player_vp_min` gate must sit above the 95/285 fraction a one-objective stack reaches unaided, and any phase setting `terminal_success_bonus` must keep `terminate_on_success: false` so the bonus is not scaled away.

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

At the end of each training epoch, the training loop runs evaluation episodes (controlled by `n_episodes` in `PPOConfig`; overridable via `--n-eval-episodes`). For each episode, the active phase's success criteria is checked. The resulting success rate is compared against the phase's `success_threshold`:

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
    unit_coherency.py              # Per-model payment for holding formation
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
