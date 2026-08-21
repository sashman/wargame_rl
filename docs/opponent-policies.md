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

The config is the only way to *choose* a policy, but not the only way one gets set:
`WargameEnv.set_opponent_policy(policy)` replaces it for the rest of the episode. That
exists for `just debug`, which wraps the configured policy in `OverridableOpponentPolicy`
(`envs/debug/overrides.py`) so a human can take individual opponent models off it. The
wrapper is deliberately **not** in the registry — it is not a scenario, and nothing should
be able to select it from YAML.

### Opponent model config fields

Each entry in `opponent_models` uses `ModelConfig`, the same schema as player models:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `x` | int | `null` | X coordinate. If `null`, placed randomly in opponent deployment zone. |
| `y` | int | `null` | Y coordinate. Must be set together with `x` or both omitted. |
| `group_id` | int | `0` | Group this model belongs to. |
| `max_wounds` | int | `1` | Maximum wound pool for this model. |
| `toughness` | int | `3` | Compared against attacker strength on the wound roll. |
| `save` | int (2–7) | `4` | Base armour save (`4` means 4+, `7` means no armour). |
| `weapons` | list | `[]` | `WeaponProfile` entries (see [shooting.md](shooting.md)). Empty = cannot shoot. |

## Turn Order

Each `env.step()` call advances the player through **one battle phase** (command, movement, shooting, charge, or fight). After the player completes their turn, the opponent's entire turn is auto-executed before the observation is returned: the clock runs through all five phases, but the policy is invoked only in the phases the player also plays (skipped phases advance the clock without an opponent action, so the opponent gets exactly as many decisions per round as the player). By default, non-movement phases are skipped (`skip_phases` config), so the player takes 1 step per round. Set `skip_phases: []` for full per-phase stepping (5 steps per round).

The `turn_order` field controls which side takes the first turn each round:

| Value | Behaviour |
|-------|-----------|
| `player` | The RL agent takes the first turn each round (agent is `player_1` on the game clock). This is the default. |
| `opponent` | The opponent takes the first turn each round. On `reset()` and after each player turn, the opponent's turn is auto-executed before the agent acts. |
| `random` | A coin flip at each `reset()` determines which side goes first for the episode. Reproducible with a fixed seed. |

## Available Policies

### `random`

Each opponent model selects a uniformly random action from the ones its action mask allows. In the movement phase that is stay or any angle/speed combination; in the shooting phase it is stay or any valid target, so `random` **does shoot back** when the shooting phase is active (it declares `shoots = True`).

```yaml
opponent_policy:
  type: random
```

**Parameters:** none.

**Use case:** Baseline opponent for initial training. Provides unpredictable but non-strategic opposition, useful for verifying the environment works before introducing smarter opponents.

**Note:** this policy samples with the global `np.random`, not `env.np_random`, so its choices are *not* reproducible from an episode seed.

### `scripted_advance_to_objective`

Each opponent model moves toward the nearest objective while keeping the group together. Each model's desired direction is a weighted blend of the vector to its closest objective and the vector to the centroid of the alive models; the polar-coordinate action with the best matching direction is selected. Step length is capped by distance to the objective boundary so models reduce speed when close and do not overshoot. A model already inside an objective's radius stays put.

```yaml
opponent_policy:
  type: scripted_advance_to_objective
  params: { cohesion_weight: 0.3 }   # optional
```

**Parameters:** `cohesion_weight` (0–1, default `0.3`) — 0 is pure objective seeking, 1 is pure flocking toward the centroid.

**Use case:** Provides goal-directed opposition that competes for the same objectives as the player. Good for training agents that need to learn to reach objectives before the opponent does.

**Note:** this policy never fires, even when its models carry weapons. Use `scripted_advance_and_shoot` for an opponent that shoots back.

### `scripted_advance_and_shoot`

Identical movement to `scripted_advance_to_objective`, which it subclasses. In each shooting phase every alive model fires at a uniformly random target drawn from the ones its action mask allows — alive, in weapon range, line of sight clear, and not locked in engagement range. A model with no valid target holds fire.

```yaml
opponent_policy:
  type: scripted_advance_and_shoot
  params: { cohesion_weight: 0.3 }   # optional, inherited from the movement policy
```

**Parameters:** `cohesion_weight` (default `0.3`), as for `scripted_advance_to_objective`.

Target choice is uniform rather than nearest-first. Nearest-first would concentrate fire and make the opponent a sharper threat than "returns fire" warrants; uniform keeps it a plain two-sided-game fixture whose damage output is easy to reason about. Targets are drawn from `env.np_random`, so a seeded episode replays exactly.

**Use case:** the only *goal-directed* opponent that makes the game two-sided (`random` also fires, but wanders). `scripted_advance_to_objective` leaves the player facing an enemy that cannot answer, which is why the `squad_march_shoot` baseline scores 1.00 on 25v25 — a bar set against a defenceless opponent. **Adopting this policy in a config invalidates any baseline or agent score measured on it; re-run `just measure-baselines` before reading a result.**

Requires the shooting phase to be active (`skip_phases` must not contain `shooting`) and `opponent_models` entries to carry `weapons`.

### `scripted_baseline`

Plays any registered **baseline** policy (`baseline/registry.py`) on the opponent side. The baselines are the strongest scripted play in the repo, but they were written to drive the *player* and the two hierarchies are separate; this adapter closes the gap without a second copy of any policy. The baseline is handed a side-swapped view of the env (`_MirroredEnv`), so the same code plays the opponent.

```yaml
opponent_policy:
  type: scripted_baseline
  params:
    baseline: squad_march_take
```

**Parameters:** `baseline` (**required**) — a name from the baseline registry: `random`, `hold_deployment`, `greedy_nearest`, `split_evenly`, `squad_march`, `squad_march_shoot`, `squad_march_deny`, `squad_march_take`, `contest_and_spread`. Any further params are forwarded to the baseline's constructor. An unknown name raises at construction.

`shoots` is derived from whether the baseline overrides `select_shooting`, so a shooting baseline gets its mask refined with range, line-of-sight and engagement validity and a movement-only one does not pay for the check. It is not a performance switch: declared wrongly, shots would be applied unchecked.

**Use case:** the opponent was the ceiling. Measured on `configs/golden/25v25_maps_coherency.yaml` and the same config with `squad_march_take` as the opponent (`configs/experiments/25v25_maps_take_opponent.yaml`) — same 60 layouts, seeds 700000+, `vp_margin` with win rate in brackets.

⚠ **These absolute figures were measured on the hand-traced evaluation tables and are superseded** — the tables were regenerated on 2026-08-20. The *comparison* stands, because both columns were measured on the same tables. Re-measured on the generated tables (all 45, n=30, seeds 700000+), the left column reads: `random` −59.6, `squad_march_shoot` +116.7, `contest_and_spread` +112.8, `squad_march_take` +126.2, `squad_march_deny` +121.6.

| player | vs `scripted_advance_and_shoot` | coherent | vs `squad_march_take` | coherent |
|---|---|---|---|---|
| `random` | −23.1 (0.42) | 0.026 | −215.2 (0.00) | 0.168 |
| `greedy_nearest` | +3.2 (0.30) | 0.846 | −161.6 (0.00) | 0.897 |
| `split_evenly` | −24.6 (0.40) | 0.125 | −204.2 (0.00) | 0.202 |
| `squad_march` | +81.1 (0.93) | 0.843 | −168.8 (0.00) | 0.790 |
| `squad_march_shoot` (the old bar) | **+104.9 (0.98)** | 0.830 | **−5.6 (0.48)** | 0.795 |
| `contest_and_spread` | +102.3 (0.98) | 0.806 | −48.8 (0.20) | 0.729 |
| `squad_march_take` | +108.1 | — | +9.9 (mirror) | — |

Six of the seven policies beat the old opponent and none of them beats this one. The bar loses 110 points of margin and drops to an even game; the ladder below it collapses from "roughly even" to "annihilated" (`random`'s surviving force falls from 0.85 of its army to 0.11). `squad_march_take` against itself sits near zero, as a mirror match should.

**A competent opponent costs formation.** Every policy that holds formation at all holds it *less* against `squad_march_take` — `squad_march` 0.843 → 0.790, the bar 0.830 → 0.795, `contest_and_spread` 0.806 → 0.729 — because units get shot apart faster and casualties break the chain. Worth knowing before any coherency figure measured on the old opponent is carried across.

**`random` is the reading-rule trap, live.** Its rate *rises* 0.026 → 0.168 against the stronger opponent, and its `adrift` falls 21.7 → 10.9 — not because it plays better but because it is dead: `alive` goes 0.849 → 0.105, and a unit shot down to one model is coherent by definition. Read the rate with `adrift` and `alive`, never alone.

Three caveats on the table. `squad_march_take` and `squad_march_shoot` are **not** separated at n=60 on either opponent — paired difference 3.2 ± 4.7 (t = 0.68) on the old one and 15.6 ± 13.8 (t = 1.13) on the new one — so "the strongest scripted policy" is a claim about where it was first measured, not one this table establishes. The `squad_march_take` rows come from a paired run rather than the ladder, because `scripts/measure_baselines.py`'s `BASELINES` tuple does not include it, which is also why they have no coherency figure. And `random`'s row moves between runs at fixed seeds: it draws from the global `np.random` rather than `env.np_random`, so it is the one policy here that is not reproducible from an episode seed.

Two honest caveats on that table. `squad_march_take` and `squad_march_shoot` are **not** separated at n=60 on either opponent — paired difference 3.2 ± 4.7 (t = 0.68) on the old one and 15.6 ± 13.8 (t = 1.13) on the new one — so "the strongest scripted policy" is a claim about where it was first measured, not one this table establishes. And the `squad_march_take` rows come from a paired run rather than the ladder, because `scripts/measure_baselines.py`'s `BASELINES` tuple does not include it.

**Adopting this policy voids every baseline and agent score measured on that config** — the whole ladder, not just the bar. Re-run `just measure-baselines <config> 100 "" 700000` first.

## Planned Policies (Not Yet Implemented)

The following policies are designed in the architecture but have no class and are not registered — naming one in YAML raises `ValueError: Unknown opponent policy type` from `build_opponent_policy`:

| Type key | Description |
|----------|-------------|
| `human` | Read actions from the renderer (keyboard/mouse input). Enables human-vs-agent play. |
| `model` | Load a pre-trained checkpoint and use it as the opponent. Enables self-play and agent-vs-agent evaluation. |

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
        "wargame_rl.wargame.envs.opponent.scripted_advance_and_shoot_policy",
        "wargame_rl.wargame.envs.opponent.scripted_hold_position_policy",  # new
    ):
        importlib.import_module(mod)
```

4. Use it in a YAML config:

```yaml
opponent_policy:
  type: scripted_hold_position
```

The `select_action` method receives the list of opponent `WargameModel` instances, the full `WargameEnv`, and optionally `action_mask` (phase-aware valid actions), giving access to objectives, board dimensions, and any other env state needed to compute actions. Reach the opponent's handler through the public `env.opponent_action_handler` property — it exposes `best_action_toward()` and `shooting_slice`. `env.last_player_shooting_results` / `env.last_opponent_shooting_results` expose the shots resolved in the most recent step, for a policy that wants to react to incoming fire.

### Policies that shoot

Set the class attribute `shoots = True` on any policy that emits shooting-slice actions. The env only refines that policy's action mask with range, line-of-sight and engagement-range validity when the flag is set, because doing so costs up to `n_opponent × n_player` line-of-sight traces per shooting phase and most policies never fire. Without the flag the mask allows any target and `domain.shooting.resolve_shooting_phase` applies the shot unchecked — a policy could shoot through terrain from across the board.

Given the flag, honouring `action_mask` is all a policy needs to do to play by the same rules the player does.

### Naming convention

Scripted policies are prefixed with `Scripted` in the class name and `scripted_` in the registry key (e.g. `ScriptedFlankPolicy` / `"scripted_flank"`). This distinguishes hand-coded behaviour from learned or external policies.

## Observation Impact

When opponents are present, the player agent's observation includes opponent model positions as a separate list (`opponent_models`). This is converted to 6 tensors in the observation pipeline:

| Tensor index | Content | Shape |
|--------------|---------|-------|
| 0 | Game state | `(6,)` — placeholder, normalized_round, phase_index, player_vp, opponent_vp, player_vp_delta (see `envs/types/env_observation.py`, `model/common/observation.py`) |
| 1 | Objectives | `(n_objectives, 2)` |
| 2 | Player models | `(n_player_models, features)` |
| 3 | Opponent models | `(n_opponent_models, features)` |
| 4 | Terrain | `(n_terrain, 17)` — normalized outline vertices padded to 8, plus the vertex count |
| 5 | Action mask | `(n_models, n_actions)` — bool, valid actions per model |

When there are no opponents, tensor 3 has shape `(0, features)`; likewise tensor 4 is `(0, 17)` with no terrain. The transformer handles both gracefully, as an empty token sequence.

## File Layout

```
wargame_rl/wargame/envs/opponent/
  __init__.py                                   # Module exports
  policy.py                                     # OpponentPolicy ABC
  registry.py                                   # Type-string -> class registry + factory
  random_policy.py                              # RandomPolicy
  scripted_advance_to_objective_policy.py        # ScriptedAdvanceToObjectivePolicy
  scripted_advance_and_shoot_policy.py           # ScriptedAdvanceAndShootPolicy

configs/golden/
  25v25_shooting_opponent.yaml                  # vs scripted_advance_and_shoot
  25v25_single_phase.yaml                       # vs scripted_advance_to_objective
```
