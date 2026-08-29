# PyTorch / RL Model Patterns

Applies to everything under `wargame_rl/wargame/model/`.

## Networks

`TransformerNetwork` (in `net.py`) is the only network implementing the `RL_Network`
protocol. It exposes `policy_from_env(env)` and `from_checkpoint(env, path)`, plus
`from_spec(spec, is_policy)` and the module-level `spec_from_observation(observation,
action_handler, objective_budget, max_groups)`. The last two exist because `from_env` **resets the
env** and reads `env._action_handler`: an opponent policy is constructed inside
`WargameEnv.__init__`, so it can do neither, and it must size from the *opponent's*
handler rather than the player's.

DQN and `MLPNetwork` were removed once neither had been trained in months. Two
things survived that removal because the transformer needs them: `common/layers.py`
(`Block`, `LayerNorm`, `SelfAttention`, and `MLP` — the *feed-forward half of a
transformer block*, unrelated to the deleted `MLPNetwork`), and `common/argmax_agent.py`.
`net.convert_state_dict` still strips the DQN Lightning prefix `policy_net.`, so
checkpoints trained before the removal still load — the weights are a
`TransformerNetwork` either way. `git log -- wargame_rl/wargame/model/dqn` restores
the rest.

**Moving a module that a checkpoint pickles breaks every checkpoint.** Lightning
stores the whole `PPO_Transformer` in `hyper_parameters`, so a checkpoint records
the *import path* of `Block` and `LayerNorm` as of the day it was written — and
`torch.load` then raises `ModuleNotFoundError`, not a warning. `common/layers.py`
ends with a `sys.modules` alias for its old path for exactly this reason;
`tests/test_checkpoint_module_alias.py` pins it, and it was found only by scoring
a real trained run, since nothing in the suite loaded a pre-move checkpoint.
Anything reachable from a Lightning module's constructor args carries the same
hazard — `SelfPlayConfig` is the newest one, pickled into `hyper_parameters` on
every checkpoint written since wave 4, so `model/common/self_play.py` may not
move without an alias.

## PPO (`model/ppo/`)

- `PPOLightning` — PyTorch Lightning module: actor-critic training, GAE, clipped surrogate objective
- `PPOConfig` / `PPOTrainingConfig` — Pydantic config (lr, gamma, GAE lambda, clip epsilon, etc.)
- `PPO_Transformer` — actor-critic model with shared transformer backbone, separate policy and value heads
- `PPOModel` — wraps policy net + value net. **`forward` casts both heads to float32** (a no-op at default precision). Under `--precision bf16-mixed` the importance ratio `exp(new_log_prob − old_log_prob)` must resolve ~0.007 nats around a log-prob of −4.8, where bf16 steps 0.0156 — the change would round away entirely and every ratio would read exactly 1, training on nothing at full speed. Keep the cast at the head, not at each `Categorical`: there are four construction sites and the value loss besides
- PPO runs on `TransformerNetwork`

## Shared (`model/common/`)

- `create_environment()` — factory for `WargameEnv` from config (optionally with `state_exporters`)
- **Rollout and eval envs are built with `build_info=False`.** The Gymnasium info dict costs ~0.2 ms of every step — 50 dataclasses, a Pydantic model and a `model_dump()` — and both loops discard it. Anything reading `info` on the training path will get `{}`; read the env's properties instead
- `observation_to_tensor` / `observations_to_tensor_batch` — observation conversion
- `WargameLightningBase` (`lightning_base.py`) — base for `PPOLightning`: evaluation, baseline logging, reward-phase advancement
- `BaseAgent` (`agent_base.py`) — shared episode-running agent interface
- `ArgmaxAgent` (`argmax_agent.py`) — plays the best valid action of a bare `RL_Network`. `simulate.py` and `scripts/measure_phase_gates.py` load weights into a `TransformerNetwork` rather than a `PPOModel`, so they cannot use `ppo/agent.py`
- `TransformerConfig` — shared transformer hyperparameters
- `Device` / `get_device()` — device management
- Wandb integration in `wandb.py` — all logging goes through Lightning logger

### Evaluation is batched, not sequential

`WargameLightningBase._run_episodes_batched` runs eval episodes in **lockstep waves**: it holds a reusable pool of eval envs (`_ensure_eval_envs`), steps them together, and scores each wave with one batched forward pass via `_batch_greedy_actions`. Every episode is `max_turns` steps, so a wave costs `max_turns` forward passes rather than `max_turns × n_episodes`. Subclasses that cannot score a batch return `None` from `_batch_greedy_actions` and fall back to the sequential `_run_episode_eval` path (also used for epsilon > 0).

## Callbacks

- `get_checkpoint_callback()` — builds two callbacks for a run: a monitored `ModelCheckpoint` keeping the top-3 by training reward, and a `PeriodicLastCheckpoint` owning `last.ckpt`. They are separate because one `ModelCheckpoint` cannot do both jobs — with `monitor` set, `save_last=True` fires only on epochs that enter the top-k, so `last.ckpt` silently becomes `best.ckpt`. And an *unmonitored* `ModelCheckpoint(save_top_k=0, save_last=True)` writes only at `on_train_end`, which leaves a killed run with no `last.ckpt` at all
- `RecordEpisodeCallback` — records MP4 episodes during training
- `EnvConfigCallback` — persists env YAML config alongside checkpoints
- `EventLogCallback` — records a match event log during training (`--record-events`)

## Observation Tensor Pipeline

`model/common/observation.py`: `WargameEnvObservation` → **6 tensors**, in this order:

| # | Tensor | Shape |
|---|---|---|
| 0 | game features | `(6,)` — placeholder, normalized_round, normalized_phase, player_vp, opponent_vp, player_vp_delta |
| 1 | objectives | `(n_objectives, 2)` normalized to `[-1, 1]`, or `(n_objectives, 5)` with `observe_objective_control` — location plus normalized player count, opponent count and radius; `+1` trailing `present` column with `objective_budget` |
| 2 | player models | `(n_models, feature_dim)` |
| 3 | opponent models | `(n_opponent_models, feature_dim)` — 0 rows when no opponents |
| 4 | terrain | `(n_terrain, 17)` — `2 * TERRAIN_VERTEX_BUDGET + 1`: normalized outline vertices, padded to 8, plus the real vertex count. 0 rows when no terrain; the *sequence* is padded to `terrain_budget` when set |
| 5 | action mask | `(n_models, n_actions)`, bool |

`feature_dim = base + n_opponent`, where base covers normalized location, distances to objectives, group_id one-hot, closest same-group **live** distance, the unit-strength column when `observe_unit_strength` is set, the two coherency columns when `observe_coherency` is set (spread ratio and component fraction) — each already a fraction, so no `NORM_` constant, and all of them inside `core`, ahead of `alive`, per the rule below — wound features (alive, wound_ratio, max_wounds_norm), and combat stats (attacks, bs, strength, ap, damage, toughness, save — each divided by its `NORM_*` constant). The trailing `n_opponent` columns are expected damage per target (player models) or zero-padding (opponent models).

The expected-damage block comes from `domain.shooting.expected_damage_matrix`, which calls the scalar `expected_damage` once per **distinct** stat pair rather than once per model pair. Every input is static YAML (no wound-based degradation — `take_damage` writes only `current_wounds`, which `expected_damage` never reads), so an army from one profile has a single distinct pair and a 25×25 block costs one call instead of 625. Keep the zero-toughness guard: `wound_roll_threshold` takes the `2 * toughness <= strength` branch at T=0 and returns 2, so padding rows would otherwise report the highest expected damage on the board.

**The block is the open-ground expectation, and that is a choice.** `expected_damage` takes `in_cover` (the shooting path passes it; the matrix does not), so every entry here assumes the target is not in cover. Applying it would make the value a function of the *pair of positions* rather than the pair of stat lines — one distinct value per model pair, which is exactly the memoisation above collapsing — and it would change the network's input on every config that has terrain and bases. If it is ever wanted, it is a scenario change to screen over two seeds, not a correction: see `docs/shooting.md` § Expected damage.

**The objective token has no such trap.** `TransformerNetwork.from_env` reads `objective_size` straight off `tensors[1].shape[-1]`, so widening it resizes the embedding automatically and leaves `_alive_feature_index` untouched. That is why `observe_objective_control` adds three columns there rather than to the per-model block.

**Objective count is a hard input dimension, and `objective_budget` is what removes it.** The per-model block is `2 + n_objectives * 2` wide, so a model token is 49 columns at three objectives, 53 at five and 55 at six — one network cannot span layouts with different counts, and a checkpoint trained at three fails `load_state_dict` outright on the real tables, which carry five or six. Setting `objective_budget` pads every objective-derived input to a fixed size; `terrain_budget` does the same for the terrain *sequence* (15 or 16 pieces across the shipped maps), which `observations_to_tensor_batch` otherwise cannot stack. Padding is explicitly marked in both cases and dropped from attention: a `present` column on the objective token (which also makes "the row is entirely zero" a safe padding test, since a real objective could otherwise sit at the exact board centre), and the existing vertex-count column on terrain, which is zero only on padding. **The objective presence flags are repeated per model on purpose** — they qualify the padded distance pairs, which are per model, and a padding slot's `(0, 0)` delta otherwise reads as "this model is standing on that objective". Both budgets default to None and are then exact no-ops; setting either changes the embedding shapes, so old checkpoints fail loudly. The network is told which regime it is in by a constructor flag (`objective_padding`, set in `from_env` from the config) rather than by sniffing the tensor.

**The same-group distance excludes the dead, and did not always.** It read every model's location with no alive filter while `take_damage` writes only `current_wounds` — a destroyed model keeps its position forever — so a model could be told its nearest squadmate was adjacent when that squadmate was a corpse. Measured on the golden shooting config, **24% of live models read a wrong value, rising to 33% after step 30**, mean error 0.056 of the column's range against a 2" coherency band worth 0.027 of it. The `group_cohesion` *reward* always masked the dead, so the observation and the reward disagreed about who was in the unit. Fixed unconditionally on 2026-08-12 with a deliberate `test_observation_golden` regeneration — the diff is confined to that one column, and old checkpoints still *load* (the width is unchanged) while scoring differently.

**Any new per-model column goes inside `core`, before `alive` — never on the end.** `TransformerNetwork._alive_feature_index` locates `alive` by counting *backwards* from the last column. A column appended after the combat stats shifts that index, and the key-padding mask then reads `wound_ratio` as `alive`: dead models stay attendable and live ones drop out. Nothing raises, so nothing tells you. This bit the one feature that was added here (`observe_threat_count`, since removed) and will bite the next one.

The docstring on `observation_to_tensor` is the source of truth — keep it, this table, and `docs/opponent-policies.md` (Observation Impact) in sync when tensor count or shapes change.

To add a new entity:
1. Add to `WargameEnvObservation` + obs builder
2. Extend `_observation_to_numpy` tuple
3. Update `observation_to_tensor` / `observations_to_tensor_batch`
4. Update `TransformerNetwork.forward()` — a dedicated embedding plus updated token ordering; player tokens must stay extractable for per-model action heads
5. Fix unpacking in tests (`test_state.py`)

## Conventions

- Use `torch.Tensor` for all neural network operations
- Observation tensors built by `observation.py` from `WargameEnvObservation`
- PPO is the only algorithm
- Device management via `device.py` utility
- Wandb integration in `wandb.py` — all logging goes through Lightning logger
- **The trunk size is a parameter, and `None` is the production one.** `TransformerNetwork.from_spec`/`from_env`, `policy_from_env`/`value_from_env` and `PPO_Transformer.from_env` all take `transformer_config`, threaded to the one place it used to be hardcoded. `None` means `TransformerConfig()` — bit-identical to before the parameter existed, and kept as `None` rather than an equal object so the untouched path stays literally untouched. `train()` exposes `--n-layers` and `--embedding-size` — and deliberately **not** `--n-heads`: the head count changes no parameter shape, so a checkpoint written at another one would load *silently* and compute differently. Every size the flags can write, `trunk_config_from_state_dict` reads back, which `from_state_dict` now uses so a non-default checkpoint loads at its own shape instead of failing with a wall of missing keys that names every layer and never the cause. The flags still reject an `embedding_size` the default 8 heads cannot divide (Pydantic cannot catch that: both fields are valid ints alone and only their ratio is wrong, so it would surface as a reshape inside attention). ⚠ **A network built at another size is a different network** — its checkpoint will not load into a default run and its scores are comparable to nothing on file. It exists so the *tests* can stop paying ~12.7M parameters on a 2-model 20x20 board, which made the five slowest tests trunk-bound; `tests/test_network_size.py` pins the shipped default so this can never drift into production
- **`get_logger` sets `log_model=False`** — checkpoints are never uploaded. Nothing reads a model artifact back (every consumer takes a local `checkpoints/` path), and the uploads filled the storage quota at ~591 MB per run. `checkpoints/` is therefore the only copy of any trained weights
- When changing the observation tuple length, expect unpacking errors in tests (`test_state.py`) — fix them as part of the same change
