# PyTorch / RL Model Patterns

Applies to everything under `wargame_rl/wargame/model/`.

## Networks

Two network variants implementing `RL_Network` protocol (in `net.py`):
- `TransformerNetwork` — NanoGPT-style self-attention (default, actively developed)
- `MLPNetwork` — standard feed-forward (legacy, will be dropped)

Both expose `policy_from_env(env)` and `from_checkpoint(env, path)` class methods.

## DQN (`model/dqn/`)

- `DQNLightning` — PyTorch Lightning module: training loop, epsilon-greedy exploration, target network sync, experience replay
- `DQNConfig` / `DQNTrainingConfig` — Pydantic config models
- `ReplayBuffer` (`experience_replay.py`) — experience replay with `Experience` named tuples
- `Agent` — runs episodes with epsilon-greedy policy
- `layers.py` — `Block`, `LayerNorm`, `SelfAttention`, `MLP` shared by `TransformerNetwork`

## PPO (`model/ppo/`)

- `PPOLightning` — PyTorch Lightning module: actor-critic training, GAE, clipped surrogate objective
- `PPOConfig` / `PPOTrainingConfig` — Pydantic config (lr, gamma, GAE lambda, clip epsilon, etc.)
- `PPO_Transformer` — actor-critic model with shared transformer backbone, separate policy and value heads
- `PPOModel` — wraps policy net + value net. **`forward` casts both heads to float32** (a no-op at default precision). Under `--precision bf16-mixed` the importance ratio `exp(new_log_prob − old_log_prob)` must resolve ~0.007 nats around a log-prob of −4.8, where bf16 steps 0.0156 — the change would round away entirely and every ratio would read exactly 1, training on nothing at full speed. Keep the cast at the head, not at each `Categorical`: there are four construction sites and the value loss besides
- PPO only supports `TransformerNetwork` (no MLP variant)

## Shared (`model/common/`)

- `create_environment()` — factory for `WargameEnv` from config (optionally with `state_exporters`)
- **Rollout and eval envs are built with `build_info=False`.** The Gymnasium info dict costs ~0.2 ms of every step — 50 dataclasses, a Pydantic model and a `model_dump()` — and both loops discard it. Anything reading `info` on the training path will get `{}`; read the env's properties instead
- `observation_to_tensor` / `observations_to_tensor_batch` — observation conversion
- `WargameLightningBase` (`lightning_base.py`) — base for `DQNLightning` / `PPOLightning`: evaluation, baseline logging, reward-phase advancement
- `RLDataset` (`dataset.py`) — generic RL `IterableDataset`
- `BaseAgent` (`agent_base.py`) — shared episode-running agent interface
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
| 1 | objectives | `(n_objectives, 2)` normalized to `[-1, 1]`, or `(n_objectives, 5)` with `observe_objective_control` — location plus normalized player count, opponent count and radius |
| 2 | player models | `(n_models, feature_dim)` |
| 3 | opponent models | `(n_opponent_models, feature_dim)` — 0 rows when no opponents |
| 4 | terrain | `(n_terrain, 4)` normalized footprint corners — 0 rows when no terrain |
| 5 | action mask | `(n_models, n_actions)`, bool |

`feature_dim = base + n_opponent`, where base covers normalized location, distances to objectives, group_id one-hot, closest same-group distance, wound features (alive, wound_ratio, max_wounds_norm), and combat stats (attacks, bs, strength, ap, damage, toughness, save — each divided by its `NORM_*` constant). The trailing `n_opponent` columns are expected damage per target (player models) or zero-padding (opponent models).

The expected-damage block comes from `domain.shooting.expected_damage_matrix`, which calls the scalar `expected_damage` once per **distinct** stat pair rather than once per model pair. Every input is static YAML (no wound-based degradation — `take_damage` writes only `current_wounds`, which `expected_damage` never reads), so an army from one profile has a single distinct pair and a 25×25 block costs one call instead of 625. Keep the zero-toughness guard: `wound_roll_threshold` takes the `2 * toughness <= strength` branch at T=0 and returns 2, so padding rows would otherwise report the highest expected damage on the board.

**The objective token has no such trap.** `TransformerNetwork.from_env` reads `objective_size` straight off `tensors[1].shape[-1]`, so widening it resizes the embedding automatically and leaves `_alive_feature_index` untouched. That is why `observe_objective_control` adds three columns there rather than to the per-model block.

**Any new per-model column goes inside `core`, before `alive` — never on the end.** `TransformerNetwork._alive_feature_index` locates `alive` by counting *backwards* from the last column. A column appended after the combat stats shifts that index, and the key-padding mask then reads `wound_ratio` as `alive`: dead models stay attendable and live ones drop out. Nothing raises, so nothing tells you. This bit the one feature that was added here (`observe_threat_count`, since removed) and will bite the next one.

The docstring on `observation_to_tensor` is the source of truth — keep it, this table, and `docs/opponent-policies.md` (Observation Impact) in sync when tensor count or shapes change.

To add a new entity:
1. Add to `WargameEnvObservation` + obs builder
2. Extend `_observation_to_numpy` tuple
3. Update `observation_to_tensor` / `observations_to_tensor_batch`
4. Update **both** networks' `forward()` — Transformer needs dedicated embedding + updated token ordering; player tokens must stay extractable for per-model action heads
5. Fix unpacking in tests (`test_state.py`, `test_dqn.py`)

## Conventions

- Use `torch.Tensor` for all neural network operations
- Observation tensors built by `observation.py` from `WargameEnvObservation`
- Default algorithm is PPO; DQN is available but not the primary focus
- Device management via `device.py` utility
- Wandb integration in `wandb.py` — all logging goes through Lightning logger
- **`get_logger` sets `log_model=False`** — checkpoints are never uploaded. Nothing reads a model artifact back (every consumer takes a local `checkpoints/` path), and the uploads filled the storage quota at ~591 MB per run. `checkpoints/` is therefore the only copy of any trained weights
- When changing the observation tuple length, expect unpacking errors in tests (`test_state.py`, `test_dqn.py`) — fix them as part of the same change
