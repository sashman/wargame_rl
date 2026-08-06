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
- `PPOModel` — wraps policy net + value net
- PPO only supports `TransformerNetwork` (no MLP variant)

## Shared (`model/common/`)

- `create_environment()` — factory for `WargameEnv` from config (optionally with `state_exporters`)
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

- `get_checkpoint_callback()` — builds the Lightning `ModelCheckpoint` for a run
- `RecordEpisodeCallback` — records MP4 episodes during training
- `EnvConfigCallback` — persists env YAML config alongside checkpoints
- `EventLogCallback` — records a match event log during training (`--record-events`)

## Observation Tensor Pipeline

`model/common/observation.py`: `WargameEnvObservation` → **6 tensors**, in this order:

| # | Tensor | Shape |
|---|---|---|
| 0 | game features | `(6,)` — placeholder, normalized_round, normalized_phase, player_vp, opponent_vp, player_vp_delta |
| 1 | objectives | `(n_objectives, 2)`, normalized to `[-1, 1]` |
| 2 | player models | `(n_models, feature_dim)` |
| 3 | opponent models | `(n_opponent_models, feature_dim)` — 0 rows when no opponents |
| 4 | terrain | `(n_terrain, 4)` normalized footprint corners — 0 rows when no terrain |
| 5 | action mask | `(n_models, n_actions)`, bool |

`feature_dim = base + n_opponent`, where base covers normalized location, distances to objectives, group_id one-hot, closest same-group distance, wound features (alive, wound_ratio, max_wounds_norm), and combat stats (attacks, bs, strength, ap, damage, toughness, save — each divided by its `NORM_*` constant). The trailing `n_opponent` columns are expected damage per target (player models) or zero-padding (opponent models).

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
- **`get_logger` sets `log_model=False`** — checkpoints are never uploaded. Nothing reads a model artifact back (every consumer takes a local `checkpoints/` path), and the uploads filled the storage quota at ~591 MB per run. `artifact_retention.py` + `just prune-artifacts` clear the historical backlog; they are a cleanup tool, not a live retention policy
- When changing the observation tuple length, expect unpacking errors in tests (`test_state.py`, `test_dqn.py`) — fix them as part of the same change
