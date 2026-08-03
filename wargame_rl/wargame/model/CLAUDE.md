# PyTorch / RL Model Patterns

Applies to everything under `wargame_rl/wargame/model/`.

## Networks

Two network variants implementing `RL_Network` protocol (in `net.py`):
- `TransformerNetwork` — NanoGPT-style self-attention (default, actively developed)
- `MLPNetwork` — standard feed-forward (legacy, will be dropped)

Both expose `policy_from_env(env)` and `from_checkpoint(env, path)` class methods.

## DQN (`model/dqn/`)

- `DQNLightning` — PyTorch Lightning module: training loop, epsilon-greedy exploration, target network sync, experience replay
- `DQNConfig` / `TrainingConfig` — Pydantic config models
- `ReplayBuffer` — experience replay with `Experience` named tuples
- `RLDataset` — PyTorch `IterableDataset` wrapping the replay buffer
- `Agent` — runs episodes with epsilon-greedy policy

## PPO (`model/ppo/`)

- `PPOLightning` — PyTorch Lightning module: actor-critic training, GAE, clipped surrogate objective
- `PPOConfig` — Pydantic config (lr, gamma, GAE lambda, clip epsilon, etc.)
- `PPO_Transformer` — actor-critic model with shared transformer backbone, separate policy and value heads
- `PPOModel` — wraps policy net + value net
- PPO only supports `TransformerNetwork` (no MLP variant)

## Shared (`model/common/`)

- `create_environment()` — factory for `WargameEnv` from config
- `observation_to_tensor` / `observations_to_tensor_batch` — observation conversion
- `RLDataset` — generic RL dataset
- `TransformerConfig` — shared transformer hyperparameters
- `Device` / `get_device()` — device management
- Wandb integration in `wandb.py` — all logging goes through Lightning logger

## Callbacks

- `CheckpointCallback` — saves model checkpoints
- `RecordEpisodeCallback` — records MP4 episodes during training
- `EnvConfigCallback` — persists env YAML config alongside checkpoints

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
- When changing the observation tuple length, expect unpacking errors in tests (`test_state.py`, `test_dqn.py`) — fix them as part of the same change
