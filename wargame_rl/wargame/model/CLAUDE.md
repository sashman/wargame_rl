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

## The set network (`model/per_model/`) — the second family

The network for the per-model facade (`envs/per_model/`, issue #283 stage 2,
#285). **Standalone `nn.Module`, not an `RL_Network`**: that base fixes a
list-of-tensors forward, a policy/value split and a `(B, n_models)` value,
all of which are the whole-army step's contract. One module is encoder +
selector + value + three heads; `SetNetwork.from_env(env, config=None)` and
`from_handler(handler, config=None)` read only the action encoding
(`n_move_actions`, `advance_slice.size`) and **no entity count** — one set of
weights serves any army, unit, objective or terrain count, pinned by
`tests/test_set_network.py` (state-dict shapes are identical across scenario
sizes; one instance plays 6/2 and 10/5 with no reload).

- **Input is the token observation** built by `envs/per_model/tokens.py`
  (numpy, env-side; see `envs/CLAUDE.md`): the acting seat's models as
  queries, everything else — game, both sides' units, enemy models,
  objectives, terrain — as one right-padded **context** with a kind per row
  and the game token at row 0, plus a 16-slot **relation** vector on every
  (model, model) and (model, context) pair. `batch.collate` pads to the
  **batch maximum, never a config budget**; padding and death share the key
  mask. ⚠ The coherency columns of the whole-army token (nearest squadmate,
  spread, component, unit offset) are **dropped here on purpose** — the
  same-unit relation and the offsets carry it — a recorded departure from
  the #283 audit table.
- **Relations live in the attention.** One `Linear(16, n_heads)` per stream
  (self, cross), shared by every block, adds a per-head score pre-softmax;
  the unit pointer re-reads the same vector through a `Linear(16, 1)`, so
  reach, sight and expected damage inform a target choice directly. Context
  tokens are embedded and read, never updated.
- **Which head a step uses is a function of `(kind, phase)`, never learned**
  (`tokens.head_for`): `open` → `declaration (B, 4)`; `act` in movement /
  charge / pile-in / consolidate → `displacement (B, 1 + n_move + n_advance)`;
  `act` in shooting / fight and every `target` → `unit (B, 1 + U)`, a pointer
  over the enemy-unit tokens in **sorted distinct group-id order**. **Column
  0 of the pointer is the no-target option**: STAY on an `act` step (legal for
  a shooter holding fire, **never legal in the fight phase** — a selected
  striker must strike), `CHARGE_TARGET_DECLINE` on a `target` step. Every head
  is masked from the batch; a fully masked row is impossible by construction
  and is asserted, never patched.
- **Value is one scalar per row**, from `[mean of alive real player latents
  ‖ game latent]`, never a model latent. The closing step runs a forward for
  the value and has no policy factor (log-prob 0).
- **`SetAgent`** (`agent.py`) is the seat: builds the tokens (caching each
  env's `TokenScenario` on `env.episode_id`, in a `WeakKeyDictionary` keyed on
  the env object — an `id(env)` key would serve a freed env's terrain to a new
  one at the same address), samples selector then head
  **on the CPU whatever the device**, decodes the two columns, and **raises**
  if `point.why_illegal` refuses the result — a builder/decoder disagreement
  is a bug, not a resample. `declaration_counts` tallies `"<phase>:<option>"`
  on opening steps: the skip declarations gate a whole unit through one logit,
  so their share is the first thing to read at the do-nothing fingerprint.
- **Default trunk 4 layers × 128 × 8 heads (~1.15M params)**, `SetNetworkConfig`,
  `None` = the default exactly as `TransformerConfig` does. Fixtures use
  `SetNetworkConfig(embedding_size=32, n_layers=2, n_heads=4)`; unlike the
  transformer, **`n_heads` IS recoverable** from this family's state dict (the
  relation bias is `Linear(16, n_heads)`), which is why its fixture may shrink
  the head count. Its checkpoints (`checkpoint.py`) are plain tensors plus
  config dicts, loadable with `weights_only=True`, so **nothing pickles this
  package's path** and it may move — unlike the Lightning route.
- `just play-per-model <config> set_network` plays it at fresh weights;
  `just play-per-model <config> <run>/last.pt` plays a trained one, greedily.

## PPO over decision steps (`model/per_model/ppo.py`, `train_per_model.py`) — stage 3

Issue #286. Ordinary single-action PPO — one scalar reward, one value, one
ratio per transition — over the per-model facade, in a **standalone Typer
loop** (`train_per_model.py`), not a Lightning module: `WargameLightningBase`'s
evaluation, baselines and checkpoint callbacks all assume a whole-army env and
a `(batch, n_models)` greedy action. It logs the whole-phase trainer's metric
names, one row per update, with `rounds` and `epoch_equivalent = rounds / 1024`
as columns so either x-axis lines up with the old dashboards; the same rows go
to `<run>/metrics.jsonl`.

- **Two kinds of step.** A *decision step* (`open`, `act`, `target`) carries a
  policy factor (`StepDecision.column >= 0`) and an advantage. A *closing step*
  (`close_turn`, or any step that terminates the episode) carries no policy
  factor: value loss only, excluded from the surrogate, the entropies, the
  ratio statistics and the advantage normalisation. ⚠ "Has a policy" is
  `column >= 0`, never `is_close` — a decision step that terminates is closing
  for the discount and still carries its sampled factor.
- **The clock is in rounds.** `compute_gae` applies `gamma` and `gae_lambda`
  only ACROSS a closing step; every decision within a turn is the same instant
  (step gamma 1.0), so the horizon counts rounds and does not drift as models
  die or shrink with the army. A rollout cut mid-turn bootstraps `r + V(next)`
  at 1.0. The rollout budget is `--rollout-rounds` closing steps per env; the
  eval and checkpoint cadences must be multiples of `rollout_rounds × envs`
  (asserted at construction, or two runs with different budgets evaluate at
  different round counts). The driver counts rounds NOMINALLY per update —
  lockstep envs can close in the same iteration, overshooting by up to
  `n_envs − 1` — and logs the real count as `train/closes`.
- **Lockstep rollouts, one forward per step.** Every env always has exactly one
  pending decision (the close included), so `SetAgent.act_batch` collates N
  observations and draws N rows; `collate` already pads across scenario sizes,
  so envs of different sizes share a batch. Envs persist across updates and are
  **never reset between rollouts** (a budget shorter than the episode must still
  visit its later rounds); a terminating env resets inline with
  `augment_start`, as the shipped collector does, after its outcome is read.
- **The reward is the re-timed stream** from `envs/per_model/reward_timing.py`
  (`PerStepReward`, one per env, own calculator instances): see
  `docs/reward-phases.md` § Where each term is paid under the per-model step.
- **`evaluate_transitions` reproduces the sampled joint log-prob exactly**
  (pinned): selector `log_softmax` over policy rows only (a closing row's
  selector is all `-inf` and would poison the batch), the head gathered by
  `(head, column)`, entropies through a clamped `p · log p` (the naive
  `where` leaks NaN through the unselected branch's gradient). Dropout is
  refused: rollouts sample in train mode, so sampled and recomputed
  log-probs would come from different masks.
- **Two entropy coefficients**, `ent_coef` for the head and
  `selector_ent_coef` for the order (defaults to `ent_coef`), per the design.
- ⚠ **`gamma` 0.9, `gae_lambda` 0.95 and the 16-round budget are carried over
  from the phase facade, where they were measured at two steps per round.**
  The `gamma` comment in `model/ppo/config.py` says verbatim to retest when the
  reward's time structure changes. They are #288's to calibrate
  (`--gamma --gae-lambda --rollout-rounds`) before any race number is quoted.
- **Refused:** curriculum configs (more than one reward phase — `try_advance`
  counts epochs and there is no epoch here yet). **Deferred:** warm start,
  resume, batched eval and recording (#287).
- **Checkpoints** are periodic, not exit-hooked: `pm-<rounds>.pt` and `last.pt`
  every `--checkpoint-every-rounds` (SIGKILL is the prescribed stop and triggers
  no handler, so `last.pt` is at most one interval stale). `load_checkpoint`
  rebuilds the network from the stored config and head sizes and refuses a
  head-size mismatch by name. No top-k: the eval rows in `metrics.jsonl` say
  which periodic checkpoint scored best. The run directory also holds
  `env_config.yaml` (verbatim) and `provenance.json` (revision `+dirty`,
  device, threads, seed bands, both configs).
- **Shared with `train.py`, imported from neither:** the Typer-default
  unwrappers and the config loader live in `model/common/cli.py`, and the
  seed bands and the scripted bar in `model/common/eval_constants.py` (a module
  that imports nothing), so the per-model driver never loads Lightning or the
  phase facade at start. `train.py` and `lightning_base.py` re-export the old
  names.
- **Seeds:** rollout envs reset at `seed × 100 + env_idx` (below the 10000+
  baseline band, derived from `--seed` so two arms at one seed share their
  layouts, unlike the shipped loop's fixed base); in-run eval on `500000+`,
  the scripted bar on `10000+` through the whole-phase facade exactly as
  `on_train_start` measures it.

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
