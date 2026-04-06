# Codebase Concerns

**Analysis Date:** 2026-04-06

## Tech Debt

**Debug print statements scattered through production code:**
- Issue: Raw `print()` calls used instead of the project's `loguru` logger.
- Files: `wargame_rl/wargame/model/net.py` (lines 123–126, 197, 407–412), `wargame_rl/wargame/model/dqn/layers.py` (line 58), `wargame_rl/wargame/model/common/wandb.py` (lines 49, 70)
- Impact: Noisy stdout during training; not controllable via log levels; inconsistent with the rest of the codebase that uses `loguru.logger`.
- Fix approach: Replace all `print()` calls with `logger.info()` / `logger.debug()` / `logger.warning()` as appropriate.

**Excessive `# type: ignore` suppressions (30+ across source):**
- Issue: Many `# type: ignore` comments paper over real type issues instead of fixing the underlying type mismatch.
- Files: `wargame_rl/wargame/model/dqn/lightning.py` (8 suppressions for `self.hparams.*`), `wargame_rl/wargame/model/ppo/lightning.py` (9 suppressions), `wargame_rl/wargame/model/net.py` (3), `wargame_rl/wargame/model/common/lightning_base.py` (2), `wargame_rl/wargame/envs/wargame.py` (1), `wargame_rl/wargame/envs/domain/placement.py` (2)
- Impact: Type errors may be silently introduced; strict mypy loses effectiveness.
- Fix approach: For `self.hparams.*` accesses, store config values as typed instance attributes instead of relying on Lightning's untyped `hparams` dict. For others, add proper overloads or intermediate typed variables.

**Deprecated env-level `terminal_success_bonus` / `terminal_vp_bonus` fields still present:**
- Issue: These top-level `WargameEnvConfig` fields are documented as deprecated in favor of per-phase `RewardPhaseConfig` equivalents, but the backward-compat backfill validator and the fields themselves remain.
- Files: `wargame_rl/wargame/envs/types/config.py` (lines 247–256, 368–391)
- Impact: Two sources of truth for the same setting; confusing for config authors; dead code path once all configs migrate.
- Fix approach: Audit existing YAML configs to confirm none use the top-level fields, then remove the fields and the `apply_legacy_terminal_bonus_defaults` validator.

**Commented-out `configure_optimizers` method in `TransformerNetwork`:**
- Issue: A large block of commented-out code (lines 367–391) implements a weight-decay-aware optimizer that was never integrated.
- Files: `wargame_rl/wargame/model/net.py` (lines 367–391)
- Impact: Dead code clutters the file and may mislead contributors.
- Fix approach: Delete the commented block. If the approach is needed later, it can be recovered from git history or reimplemented.

**`BaseAgent` has a hard dependency on `ReplayBuffer`:**
- Issue: `BaseAgent.__init__` imports and references `ReplayBuffer` from the DQN module even though PPO agents never use it (it's always `None`).
- Files: `wargame_rl/wargame/model/common/agent_base.py` (line 10, 21, 43–44)
- Impact: Unnecessary coupling between the shared agent base and the DQN-specific replay buffer; violates the layering principle.
- Fix approach: Remove the `replay_buffer` attribute from `BaseAgent`. Let the DQN `Agent` subclass own the replay buffer directly.

**`MLPNetwork.from_env` calls `env.reset()` as a side effect:**
- Issue: Both `MLPNetwork.from_env` and `TransformerNetwork.from_env` reset the environment to introspect observation shapes, which is a hidden side effect of a factory method.
- Files: `wargame_rl/wargame/model/net.py` (lines 118–126, 394–421)
- Impact: Re-initializes env state unexpectedly; can cause subtle bugs if called after meaningful env state has been established.
- Fix approach: Pass observation metadata (sizes) directly to the factory instead of requiring a live env reset.

**`WargameEnv.render()` returns `None` explicitly:**
- Issue: The `render()` method has an unnecessary `return None` after the conditional block.
- Files: `wargame_rl/wargame/envs/wargame.py` (line 439)
- Impact: Minor; redundant code.
- Fix approach: Remove the explicit `return None`.

## Known Bugs

**DQN replay buffer warm-up is hardcoded to 200 episodes:**
- Issue: `DQNLightning.populate()` runs exactly 200 random episodes regardless of buffer size or env episode length, with no early-exit once the buffer is sufficiently full.
- Files: `wargame_rl/wargame/model/dqn/lightning.py` (lines 74–86)
- Trigger: Starting DQN training. With short episodes (< 50 steps), 200 episodes may overflow the 10k default buffer. With long episodes, it may take minutes.
- Workaround: Manually adjust the `replay_size` or modify the loop.

**`_collect_experiences` breakdown scaling is inaccurate:**
- Issue: When the collected rollout is truncated to `n_steps`, the breakdown sums are scaled by `used_steps / total_steps`, but the sums were already weighted by episode lengths, making the correction approximate rather than exact.
- Files: `wargame_rl/wargame/model/ppo/lightning.py` (lines 606–617)
- Symptoms: Logged `reward/components/*` metrics may be slightly biased when the last episode is partially used.
- Workaround: None needed for training correctness; only affects logged metrics.

## Security Considerations

**No secrets in codebase:**
- Risk: Low. `.env` files not detected; wandb API key handled externally.
- Files: N/A
- Current mitigation: Wandb auth is handled by the wandb CLI, not by config files in the repo.
- Recommendations: Continue keeping secrets out of YAML configs and source.

**`torch.load` with `weights_only=False`:**
- Risk: `weights_only=False` allows arbitrary pickle deserialization, which can execute arbitrary code if a malicious checkpoint file is loaded.
- Files: `wargame_rl/wargame/model/net.py` (line 49), `train.py` (line 117)
- Current mitigation: Checkpoints are only loaded from local files the user provides via CLI.
- Recommendations: Switch to `weights_only=True` where possible. For Lightning checkpoints that include non-tensor state, add explicit safe loading allowlists.

## Performance Bottlenecks

**Per-step Python loop over environments in parallel rollout:**
- Problem: `_collect_rollout_parallel` steps each environment in a serial Python `for` loop per timestep, negating much of the benefit of batched policy inference.
- Files: `wargame_rl/wargame/model/ppo/lightning.py` (lines 521–534)
- Cause: Gymnasium envs run Python-level `step()` calls that cannot be vectorized without a true subprocess vector env.
- Improvement path: Use `gymnasium.vector.AsyncVectorEnv` or `SyncVectorEnv` to parallelize env stepping across processes, keeping the batched policy inference.

**Observation-to-tensor conversion allocates many small NumPy arrays per step:**
- Problem: `_observation_to_numpy` builds fresh arrays for every observation, including per-model one-hot group encoding, normalization, and stacking.
- Files: `wargame_rl/wargame/model/common/observation.py` (lines 101–163)
- Cause: Functional style with no pre-allocated buffers; each call creates ~10 temporary arrays per model.
- Improvement path: Pre-allocate output buffers sized for the known number of models/objectives and fill them in-place.

**Distance cache recomputes from scratch every step:**
- Problem: `compute_distances` builds full model-objective distance matrices every step even when only a few models moved.
- Files: `wargame_rl/wargame/envs/env_components/distance_cache.py` (lines 70–108)
- Cause: The cache is not persistent; it's recreated each step.
- Improvement path: Persist the cache and only update rows for models that moved (track via `previous_location != location`).

**HumanRender creates a new `pygame.font.Font` every frame:**
- Problem: `_draw_north_panel`, `_draw_south_panel`, and `_draw_model_tooltip` each call `pygame.font.Font(None, ...)` on every render, which involves system font lookup.
- Files: `wargame_rl/wargame/envs/renders/human.py` (lines 274, 301, 413, 546)
- Cause: Fonts not cached as instance attributes.
- Improvement path: Create fonts once in `__init__` or `setup()` and reuse them.

## Fragile Areas

**Observation tensor pipeline (shape coupling across 5 files):**
- Files: `wargame_rl/wargame/envs/env_components/observation_builder.py`, `wargame_rl/wargame/model/common/observation.py`, `wargame_rl/wargame/model/net.py`, `wargame_rl/wargame/model/ppo/networks.py`, `wargame_rl/wargame/model/dqn/lightning.py`
- Why fragile: Adding a new feature column to observations requires coordinated changes across the observation builder (Python dicts), the numpy converter (feature_dim calculation), the tensor converter (indices 0–4), and all neural network input embeddings. The feature dimension is computed dynamically from `n_objectives`, `max_groups`, and hardcoded offsets (+3 for alive/wounds), making it easy to get wrong.
- Safe modification: Always run `just test` after any observation change; verify the `feature_dim` calculation in `_observation_to_numpy` matches `_models_to_features`.
- Test coverage: `tests/test_env.py`, `tests/test_ppo.py`, `tests/test_dqn.py` exercise the pipeline but do not explicitly assert on tensor shapes or feature dimensions.

**`hparams` access pattern in Lightning modules:**
- Files: `wargame_rl/wargame/model/dqn/lightning.py`, `wargame_rl/wargame/model/ppo/lightning.py`
- Why fragile: `self.hparams.gamma`, `self.hparams.lr`, etc. are dynamically typed `Any` values coming from `save_hyperparameters()`. Renaming a constructor parameter or changing its type silently breaks training.
- Safe modification: After any constructor signature change, verify all `self.hparams.*` accesses still resolve to the correct name and type.
- Test coverage: `tests/test_dqn.py` and `tests/test_ppo.py` test training steps but only catch runtime errors, not type mismatches.

**`WargameEnv` exposes private attributes accessed by external code:**
- Files: `wargame_rl/wargame/model/net.py` (accesses `env._action_handler.n_actions`), `wargame_rl/wargame/envs/opponent/scripted_advance_to_objective_policy.py` (accesses `env._opponent_action_handler`)
- Why fragile: Renaming or restructuring `_action_handler` / `_opponent_action_handler` breaks the network factory and opponent policies.
- Safe modification: Expose `n_actions` as a public property on `WargameEnv` (already partially done); for opponent policies, pass the action handler explicitly rather than reaching into `env._*`.
- Test coverage: Tests rely on these access patterns indirectly; breakage would surface as AttributeError.

**Record episode callback depends on internal module structure:**
- Files: `wargame_rl/wargame/model/common/record_episode_callback.py` (lines 186–213)
- Why fragile: Uses `getattr(pl_module, "policy_net")` and `getattr(pl_module, "ppo_model")` with fallback chains. Adding a new algorithm or renaming attributes silently disables recording.
- Safe modification: Define a protocol or abstract property `_policy_model()` on the base Lightning module (already exists in `lightning_base.py`) and use that instead of getattr chains.
- Test coverage: No dedicated test for the recording callback logic.

## Scaling Limits

**Single-process env stepping:**
- Current capacity: 1–8 env instances stepped sequentially in Python (PPO parallel rollout).
- Limit: CPU-bound on env stepping; GPU sits idle during rollout collection.
- Scaling path: Migrate to `gymnasium.vector.AsyncVectorEnv` for true multiprocess stepping, or implement env stepping in C/Cython for complex game logic.

**Replay buffer is in-memory only:**
- Current capacity: 10k experiences in `deque` (default DQN config).
- Limit: Memory grows linearly with buffer size; each `Experience` holds two full `WargameEnvObservation` objects.
- Scaling path: Store only observation indices and reconstruct from a ring buffer of raw states, or offload to memory-mapped storage.

## Dependencies at Risk

**`pydantic-yaml` may be deprecated or stale:**
- Risk: `pydantic-yaml` (`parse_yaml_raw_as`) has had irregular maintenance. Pydantic v2's own YAML support or `pydantic-settings` may supersede it.
- Impact: Config loading in `train.py` (line 77) would break.
- Migration plan: Switch to `yaml.safe_load()` + `WargameEnvConfig(**data)` which is already Pydantic v2-native.

**`imageio` with FFMPEG backend for video recording:**
- Risk: The `imageio[ffmpeg]` plugin requires FFMPEG binaries on the system. Not guaranteed in containerized or CI environments.
- Impact: `RecordEpisodeCallback` fails silently (recording process crashes but training continues).
- Migration plan: Add FFMPEG to Docker/CI images, or switch to `cv2.VideoWriter` which bundles its own codec.

## Missing Critical Features

**No action masking applied during PPO policy inference:**
- Problem: PPO's `_collect_experiences` and `_collect_rollout_parallel` sample actions from `Categorical(logits=logits)` without applying the action mask. The action mask is included in observations but not used to mask invalid actions during training rollouts.
- Files: `wargame_rl/wargame/model/ppo/lightning.py` (lines 508–511, 590–596), `wargame_rl/wargame/model/ppo/networks.py` (line 90)
- Blocks: Correct multi-phase behavior; the agent may select movement actions during shooting phase (or vice versa) wasting steps.

**No learning rate scheduler:**
- Problem: Both DQN and PPO use a fixed learning rate throughout training. No warmup, decay, or cosine annealing is configured.
- Files: `wargame_rl/wargame/model/ppo/lightning.py` (line 171), `wargame_rl/wargame/model/dqn/lightning.py` (lines 189–195)
- Blocks: Long training runs may benefit from LR decay to stabilize late-training performance.

## Test Coverage Gaps

**No tests for `RecordEpisodeCallback`:**
- What's not tested: The entire video recording pipeline — subprocess spawning, SDL dummy driver, frame capture, MP4 writing, and wandb logging.
- Files: `wargame_rl/wargame/model/common/record_episode_callback.py`
- Risk: Recording can silently break (and has, based on the defensive coding style) without any test catching it.
- Priority: Medium — recording is not critical path, but debugging async subprocess failures is painful.

**No tests for PPO parallel rollout collection:**
- What's not tested: `_collect_rollout_parallel` with `num_rollout_envs > 1`, including multi-env stepping, reward breakdown aggregation, and last-value computation.
- Files: `wargame_rl/wargame/model/ppo/lightning.py` (lines 435–572)
- Risk: Shape mismatches or off-by-one errors in the 2D rollout arrays would go undetected.
- Priority: High — this is the primary training path for PPO.

**No tests for warm-start checkpoint loading:**
- What's not tested: `_apply_warm_start_weights` in `train.py` — specifically that partial state dict loading works correctly and that DQN target/online alignment is preserved.
- Files: `train.py` (lines 113–127)
- Risk: Warm-starting from an incompatible checkpoint could silently load partial weights.
- Priority: Medium.

**No tests for the `wandb` integration module:**
- What's not tested: `init_wandb` context manager, run naming, group assignment, error handling, and `get_logger` logger selection.
- Files: `wargame_rl/wargame/model/common/wandb.py`
- Risk: Wandb initialization failures may not be caught until a real training run.
- Priority: Low — wandb integration is simple and rarely changes.

**Observation pipeline lacks explicit shape assertions in tests:**
- What's not tested: The exact tensor shapes produced by `observation_to_tensor` and `observations_to_tensor_batch` for various model/objective counts.
- Files: `wargame_rl/wargame/model/common/observation.py`
- Risk: Adding a new observation feature silently changes tensor dimensions, breaking networks at runtime.
- Priority: High — this is the most common source of shape-mismatch bugs.

---

*Concerns audit: 2026-04-06*
