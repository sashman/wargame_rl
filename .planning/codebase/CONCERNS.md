# Codebase Concerns

**Analysis Date:** 2026-04-02

## Tech Debt

**Dockerfile broken and out of sync:**
- Issue: `ADD` on line 19 is malformed (`warghame_rlghame_rl` concatenated path); image build fails or copies incorrectly.
- Files: `Dockerfile`
- Impact: Docker-based deploy or `just` docker targets are unreliable.
- Fix approach: Use a single valid `COPY`/`ADD` line (e.g. `COPY wargame_rl /app/wargame_rl`), align `WORKDIR` and `uv sync` with package layout, and fix `CMD` to match an actual entrypoint.

**Docker `CMD` does not match `main.py`:**
- Issue: `CMD ["python", "main.py", "--number", "10"]` references `--number`, which does not exist on `main.py`.
- Files: `Dockerfile`, `main.py`
- Impact: Container default command fails at runtime.
- Fix approach: Point `CMD` at `train.py`/`simulate.py` Typer apps or valid `main.py` flags (`--env_test`, etc.).

**Project / image naming typo:**
- Issue: PyPI/project name is `warghame-rl` (typo for “wargame”); propagates to `Justfile` docker tag `warghame-rl`.
- Files: `pyproject.toml`, `Justfile`
- Impact: Confusing branding; harder to discover; rename is a breaking change for published artifacts.
- Fix approach: Rename in `pyproject.toml` and docs when ready for a major/version bump; update Docker tags consistently.

**Legacy env-level terminal bonus fields:**
- Issue: `terminal_success_bonus` and `terminal_vp_bonus` on `WargameEnvConfig` are deprecated in favour of per-phase fields; `apply_legacy_terminal_bonus_defaults` backfills for compatibility.
- Files: `wargame_rl/wargame/envs/types/config.py`, `tests/test_reward_phases.py`
- Impact: Two sources of truth until old YAML is migrated; validators must stay in sync.
- Fix approach: Migrate example configs under `examples/env_config/` to phase-level bonuses; schedule removal behind a deprecation window.

**MLP network marked legacy in product docs:**
- Issue: Documentation describes `MLPNetwork` as legacy while training, simulation, recording, and tests still depend on it.
- Files: `wargame_rl/wargame/model/net.py`, `train.py`, `simulate.py`, `wargame_rl/wargame/model/common/record_episode_callback.py`, `tests/conftest.py`
- Impact: Dual maintenance and test surface until MLP is removed or recording defaults to transformer-only.
- Fix approach: Either drop MLP in a dedicated phase or document it as supported until parity tests exist for transformer-only paths.

**Widespread `type: ignore` / Lightning `hparams` friction:**
- Issue: Multiple `# type: ignore` comments around Lightning `hparams`, `wandb`, gym spaces, and dynamic opponent registry construction.
- Files: `wargame_rl/wargame/model/dqn/lightning.py`, `wargame_rl/wargame/model/ppo/lightning.py`, `wargame_rl/wargame/model/common/lightning_base.py`, `wargame_rl/wargame/envs/wargame.py`, `wargame_rl/wargame/envs/opponent/registry.py`, others per ripgrep
- Impact: Mypy no longer guards those lines; regressions can slip through.
- Fix approach: Narrow ignores with specific codes, typed wrappers for `hparams`, and stricter `OpponentPolicyConfig` validation so `build_opponent_policy` typing is sound.

**Commented CUDA override in training entrypoint:**
- Issue: `# os.environ["CUDA_VISIBLE_DEVICES"] = ""` in `train.py` suggests local CUDA pain; behaviour depends on machine.
- Files: `train.py`
- Impact: Inconsistent runs across dev/CI; users hit opaque CUDA errors.
- Fix approach: Document in README; optional CLI flag or env-based CPU forcing without editing source.

**Dual entrypoints (`main.py` vs Typer CLIs):**
- Issue: `main.py` is a thin legacy/demo path; primary workflows use `train.py` and `simulate.py`.
- Files: `main.py`, `train.py`, `simulate.py`
- Impact: Confusion about canonical commands; `argparse` `type=bool` on flags is a known footgun.
- Fix approach: Deprecate or redirect `main.py` to documented CLIs; align Docker and docs.

## Known Bugs

**Dockerfile build/runtime (see Tech Debt):**
- Symptoms: Invalid `ADD` syntax; default `CMD` fails.
- Files: `Dockerfile`
- Trigger: `docker build` or default container start.
- Workaround: Fix Dockerfile locally or run `uv run python train.py` outside Docker.

## Security Considerations

**Checkpoint loading with full unpickling:**
- Risk: `RL_Network.from_checkpoint` uses `torch.load(..., weights_only=False)`, enabling arbitrary code execution if a checkpoint file is malicious.
- Files: `wargame_rl/wargame/model/net.py`
- Current mitigation: Trust model — checkpoints are expected to be local/trusted artifacts.
- Recommendations: Prefer `weights_only=True` once checkpoint format is pure `state_dict`; document “never load untrusted checkpoints”; align with `record_episode_callback.py` which already uses `weights_only=True` for policy snapshots.

**Wandb and third-party credentials:**
- Risk: API keys in environment (standard Wandb flow).
- Files: Not read (secrets); integration in `wargame_rl/wargame/model/common/wandb.py`, `train.py`
- Current mitigation: Typical `WANDB_API_KEY` / login flow; `--no-wandb` for offline/CI.
- Recommendations: Keep keys out of repo; CI uses `no_wandb` in smoke tests (`tests/test_z_e2e_training.py`).

## Performance Bottlenecks

**Human renderer and episode recording:**
- Problem: `HumanRender` is large and runs full pygame + frame capture; recording spawns a child process with dummy SDL to avoid EGL conflicts with PyTorch.
- Files: `wargame_rl/wargame/envs/renders/human.py`, `wargame_rl/wargame/model/common/record_episode_callback.py`
- Cause: Headless-safe recording requires process isolation and per-frame RGB arrays.
- Improvement path: Optional lower-FPS recording, smaller board for demos, or headless framebuffer-only path without full UI.

**PPO Lightning module size:**
- Problem: `PPOLightning` is a long module mixing optimisation, rollout, and logging.
- Files: `wargame_rl/wargame/model/ppo/lightning.py`
- Cause: Single-class accumulation of features.
- Improvement path: Extract rollout collection and loss computation into smaller units (testability and profiling).

## Fragile Areas

**Multiprocessing recording callback:**
- Files: `wargame_rl/wargame/model/common/record_episode_callback.py`
- Why fragile: Spawned daemon process, temp policy files, SDL driver ordering, and interaction with Wandb video logging; race if epochs advance faster than recording completes (mitigated by skip-if-alive).
- Safe modification: Preserve `SDL_VIDEODRIVER=dummy` before pygame import; keep `weights_only=True` on child load; extend timeouts in `on_train_end` if videos grow.
- Test coverage: Behaviour is integration-heavy; smoke does not assert video bytes.

**Opponent policy wiring:**
- Files: `wargame_rl/wargame/envs/wargame.py`, `wargame_rl/wargame/envs/opponent/registry.py`, `wargame_rl/wargame/envs/types/config.py`
- Why fragile: `build_opponent_policy(config.opponent_policy, ...)` relies on Pydantic validation that `opponent_policy` is present when opponents exist; mypy uses `type: ignore[arg-type]`.
- Safe modification: Add or extend model validators on `WargameEnvConfig` so invalid combos fail at parse time; tighten registry typing.
- Test coverage: `tests/test_opponents.py` is large and primary guard.

**Reward phase and curriculum logic:**
- Files: `wargame_rl/wargame/envs/reward/phase_manager.py`, `wargame_rl/wargame/envs/reward/phase.py`, `tests/test_reward_phases.py`
- Why fragile: Many edge cases (thresholds, min epochs, success criteria registries).
- Safe modification: Change one calculator or criterion at a time; extend `tests/test_reward_phases.py` parametrisation.
- Test coverage: Strong unit coverage in `tests/test_reward_phases.py` (large file).

## Scaling Limits

**Transformer without positional encoding (roadmap gap):**
- Current capacity: Sequence order is learned implicitly; roadmap calls out adding positional encodings.
- Files: `wargame_rl/wargame/model/net.py`, `docs/goals-and-roadmap.md`
- Limit: May cap performance as model/objective counts grow or permutation sensitivity matters.
- Scaling path: Implement learned or sinusoidal PE per roadmap Phase 1.

**Combat and wounds not simulated:**
- Current capacity: `max_wounds` / `current_wounds` appear in model stats and observation space but damage/removal is not implemented.
- Files: `wargame_rl/wargame/envs/domain/entities.py`, `wargame_rl/wargame/envs/domain/battle_factory.py`, `wargame_rl/wargame/envs/types/config.py`
- Limit: Observation includes wound fields that are effectively static during an episode.
- Scaling path: Roadmap Phase 2 (shooting) and related mechanics.

## Dependencies at Risk

**PyTorch / Lightning / CUDA stack:**
- Risk: GPU driver and CUDA mismatches; documented in workspace rules.
- Impact: Training fails or falls back unpredictably without `CUDA_VISIBLE_DEVICES=""`.
- Migration plan: Pin torch builds to CI Python 3.12; document CPU-only training.

**Pinned Plotly:**
- Risk: `plotly==5.24.1` in `pyproject.toml` may lag security or compatibility fixes.
- Impact: Low for a research codebase; possible install conflicts over time.
- Migration plan: Periodically bump with `uv add 'plotly>=...'` and regression-test `wargame_rl/plotting/training.py`.

## Missing Critical Features

**Roadmap vs implementation (selected gaps):**
- Problem: Shooting, terrain, two-player self-play, web replay, and hyperparameter sweep tooling are not implemented; docs describe intended direction.
- Blocks: Production-like adversarial balance and richer tactics until those phases land.
- Files: `docs/goals-and-roadmap.md`, `docs/tabletop-rules-reference.md`

## Test Coverage Gaps

**No enforced minimum coverage:**
- What's not tested: Global coverage threshold is not set in `pytest`/`Justfile`; CI reports coverage to PR comments but does not fail on regression.
- Files: `Justfile`, `.github/workflows/main-validate.yaml`
- Risk: Coverage can drift downward without blocking merge.
- Priority: Medium — add `--cov-fail-under` when baseline is established.

**Docker and container paths:**
- What's not tested: No CI step builds `Dockerfile`; breakage went undetected.
- Files: `Dockerfile`, `.github/workflows/main-validate.yaml`
- Risk: Release or onboarding via Docker fails silently until manual build.
- Priority: High until Dockerfile is repaired.

**Interactive / pygame-heavy paths:**
- What's not tested: Full human render loop and MP4 encoding paths are hard to run headless in unit tests.
- Files: `wargame_rl/wargame/envs/renders/human.py`, `wargame_rl/wargame/model/common/record_episode_callback.py`
- Risk: Recording regressions surface only during training with `--record-during-training`.
- Priority: Medium — optional integration job with xvfb or mocked frames.

---

*Concerns audit: 2026-04-02*
