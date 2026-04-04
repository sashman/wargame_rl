# Testing Patterns

**Analysis Date:** 2026-04-02

## Test Framework

**Runner:**

- **pytest** `>=8.3.4` (declared in `pyproject.toml` dependency group `dev`).
- No `pytest.ini` or `[tool.pytest.ini_options]` in `pyproject.toml`; behavior is pytest defaults plus CLI flags from `Justfile`.

**Assertion library:**

- Plain `assert` statements (pytest introspection).

**Plugins:**

- **pytest-cov** `>=7.0.0` — coverage reports.
- **pytest-rerunfailures** `>=14.0` — used with `@pytest.mark.flaky` for occasional nondeterminism (see below).

**Run commands:**

```bash
just test               # pytest verbose, color, cov on wargame_rl, term-missing + coverage.xml
just validate           # format + lint + test (full gate)
uv run pytest tests     # minimal direct invocation (no coverage unless you add flags)
```

Coverage XML path: `coverage.xml` (from `Justfile` `test` recipe).

## Test File Organization

**Location:**

- All tests live under repository-root `tests/`, not co-located with package sources.

**Naming:**

- `test_*.py` modules. End-to-end or intentionally late-collected files may use a `test_z_*` prefix so `pytest tests` runs heavier tests after lighter ones (`tests/test_z_e2e_training.py` documents this).

**Structure:**

```
tests/
├── conftest.py              # shared fixtures (env, networks, replay buffer, experiences)
├── test_env.py              # WargameEnv reset/step
├── test_reward_phases.py    # reward calculators, criteria, phase manager
├── test_integration.py      # cross-component gaps
├── test_dqn.py / test_ppo.py / test_agent.py
├── test_simulate.py         # simulate CLI + checkpoint discovery
├── test_z_e2e_training.py   # smoke: train() one epoch
└── ...
```

## Test Structure

**Suite organization:**

- Group related cases in one module with section comments and focused `test_*` functions.
- Use descriptive test names: `test_reset_with_seed_is_reproducible`, `test_step_returns_five_tuple` (`tests/test_env.py`).

**Fixtures:**

- **Global shared fixtures** in `tests/conftest.py`: `env`, `n_steps`, `experiences`, `replay_buffer`, `dqn_mlp_net`, `dqn_transformer_net`, parametrized `policy_net` and `ppo_net`.
- **Module-local fixtures** override or specialize `env` / `env_config` (e.g. `tests/test_env.py`).
- **`@lru_cache` on fixtures** (`conftest.py`) to avoid rebuilding large networks repeatedly when the fixture is reused.

**Parametrization:**

- `@pytest.mark.parametrize(..., indirect=True)` to select which network fixture to inject (`tests/test_dqn.py` for `policy_net`).
- Table-style parametrize for multiple scenarios in one test function where appropriate (`tests/test_reward_phases.py` patterns throughout the file).

**Flaky tests:**

- Mark specific parametrized cases with `pytest.mark.flaky(reruns=3)` (e.g. transformer branch in `tests/test_dqn.py` `test_dqn_loss`).

## Mocking

**Framework:** pytest built-ins preferred over `unittest.mock`.

**Patterns:**

- **`monkeypatch`**: replace `torch.cuda.is_available` and clear `lru_cache` on device helpers so CI stays CPU-only (`tests/test_simulate.py`).

```python
def test_simulate_latest_discovers_checkpoint_and_runs_episode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from wargame_rl.wargame.model.common import auto_device

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    auto_device.cache_clear()
```

- **`pytest.raises`**: assert expected exceptions and messages (`tests/test_simulate.py` `test_get_latest_checkpoint_raises_when_no_checkpoints_dir`).

**What to mock:**

- Hardware / environment details (CUDA), not core game logic, unless isolating CLI or filesystem.

**What not to mock:**

- Prefer real `WargameEnv`, real replay buffers, and real Lightning modules for integration-style tests (`tests/test_integration.py`, `tests/test_dqn.py`). The project guidelines discourage mocking except when necessary (external APIs, non-deterministic hardware).

## Fixtures and Factories

**Test data:**

- Build `WargameEnv` and `WargameEnvConfig` inline or via fixtures; use `ModelConfig` / `ObjectiveConfig` for precise placements (`tests/test_integration.py` `test_termination_when_all_at_objective`).
- Roll synthetic trajectories with `env.action_space.sample()` and wrap in `Experience` (`tests/conftest.py` `experiences` fixture).

**Temporary filesystem:**

- Use `tmp_path` for checkpoint dirs and chdir isolation (`tests/test_simulate.py`).

**Checkpoint artifacts:**

- Write minimal Lightning `state_dict` and `env_config.yaml` with `pydantic_yaml.to_yaml_str` to exercise simulate path (`tests/test_simulate.py`).

## Coverage

**Requirements:** No enforced percentage threshold in repo config; coverage is produced every `just test` run.

**View coverage:**

```bash
just test                    # includes --cov-report=term-missing
# XML for CI/tools: coverage.xml
```

## Test Types

**Unit tests:**

- Reward calculator/criteria behavior, registries, distance cache, config edge cases (`tests/test_reward_phases.py`, `tests/test_objective_control_distance_cache.py`).

**Integration tests:**

- Env + replay + dataset wiring (`tests/test_integration.py`); multi-step env with rewards (`test_phased_reward_produces_nonzero_rewards`).

**Training / Lightning tests:**

- Forward shapes, loss decreases after optimization steps, dataloaders (`tests/test_dqn.py`, `tests/test_ppo.py`).

**E2E / smoke:**

- `tests/test_z_e2e_training.py` calls `train()` from `train.py` for **one epoch**, `no_wandb=True`, forces CPU via `CUDA_VISIBLE_DEVICES=""`, uses `examples/env_config/4_models_2_objectives_fixed.yaml`.

**Import smoke:**

- `tests/test_imports.py` exercises PPO imports (legacy style with prints and `return True`); treat as smoke, not a pattern for new tests.

## Common Patterns

**Async testing:** Not used in current suite (synchronous pytest only).

**Error testing:**

```python
with pytest.raises(FileNotFoundError, match="Checkpoints directory not found"):
    sim_module.get_latest_checkpoint()
```

**Headless / CI safety:**

- `render_mode=None` on configs by default in tests.
- For demos that might open pygame, null out `harness.env.renderer` (`tests/test_interactive_demo.py`).

**Determinism:**

- Pass explicit `seed` to `env.reset(seed=...)` when asserting reproducibility (`tests/test_env.py` `test_reset_with_seed_is_reproducible`).
- `torch.manual_seed(42)` before optimizer tests (`tests/test_dqn.py`).

**Property-based / Hypothesis:** Not present in dependencies or tests; `.gitignore` lists `.hypothesis/` for optional local use. Prefer pytest parametrize for combinatorial cases unless you add hypothesis deliberately.

---

*Testing analysis: 2026-04-02*
