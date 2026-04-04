# External Integrations

**Analysis Date:** 2026-04-02

## APIs & External Services

**Experiment tracking (Weights & Biases):**

- Wandb — metrics, model checkpoints (`log_model=True`), and optional MP4 episode videos logged from `wargame_rl/wargame/model/common/record_episode_callback.py` (`wandb.Video`).

- SDK: `wandb` Python package.

- Initialization: `wargame_rl/wargame/model/common/wandb.py` — `wandb.init` with `project="wargame_rl"`, `entity="wargame_rl"`, optional `group` for multi-run grouping (`train.py` `--wandb-group`, `Justfile` `train-multi`).

- Auth: Wandb CLI / environment variables as per upstream Wandb docs (e.g. `WANDB_API_KEY`); no secrets committed in-repo.

- Disable path: `train.py` `--no-wandb` uses `CSVLogger` from PyTorch Lightning instead (`wargame_rl/wargame/model/common/wandb.py` `get_logger`).

**Not detected:**

- No Stripe, Supabase, OpenAI, or other third-party HTTP APIs in application code (no `httpx` / `requests` usage for external SaaS in `wargame_rl/`).

## Data Storage

**Databases:**

- None. No SQL/NoSQL client libraries in `pyproject.toml`; no ORM.

**File Storage:**

- Local filesystem — checkpoints (typically under `checkpoints/` per project conventions), Lightning CSV logs under `logs/` when Wandb is disabled, Wandb run artifacts under `wandb/` directory when using local sync.

- Optional dataset path via `PATH_DATASETS` (see `wargame_rl/wargame/model/common/dataset.py`).

**Caching:**

- UV build cache in Docker (`/root/.cache/uv` mount in `Dockerfile`). No application-level Redis/memcached.

## Authentication & Identity

**Auth provider:**

- None in-app. Training/simulation are offline CLI processes.

- Wandb account authentication is external to the codebase (SDK handles identity when logging is enabled).

## Monitoring & Observability

**Error tracking:**

- None (no Sentry/Rollbar).

**Logs:**

- Loguru to stdout (`main.py` and elsewhere).

- Lightning + Wandb/CSV for training metrics.

## CI/CD & Deployment

**Hosting:**

- No cloud deploy target encoded in-repo beyond optional Docker image (`Dockerfile`, `Justfile` `dockerize`).

**CI pipeline:**

- No `.github/workflows` detected in workspace snapshot; validation is local via `just validate` (format, lint, test).

**Developer workflow:**

- GitHub CLI `gh pr create` referenced in `Justfile` `ship` recipe for PR creation (requires user’s `gh` auth, not an application webhook).

## Environment Configuration

**Required env vars:**

- None strictly required for core training if Wandb is disabled (`--no-wandb`) and defaults suffice.

- For Wandb logging: user must satisfy Wandb authentication per their setup.

**Optional env vars:**

- `CUDA_VISIBLE_DEVICES` — force device visibility (e.g. CPU-only).

- `PATH_DATASETS` — dataset directory override.

- `SDL_VIDEODRIVER` — set programmatically to `dummy` during headless recording; rarely needs manual set.

**Secrets location:**

- Wandb API keys and similar: user/environment only; never commit `.env` or credential files.

## Webhooks & Callbacks

**Incoming:**

- None.

**Outgoing:**

- None (Wandb SDK uses its own upload protocol; not user-defined HTTP webhooks).

---

*Integration audit: 2026-04-02*
