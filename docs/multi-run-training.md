# Multi-run training and Wandb

## Running multiple configs in parallel

Use `just train-multi` to run several env configs at once, each in its own process:

```bash
just train-multi config1.yaml config2.yaml config3.yaml
```

- Each run gets a unique **run name** (and thus checkpoint directory) via an automatic `--run-suffix` (1, 2, 3, …).
- All runs from one `train-multi` invocation share a **Wandb group** (e.g. `train-multi-2025-03-14-12-00-00`) so they appear together in the Wandb UI.
- `train-multi` uses PPO and transformer; for other algorithm/network use `just train` or run `train.py` manually.

Each process calls `wandb.init()` independently; Wandb supports multiple concurrent runs and assigns each a unique run ID. No SDK changes are required for concurrency.

## Optional CLI options (single-run)

When running `just train` or `uv run train.py` directly, you can pass:

- **`--run-name`** — Override the base run name explicitly.
- **`--run-suffix`** — Appended to the run name so checkpoint dirs stay unique (e.g. when scripting parallel jobs yourself).
- **`--wandb-group`** — Group name in the Wandb UI for organizing related runs.

If `--run-name` is not provided, the base name is generated from training/env metadata (algorithm, network type, model/objective counts, board size, phase count, and opponent policy type when present), then timestamp/suffix are appended.

## Hyperparameter search (future)

For systematic hyperparameter search, [Wandb Sweeps](https://docs.wandb.ai/guides/sweeps) can be integrated later (sweep config + `wandb agent`), e.g. as a `just sweep` target.
