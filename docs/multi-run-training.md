# Multi-run training and Wandb

## Running multiple configs in parallel

Use `just train-multi` to run several env configs at once, each in its own process:

```bash
just train-multi config1.yaml config2.yaml config3.yaml
```

- Each run gets a unique **run name** (and thus checkpoint directory) via an automatic `--run-suffix` (1, 2, 3, …).
- All runs from one `train-multi` invocation share a **Wandb group** (e.g. `train-multi-2025-03-14-12-00-00`) so they appear together in the Wandb UI.
- Override algorithm and network type: `just train-multi config1.yaml config2.yaml algorithm=dqn model=transformer`.

Each process calls `wandb.init()` independently; Wandb supports multiple concurrent runs and assigns each a unique run ID. No SDK changes are required for concurrency.

## Optional CLI options (single-run)

When running `just train` or `uv run train.py` directly, you can pass:

- **`--run-suffix`** — Appended to the run name so checkpoint dirs stay unique (e.g. when scripting parallel jobs yourself).
- **`--wandb-group`** — Group name in the Wandb UI for organizing related runs.

## Hyperparameter search (future)

For systematic hyperparameter search, [Wandb Sweeps](https://docs.wandb.ai/guides/sweeps) can be integrated later (sweep config + `wandb agent`), e.g. as a `just sweep` target.
