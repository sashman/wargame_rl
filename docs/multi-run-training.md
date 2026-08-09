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

### Variants

| Recipe | Use |
|---|---|
| `just train-multi <configs...>` | One run per config, no epoch cap |
| `just train-multi-epochs <max_epochs> <configs...>` | Same, but every arm stops at the same epoch so the arms stay comparable |
| `just train-multi-seeds <max_epochs> <n_seeds> <configs...>` | Every config at each of N seeds, **one seed group at a time**. An arm without an error bar is unreadable: measured within-arm seed spread on win rate is 6–7pp on the 25v25 configs |
| `just train-seed <max_epochs> <seed> <group> <configs...>` | Every config once at one specific seed, into an **existing** Wandb group. For re-running a seed group that died partway — `train-multi-seeds` always mints a fresh group and always starts from seed 1 |
| `just train-arm <max_epochs> <n_seeds> <group> <tag> <flags> <configs...>` | Like `train-multi-seeds`, but passes extra `train.py` flags through and stamps `tag` into the run suffix. For arms that vary a *training* flag (e.g. a PPO override) rather than a config field, which would otherwise collide on run name and checkpoint directory |

Seed groups run sequentially, not all at once: a PPO transformer run holds ~3.8 GB of VRAM, so eight concurrent runs overflow a 24 GB card. They fail *after* startup, on a small allocation partway through training, which is easy to mistake for a clean launch.

### Checkpoints are not uploaded to Wandb

`get_logger` sets `log_model=False`. Nothing in the repo reads a model artifact back — every consumer (`simulate`, `record-sim`, `measure-checkpoint`, `measure-phase-gates`, `--resume-ckpt-path`, `--warm-start-ckpt-path`) takes a local path under `checkpoints/` — while each run uploaded roughly 591 MB, which exhausted the project's storage quota. Metrics, history and recorded videos still log normally.

**`checkpoints/` is therefore the only copy of any trained weights**, which makes `just clean` destructive rather than merely inconvenient.

## Optional CLI options (single-run)

Pass these to `uv run train.py` directly, or as trailing extra arguments to `just train`
(they come after the recipe's five positional arguments — e.g.
`just train config.yaml ppo transformer '' '' --run-name my-run`):

- **`--run-name`** — Override the base run name explicitly.
- **`--run-suffix`** — Appended to the run name so checkpoint dirs stay unique (e.g. when scripting parallel jobs yourself).
- **`--wandb-group`** — Group name in the Wandb UI for organizing related runs.
- **`--seed`** — Seeds weight init, rollout and eval via `seed_everything`. Omitted, runs are seeded from OS entropy: replicates still differ, but neither is reproducible.
- **`--lr`** / **`--max-grad-norm`** — Override the PPO learning rate and the gradient-clipping threshold. Measured on `25v25_shooting_opponent.yaml`: clipping binds on **100% of minibatches** at the 0.5 default for a whole run, so `max_grad_norm` — not `lr` — currently sets the effective step size. See [training-throughput.md](training-throughput.md) and `train/grad_clipped_fraction`.
- **`--no-tf32`** — Keep matmuls at full fp32. TF32 is on by default on sm_80+, and reproducing a run requires matching this flag as well as the seed. See [docs/training-throughput.md](training-throughput.md).
- **`--eval-every-n-epochs`** — Evaluate every Nth epoch instead of every one; ~16% of wall-clock at N=4. Single-phase configs only (raises on a curriculum config). Arms being compared must share the value.
- **`--precision`** — Lightning precision, default `32-true`. `bf16-mixed` is 2.4x on the PPO update but its effect on learning is unmeasured; arms must not mix settings.

If `--run-name` is not provided, the base name is generated from training/env metadata: the env config's **`config_name` first**, then algorithm, network type, model/objective counts, board size, phase count, and opponent policy type when present; timestamp and suffix are appended.

**`config_name` leads the name because everything after it describes the *scenario*, which the arms of an experiment deliberately share.** Four configs differing only in an observation flag once produced byte-identical run names, so every arm wrote checkpoints into one directory and `measure-checkpoint` scored whichever process saved last. `tests/test_train_run_name.py` asserts the arms of a batch stay distinct.

## Hyperparameter search (future)

For systematic hyperparameter search, [Wandb Sweeps](https://docs.wandb.ai/guides/sweeps) can be integrated later (sweep config + `wandb agent`), e.g. as a `just sweep` target.
