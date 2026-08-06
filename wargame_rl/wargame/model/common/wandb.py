from contextlib import contextmanager
from datetime import datetime
from types import SimpleNamespace
from typing import Any, Generator

from pytorch_lightning.loggers import CSVLogger, WandbLogger

import wandb
from wargame_rl.wargame.model.common.artifact_retention import (
    DEFAULT_KEEP,
    MODEL_ARTIFACT_TYPE,
    prune_model_artifacts,
)

# mypy: disable-error-code=attr-defined

PROJECT_NAME = "wargame_rl"
DEFAULT_NAME = "policy-dqn-env-v2"
ENTITY = "wargame_rl"


def _make_run_name(name: str | None, run_suffix: str | None = None) -> str:
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    base = name if name else DEFAULT_NAME
    run_name = f"{base}-{timestamp}"
    if run_suffix is not None:
        run_name = f"{run_name}-{run_suffix}"
    return run_name


@contextmanager
def init_wandb(
    config: dict | None = None,
    name: str | None = None,
    disabled: bool = False,
    group: str | None = None,
    run_suffix: str | None = None,
) -> Generator[Any, None, None]:
    """Initialize a wandb run or yield a lightweight stub when disabled.

    When disabled, yields a SimpleNamespace with a `.name` attribute so
    callers can use the same interface for checkpoint / callback naming.
    """
    run_name = _make_run_name(name, run_suffix)

    if disabled:
        yield SimpleNamespace(name=run_name)
        return

    if config is None:
        config = {}

    if wandb.run is not None:
        print("Warning: wandb run already exists, finishing previous run")
        wandb.finish()

    init_kwargs: dict[str, Any] = {
        "project": PROJECT_NAME,
        "config": config,
        "name": run_name,
        "entity": ENTITY,
    }
    if group is not None:
        init_kwargs["group"] = group

    try:
        run = wandb.init(**init_kwargs)

        if run is None:
            raise RuntimeError("Failed to initialize wandb run")

        yield run

    except Exception as e:
        print(f"Error during wandb initialization or execution: {e}")
        raise
    finally:
        if wandb.run is not None:
            wandb.finish()


def get_logger(run: Any = None, *, disabled: bool = False) -> WandbLogger | CSVLogger:
    """Return the Lightning logger for a run.

    `log_model` is off: nothing in this repo ever reads a model artifact back.
    Every checkpoint consumer -- `simulate`, `record-sim`, `measure-checkpoint`,
    `measure-phase-gates`, `--resume-ckpt-path`, `--warm-start-ckpt-path` --
    takes a local path under `checkpoints/`. Uploading was write-only, and at
    ~148 MB a checkpoint and four kept per run it filled the storage quota.
    Metrics, history and recorded videos still go to Wandb.
    """
    if disabled:
        return CSVLogger(save_dir="logs")
    return WandbLogger(log_model=False, run=run)


def _model_artifact_collections() -> list[Any]:
    """Return every `model`-type artifact collection in the project."""
    api = wandb.Api()
    project = api.project(name=PROJECT_NAME, entity=ENTITY)
    return [
        collection
        for artifact_type in project.artifacts_types()
        if artifact_type.name == MODEL_ARTIFACT_TYPE
        for collection in artifact_type.collections()
    ]


def prune_wandb_model_artifacts(
    keep: int = DEFAULT_KEEP,
    *,
    dry_run: bool = False,
) -> list[str]:
    """Delete all but the `keep` most recent model artifacts in the project.

    Args:
        keep: How many of the most recent runs' checkpoints to retain.
        dry_run: Report what would be deleted without deleting it.

    Returns:
        The names that were deleted (or would have been, under `dry_run`).
    """
    return prune_model_artifacts(_model_artifact_collections(), keep, dry_run=dry_run)
