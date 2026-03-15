from contextlib import contextmanager
from datetime import datetime
from types import SimpleNamespace
from typing import Any, Generator

from pytorch_lightning.loggers import CSVLogger, WandbLogger

import wandb

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
    if disabled:
        return CSVLogger(save_dir="logs")
    return WandbLogger(log_model=True, run=run)
