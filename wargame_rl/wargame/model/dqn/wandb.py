from contextlib import contextmanager
from datetime import datetime
from typing import Generator

from pytorch_lightning.loggers import WandbLogger

import wandb

# mypy: disable-error-code="attr-defined,name-defined"

PROJECT_NAME = "wargame_rl"
DEFAULT_NAME = "policy-dqn-env-v2"
ENTITY = "wargame_rl"


@contextmanager
def init_wandb(
    config: dict | None = None, name: str | None = None
) -> Generator[wandb.Run, None, None]:
    if config is None:
        config = {}
    if name is None:
        name = DEFAULT_NAME + "-" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    # Check if wandb is already initialized
    if wandb.run is not None:
        print("Warning: wandb run already exists, finishing previous run")
        wandb.finish()

    try:
        run = wandb.init(project=PROJECT_NAME, config=config, name=name, entity=ENTITY)

        if run is None:
            raise RuntimeError("Failed to initialize wandb run")

        yield run

    except Exception as e:
        print(f"Error during wandb initialization or execution: {e}")
        raise
    finally:
        if wandb.run is not None:
            wandb.finish()


def get_logger(run: wandb.Run) -> WandbLogger:
    # log_model=True -> log the model at the end of the training
    wandb_logger = WandbLogger(log_model=True, run=run)
    return wandb_logger
