from contextlib import contextmanager
from datetime import datetime

from pytorch_lightning.loggers import WandbLogger

import wandb

PROJECT_NAME = "wargame-rl"
DEFAULT_NAME = "policy-dqn-env-v2"


@contextmanager
def init_wandb(config: dict | None = None, name: str | None = None):
    if config is None:
        config = {}
    if name is None:
        name = DEFAULT_NAME + "-" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    # Check if wandb is already initialized
    if wandb.run is not None:  # type: ignore
        print("Warning: wandb run already exists, finishing previous run")
        wandb.finish()  # type: ignore

    try:
        run = wandb.init(project=PROJECT_NAME, config=config, name=name)  # type: ignore

        if run is None:
            raise RuntimeError("Failed to initialize wandb run")

        yield run

    except Exception as e:
        print(f"Error during wandb initialization or execution: {e}")
        raise
    finally:
        if wandb.run is not None:  # type: ignore
            wandb.finish()  # type: ignore


def get_logger(run) -> WandbLogger:
    # log_model=True -> log the model at the end of the training
    wandb_logger = WandbLogger(log_model=True, run=run)
    return wandb_logger
