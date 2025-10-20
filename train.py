import os

import typer
from pydantic_yaml import parse_yaml_raw_as
from pytorch_lightning import Trainer

from wargame_rl.wargame.envs.types import WargameEnvConfig
from wargame_rl.wargame.model.dqn.callback import get_checkpoint_callback
from wargame_rl.wargame.model.dqn.config import DQNConfig, TrainingConfig
from wargame_rl.wargame.model.dqn.dqn import DQN
from wargame_rl.wargame.model.dqn.factory import create_environment
from wargame_rl.wargame.model.dqn.lightning import DQNLightning
from wargame_rl.wargame.model.dqn.wandb import get_logger, init_wandb

app = typer.Typer(pretty_exceptions_enable=False)


def get_env_config(
    env_config_path: str | None, render_mode: str | None
) -> WargameEnvConfig:
    if env_config_path is None:
        return WargameEnvConfig(render_mode=render_mode)

    if not os.path.exists(env_config_path):
        raise FileNotFoundError(f"Environment config file not found: {env_config_path}")

    return parse_yaml_raw_as(WargameEnvConfig, open(env_config_path).read())  # pyright: ignore[reportUndefinedVariable]


@app.command()
def train(
    render_mode: str = typer.Option(None, help="Render mode for the environment"),
    env_config_path: str = typer.Option(
        None, help="Path to the environment config file"
    ),
):
    """Train the DQN agent."""

    dqn_config = DQNConfig()
    training_config = TrainingConfig()

    env_config = get_env_config(env_config_path, render_mode)

    env = create_environment(env_config=env_config)

    net = DQN.from_env(env)
    model = DQNLightning(env=env, policy_net=net, **dqn_config.model_dump())

    config = {
        "wargame": env_config.model_dump(),
        "dqn": dqn_config.model_dump(),
        "training": training_config.model_dump(),
    }

    with init_wandb(config=config) as run:
        callbacks = get_checkpoint_callback(run.name)
        logger = get_logger(run)
        trainer = Trainer(
            accelerator="auto",
            max_epochs=training_config.max_epochs,
            val_check_interval=training_config.val_check_interval,
            logger=logger,
            callbacks=callbacks,
        )

        trainer.fit(model)


if __name__ == "__main__":
    app()
