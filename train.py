import os

import typer
from pydantic_yaml import parse_yaml_raw_as
from pytorch_lightning import Trainer

from wargame_rl.wargame.envs.types import WargameEnvConfig
from wargame_rl.wargame.model.dqn.checkpoint_callback import get_checkpoint_callback
from wargame_rl.wargame.model.dqn.config import DQNConfig, NetworkType, TrainingConfig
from wargame_rl.wargame.model.dqn.dqn import DQN_MLP, DQN_Transformer
from wargame_rl.wargame.model.dqn.env_config_callback import EnvConfigCallback
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

    with open(env_config_path) as f:
        env_config = parse_yaml_raw_as(WargameEnvConfig, f.read())  # pyright: ignore[reportUndefinedVariable]

    # Override render_mode with CLI argument (including None)
    env_config.render_mode = render_mode

    return WargameEnvConfig(**env_config.model_dump())


@app.command()
def train(
    render_mode: str | None = typer.Option(
        None, help="Render mode for the environment"
    ),
    env_config_path: str | None = typer.Option(
        None, help="Path to the environment config file"
    ),
    network_type: NetworkType = typer.Option(
        NetworkType.TRANSFORMER, help="Network type to use"
    ),
) -> None:
    """Train the DQN agent."""

    dqn_config = DQNConfig()
    training_config = TrainingConfig()

    env_config = get_env_config(env_config_path, render_mode)

    env = create_environment(env_config=env_config)
    if network_type == NetworkType.TRANSFORMER:
        net = DQN_Transformer.from_env(env)
    else:
        net = DQN_MLP.from_env(env)
    model = DQNLightning(env=env, policy_net=net, **dqn_config.model_dump())

    config = {
        "wargame": env_config.model_dump(),
        "dqn": dqn_config.model_dump(),
        "training": training_config.model_dump(),
    }

    with init_wandb(config=config) as run:
        env_config_callback = EnvConfigCallback(run.name, env_config)
        callbacks = [env_config_callback] + get_checkpoint_callback(run.name)
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
