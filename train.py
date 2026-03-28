import os
from enum import Enum
from typing import cast

import typer
from pydantic_yaml import parse_yaml_raw_as
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback

from wargame_rl.wargame.envs.types import WargameEnvConfig
from wargame_rl.wargame.model.common import (
    EnvConfigCallback,
    get_checkpoint_callback,
    get_logger,
    init_wandb,
)
from wargame_rl.wargame.model.common.factory import create_environment
from wargame_rl.wargame.model.common.record_episode_callback import (
    RecordEpisodeCallback,
)
from wargame_rl.wargame.model.dqn.config import (
    DQNConfig,
    DQNTrainingConfig,
    NetworkType,
)
from wargame_rl.wargame.model.dqn.lightning import DQNLightning
from wargame_rl.wargame.model.net import MLPNetwork, RL_Network, TransformerNetwork
from wargame_rl.wargame.model.ppo.config import PPOConfig, PPOTrainingConfig
from wargame_rl.wargame.model.ppo.lightning import PPOLightning
from wargame_rl.wargame.model.ppo.ppo import PPO_Transformer

# os.environ["CUDA_VISIBLE_DEVICES"] = ""


app = typer.Typer(pretty_exceptions_enable=False)


class AlgorithmType(str, Enum):
    """Type of algorithm to train."""

    DQN = "dqn"
    PPO = "ppo"


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
        "examples/env_config/4_models_2_objectives_fixed.yaml",
        help="Path to the environment config file",
    ),
    algorithm: AlgorithmType = typer.Option(
        AlgorithmType.PPO, help="Algorithm to use for training"
    ),
    network_type: NetworkType = typer.Option(
        NetworkType.TRANSFORMER, help="Network type to use"
    ),
    record_during_training: bool = typer.Option(
        False,
        help="Record a single episode as MP4 when new checkpoints are saved (async, human render)",
    ),
    record_after_epoch: int = typer.Option(
        10,
        help="Start recording only after this many epochs (when record_during_training is enabled)",
    ),
    record_every_n_epochs: int = typer.Option(
        20,
        help="Record every N epochs after recording starts",
    ),
    max_epochs: int | None = typer.Option(
        None,
        help="Override max training epochs (defaults to TrainingConfig value)",
    ),
    no_wandb: bool = typer.Option(
        False,
        help="Disable wandb logging (use local CSV logger instead)",
    ),
    no_inner_progress: bool = typer.Option(
        False,
        help="Disable rollout/PPO tqdm progress bars (e.g. for CI or log redirection)",
    ),
    run_suffix: str | None = typer.Option(
        None,
        help="Optional suffix appended to run name (for unique checkpoint dirs when running multiple jobs in parallel)",
    ),
    wandb_group: str | None = typer.Option(
        None,
        help="Wandb group name to organize runs in the UI (e.g. when running multiple configs in parallel)",
    ),
) -> None:
    """Train the agent."""

    env_config = get_env_config(env_config_path, render_mode)

    # Ensure training episodes are long enough: override to at least 100 rounds
    # unless the config explicitly requests more.
    env_config.number_of_battle_rounds = max(env_config.number_of_battle_rounds, 100)

    env = create_environment(env_config=env_config)

    if algorithm == AlgorithmType.DQN:
        dqn_config = DQNConfig()
        training_config = DQNTrainingConfig(
            record_during_training=record_during_training,
            record_after_epoch=record_after_epoch,
            record_every_n_epochs=record_every_n_epochs,
        )
        if max_epochs is not None:
            training_config.max_epochs = max_epochs

        if network_type == NetworkType.TRANSFORMER:
            net: RL_Network = TransformerNetwork.policy_from_env(env)
        else:
            net = MLPNetwork.policy_from_env(env)
        model = DQNLightning(env=env, policy_net=net, **dqn_config.model_dump())

        config = {
            "wargame": env_config.model_dump(),
            "dqn": dqn_config.model_dump(),
            "training": training_config.model_dump(),
        }

        with init_wandb(
            config=config,
            name=env_config.config_name,
            disabled=no_wandb,
            group=wandb_group,
            run_suffix=run_suffix,
        ) as run:
            env_config_callback = EnvConfigCallback(run.name, env_config)
            dqn_callbacks = cast(
                list[Callback],
                [env_config_callback]
                + get_checkpoint_callback(run.name, filename_prefix="dqn"),
            )
            if training_config.record_during_training:
                dqn_callbacks.append(
                    RecordEpisodeCallback(
                        run.name,
                        env_config,
                        record_during_training=training_config.record_during_training,
                        record_after_epoch=training_config.record_after_epoch,
                        record_every_n_epochs=training_config.record_every_n_epochs,
                        filename_prefix="dqn",
                    )
                )
            logger = get_logger(run, disabled=no_wandb)
            trainer = Trainer(
                accelerator="auto",
                max_epochs=training_config.max_epochs,
                val_check_interval=training_config.val_check_interval,
                logger=logger,
                callbacks=dqn_callbacks,
            )

            trainer.fit(model)

    elif algorithm == AlgorithmType.PPO:
        ppo_config = PPOConfig()
        if no_inner_progress:
            ppo_config.show_inner_progress = False
        ppo_training_config = PPOTrainingConfig(
            record_during_training=record_during_training,
            record_after_epoch=record_after_epoch,
            record_every_n_epochs=record_every_n_epochs,
        )
        if max_epochs is not None:
            ppo_training_config.max_epochs = max_epochs

        if network_type == NetworkType.TRANSFORMER:
            ppo_net = PPO_Transformer.from_env(env)
        else:
            raise NotImplementedError("We will probably never do this.")
        ppo_model = PPOLightning(env=env, ppo_model=ppo_net, **ppo_config.model_dump())

        config = {
            "wargame": env_config.model_dump(),
            "ppo": ppo_config.model_dump(),
            "training": ppo_training_config.model_dump(),
        }

        with init_wandb(
            config=config,
            name=env_config.config_name,
            disabled=no_wandb,
            group=wandb_group,
            run_suffix=run_suffix,
        ) as run:
            env_config_callback = EnvConfigCallback(run.name, env_config)
            ppo_callbacks = cast(
                list[Callback],
                [env_config_callback]
                + get_checkpoint_callback(run.name, filename_prefix="ppo"),
            )
            if ppo_training_config.record_during_training:
                ppo_callbacks.append(
                    RecordEpisodeCallback(
                        run.name,
                        env_config,
                        record_during_training=ppo_training_config.record_during_training,
                        record_after_epoch=ppo_training_config.record_after_epoch,
                        record_every_n_epochs=ppo_training_config.record_every_n_epochs,
                        filename_prefix="ppo",
                    )
                )
            logger = get_logger(run, disabled=no_wandb)
            trainer = Trainer(
                accelerator="auto",
                max_epochs=ppo_training_config.max_epochs,
                val_check_interval=ppo_training_config.val_check_interval,
                logger=logger,
                callbacks=ppo_callbacks,
                log_every_n_steps=1,
            )

            trainer.fit(ppo_model)


if __name__ == "__main__":
    app()
