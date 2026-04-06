import os
from enum import Enum
from typing import Any, cast

import torch
import typer
from pydantic_yaml import parse_yaml_raw_as
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import Callback
from typer.models import OptionInfo

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


def _build_default_run_base_name(
    algorithm: AlgorithmType,
    network_type: NetworkType,
    env_config: WargameEnvConfig,
) -> str:
    """Build a descriptive run name base from training and env metadata."""
    parts = [
        algorithm.value,
        network_type.value,
        f"m{env_config.number_of_wargame_models}",
        f"opp{env_config.number_of_opponent_models}",
        f"obj{env_config.number_of_objectives}",
        f"b{env_config.board_width}x{env_config.board_height}",
        f"ph{len(env_config.reward_phases)}",
    ]
    if env_config.opponent_policy is not None:
        parts.append(f"vs-{env_config.opponent_policy.type}")
    return "-".join(parts)


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


def _validate_checkpoint_mode(
    resume_ckpt_path: str | None,
    warm_start_ckpt_path: str | None,
) -> None:
    if resume_ckpt_path is not None and warm_start_ckpt_path is not None:
        raise ValueError(
            "resume_ckpt_path and warm_start_ckpt_path are mutually exclusive"
        )
    for checkpoint_path in (resume_ckpt_path, warm_start_ckpt_path):
        if checkpoint_path is not None and not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")


def _extract_state_dict(payload: dict[str, Any]) -> dict[str, torch.Tensor]:
    if "state_dict" in payload and isinstance(payload["state_dict"], dict):
        state_dict = payload["state_dict"]
    else:
        state_dict = payload
    if not isinstance(state_dict, dict):
        raise ValueError(
            "Unsupported checkpoint payload: missing state_dict dictionary"
        )
    output = {k: v for k, v in state_dict.items() if isinstance(v, torch.Tensor)}
    if not output:
        raise ValueError("No tensor weights found in checkpoint state_dict")
    return output


def _apply_warm_start_weights(
    model: LightningModule,
    warm_start_ckpt_path: str,
) -> None:
    loaded = torch.load(warm_start_ckpt_path, map_location="cpu", weights_only=False)
    if not isinstance(loaded, dict):
        raise ValueError(
            f"Unsupported checkpoint format for warm start: {warm_start_ckpt_path}"
        )
    state_dict = _extract_state_dict(loaded)
    model.load_state_dict(state_dict, strict=False)
    if isinstance(model, DQNLightning):
        # Keep online/target aligned when warm-starting from online weights only.
        model.target_net.load_state_dict(model.policy_net.state_dict())
        model.target_net.eval()


def _fit_with_optional_resume(
    trainer: Trainer,
    model: LightningModule,
    resume_ckpt_path: str | None,
) -> None:
    if resume_ckpt_path is None:
        trainer.fit(model)
        return
    trainer.fit(model, ckpt_path=resume_ckpt_path)


def _resolve_optional_str(value: str | OptionInfo | None) -> str | None:
    if isinstance(value, OptionInfo):
        return None
    return value


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
    n_steps: int | None = typer.Option(
        None,
        help="Override PPO rollout steps (defaults to PPOConfig value)",
    ),
    n_eval_episodes: int | None = typer.Option(
        None,
        help="Override number of evaluation episodes per epoch (defaults to config value)",
    ),
    resume_ckpt_path: str | None = typer.Option(
        None,
        help="Resume full training state (model, optimizer, epoch/step) from a Lightning checkpoint.",
    ),
    warm_start_ckpt_path: str | None = typer.Option(
        None,
        help="Load model weights from checkpoint and start fresh optimizer/epoch state.",
    ),
    run_name: str | None = typer.Option(
        None,
        help="Optional run name base. If omitted, a descriptive name is generated from algorithm/network/env settings.",
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
    render_mode = _resolve_optional_str(render_mode)
    env_config_path = _resolve_optional_str(env_config_path)
    run_name = _resolve_optional_str(run_name)
    run_suffix = _resolve_optional_str(run_suffix)
    wandb_group = _resolve_optional_str(wandb_group)
    resume_ckpt_path = _resolve_optional_str(resume_ckpt_path)
    warm_start_ckpt_path = _resolve_optional_str(warm_start_ckpt_path)

    _validate_checkpoint_mode(resume_ckpt_path, warm_start_ckpt_path)

    env_config = get_env_config(env_config_path, render_mode)

    # Ensure training episodes are long enough: override to at least 100 rounds
    # unless the config explicitly requests more.
    env_config.number_of_battle_rounds = max(env_config.number_of_battle_rounds, 100)
    default_run_name = _build_default_run_base_name(algorithm, network_type, env_config)
    run_name_base = run_name if run_name else default_run_name

    env = create_environment(env_config=env_config)

    if algorithm == AlgorithmType.DQN:
        dqn_config = DQNConfig()
        if n_eval_episodes is not None:
            dqn_config.n_episodes = n_eval_episodes
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
            name=run_name_base,
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

            if warm_start_ckpt_path is not None:
                _apply_warm_start_weights(model, warm_start_ckpt_path)
            _fit_with_optional_resume(trainer, model, resume_ckpt_path)

    elif algorithm == AlgorithmType.PPO:
        ppo_config = PPOConfig()
        if no_inner_progress:
            ppo_config.show_inner_progress = False
        if n_steps is not None:
            ppo_config.n_steps = n_steps
        if n_eval_episodes is not None:
            ppo_config.n_episodes = n_eval_episodes
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
            name=run_name_base,
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

            if warm_start_ckpt_path is not None:
                _apply_warm_start_weights(ppo_model, warm_start_ckpt_path)
            _fit_with_optional_resume(trainer, ppo_model, resume_ckpt_path)


if __name__ == "__main__":
    app()
