import os
from typing import Any, cast

import torch
import typer
from loguru import logger as log
from pydantic_yaml import parse_yaml_raw_as
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import Callback
from typer.models import OptionInfo

from wargame_rl.wargame.envs.state import EventLogExporter
from wargame_rl.wargame.envs.types import WargameEnvConfig
from wargame_rl.wargame.model.common import (
    EnvConfigCallback,
    get_checkpoint_callback,
    get_logger,
    init_wandb,
)
from wargame_rl.wargame.model.common.event_log_callback import EventLogCallback
from wargame_rl.wargame.model.common.factory import create_environment
from wargame_rl.wargame.model.common.performance import configure_matmul_precision
from wargame_rl.wargame.model.common.record_episode_callback import (
    RecordEpisodeCallback,
)
from wargame_rl.wargame.model.ppo.config import PPOConfig, PPOTrainingConfig
from wargame_rl.wargame.model.ppo.lightning import PPOLightning
from wargame_rl.wargame.model.ppo.ppo import PPO_Transformer

# os.environ["CUDA_VISIBLE_DEVICES"] = ""


app = typer.Typer(pretty_exceptions_enable=False)

# PPO on a transformer is the only combination this trains, but the pair stays in
# the run name: it is the prefix every wandb run and checkpoint directory has
# carried, and dropping it would split the naming scheme across the point where
# DQN and the MLP policy were removed.
RUN_NAME_PREFIX_PARTS = ("ppo", "transformer")


def _build_default_run_base_name(env_config: WargameEnvConfig) -> str:
    """Build a descriptive run name base from training and env metadata.

    `config_name` leads when set, because everything after it describes the
    *scenario* — board size, force sizes, phase count, opponent — and arms of an
    experiment deliberately share all of those. Four configs differing only in
    an observation flag or one reward term produced byte-identical names, so
    every arm in a batch wrote its checkpoints into one directory and scored
    whichever process happened to save last.
    """
    parts = list(RUN_NAME_PREFIX_PARTS)
    if env_config.config_name:
        parts.append(env_config.config_name)
    parts += [
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


def _resolve_optional_float(value: float | OptionInfo | None) -> float | None:
    """Unwrap a Typer default, as `_resolve_optional_int` does for ints."""
    if isinstance(value, OptionInfo):
        return None
    return value


def _resolve_optional_int(value: int | OptionInfo | None) -> int | None:
    """Unwrap a Typer default for callers that invoke `train()` directly.

    Typer only substitutes real values when it parses argv. Called as a plain
    function -- from tests, or any other Python caller -- the parameter still
    holds the `OptionInfo` sentinel, which is truthy and is not `None`.
    """
    if isinstance(value, OptionInfo):
        return None
    return value


@app.command()
def train(
    render_mode: str | None = typer.Option(
        None, help="Render mode for the environment"
    ),
    env_config_path: str | None = typer.Option(
        "configs/dev/tiny.yaml",
        help="Path to the environment config file",
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
    tf32: bool = typer.Option(
        False,
        help=(
            "Trade result quality for speed: TF32 matmuls are 17.8% faster end "
            "to end but cost ~8.5 vp_margin on 25v25 (measured, two seeds). Off "
            "by default; use for smoke and profiling runs, not for results"
        ),
    ),
    precision: str = typer.Option(
        "32-true",
        help=(
            "Lightning precision. '32-true' is the default and the only setting "
            "any published result was trained under. 'bf16-mixed' is 2.4x on the "
            "PPO update on an RTX 4090 but its effect on learning is unmeasured "
            "-- A/B it over two seeds before trusting a run"
        ),
    ),
    n_steps: int | None = typer.Option(
        None,
        help="Override PPO rollout steps (defaults to PPOConfig value)",
    ),
    n_eval_episodes: int | None = typer.Option(
        None,
        help="Override number of evaluation episodes per epoch (defaults to config value)",
    ),
    eval_every_n_epochs: int | None = typer.Option(
        None,
        help=(
            "Evaluate every Nth epoch instead of every one. At the 30 episodes "
            "the seeded recipes use, evaluation is ~22% of a real epoch. "
            "Single-phase configs only — raises on a curriculum config, where "
            "it would change reward-phase advancement"
        ),
    ),
    gamma: float | None = typer.Option(
        None,
        help="Override PPO discount factor (defaults to PPOConfig value)",
    ),
    gae_lambda: float | None = typer.Option(
        None,
        help=(
            "Override GAE lambda (defaults to PPOConfig value). Raise this WITH "
            "--gamma or the horizon barely moves: the advantage window is "
            "1/(1 - gamma*lambda) steps, so at the default lambda 0.95 a gamma "
            "of 0.9 sees 6.9 steps and 0.99 only reaches 16.8 -- both under half "
            "a 40-step episode. gamma 0.99 with lambda 0.99 gives 50."
        ),
    ),
    ent_coef: float | None = typer.Option(
        None,
        help="Override PPO entropy coefficient (defaults to PPOConfig value)",
    ),
    lr: float | None = typer.Option(
        None,
        help="Override PPO learning rate (defaults to PPOConfig value)",
    ),
    max_grad_norm: float | None = typer.Option(
        None,
        help=(
            "Override the gradient-clipping threshold. Measured as the binding "
            "constraint on this scenario -- train/grad_clipped_fraction is 1.0 "
            "for the whole run at the 0.5 default -- so it, not --lr, is what "
            "currently sets the effective step size"
        ),
    ),
    num_rollout_envs: int | None = typer.Option(
        None,
        help=(
            "Override the number of parallel rollout envs. <=0 auto-detects "
            "from hardware (defaults to PPOConfig value)"
        ),
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
    record_events: bool = typer.Option(
        False,
        help="Record the last training episode as a JSON event log (written to recordings/)",
    ),
    seed: int | None = typer.Option(
        None,
        help="Seed weight init, rollout and eval. Omitted, runs are seeded from OS "
        "entropy: replicates still differ, but neither is reproducible. Set it "
        "whenever running several seeds per arm, so a result can be re-run.",
    ),
) -> None:
    """Train the agent."""
    resolved_seed = _resolve_optional_int(seed)
    if resolved_seed is not None:
        # Covers torch, numpy and python's RNGs in one call. Without it nothing
        # in the training path seeds anything, so a run cannot be reproduced.
        seed_everything(resolved_seed, workers=True)
        log.info("Seeded run with {}", resolved_seed)
    render_mode = _resolve_optional_str(render_mode)
    env_config_path = _resolve_optional_str(env_config_path)
    run_name = _resolve_optional_str(run_name)
    run_suffix = _resolve_optional_str(run_suffix)
    wandb_group = _resolve_optional_str(wandb_group)
    resume_ckpt_path = _resolve_optional_str(resume_ckpt_path)
    warm_start_ckpt_path = _resolve_optional_str(warm_start_ckpt_path)

    _validate_checkpoint_mode(resume_ckpt_path, warm_start_ckpt_path)

    # Process-wide, so it is set before any model is constructed.
    configure_matmul_precision(enabled=tf32)
    resolved_precision = _resolve_optional_str(precision) or "32-true"

    env_config = get_env_config(env_config_path, render_mode)

    default_run_name = _build_default_run_base_name(env_config)
    run_name_base = run_name if run_name else default_run_name

    event_exporter: EventLogExporter | None = None
    if record_events:
        event_exporter = EventLogExporter(anchor_interval=10)

    env = create_environment(
        env_config=env_config,
        state_exporters=[event_exporter] if event_exporter else None,
    )

    ppo_config = PPOConfig()
    if no_inner_progress:
        ppo_config.show_inner_progress = False
    if n_steps is not None:
        ppo_config.n_steps = n_steps
    if n_eval_episodes is not None:
        ppo_config.n_episodes = n_eval_episodes
    # Resolved rather than compared to None directly: called as a plain
    # function the parameter still holds Typer's `OptionInfo` sentinel,
    # which is not None and would be assigned straight into the config.
    resolved_eval_interval = _resolve_optional_int(eval_every_n_epochs)
    if resolved_eval_interval is not None:
        ppo_config.eval_every_n_epochs = resolved_eval_interval
    if gamma is not None:
        ppo_config.gamma = gamma
    resolved_gae_lambda = _resolve_optional_float(gae_lambda)
    if resolved_gae_lambda is not None:
        ppo_config.gae_lambda = resolved_gae_lambda
    if ent_coef is not None:
        ppo_config.ent_coef = ent_coef
    # Resolved rather than compared to None: called as a plain function the
    # parameter still holds Typer's `OptionInfo` sentinel, which is not None
    # and would otherwise be assigned straight into the config.
    resolved_lr = _resolve_optional_float(lr)
    if resolved_lr is not None:
        ppo_config.lr = resolved_lr
    resolved_max_grad_norm = _resolve_optional_float(max_grad_norm)
    if resolved_max_grad_norm is not None:
        ppo_config.max_grad_norm = resolved_max_grad_norm
    if num_rollout_envs is not None:
        ppo_config.num_rollout_envs = num_rollout_envs
    ppo_training_config = PPOTrainingConfig(
        record_during_training=record_during_training,
        record_after_epoch=record_after_epoch,
        record_every_n_epochs=record_every_n_epochs,
    )
    if max_epochs is not None:
        ppo_training_config.max_epochs = max_epochs

    # The config decides whether the model gets a shooting slice, and
    # therefore whether it decodes targets autoregressively.
    ppo_net = PPO_Transformer.from_env(env, ppo_config)
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
        if event_exporter is not None:
            ppo_callbacks.append(EventLogCallback(run_name_base, event_exporter))
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
            precision=resolved_precision,  # type: ignore[arg-type]
        )

        if warm_start_ckpt_path is not None:
            _apply_warm_start_weights(ppo_model, warm_start_ckpt_path)
        _fit_with_optional_resume(trainer, ppo_model, resume_ckpt_path)

    if event_exporter and len(event_exporter.log) > 0:
        _write_event_log(event_exporter, run_name_base)


def _write_event_log(exporter: EventLogExporter, run_name: str) -> None:
    """Serialise the last recorded episode to a JSONL file in recordings/.

    Shares its implementation with the per-epoch `EventLogCallback` so both
    write the same path, and this final call simply supersedes the last one.
    """
    callback = EventLogCallback(run_name, exporter)
    if callback.write():
        log.info(
            f"Event log written to {callback.output_path} ({len(exporter.log)} events)"
        )


if __name__ == "__main__":
    app()
