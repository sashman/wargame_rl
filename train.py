import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, cast

import torch
import typer
from loguru import logger as log
from pydantic_yaml import parse_yaml_raw_as
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import Callback
from typer.models import OptionInfo

from wargame_rl.wargame.envs.renders.v2.control import ThreatOptions
from wargame_rl.wargame.envs.state import EventLogExporter
from wargame_rl.wargame.envs.types import WargameEnvConfig
from wargame_rl.wargame.model.common import (
    EnvConfigCallback,
    get_checkpoint_callback,
    get_logger,
    init_wandb,
    make_run_name,
)
from wargame_rl.wargame.model.common.config import TransformerConfig
from wargame_rl.wargame.model.common.event_log_callback import EventLogCallback
from wargame_rl.wargame.model.common.factory import create_environment
from wargame_rl.wargame.model.common.performance import configure_matmul_precision
from wargame_rl.wargame.model.common.record_episode_callback import (
    RecordEpisodeCallback,
)
from wargame_rl.wargame.model.common.self_play import SelfPlayConfig
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


def _resolve_transformer_config(
    n_layers: int | None,
    embedding_size: int | None,
) -> TransformerConfig | None:
    """The trunk size to build, or None for the production default.

    Returns **None** when nothing was overridden, rather than an equal
    `TransformerConfig()`. The two build the same network, but None is what
    every caller before these flags existed passed, so the untouched path stays
    literally the untouched path.

    ⚠ Typer fills a default only when *it* invokes the command. Called directly
    -- as `tests/test_z_e2e_training.py` does -- an option nobody passed is an
    `OptionInfo`, which is `not None` and would sail through a plain guard and
    reach Pydantic as a field value. Every override in this file has that trap;
    this one checks the *type* rather than the identity, so a direct caller that
    omits these is right rather than merely lucky.
    """
    # `isinstance(..., int)` rather than `is not None`: see the docstring.
    given = [n_layers, embedding_size]
    if not any(isinstance(value, int) for value in given):
        return None

    defaults = TransformerConfig()
    config = TransformerConfig(
        n_layers=n_layers if isinstance(n_layers, int) else defaults.n_layers,
        embedding_size=(
            embedding_size
            if isinstance(embedding_size, int)
            else defaults.embedding_size
        ),
    )
    # ⚠ There is no --n-heads, on purpose: the head count changes no parameter
    # shape, so a checkpoint written at another one would load SILENTLY and
    # compute differently. Every size these flags can write,
    # `trunk_config_from_state_dict` can read back. The divisibility check
    # remains because the default 8 heads still have to divide a custom width,
    # and Pydantic cannot catch it -- both fields are valid ints alone and only
    # their ratio is wrong, so it would surface as a reshape inside attention.
    if config.embedding_size % config.n_heads != 0:
        raise ValueError(
            f"embedding_size {config.embedding_size} is not divisible by "
            f"n_heads {config.n_heads}"
        )
    log.warning(
        "⚠ NON-DEFAULT NETWORK: {}. Its checkpoint will not load into a default "
        "run and its scores are not comparable to any recorded number.",
        config,
    )
    return config


def _anchor_list(value: str) -> list[str]:
    """Split a comma-separated `--pool-anchor` into the pool's fixed floor.

    Comma-separated rather than a repeated option so every existing invocation
    and Justfile recipe keeps working unchanged: a single name parses to a
    one-element list, which is exactly what the field held before it was a list.
    """
    names = [name.strip() for name in value.split(",") if name.strip()]
    if not names:
        raise ValueError("--pool-anchor needs at least one policy name")
    return names


def _validate_kl_anchor(
    kl_ref_coef: float | None,
    warm_start_ckpt_path: str | None,
) -> None:
    """Refuse a KL anchor with nothing to anchor to.

    Caught at startup rather than at the first minibatch, because the failure
    is otherwise silent and expensive: the run would anchor to whatever random
    initialisation `seed_everything` produced and spend a training window
    holding itself near it.
    """
    if kl_ref_coef is not None and kl_ref_coef > 0.0 and warm_start_ckpt_path is None:
        raise ValueError(
            "kl_ref_coef > 0 requires warm_start_ckpt_path: the anchor is the "
            "weights the run starts from, and there are none to hold onto"
        )


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


@contextmanager
def _trusted_checkpoint_load() -> Iterator[None]:
    """Let `torch.load` unpickle our own checkpoints during a Lightning resume.

    PyTorch 2.6 flipped `torch.load`'s `weights_only` default to True, and a
    checkpoint here pickles the whole `WargameEnv` (it is a Lightning hparam),
    so Lightning's internal restore raises `UnpicklingError: Unsupported global
    ... WargameEnv` on **every** checkpoint this repo has ever written. That
    made `--resume-ckpt-path` unusable, silently: the run dies in seconds and
    the launcher still exits 0.

    Scoped to the resume call and restored afterwards, rather than allowlisting
    globals, because the allowlist would have to name every config and geometry
    type a checkpoint happens to contain and would break again on the next one.
    `TransformerNetwork.from_checkpoint` already loads these files with
    `weights_only=False` for the same reason -- this makes the resume path
    consistent with the read path.
    """
    original = torch.load

    def trusted(*args: Any, **kwargs: Any) -> Any:
        kwargs["weights_only"] = False
        return original(*args, **kwargs)

    torch.load = trusted  # type: ignore[assignment]
    try:
        yield
    finally:
        torch.load = original  # type: ignore[assignment]


def _fit_with_optional_resume(
    trainer: Trainer,
    model: LightningModule,
    resume_ckpt_path: str | None,
) -> None:
    if resume_ckpt_path is None:
        trainer.fit(model)
        return
    with _trusted_checkpoint_load():
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


def _resolve_default(value: Any, default: Any) -> Any:
    """Unwrap a Typer default to a concrete fallback, for direct callers.

    `_resolve_optional_int` returns `None`, which is right for options whose
    absence means "unset". These have real defaults, and a `SelfPlayConfig`
    handed an `OptionInfo` fails validation rather than falling back.
    """
    return default if isinstance(value, OptionInfo) else value


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
    record_threat_range: bool = typer.Option(
        False,
        "--record-threat-range",
        help=(
            "Draw each side's shooting threat footprint in training recordings. "
            "A video has no keyboard, so this is the only way to get it there."
        ),
    ),
    record_threat_field: bool = typer.Option(
        False,
        "--record-threat-field",
        help=(
            "Draw the opponent's NEXT-turn threat field -- after they move -- in "
            "training recordings, as danger bands. Different from "
            "--record-threat-range, which draws only what bears this instant. "
            "Costs a per-layout visibility cache on the first recorded frame."
        ),
    ),
    record_engagement_range: bool = typer.Option(
        False,
        "--record-engagement-range",
        help="Draw each side's engagement range in training recordings.",
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
    n_layers: int | None = typer.Option(
        None,
        help=(
            "Transformer depth. ⚠ CHANGES THE NETWORK: a checkpoint trained at "
            "another size will not load into a default run, and no score is "
            "comparable across it. Defaults to TransformerConfig's 8."
        ),
    ),
    embedding_size: int | None = typer.Option(
        None,
        help=(
            "Transformer width. ⚠ Changes the network -- see --n-layers. "
            "Defaults to TransformerConfig's 256."
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
    kl_ref_target: float | None = typer.Option(
        None,
        help="Drift to hold, in nats per model, adapting --kl-ref-coef each "
        "epoch to reach it. 0.0 (the default) keeps the coefficient fixed. "
        "Unpenalised PPO from a clone drifts to 2.0-2.6 nats here.",
    ),
    kl_ref_coef: float | None = typer.Option(
        None,
        help="Weight on KL(policy || the warm-started weights), holding the run "
        "near the policy it started from. 0.0 (the default) builds no reference "
        "network and adds no term, so a control run is bit-identical to a build "
        "without the anchor. Requires --warm-start-ckpt-path: anchoring to a "
        "random initialisation is meaningless.",
    ),
    self_play: bool = typer.Option(
        False,
        help="Train against a pool of the run's own frozen snapshots, sampled by "
        "PFSP. OFF by default and a no-op when off: no scheduler is built and no "
        "stream is drawn, so a control run is bit-identical to one on a build "
        "without the feature. Do NOT start one on a scenario whose "
        "`just measure-seat-parity` gate fails -- the learner only ever trains "
        "the player seat, so a snapshot on the other seat plays a game it never "
        "practised.",
    ),
    snapshot_every_n_epochs: int = typer.Option(
        25,
        help="How often to freeze the learner into the pool. Written from a "
        "Lightning hook, so SIGKILL writes nothing and a pool is routinely this "
        "many epochs stale -- the same hazard last.ckpt has.",
    ),
    pool_capacity: int = typer.Option(
        8,
        help="How many snapshots to keep, the anchor included. A snapshot is a "
        "full checkpoint on local disk and `checkpoints/` is the only copy of any "
        "weights here, so this is a disk budget before it is a statistical one.",
    ),
    pool_anchor: str = typer.Option(
        "squad_march_take",
        help="Pool entry zero, never evicted -- the floor a pool of nothing but "
        "recent selves would not have.",
    ),
    pfsp_mode: str = typer.Option(
        "hard",
        help="Which opponents to prefer: `hard` (the ones the learner loses to), "
        "`even` (level matchups), or `uniform` -- the CONTROL, which is what to "
        "run an arm against, since a pool changes training on its own and the "
        "schedule is a separate claim.",
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
    _validate_kl_anchor(_resolve_optional_float(kl_ref_coef), warm_start_ckpt_path)

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
    transformer_config = _resolve_transformer_config(n_layers, embedding_size)
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
    resolved_kl_ref_coef = _resolve_optional_float(kl_ref_coef)
    if resolved_kl_ref_coef is not None:
        ppo_config.kl_ref_coef = resolved_kl_ref_coef
    resolved_kl_ref_target = _resolve_optional_float(kl_ref_target)
    if resolved_kl_ref_target is not None:
        ppo_config.kl_ref_target = resolved_kl_ref_target
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
    ppo_net = PPO_Transformer.from_env(env, ppo_config, transformer_config)
    run_name, snapshot_dir = _run_paths(run_name_base, run_suffix)
    self_play_config = SelfPlayConfig(
        enabled=_resolve_default(self_play, False),
        snapshot_every_n_epochs=_resolve_default(snapshot_every_n_epochs, 25),
        pool_capacity=_resolve_default(pool_capacity, 8),
        anchors=_anchor_list(_resolve_default(pool_anchor, "squad_march_take")),
        sampling=cast(Any, _resolve_default(pfsp_mode, "hard")),
    )
    ppo_model = PPOLightning(
        env=env,
        ppo_model=ppo_net,
        self_play=self_play_config,
        snapshot_dir=snapshot_dir,
        seed=resolved_seed or 0,
        **ppo_config.model_dump(),
    )

    config = {
        "wargame": env_config.model_dump(),
        "ppo": ppo_config.model_dump(),
        "training": ppo_training_config.model_dump(),
        "self_play": self_play_config.model_dump(),
    }

    with init_wandb(
        config=config,
        disabled=no_wandb,
        group=wandb_group,
        run_name=run_name,
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
                    overlays=ThreatOptions(
                        show_threat=record_threat_range,
                        show_engagement=record_engagement_range,
                        show_threat_field=record_threat_field,
                    ),
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
            # Inside this branch, not after it: the anchor is the weights the
            # run began FROM, and `_validate_kl_anchor` has already refused a
            # coefficient without a warm start to hold onto.
            ppo_model.attach_kl_reference()
        _fit_with_optional_resume(trainer, ppo_model, resume_ckpt_path)

    if event_exporter and len(event_exporter.log) > 0:
        _write_event_log(event_exporter, run_name_base)


def _run_paths(run_name_base: str, run_suffix: str | None) -> tuple[str, Path]:
    """This run's name, and the directory its self-play pool goes in.

    Derived together from **one** call to `make_run_name`, which is the whole
    point of the helper: that function stamps wall-clock time, so calling it
    twice can land in the next second and put the pool somewhere the
    checkpoints are not.

    ⚠ The pool used to be named from `run_name_base` -- no timestamp and no
    `--run-suffix`, while the checkpoint directory carries both. Two
    consequences, both silent. The documented path `checkpoints/<run>/pool/` was
    empty, so a reader checking the mechanism found nothing and would conclude
    it had failed. And **every self-play run on one env config wrote the same
    filenames into one shared directory**: a pool entry holds a path that is
    loaded lazily when the opponent is seated, so two concurrent runs would seat
    each other's weights as their own past selves, with nothing raised.
    """
    run_name = make_run_name(run_name_base, run_suffix)
    return run_name, Path("checkpoints") / run_name / "pool"


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
