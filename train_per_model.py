"""Train the set network over the per-model facade with PPO (issue #286).

A standalone loop beside `train.py`, not a Lightning module: the whole-phase
trainer's evaluation, baselines and checkpoint callbacks assume a whole-army
env and a `(batch, n_models)` greedy action, and a per-model driver fits none
of that. It logs the whole-phase trainer's metric names, one row per update,
with `rounds` and `epoch_equivalent = rounds / 1024` as columns so either
x-axis lines up with the old dashboards.

    uv run train_per_model.py --env-config-path configs/golden/25v25_maps_two_mode.yaml \\
        --rounds 4096 --num-rollout-envs 4

Everything is counted in ROUNDS -- the rollout budget, the eval and checkpoint
cadences, the total -- because that is the unit that does not scale with the
army (#283). Curriculum configs are refused.

⚠ **The regime is `rollout_rounds x num_rollout_envs` rounds per update, and
it is logged on every row as `train/rounds_per_update`.** Left to
auto-detect, the env count is clamped by CPU affinity, and a pinned launch
trains on ONE env -- the 2026-09-07 calibration sweep updated every 16-32
rounds against the phase facade's 1024 and learned nothing
(`reports/2026-09-14-one-episode-per-update.md`). Pass `--num-rollout-envs`.

Two ways back into a run. `--resume-from <run_dir | *.pt>` continues IN PLACE:
same directory, `metrics.jsonl` appended, optimizer and sampling generator
restored, and every knob the checkpoint carries is refused if changed -- a
resumed run is one run. `--warm-start-from <*.pt>` takes the weights alone
onto any scenario whose displacement head matches (the set network reads no
entity count, so a three-model rung's checkpoint starts a twenty-four-model
one) with a fresh optimizer and a new run directory.
"""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import typer
from loguru import logger

from wargame_rl.wargame.envs.baseline.evaluate import evaluate_baseline
from wargame_rl.wargame.envs.baseline.registry import build_baseline_policy
from wargame_rl.wargame.envs.evaluation import format_optional_metric
from wargame_rl.wargame.envs.per_model.env import PerModelEnv
from wargame_rl.wargame.envs.per_model.recording import record_episode
from wargame_rl.wargame.envs.per_model.reward_timing import Credit, PerStepReward
from wargame_rl.wargame.envs.per_model.types import (
    BatchChooser,
    PerModelAction,
    PerModelObservation,
)
from wargame_rl.wargame.envs.types import WargameEnvConfig
from wargame_rl.wargame.model.common.cli import (
    get_env_config,
    resolve_default,
    resolve_optional_float,
    resolve_optional_int,
    resolve_optional_str,
)
from wargame_rl.wargame.model.common.eval_constants import (
    BASELINE_EPISODES,
    BASELINE_POLICIES,
    BASELINE_SEED_BASE,
    EVAL_SEED_BASE,
    EVAL_WAVE_SIZE,
)
from wargame_rl.wargame.model.common.factory import create_environment
from wargame_rl.wargame.model.common.wandb import init_wandb
from wargame_rl.wargame.model.per_model.agent import SetAgent
from wargame_rl.wargame.model.per_model.checkpoint import (
    LAST_CHECKPOINT,
    LoadedCheckpoint,
    TrainingState,
    load_checkpoint,
    load_training_state,
    periodic_checkpoint_name,
    save_checkpoint,
)
from wargame_rl.wargame.model.per_model.config import SetNetworkConfig
from wargame_rl.wargame.model.per_model.evaluate import EvalResult, evaluate_per_model
from wargame_rl.wargame.model.per_model.net import SetNetwork
from wargame_rl.wargame.model.per_model.ppo import (
    EpisodeOutcome,
    PerModelPPOConfig,
    Rollout,
    RolloutEntropy,
    UpdateStats,
    adapt_kl_coef,
    affinity_cpu_count,
    auto_num_rollout_envs,
    check_trainable,
    collect_rollout,
    compute_gae,
    device_max_rollout_envs,
    ppo_update,
    rollout_entropy,
)
from wargame_rl.wargame.rating.elo import rating_from_score
from wargame_rl.wargame.rating.score import margin_score

app = typer.Typer(pretty_exceptions_enable=False)

# A whole-phase epoch is 2048 steps at two steps per round.
ROUNDS_PER_EPOCH_EQUIVALENT = 1024
CHECKPOINT_ROOT = Path("checkpoints") / "per_model"
# Rollout envs are seeded off `--seed`, below the 10000+ baseline band, so
# two arms at one seed share their layouts (the shipped loop's base is fixed).
ROLLOUT_SEED_STRIDE = 100
# The knobs a resumed run may not change: the checkpoint's copy wins, and an
# explicit flag that disagrees is refused by name rather than silently
# overridden either way.
_PPO_KNOBS = (
    "rollout_rounds",
    "num_rollout_envs",
    "gamma",
    "gae_lambda",
    "ent_coef",
    "selector_ent_coef",
    "lr",
    "max_grad_norm",
    "n_epochs",
    "batch_size",
    "kl_ref_coef",
    "kl_ref_target",
    "credit",
)


def resolve_device(name: str) -> torch.device:
    """`auto` picks CUDA when it is usable."""
    if name != "auto":
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def git_revision() -> str:
    """The short revision, `+dirty` when the tree has changes; `unknown` off git."""
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return f"{revision}+dirty" if dirty else revision


def refuse_curriculum(config: WargameEnvConfig) -> None:
    """One reward phase only; advancement is a follow-up to #286."""
    if len(config.reward_phases) != 1:
        raise ValueError(
            f"the per-model driver trains a single reward phase; this config "
            f"carries {len(config.reward_phases)} (curriculum advancement is a "
            "follow-up to #286)"
        )


def check_cadence(every: int, rollout_total: int, what: str) -> None:
    """A cadence that is not a multiple of the rollout drifts with the
    overshoot, so two runs with different budgets would evaluate at different
    round counts."""
    if every < rollout_total or every % rollout_total != 0:
        raise ValueError(
            f"{what} ({every} rounds) must be a positive multiple of the rollout "
            f"({rollout_total} rounds = rollout_rounds x envs)"
        )


def affinity_clamp_warning(n_envs: int, device: torch.device) -> str | None:
    """The warning an auto-detected env count earns when affinity clamped it.

    None when the count reached the device ceiling. The message names the
    visible CPU count because that is what a pinned launch (`taskset`, a
    cgroup, a container limit) changed without anyone asking for it.
    """
    ceiling = device_max_rollout_envs(device)
    if n_envs >= ceiling:
        return None
    return (
        f"⚠ num_rollout_envs auto-clamped to {n_envs} by CPU affinity "
        f"({affinity_cpu_count()} CPUs visible to this process; the {device.type} "
        f"ceiling is {ceiling}). Rounds per update scale with the env count, so "
        "this run trains in a different regime from one launched unpinned -- "
        "pass --num-rollout-envs to say which regime you mean."
    )


class MetricsLog:
    """One row per event, to a local JSONL and (optionally) to wandb."""

    def __init__(self, path: Path, *, to_wandb: bool) -> None:
        self.path = path
        self.to_wandb = to_wandb

    def log(self, row: dict[str, Any]) -> None:
        with self.path.open("a") as handle:
            handle.write(json.dumps(row) + "\n")
        if self.to_wandb:
            import wandb

            wandb.log(row)  # type: ignore[attr-defined]


def baseline_rows(env_config: WargameEnvConfig) -> dict[str, float]:
    """The scripted bar, measured through the whole-phase facade exactly as
    `WargameLightningBase.on_train_start` measures it."""
    env = create_environment(env_config=env_config)
    seeds = [BASELINE_SEED_BASE + i for i in range(BASELINE_EPISODES)]
    rows: dict[str, float] = {}
    for name in BASELINE_POLICIES:
        result = evaluate_baseline(build_baseline_policy(name), env, seeds)
        rows[f"eval/baseline_{name}_win_rate"] = 100.0 * result.win_rate
        rows[f"eval/baseline_{name}_vp_margin"] = result.vp_margin
        rows[f"eval/baseline_{name}_at_objectives"] = (
            result.final_fraction_at_objectives
        )
        rows[f"eval/baseline_{name}_fraction_alive"] = result.final_fraction_alive
        rows[f"eval/baseline_{name}_objectives_held"] = result.objectives_held
    env.close()
    return rows


def eval_rows(result: EvalResult, prefix: str = "") -> dict[str, float]:
    """The whole-phase trainer's eval keys, read off the shared result.

    `mean_episode_decisions`, not `mean_episode_steps`: this facade's unit is
    the decision, and logging it under the phase facade's key was a same-key
    different-unit trap. Coherency is the tracker's, on the phase facade's
    grid; a column this facade does not measure is omitted, not zeroed. The
    passive pair (`eval/stationary_share`, `eval/hold_fire_share`) is the
    do-nothing fingerprint, and it is the GREEDY policy's -- the one a score
    reports -- where `train/declaration/*` is the sampled one's.
    """
    suffix = f"{prefix}_" if prefix else ""
    margins = np.array(result.vp_margin_per_episode, dtype=np.float64)
    elo = float(rating_from_score(float(margin_score(margins).mean())))
    rows: dict[str, float] = {
        f"eval/{suffix}vp_player": result.player_vp,
        f"eval/{suffix}vp_opponent": result.opponent_vp,
        f"eval/{suffix}vp_margin": result.vp_margin,
        f"eval/{suffix}win_rate": 100.0 * result.win_rate,
        f"eval/{suffix}elo": elo,
        f"eval/{suffix}fraction_alive": result.final_fraction_alive,
        f"eval/{suffix}at_objectives": result.final_fraction_at_objectives,
        f"eval/{suffix}objectives_held": result.objectives_held,
    }
    optional: dict[str, float | None] = {
        f"eval/{suffix}vp_margin_se": result.vp_margin_se,
        f"eval/{suffix}stationary_share": result.stationary_share,
        f"eval/{suffix}hold_fire_share": result.hold_fire_share,
        f"eval/{suffix}coherency_rate": result.coherency_rate,
        f"eval/{suffix}models_out_of_coherency": result.models_out_of_coherency,
        f"reward/{suffix}mean_episode_reward": result.mean_reward,
        f"reward/{suffix}max_episode_reward": result.max_reward,
        f"reward/{suffix}min_episode_reward": result.min_reward,
        f"{suffix}mean_episode_decisions": result.mean_decisions,
        f"{suffix}success_rate": (
            None if result.success_rate is None else 100.0 * result.success_rate
        ),
        # Turns-to-success on the phase clock, so a curriculum rung's speed
        # readout has an in-run curve and not only its final n=100 read.
        f"eval/{suffix}mean_turns": result.mean_turns,
    }
    rows.update({key: value for key, value in optional.items() if value is not None})
    return rows


def update_rows(
    stats: UpdateStats,
    rollout: Rollout,
    *,
    entropy: RolloutEntropy,
    declarations: Counter[str],
    rollout_s: float,
    update_s: float,
    rounds_per_update: int,
    rounds_done: int,
    approx_kl_cumulative: float,
) -> dict[str, float]:
    """One update's row: the losses, the health panel, the entropies by phase
    and by head, the reward breakdown per round, and the sampled declaration
    shares. `docs/metrics.md` § The per-model health panel says what each
    panel key reads as when healthy."""
    rows: dict[str, float] = {
        "steps": float(rollout.n_steps),
        "loss/train_loss": stats.train_loss,
        "loss/policy_loss": stats.policy_loss,
        "loss/value_loss": stats.value_loss,
        "loss/entropy_loss": stats.entropy_loss,
        "train/clip_fraction": stats.clip_fraction,
        "train/approx_kl": stats.approx_kl,
        "train/approx_kl_cumulative": approx_kl_cumulative,
        "train/approx_kl_per_1k_rounds": (
            1000.0 * approx_kl_cumulative / max(1, rounds_done)
        ),
        "train/kl_ref": stats.kl_ref,
        "train/kl_ref_coef": stats.kl_ref_coef,
        "train/explained_variance": stats.explained_variance,
        "train/grad_norm": stats.grad_norm,
        "train/grad_clipped_fraction": stats.grad_clipped_fraction,
        "train/advantage_mean": stats.advantage_mean,
        "train/advantage_std": stats.advantage_std,
        "train/advantage_abs_max": stats.advantage_abs_max,
        "train/return_mean": stats.return_mean,
        "train/return_std": stats.return_std,
        "train/value_mean": stats.value_mean,
        "train/value_std": stats.value_std,
        "train/ratio_p01": stats.ratio_p01,
        "train/ratio_p99": stats.ratio_p99,
        "train/rounds_per_update": float(rounds_per_update),
        "train/gradient_steps_per_round": stats.n_minibatches
        / max(1, rounds_per_update),
        "train/num_rollout_envs_resolved": float(rollout.n_envs),
        "train/entropy/selector": entropy.selector,
        "perf/rollout_s": rollout_s,
        "perf/update_s": update_s,
        "perf/epoch_s": rollout_s + update_s,
        "perf/env_steps_per_s": rollout.n_steps / max(rollout_s, 1e-9),
        "perf/update_ms_per_minibatch": 1000.0 * update_s / max(1, stats.n_minibatches),
        "train/episodes": float(len(rollout.episodes)),
    }
    for phase, value in entropy.by_phase.items():
        rows[f"train/entropy/{phase}"] = value
    for head, value in entropy.by_head.items():
        rows[f"train/entropy/head/{head}"] = value
    if rollout.episodes:
        rows["train/vp_margin"] = float(
            np.mean([e.player_vp - e.opponent_vp for e in rollout.episodes])
        )
    # Per ROUND, not per step: under re-timing a per-step mean would depend
    # on the army size.
    for key, value in rollout.breakdown.items():
        rows[f"reward/components/{key}"] = value / max(1, rollout.closes)
    total = sum(declarations.values())
    for key, count in declarations.items():
        rows[f"train/declaration/{key}"] = count / max(1, total)
    return rows


@dataclass(frozen=True)
class ResumePoint:
    """A run being continued in place: its directory, its last checkpoint,
    and the training state that checkpoint carried."""

    checkpoint: Path
    run_dir: Path
    loaded: LoadedCheckpoint
    state: TrainingState


def resolve_resume(spec: str) -> ResumePoint:
    """`spec` is a run directory (its `last.pt`) or a checkpoint inside one."""
    path = Path(spec)
    checkpoint = path / LAST_CHECKPOINT if path.is_dir() else path
    if not checkpoint.exists():
        raise ValueError(f"nothing to resume at {checkpoint}")
    return ResumePoint(
        checkpoint=checkpoint,
        run_dir=checkpoint.parent,
        loaded=load_checkpoint(checkpoint),
        state=load_training_state(checkpoint),
    )


def refuse_changed_knobs(
    overrides: dict[str, Any], stored: PerModelPPOConfig, checkpoint: Path
) -> None:
    """A resumed run is one run: every explicit PPO flag must agree with the
    checkpoint's copy. Silently taking either side would make the metrics
    file lie about half of itself."""
    stored_values = stored.model_dump()
    changed = {
        key: (value, stored_values[key])
        for key, value in overrides.items()
        if value is not None and key in _PPO_KNOBS and value != stored_values[key]
    }
    if changed:
        described = ", ".join(
            f"{key}={asked!r} (checkpoint {kept!r})"
            for key, (asked, kept) in changed.items()
        )
        raise ValueError(
            f"--resume-from {checkpoint} refuses changed knobs: {described}. "
            "A resumed run keeps its regime; warm-start into a new run to "
            "change one."
        )


def refuse_changed_trunk(
    requested: dict[str, int], stored: SetNetworkConfig, checkpoint: Path
) -> None:
    """The checkpoint's trunk is the network; a trunk flag that disagrees
    names a network the weights cannot load into."""
    stored_values = stored.model_dump()
    changed = {
        key: (value, stored_values[key])
        for key, value in requested.items()
        if value != stored_values[key]
    }
    if changed:
        described = ", ".join(
            f"{key}={asked} (checkpoint {kept})"
            for key, (asked, kept) in changed.items()
        )
        raise ValueError(
            f"{checkpoint} was trained with a different trunk: {described}"
        )


@app.command()
def train(
    env_config_path: str | None = typer.Option(
        None, help="Path to the env config (taken from the run on --resume-from)."
    ),
    rounds: int = typer.Option(1024, help="Total training budget, in rounds."),
    rollout_rounds: int | None = typer.Option(
        None, help="Closing steps per env per update (default 16)."
    ),
    num_rollout_envs: int | None = typer.Option(
        None,
        help="Lockstep rollout envs; 0 or unset auto-detects (and warns when "
        "CPU affinity clamps it). Rounds per update = rollout_rounds x envs.",
    ),
    gamma: float | None = typer.Option(None, help="Per-ROUND discount."),
    gae_lambda: float | None = typer.Option(None, help="Per-ROUND GAE decay."),
    ent_coef: float | None = typer.Option(None, help="Head entropy coefficient."),
    selector_ent_coef: float | None = typer.Option(
        None, help="Selector entropy coefficient (default: ent_coef)."
    ),
    kl_ref_coef: float | None = typer.Option(
        None,
        "--kl-ref-coef",
        help="Weight on the KL anchor to the run's starting weights (#332); "
        "0 builds no reference.",
    ),
    kl_ref_target: float | None = typer.Option(
        None,
        "--kl-ref-target",
        help="Drift to hold, in nats per decision; makes the coefficient "
        "adaptive. 0 keeps it fixed.",
    ),
    credit: str | None = typer.Option(
        None,
        "--credit",
        help="Who a payment reaches: `mean` (the bridge accounting, default) or "
        "`actor` (each model its own action term undivided and its own state "
        "credit on its own step).",
    ),
    lr: float | None = typer.Option(None),
    max_grad_norm: float | None = typer.Option(None),
    n_epochs: int | None = typer.Option(None),
    batch_size: int | None = typer.Option(None),
    eval_every_rounds: int = typer.Option(
        256, help="Evaluate every N rounds (a multiple of the rollout)."
    ),
    n_eval_episodes: int = typer.Option(20),
    eval_seed_base: int = typer.Option(
        EVAL_SEED_BASE, help="First in-run eval seed; written to provenance."
    ),
    eval_wave_size: int = typer.Option(
        EVAL_WAVE_SIZE, help="Eval episodes stepped in lockstep per wave."
    ),
    checkpoint_every_rounds: int = typer.Option(
        256, help="Write pm-NNNNNNNN.pt and last.pt every N rounds."
    ),
    record_every_rounds: int | None = typer.Option(
        None,
        help="Record one GREEDY episode (an event log at decision cadence, "
        "under <run_dir>/recordings/) every N rounds. Default: the checkpoint "
        "cadence, so every checkpoint has a recording beside it; 0 disables.",
    ),
    record_seed: int = typer.Option(
        EVAL_SEED_BASE,
        help="The seed every recording plays (the in-run eval band's first).",
    ),
    video_every_rounds: int | None = typer.Option(
        None,
        help="Render one of the recordings to an MP4 and log it to Wandb as "
        "`episode_recording` every N rounds (a multiple of the recording "
        "cadence). Default: every twentieth recording, the whole-army trainer's "
        "cadence; 0 disables. The MP4 lands beside its event log.",
    ),
    video_fps: int = typer.Option(5, help="Frames per second of the rendered MP4."),
    video_theme: str = typer.Option(
        "tabletop", help="v2 theme the MP4 is drawn in: 'default' or 'tabletop'."
    ),
    backward_start: int = typer.Option(
        0,
        help="The backward start curriculum: begin at this many squads already "
        "standing on distinct objectives and walk the count down to 0 as the "
        "rollouts succeed (#340). 0 is off. Evaluation always starts from "
        "deployment.",
    ),
    backward_start_share: float = typer.Option(
        0.75,
        help="The share of rollout episodes that start at the current level; "
        "the rest start from deployment so the walk in is never forgotten.",
    ),
    backward_start_advance: float = typer.Option(
        0.8,
        help="Step the level down once the level's own episodes succeed at this "
        "rate over the window.",
    ),
    backward_start_window: int = typer.Option(
        8, help="Rollouts over which the level's success rate is read."
    ),
    n_layers: int | None = typer.Option(None, help="Trunk depth (default 4)."),
    embedding_size: int | None = typer.Option(None, help="Trunk width (default 128)."),
    seed: int | None = typer.Option(None),
    torch_threads: int = typer.Option(
        2, help="torch.set_num_threads; a thread setting moved a score by 2 vp."
    ),
    device: str = typer.Option("auto"),
    no_wandb: bool = typer.Option(False, "--no-wandb"),
    run_name: str | None = typer.Option(None),
    run_suffix: str | None = typer.Option(None),
    wandb_group: str | None = typer.Option(None),
    checkpoint_root: str = typer.Option(str(CHECKPOINT_ROOT)),
    resume_from: str | None = typer.Option(
        None,
        "--resume-from",
        help="A run directory or its .pt: continue that run in place, with "
        "its optimizer, generator and knobs; --rounds must exceed its count.",
    ),
    warm_start_from: str | None = typer.Option(
        None,
        "--warm-start-from",
        help="A .pt whose weights start a NEW run (fresh optimizer) on any "
        "scenario with the same displacement head.",
    ),
) -> Path:
    """Train; returns the run directory."""
    resume_spec = resolve_optional_str(resume_from)
    warm_spec = resolve_optional_str(warm_start_from)
    if resume_spec is not None and warm_spec is not None:
        raise ValueError(
            "--resume-from continues a run and --warm-start-from begins one; "
            "pass one or the other"
        )
    resume = None if resume_spec is None else resolve_resume(resume_spec)

    resolved_seed = resolve_optional_int(seed)
    if resume is not None:
        if resolved_seed is not None and resolved_seed != resume.loaded.seed:
            raise ValueError(
                f"--resume-from {resume.checkpoint} refuses seed {resolved_seed}: "
                f"the run's seed is {resume.loaded.seed!r}"
            )
        resolved_seed = resume.loaded.seed
    if resolved_seed is not None:
        torch.manual_seed(resolved_seed)
        np.random.seed(resolved_seed)
    torch.set_num_threads(int(resolve_default(torch_threads, 2)))
    resolved_device = resolve_device(str(resolve_default(device, "auto")))

    config_path = resolve_optional_str(env_config_path)
    if resume is not None:
        env_config, config_path = _resumed_env_config(resume, config_path)
    elif config_path is None:
        raise ValueError("--env-config-path is required unless --resume-from")
    else:
        env_config = get_env_config(config_path, None)
    refuse_curriculum(env_config)

    overrides: dict[str, Any] = {
        "rollout_rounds": resolve_optional_int(rollout_rounds),
        "num_rollout_envs": resolve_optional_int(num_rollout_envs),
        "gamma": resolve_optional_float(gamma),
        "gae_lambda": resolve_optional_float(gae_lambda),
        "ent_coef": resolve_optional_float(ent_coef),
        "selector_ent_coef": resolve_optional_float(selector_ent_coef),
        "lr": resolve_optional_float(lr),
        "max_grad_norm": resolve_optional_float(max_grad_norm),
        "n_epochs": resolve_optional_int(n_epochs),
        "batch_size": resolve_optional_int(batch_size),
        "kl_ref_coef": resolve_optional_float(kl_ref_coef),
        "kl_ref_target": resolve_optional_float(kl_ref_target),
        "credit": Credit(credit) if resolve_optional_str(credit) else None,
    }
    if resume is not None:
        refuse_changed_knobs(overrides, resume.loaded.ppo_config, resume.checkpoint)
        ppo_config = resume.loaded.ppo_config
        n_envs = ppo_config.num_rollout_envs
    else:
        ppo_config = PerModelPPOConfig().model_copy(
            update={k: v for k, v in overrides.items() if v is not None}
        )
        ppo_config = PerModelPPOConfig(**ppo_config.model_dump())
        n_envs = ppo_config.num_rollout_envs
        if n_envs <= 0:
            n_envs = auto_num_rollout_envs(resolved_device)
            warning = affinity_clamp_warning(n_envs, resolved_device)
            if warning is not None:
                logger.warning(warning)
        # Written back so the persisted config and every checkpoint say how
        # many envs produced the run, not the `0` that asked for auto-detection.
        ppo_config = PerModelPPOConfig(
            **ppo_config.model_copy(update={"num_rollout_envs": n_envs}).model_dump()
        )
    rollout_total = ppo_config.rollout_rounds * n_envs
    eval_every = int(resolve_default(eval_every_rounds, 256))
    checkpoint_every = int(resolve_default(checkpoint_every_rounds, 256))
    record_every = (
        checkpoint_every
        if resolve_optional_int(record_every_rounds) is None
        else int(resolve_optional_int(record_every_rounds) or 0)
    )
    record_seed_value = int(resolve_default(record_seed, EVAL_SEED_BASE))
    video_every = (
        VIDEO_EVERY_RECORDINGS * record_every
        if resolve_optional_int(video_every_rounds) is None
        else int(resolve_optional_int(video_every_rounds) or 0)
    )
    video_fps_value = int(resolve_default(video_fps, 5))
    video_theme_value = str(resolve_default(video_theme, "tabletop"))
    backward = BackwardStart(
        level=int(resolve_default(backward_start, 0)),
        share=float(resolve_default(backward_start_share, 0.75)),
        advance_at=float(resolve_default(backward_start_advance, 0.8)),
        window=int(resolve_default(backward_start_window, 8)),
        rng=np.random.default_rng(resolve_optional_int(seed) or 0),
    )
    if backward.level > 0:
        n_groups = len({m.group_id for m in PerModelEnv(env_config).wargame_models})
        if backward.level > min(n_groups, env_config.number_of_objectives):
            raise ValueError(
                f"backward_start ({backward.level}) exceeds the squads ({n_groups}) "
                f"or objectives ({env_config.number_of_objectives}) on this config"
            )
    total_rounds = int(resolve_default(rounds, 1024))
    eval_episodes = int(resolve_default(n_eval_episodes, 20))
    eval_seeds_from = int(resolve_default(eval_seed_base, EVAL_SEED_BASE))
    if total_rounds < rollout_total:
        raise ValueError(
            f"rounds ({total_rounds}) is below one rollout ({rollout_total} = "
            "rollout_rounds x envs); nothing would train"
        )
    if resume is not None and total_rounds <= resume.loaded.rounds:
        raise ValueError(
            f"--resume-from {resume.checkpoint} is at {resume.loaded.rounds} rounds; "
            f"--rounds {total_rounds} would train nothing"
        )
    check_cadence(eval_every, rollout_total, "eval_every_rounds")
    check_cadence(checkpoint_every, rollout_total, "checkpoint_every_rounds")
    if record_every > 0:
        check_cadence(record_every, rollout_total, "record_every_rounds")
    if video_every > 0 and (record_every <= 0 or video_every % record_every != 0):
        raise ValueError(
            f"video_every_rounds ({video_every}) must be a positive multiple of "
            f"record_every_rounds ({record_every}): a video is a rendered recording"
        )

    trunk: dict[str, int] = {}
    if resolve_optional_int(n_layers) is not None:
        trunk["n_layers"] = int(resolve_optional_int(n_layers) or 0)
    if resolve_optional_int(embedding_size) is not None:
        trunk["embedding_size"] = int(resolve_optional_int(embedding_size) or 0)
    network_config = SetNetworkConfig()
    if trunk:
        network_config = SetNetworkConfig(**{**network_config.model_dump(), **trunk})

    envs = [PerModelEnv(env_config) for _ in range(n_envs)]
    rounds_done = 0 if resume is None else resume.loaded.rounds
    # On a resume the envs restart at fresh episodes (their mid-episode state
    # is not checkpointed), offset by the rounds already trained so the run
    # does not replay its own first layouts.
    seed_base = (resolved_seed or 0) * ROLLOUT_SEED_STRIDE + rounds_done
    initial_levels = [backward.draw() for _ in envs]
    observations = [
        env.reset(
            seed=seed_base + index,
            options={"augment_start": True, "start_groups": initial_levels[index]},
        )[0]
        for index, env in enumerate(envs)
    ]
    retimers = [PerStepReward(env, credit=ppo_config.credit) for env in envs]
    for retimer in retimers:
        retimer.reset()
    wave = max(
        1, min(int(resolve_default(eval_wave_size, EVAL_WAVE_SIZE)), eval_episodes)
    )
    eval_envs = [PerModelEnv(env_config) for _ in range(wave)]
    eval_retimers = [PerStepReward(env, credit=ppo_config.credit) for env in eval_envs]

    expected_head = SetNetwork.n_displacements_for(envs[0].player_action_handler)
    warm: LoadedCheckpoint | None = None
    if resume is not None:
        refuse_changed_trunk(trunk, resume.loaded.network.config, resume.checkpoint)
        if resume.loaded.network.n_displacements != expected_head:
            raise ValueError(
                f"{resume.checkpoint} has a displacement head of "
                f"{resume.loaded.network.n_displacements}; the run's scenario "
                f"needs {expected_head}"
            )
        network = resume.loaded.network
        network_config = network.config
    elif warm_spec is not None:
        warm = load_checkpoint(Path(warm_spec), expected_n_displacements=expected_head)
        refuse_changed_trunk(trunk, warm.network.config, Path(warm_spec))
        network = warm.network
        network_config = network.config
        logger.info(
            "Warm start from {} ({} rounds, revision {})",
            warm_spec,
            warm.rounds,
            warm.revision,
        )
    else:
        network = SetNetwork.from_env(envs[0], network_config)
    if network_config != SetNetworkConfig():
        logger.warning(
            "⚠ NON-DEFAULT NETWORK {}: its checkpoints load into nothing else",
            network_config.model_dump(),
        )
    network = network.to(resolved_device)
    check_trainable(network)
    agent = SetAgent(network)
    optimizer = torch.optim.Adam(network.parameters(), lr=ppo_config.lr, eps=1e-5)
    generator = torch.Generator().manual_seed(resolved_seed or 0)
    declarations_seen = 0
    approx_kl_cumulative = 0.0
    if resume is not None:
        optimizer.load_state_dict(resume.state.optimizer_state)
        generator.set_state(resume.state.generator_state)
        declarations_seen = resume.state.declarations_seen
        approx_kl_cumulative = resume.state.approx_kl_cumulative
    reference: SetNetwork | None = None
    kl_ref_coef_now = ppo_config.kl_ref_coef
    if ppo_config.kl_ref_coef > 0.0:
        reference = _kl_reference(network, resume, expected_head, resolved_device)
        if resume is not None and resume.state.kl_ref_coef > 0.0:
            kl_ref_coef_now = resume.state.kl_ref_coef
        logger.info(
            "KL anchor attached: coef {} target {} nats per decision",
            kl_ref_coef_now,
            ppo_config.kl_ref_target,
        )

    base_name = f"per-model-{Path(config_path).stem}"
    resolved_run_name = resolve_optional_str(run_name) or base_name
    config = {
        "wargame": env_config.model_dump(mode="json"),
        "ppo": ppo_config.model_dump(mode="json"),
        "network": network_config.model_dump(),
        "driver": {
            "rounds": total_rounds,
            "eval_every_rounds": eval_every,
            "checkpoint_every_rounds": checkpoint_every,
            "record_every_rounds": record_every,
            "record_seed": record_seed_value,
            "video_every_rounds": video_every,
            "video_fps": video_fps_value,
            "video_theme": video_theme_value,
            "backward_start": backward.level,
            "backward_start_share": backward.share,
            "backward_start_advance": backward.advance_at,
            "backward_start_window": backward.window,
            "seed": resolved_seed,
        },
    }
    # The bar goes through the phase facade -- the first time this config
    # touches it -- so it is measured before the run directory exists and a
    # refusal leaves nothing half-written. A resumed run already logged it.
    bar = None if resume is not None else baseline_rows(env_config)
    disabled = bool(resolve_default(no_wandb, False))
    with init_wandb(
        config=config,
        name=resolved_run_name,
        disabled=disabled,
        group=resolve_optional_str(wandb_group),
        run_suffix=resolve_optional_str(run_suffix),
        run_name=None if resume is None else resume.run_dir.name,
    ) as run:
        if resume is not None:
            run_dir = resume.run_dir
        else:
            run_dir = Path(str(resolve_default(checkpoint_root, CHECKPOINT_ROOT)))
            run_dir = run_dir / str(run.name)
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "env_config.yaml").write_text(
                _config_yaml(env_config, config_path)
            )
        revision = git_revision()
        _write_provenance(
            run_dir,
            config,
            revision=revision,
            device=resolved_device,
            seed_bands={
                "rollout": seed_base,
                "eval": eval_seeds_from,
                "baselines": BASELINE_SEED_BASE,
            },
            resume=resume,
            warm=warm,
            warm_spec=warm_spec,
        )
        metrics = MetricsLog(run_dir / "metrics.jsonl", to_wandb=not disabled)
        videos = TrainingVideos(
            fps=video_fps_value, theme=video_theme_value, to_wandb=not disabled
        )
        logger.info("Run directory {}", run_dir)
        logger.info(
            "Regime: {} rounds per update ({} rollout rounds x {} envs)",
            rollout_total,
            ppo_config.rollout_rounds,
            n_envs,
        )

        if bar is not None:
            metrics.log({"rounds": 0, "epoch_equivalent": 0.0, **bar})

        env_config_dict = env_config.model_dump(mode="json")
        while rounds_done < total_rounds:
            agent.network.train()
            started = time.perf_counter()
            rollout = collect_rollout(
                envs,
                agent,
                retimers,
                observations,
                ppo_config,
                generator=generator,
                start_groups=backward.draw if backward.level > 0 else None,
                initial_start_groups=initial_levels,
            )
            initial_levels = list(rollout.start_groups)
            observations = rollout.observations
            backward.observe(rollout.episodes)
            rollout_s = time.perf_counter() - started
            entropy = rollout_entropy(network, rollout, ppo_config.batch_size)
            started = time.perf_counter()
            returns, advantages = compute_gae(rollout, ppo_config)
            stats = ppo_update(
                network,
                optimizer,
                rollout,
                returns,
                advantages,
                ppo_config,
                generator=generator,
                reference=reference,
                kl_ref_coef=kl_ref_coef_now,
            )
            kl_ref_coef_now = adapt_kl_coef(
                kl_ref_coef_now, stats.kl_ref, ppo_config.kl_ref_target
            )
            update_s = time.perf_counter() - started
            # Nominal: lockstep envs can close in the same iteration, so the
            # real count overshoots by up to n_envs - 1; the cadences are
            # multiples of the nominal budget, and the real count is logged.
            rounds_done += rollout_total
            approx_kl_cumulative += stats.approx_kl
            declarations = Counter(agent.declaration_counts)
            new_declarations = sum(declarations.values()) - declarations_seen
            declarations_seen = sum(declarations.values())
            row = {
                "rounds": rounds_done,
                "epoch_equivalent": rounds_done / ROUNDS_PER_EPOCH_EQUIVALENT,
                "rounds_per_s": rollout.closes / max(rollout_s + update_s, 1e-9),
                **update_rows(
                    stats,
                    rollout,
                    entropy=entropy,
                    declarations=declarations,
                    rollout_s=rollout_s,
                    update_s=update_s,
                    rounds_per_update=rollout_total,
                    rounds_done=rounds_done,
                    approx_kl_cumulative=approx_kl_cumulative,
                ),
            }
            row["train/declarations"] = float(new_declarations)
            row["train/closes"] = float(rollout.closes)
            row.update(backward.rows())
            agent.declaration_counts.clear()
            if rounds_done % eval_every == 0 or rounds_done >= total_rounds:
                started = time.perf_counter()
                result = evaluate_per_model(
                    eval_envs,
                    agent,
                    eval_retimers,
                    [eval_seeds_from + i for i in range(eval_episodes)],
                )
                row.update(eval_rows(result))
                row["perf/eval_s"] = time.perf_counter() - started
                logger.info(
                    "rounds {} vp_margin {:.1f} win {:.0f}% held {:.2f} coherent {} "
                    "stat {} hold {}",
                    rounds_done,
                    result.vp_margin,
                    100.0 * result.win_rate,
                    result.objectives_held,
                    format_optional_metric(result.coherency_rate),
                    format_optional_metric(result.stationary_share, 2),
                    format_optional_metric(result.hold_fire_share, 2),
                )
            if rounds_done % checkpoint_every == 0 or rounds_done >= total_rounds:
                training_state = TrainingState(
                    optimizer_state=optimizer.state_dict(),
                    generator_state=generator.get_state(),
                    declarations_seen=declarations_seen,
                    approx_kl_cumulative=approx_kl_cumulative,
                    kl_ref_coef=kl_ref_coef_now if reference is not None else 0.0,
                )
                for name in (periodic_checkpoint_name(rounds_done), LAST_CHECKPOINT):
                    save_checkpoint(
                        run_dir / name,
                        network,
                        ppo_config=ppo_config,
                        env_config=env_config_dict,
                        rounds=rounds_done,
                        seed=resolved_seed,
                        revision=revision,
                        training_state=training_state,
                    )
            if record_every > 0 and (
                rounds_done % record_every == 0 or rounds_done >= total_rounds
            ):
                started = time.perf_counter()
                recorded = record_greedy_episode(
                    network, env_config, run_dir, rounds_done, record_seed_value
                )
                row["perf/record_s"] = time.perf_counter() - started
                logger.info("rounds {} recorded {}", rounds_done, recorded)
                if video_every > 0 and (
                    rounds_done % video_every == 0 or rounds_done >= total_rounds
                ):
                    videos.start(recorded, rounds_done)
            videos.log_finished()
            metrics.log(row)
            logger.info(
                "rounds {} steps {} loss {:.3f} kl {:.4f} rollout {:.1f}s update {:.1f}s",
                rounds_done,
                rollout.n_steps,
                stats.train_loss,
                stats.approx_kl,
                rollout_s,
                update_s,
            )
        videos.finish()
        return run_dir


class BackwardStart:
    """The backward start curriculum's schedule (#340).

    At `level` k, a `share` of rollout episodes begin with k squads already
    standing on k distinct objectives (`start_groups_on_objectives`) and the
    rest from deployment. The level steps down by one once the level's OWN
    episodes -- not the deployment ones, and not the evaluation, which always
    starts from deployment -- succeed at `advance_at` over the last `window`
    rollouts with at least `window` such episodes. At level 0 it is the plain
    augmented start and draws nothing.
    """

    def __init__(
        self,
        *,
        level: int,
        share: float,
        advance_at: float,
        window: int,
        rng: np.random.Generator,
    ) -> None:
        if not 0.0 < share <= 1.0:
            raise ValueError(f"backward_start_share must be in (0, 1], got {share}")
        if not 0.0 < advance_at <= 1.0:
            raise ValueError(
                f"backward_start_advance must be in (0, 1], got {advance_at}"
            )
        if window < 1:
            raise ValueError(f"backward_start_window must be >= 1, got {window}")
        self.level = max(0, level)
        self.share = share
        self.advance_at = advance_at
        self.window = window
        self._rng = rng
        self._recent: list[list[bool]] = []
        self.advanced_at_rollout: list[int] = []
        self._rollouts = 0
        self.last_level_success: float | None = None

    def draw(self) -> int:
        """The level the next episode starts at: the current one or 0."""
        if self.level <= 0:
            return 0
        return self.level if float(self._rng.random()) < self.share else 0

    def observe(self, episodes: Sequence[EpisodeOutcome]) -> bool:
        """Read one rollout's finished episodes; True when the level stepped down."""
        self._rollouts += 1
        if self.level <= 0:
            return False
        self._recent.append(
            [e.success for e in episodes if e.start_groups == self.level]
        )
        self._recent = self._recent[-self.window :]
        outcomes = [o for rollout in self._recent for o in rollout]
        if len(self._recent) < self.window or len(outcomes) < self.window:
            self.last_level_success = float(np.mean(outcomes)) if outcomes else None
            return False
        self.last_level_success = float(np.mean(outcomes))
        if self.last_level_success < self.advance_at:
            return False
        self.level -= 1
        self.advanced_at_rollout.append(self._rollouts)
        self._recent = []
        logger.info(
            "backward start: level {} -> {} at rollout {} (level success {:.2f})",
            self.level + 1,
            self.level,
            self._rollouts,
            self.last_level_success,
        )
        return True

    def rows(self) -> dict[str, float]:
        """The metrics rows: the level, and the level's rolling success."""
        rows = {"curriculum/start_groups": float(self.level)}
        if self.last_level_success is not None:
            rows["curriculum/level_success"] = self.last_level_success
        return rows


RECORDINGS_DIR = "recordings"
# One MP4 per twenty recordings by default: at 512 rounds a recording that is
# twelve videos on a 122,880-round run, the whole-army trainer's cadence of one
# every twenty epochs.
VIDEO_EVERY_RECORDINGS = 20
VIDEO_KEY = "episode_recording"


class TrainingVideos:
    """Renders training recordings to MP4 in the background and logs them.

    The render runs in a subprocess (`replay_events.py render`) as the
    whole-army trainer's recording callback did: pygame's SDL must use the
    dummy driver and stay out of the trainer's process, and a render must not
    stall the next update. A finished MP4 is logged to Wandb under
    `episode_recording` (the whole-army key) on the next checkpoint, uncommitted,
    so it lands on that checkpoint's row; `finish` waits for the rest at the end
    of training.
    """

    def __init__(self, *, fps: int, theme: str, to_wandb: bool) -> None:
        self.fps = fps
        self.theme = theme
        self.to_wandb = to_wandb
        self._pending: list[tuple[subprocess.Popen[bytes], Path, int]] = []
        self.rendered: list[Path] = []

    def start(self, recording: Path, rounds: int) -> Path:
        """Begin rendering `recording` to an MP4 beside it; returns the MP4 path."""
        mp4 = recording.with_suffix(".mp4")
        log = recording.with_suffix(".render.log")
        env = dict(os.environ, SDL_VIDEODRIVER="dummy", SDL_AUDIODRIVER="dummy")
        command = [
            sys.executable,
            str(Path(__file__).resolve().parent / "replay_events.py"),
            "render",
            str(recording),
            "--out",
            str(mp4),
            "--theme",
            self.theme,
            "--fps",
            str(self.fps),
        ]
        with log.open("wb") as handle:
            process = subprocess.Popen(
                command, stdout=handle, stderr=subprocess.STDOUT, env=env
            )
        self._pending.append((process, mp4, rounds))
        return mp4

    def log_finished(self) -> list[Path]:
        """Log every render that has finished since the last call; returns them."""
        done: list[Path] = []
        still: list[tuple[subprocess.Popen[bytes], Path, int]] = []
        for process, mp4, rounds in self._pending:
            if process.poll() is None:
                still.append((process, mp4, rounds))
                continue
            collected = self._collect(process, mp4, rounds)
            if collected is not None:
                done.append(collected)
        self._pending = still
        return done

    def finish(self) -> list[Path]:
        """Wait for every pending render and log it."""
        for process, _mp4, _rounds in self._pending:
            process.wait()
        return self.log_finished()

    def _collect(
        self, process: subprocess.Popen[bytes], mp4: Path, rounds: int
    ) -> Path | None:
        if process.returncode != 0 or not mp4.exists():
            logger.warning(
                "render of {} failed (exit {}); see {}",
                mp4.with_suffix(".json").name,
                process.returncode,
                mp4.with_suffix(".render.log"),
            )
            return None
        self.rendered.append(mp4)
        logger.info("rounds {} rendered {}", rounds, mp4)
        if self.to_wandb:
            import wandb

            wandb.log(  # type: ignore[attr-defined]
                {VIDEO_KEY: wandb.Video(str(mp4), format="mp4")},  # type: ignore[attr-defined]
                commit=False,
            )
        return mp4


def record_greedy_episode(
    network: SetNetwork,
    env_config: WargameEnvConfig,
    run_dir: Path,
    rounds: int,
    seed: int,
) -> Path:
    """Record one greedy episode of the network as it stands, to an event log.

    Written under `<run_dir>/recordings/pm-<rounds>-seed<seed>.json` at
    DECISION cadence (one snapshot per decision; `just replay-render` draws
    it). Greedy, because that is the policy a score reports; the sampled
    policy training rolls out is read beside it by
    `just measure-per-model-eval-mode`. The network is put back in whatever
    mode it was in, so recording never changes the update that follows it.
    """
    was_training = network.training
    network.eval()
    agent = SetAgent(network, greedy=True)

    def chooser_for(_envs: Sequence[PerModelEnv]) -> BatchChooser:
        def choose(
            envs: Sequence[PerModelEnv], observations: Sequence[PerModelObservation]
        ) -> list[PerModelAction]:
            with torch.no_grad():
                return [d.action for d in agent.act_batch(envs, observations)]

        return choose

    try:
        return record_episode(
            chooser_for,
            env_config,
            seed,
            run_dir / RECORDINGS_DIR / f"pm-{rounds:08d}-seed{seed}.json",
            cadence="decision",
            driver="train_per_model",
        )
    finally:
        network.train(was_training)


def _resumed_env_config(
    resume: ResumePoint, config_path: str | None
) -> tuple[WargameEnvConfig, str]:
    """The run's own scenario, from its directory; an explicit path that
    names a different scenario is refused rather than quietly swapped in."""
    stored = WargameEnvConfig(**resume.loaded.env_config)
    run_copy = resume.run_dir / "env_config.yaml"
    path = config_path if config_path is not None else str(run_copy)
    if config_path is not None:
        asked = get_env_config(config_path, None)
        if asked.model_dump(mode="json") != stored.model_dump(mode="json"):
            raise ValueError(
                f"--resume-from {resume.checkpoint} refuses --env-config-path "
                f"{config_path}: it differs from the run's scenario"
            )
    return stored, path


def _kl_reference(
    network: SetNetwork,
    resume: ResumePoint | None,
    expected_head: int,
    device: torch.device,
) -> SetNetwork:
    """The frozen network the KL anchor measures drift against: the run's
    STARTING weights. On a fresh run that is a copy of `network` as built
    (the warm-start weights, or the fresh init); on a resume it is reloaded
    from the run's `warm_started_from` checkpoint, because the resumed
    weights have already moved and anchoring to them would let the drift
    ratchet forward with every resume."""
    if resume is None:
        reference = copy.deepcopy(network)
    else:
        provenance_path = resume.checkpoint.parent / "provenance.json"
        warm = json.loads(provenance_path.read_text()).get("warm_started_from")
        if not warm or not warm.get("checkpoint"):
            raise ValueError(
                f"{resume.checkpoint} resumes an anchored run whose provenance "
                "names no warm-start checkpoint to anchor to"
            )
        reference = load_checkpoint(
            Path(warm["checkpoint"]), expected_n_displacements=expected_head
        ).network
    reference = reference.to(device)
    reference.eval()
    for parameter in reference.parameters():
        parameter.requires_grad_(False)
    return reference


def _write_provenance(
    run_dir: Path,
    config: dict[str, Any],
    *,
    revision: str,
    device: torch.device,
    seed_bands: dict[str, int],
    resume: ResumePoint | None,
    warm: LoadedCheckpoint | None,
    warm_spec: str | None,
) -> None:
    """`provenance.json`: written fresh for a new run; on a resume the
    existing file keeps its history and gains one `resumed_from` entry."""
    path = run_dir / "provenance.json"
    provenance: dict[str, Any] = {}
    if resume is not None and path.exists():
        provenance = json.loads(path.read_text())
    provenance.update(
        {
            **config,
            "revision": revision,
            "device": str(device),
            "torch_threads": torch.get_num_threads(),
            "seed_bands": seed_bands,
        }
    )
    if resume is not None:
        resumed = list(provenance.get("resumed_from", []))
        resumed.append(
            {
                "checkpoint": str(resume.checkpoint),
                "rounds": resume.loaded.rounds,
                "revision_at_resume": revision,
            }
        )
        provenance["resumed_from"] = resumed
    if warm is not None:
        provenance["warm_started_from"] = {
            "checkpoint": warm_spec,
            "rounds": warm.rounds,
            "revision": warm.revision,
            "seed": warm.seed,
        }
    path.write_text(json.dumps(provenance, indent=2))


def _config_yaml(env_config: WargameEnvConfig, path: str) -> str:
    """The config verbatim when the path still names it, else a dump: the
    path in provenance can outlive the file it named."""
    source = Path(path)
    if source.exists():
        return source.read_text()
    from pydantic_yaml import to_yaml_str

    return str(to_yaml_str(env_config))


if __name__ == "__main__":
    app()
