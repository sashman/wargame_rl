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
army (#283). Curriculum configs are refused; warm start and resume are
follow-ups.
"""

from __future__ import annotations

import json
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
import typer
from loguru import logger

from wargame_rl.wargame.envs.baseline.evaluate import evaluate_baseline
from wargame_rl.wargame.envs.baseline.registry import build_baseline_policy
from wargame_rl.wargame.envs.per_model.env import PerModelEnv
from wargame_rl.wargame.envs.per_model.reward_timing import PerStepReward
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
)
from wargame_rl.wargame.model.common.factory import create_environment
from wargame_rl.wargame.model.common.wandb import init_wandb
from wargame_rl.wargame.model.per_model.agent import SetAgent
from wargame_rl.wargame.model.per_model.checkpoint import (
    LAST_CHECKPOINT,
    periodic_checkpoint_name,
    save_checkpoint,
)
from wargame_rl.wargame.model.per_model.config import SetNetworkConfig
from wargame_rl.wargame.model.per_model.evaluate import (
    PerModelEvalResult,
    evaluate_per_model,
)
from wargame_rl.wargame.model.per_model.net import SetNetwork
from wargame_rl.wargame.model.per_model.ppo import (
    PerModelPPOConfig,
    Rollout,
    UpdateStats,
    auto_num_rollout_envs,
    check_trainable,
    collect_rollout,
    compute_gae,
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


def eval_rows(result: PerModelEvalResult, prefix: str = "") -> dict[str, float]:
    suffix = f"{prefix}_" if prefix else ""
    margins = np.array([e.vp_margin for e in result.episodes], dtype=np.float64)
    elo = float(rating_from_score(float(margin_score(margins).mean())))
    return {
        f"eval/{suffix}vp_player": result.vp_player,
        f"eval/{suffix}vp_opponent": result.vp_opponent,
        f"eval/{suffix}vp_margin": result.vp_margin,
        f"eval/{suffix}win_rate": 100.0 * result.win_rate,
        f"eval/{suffix}elo": elo,
        f"eval/{suffix}fraction_alive": result.fraction_alive,
        f"eval/{suffix}objectives_held": result.objectives_held,
        f"eval/{suffix}coherency_rate": result.coherency_rate,
        f"reward/{suffix}mean_episode_reward": result.mean_reward,
        f"reward/{suffix}max_episode_reward": result.max_reward,
        f"reward/{suffix}min_episode_reward": result.min_reward,
        f"{suffix}mean_episode_steps": result.mean_steps,
        f"{suffix}success_rate": 100.0 * result.success_rate,
    }


def update_rows(
    stats: UpdateStats,
    rollout: Rollout,
    *,
    entropy_by_phase: dict[str, float],
    selector_entropy: float,
    declarations: Counter[str],
    rollout_s: float,
    update_s: float,
) -> dict[str, float]:
    rows: dict[str, float] = {
        "steps": float(rollout.n_steps),
        "loss/train_loss": stats.train_loss,
        "loss/policy_loss": stats.policy_loss,
        "loss/value_loss": stats.value_loss,
        "loss/entropy_loss": stats.entropy_loss,
        "train/clip_fraction": stats.clip_fraction,
        "train/approx_kl": stats.approx_kl,
        "train/explained_variance": stats.explained_variance,
        "train/grad_norm": stats.grad_norm,
        "train/grad_clipped_fraction": stats.grad_clipped_fraction,
        "train/entropy/selector": selector_entropy,
        "perf/rollout_s": rollout_s,
        "perf/update_s": update_s,
        "perf/epoch_s": rollout_s + update_s,
        "perf/env_steps_per_s": rollout.n_steps / max(rollout_s, 1e-9),
        "perf/update_ms_per_minibatch": 1000.0 * update_s / max(1, stats.n_minibatches),
        "train/episodes": float(len(rollout.episodes)),
    }
    for phase, value in entropy_by_phase.items():
        rows[f"train/entropy/{phase}"] = value
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


@app.command()
def train(
    env_config_path: str = typer.Option(
        "configs/dev/4v4_per_model_smoke.yaml", help="Path to the env config."
    ),
    rounds: int = typer.Option(1024, help="Total training budget, in rounds."),
    rollout_rounds: int | None = typer.Option(
        None, help="Closing steps per env per update (default 16)."
    ),
    num_rollout_envs: int | None = typer.Option(
        None, help="Lockstep rollout envs; 0 or unset auto-detects."
    ),
    gamma: float | None = typer.Option(None, help="Per-ROUND discount."),
    gae_lambda: float | None = typer.Option(None, help="Per-ROUND GAE decay."),
    ent_coef: float | None = typer.Option(None, help="Head entropy coefficient."),
    selector_ent_coef: float | None = typer.Option(
        None, help="Selector entropy coefficient (default: ent_coef)."
    ),
    lr: float | None = typer.Option(None),
    max_grad_norm: float | None = typer.Option(None),
    n_epochs: int | None = typer.Option(None),
    batch_size: int | None = typer.Option(None),
    eval_every_rounds: int = typer.Option(
        256, help="Evaluate every N rounds (a multiple of the rollout)."
    ),
    n_eval_episodes: int = typer.Option(20),
    checkpoint_every_rounds: int = typer.Option(
        256, help="Write pm-NNNNNNNN.pt and last.pt every N rounds."
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
) -> Path:
    """Train; returns the run directory."""
    resolved_seed = resolve_optional_int(seed)
    if resolved_seed is not None:
        torch.manual_seed(resolved_seed)
        np.random.seed(resolved_seed)
    torch.set_num_threads(int(resolve_default(torch_threads, 2)))
    resolved_device = resolve_device(str(resolve_default(device, "auto")))

    env_config = get_env_config(str(resolve_default(env_config_path, None)), None)
    refuse_curriculum(env_config)

    ppo_config = PerModelPPOConfig()
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
    }
    ppo_config = ppo_config.model_copy(
        update={k: v for k, v in overrides.items() if v is not None}
    )
    ppo_config = PerModelPPOConfig(**ppo_config.model_dump())
    n_envs = ppo_config.num_rollout_envs
    if n_envs <= 0:
        n_envs = auto_num_rollout_envs(resolved_device)
    # Written back so the persisted config and every checkpoint say how many
    # envs produced the run, not the `0` that asked for auto-detection.
    ppo_config = PerModelPPOConfig(
        **ppo_config.model_copy(update={"num_rollout_envs": n_envs}).model_dump()
    )
    rollout_total = ppo_config.rollout_rounds * n_envs
    eval_every = int(resolve_default(eval_every_rounds, 256))
    checkpoint_every = int(resolve_default(checkpoint_every_rounds, 256))
    total_rounds = int(resolve_default(rounds, 1024))
    eval_episodes = int(resolve_default(n_eval_episodes, 20))
    if total_rounds < rollout_total:
        raise ValueError(
            f"rounds ({total_rounds}) is below one rollout ({rollout_total} = "
            "rollout_rounds x envs); nothing would train"
        )
    check_cadence(eval_every, rollout_total, "eval_every_rounds")
    check_cadence(checkpoint_every, rollout_total, "checkpoint_every_rounds")

    network_config = SetNetworkConfig()
    trunk: dict[str, int] = {}
    if resolve_optional_int(n_layers) is not None:
        trunk["n_layers"] = int(resolve_optional_int(n_layers) or 0)
    if resolve_optional_int(embedding_size) is not None:
        trunk["embedding_size"] = int(resolve_optional_int(embedding_size) or 0)
    if trunk:
        network_config = SetNetworkConfig(**{**network_config.model_dump(), **trunk})
        logger.warning(
            "⚠ NON-DEFAULT NETWORK {}: its checkpoints load into nothing else",
            network_config.model_dump(),
        )

    envs = [PerModelEnv(env_config) for _ in range(n_envs)]
    seed_base = (resolved_seed or 0) * ROLLOUT_SEED_STRIDE
    observations = [
        env.reset(seed=seed_base + index, options={"augment_start": True})[0]
        for index, env in enumerate(envs)
    ]
    retimers = [PerStepReward(env) for env in envs]
    for retimer in retimers:
        retimer.reset()
    eval_env = PerModelEnv(env_config)
    eval_retimer = PerStepReward(eval_env)

    network = SetNetwork.from_env(envs[0], network_config).to(resolved_device)
    check_trainable(network)
    agent = SetAgent(network)
    optimizer = torch.optim.Adam(network.parameters(), lr=ppo_config.lr, eps=1e-5)
    generator = torch.Generator().manual_seed(resolved_seed or 0)

    base_name = f"per-model-{Path(env_config_path).stem}"
    resolved_run_name = resolve_optional_str(run_name) or base_name
    config = {
        "wargame": env_config.model_dump(mode="json"),
        "ppo": ppo_config.model_dump(),
        "network": network_config.model_dump(),
        "driver": {
            "rounds": total_rounds,
            "eval_every_rounds": eval_every,
            "checkpoint_every_rounds": checkpoint_every,
            "seed": resolved_seed,
        },
    }
    # The bar goes through the phase facade -- the first time this config
    # touches it -- so it is measured before the run directory exists and a
    # refusal leaves nothing half-written.
    bar = baseline_rows(env_config)
    disabled = bool(resolve_default(no_wandb, False))
    with init_wandb(
        config=config,
        name=resolved_run_name,
        disabled=disabled,
        group=resolve_optional_str(wandb_group),
        run_suffix=resolve_optional_str(run_suffix),
    ) as run:
        run_dir = Path(str(resolve_default(checkpoint_root, CHECKPOINT_ROOT)))
        run_dir = run_dir / str(run.name)
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "env_config.yaml").write_text(
            _config_yaml(env_config, env_config_path)
        )
        revision = git_revision()
        (run_dir / "provenance.json").write_text(
            json.dumps(
                {
                    **config,
                    "revision": revision,
                    "device": str(resolved_device),
                    "torch_threads": torch.get_num_threads(),
                    "seed_bands": {
                        "rollout": seed_base,
                        "eval": EVAL_SEED_BASE,
                        "baselines": BASELINE_SEED_BASE,
                    },
                },
                indent=2,
            )
        )
        metrics = MetricsLog(run_dir / "metrics.jsonl", to_wandb=not disabled)
        logger.info("Run directory {}", run_dir)

        metrics.log({"rounds": 0, "epoch_equivalent": 0.0, **bar})

        env_config_dict = env_config.model_dump(mode="json")
        rounds_done = 0
        declarations_seen = 0
        while rounds_done < total_rounds:
            agent.network.train()
            started = time.perf_counter()
            rollout = collect_rollout(
                envs, agent, retimers, observations, ppo_config, generator=generator
            )
            observations = rollout.observations
            rollout_s = time.perf_counter() - started
            entropy_by_phase, selector_entropy = rollout_entropy(
                network, rollout, ppo_config.batch_size
            )
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
            )
            update_s = time.perf_counter() - started
            # Nominal: lockstep envs can close in the same iteration, so the
            # real count overshoots by up to n_envs - 1; the cadences are
            # multiples of the nominal budget, and the real count is logged.
            rounds_done += rollout_total
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
                    entropy_by_phase=entropy_by_phase,
                    selector_entropy=selector_entropy,
                    declarations=declarations,
                    rollout_s=rollout_s,
                    update_s=update_s,
                ),
            }
            row["train/declarations"] = float(new_declarations)
            row["train/closes"] = float(rollout.closes)
            agent.declaration_counts.clear()
            if rounds_done % eval_every == 0 or rounds_done >= total_rounds:
                started = time.perf_counter()
                result = evaluate_per_model(
                    eval_env,
                    agent,
                    eval_retimer,
                    [EVAL_SEED_BASE + i for i in range(eval_episodes)],
                )
                row.update(eval_rows(result))
                row["perf/eval_s"] = time.perf_counter() - started
                logger.info(
                    "rounds {} vp_margin {:.1f} win {:.0f}% held {:.2f} coherent {:.3f}",
                    rounds_done,
                    result.vp_margin,
                    100.0 * result.win_rate,
                    result.objectives_held,
                    result.coherency_rate,
                )
            if rounds_done % checkpoint_every == 0 or rounds_done >= total_rounds:
                for name in (periodic_checkpoint_name(rounds_done), LAST_CHECKPOINT):
                    save_checkpoint(
                        run_dir / name,
                        network,
                        ppo_config=ppo_config,
                        env_config=env_config_dict,
                        rounds=rounds_done,
                        seed=resolved_seed,
                        revision=revision,
                    )
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
        return run_dir


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
