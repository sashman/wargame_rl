"""Train the per-model architecture (issues #286/#288): a plain PPO loop.

Deliberately not Lightning. The whole-phase trainer's Lightning module exists
to serve epochs, callbacks and a batched eval that all assume one step per
phase; here the loop *is* the spec — a rollout budget in rounds, an update, a
periodic greedy evaluation — and writing it out keeps every time constant
visible in one screen of code.

Three disciplines carried over from the whole-phase trainer's scars:

- **Checkpoints are periodic, not exit-hooked.** SIGKILL is the prescribed way
  to stop a trainer here and it triggers no handler, so ``last.pt`` is written
  every checkpoint interval and is at most that stale.
- **Metrics land in a local JSONL beside the checkpoints**, whether or not
  wandb is on — reading the wandb API while runs train has crashed runs before.
- **The CPU thread count is pinned and recorded** (issue #306): a thread
  setting moved a score by 2 vp, so two arms compared by score must state it.

Rollouts continue the in-flight episode across update boundaries; a fresh
driver process starts its rollout env at ``ROLLOUT_SEED_BASE + seed`` so two
runs at one seed see the same layout stream.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import typer
from loguru import logger
from pydantic_yaml import parse_yaml_raw_as

from wargame_rl.wargame.envs.baseline.evaluate import evaluate_baseline
from wargame_rl.wargame.envs.baseline.registry import build_baseline_policy
from wargame_rl.wargame.envs.per_model import PerModelEnv
from wargame_rl.wargame.envs.per_model.types import PerModelObservation
from wargame_rl.wargame.envs.types import WargameEnvConfig
from wargame_rl.wargame.envs.wargame import WargameEnv
from wargame_rl.wargame.model.per_model import SetAgent, SetNetwork, SetNetworkConfig
from wargame_rl.wargame.model.per_model.evaluate import evaluate_per_model
from wargame_rl.wargame.model.per_model.ppo import (
    PerModelPPOConfig,
    collect_rollout,
    ppo_update,
)
from wargame_rl.wargame.rating.elo import rating_from_score
from wargame_rl.wargame.rating.score import margin_score

app = typer.Typer(pretty_exceptions_enable=False)

# The whole-phase trainer's rollout band (model/ppo/lightning.py), reused so
# neither facade trains on the evaluation bands (500000+/700000+/900000+).
ROLLOUT_SEED_BASE = 0
# The whole-phase trainer's scripted bar, measured once per run under the same
# keys (model/common/lightning_base.py) — restated rather than imported so
# this path stays free of Lightning.
BASELINE_POLICIES = ("random", "squad_march", "squad_march_shoot")
BASELINE_EPISODES = 20
BASELINE_SEED_BASE = 10_000
# One whole-phase epoch of experience: 2048 steps at 2 steps/round. Wandb
# rows aggregate to this cadence so charts line up with the old runs'.
WANDB_LOG_EVERY_ROUNDS = 1024


def _git_revision() -> str:
    """The current code revision, `+dirty` when the tree has local edits."""
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        return f"{revision}+dirty" if status else revision
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _head_sizes(env: PerModelEnv) -> dict[str, int]:
    """The action-head widths an env implies — a checkpoint's shape contract."""
    handler = env.player_action_handler
    advance_slice = handler.advance_slice
    return {
        "n_move_actions": int(handler.n_move_actions),
        "n_advance_actions": int(advance_slice.size) if advance_slice else 0,
    }


def _save_checkpoint(
    path: Path,
    network: SetNetwork,
    network_config: SetNetworkConfig,
    ppo_config: PerModelPPOConfig,
    env: PerModelEnv,
    env_config_path: Path,
    rounds: int,
    seed: int,
    revision: str,
) -> None:
    """Write one self-describing checkpoint file (atomically, via a temp name)."""
    payload = {
        "state_dict": network.state_dict(),
        "network_config": network_config.model_dump(),
        "ppo_config": ppo_config.model_dump(),
        "head_sizes": _head_sizes(env),
        "env_config": str(env_config_path.resolve()),
        "rounds": rounds,
        "seed": seed,
        "revision": revision,
    }
    temp = path.with_suffix(".tmp")
    torch.save(payload, temp)
    temp.replace(path)


def load_per_model_checkpoint(
    path: Path, env: PerModelEnv
) -> tuple[SetNetwork, dict[str, Any]]:
    """Rebuild the network a driver checkpoint describes and load its weights.

    Returns the network and the checkpoint's own metadata (rounds, configs,
    revision) so a scorer can quote provenance without guessing.
    """
    payload = torch.load(path, map_location="cpu", weights_only=True)
    saved_heads = payload.get("head_sizes")
    if saved_heads is not None and saved_heads != _head_sizes(env):
        raise ValueError(
            f"Checkpoint {path} was trained with action heads {saved_heads}; "
            f"this env implies {_head_sizes(env)}. Score it on the scenario "
            "family it trained on — a load_state_dict shape wall would name "
            "every layer and never this cause."
        )
    network = SetNetwork.from_env(env, SetNetworkConfig(**payload["network_config"]))
    network.load_state_dict(payload["state_dict"])
    metadata = {key: value for key, value in payload.items() if key != "state_dict"}
    return network, metadata


def _mean_breakdown(breakdowns: list[dict[str, float]]) -> dict[str, float]:
    """Mean per-term episode totals over recent finished episodes."""
    if not breakdowns:
        return {}
    keys = sorted({key for breakdown in breakdowns for key in breakdown})
    return {
        key: sum(b.get(key, 0.0) for b in breakdowns) / len(breakdowns) for key in keys
    }


def _mean_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Mean of the numeric fields over a logging window; last value otherwise."""
    if not records:
        return {}
    merged: dict[str, Any] = {}
    keys = {key for record in records for key in record}
    for key in keys:
        values = [r[key] for r in records if key in r]
        if all(isinstance(v, (int, float)) for v in values):
            merged[key] = sum(values) / len(values)
        else:
            merged[key] = values[-1]
    # Counters and coordinates keep their latest value, not a window mean.
    for key in ("rounds", "env_steps", "epoch_equivalent", "train_episodes"):
        if key in records[-1]:
            merged[key] = records[-1][key]
    return merged


def _declaration_shares(counts: dict[str, int], prefix: str) -> dict[str, float]:
    """Skip-declaration shares per phase from an agent's tally window."""
    shares: dict[str, float] = {}
    for phase, skip_option, label in (
        ("movement", 1, "stationary_share"),
        ("shooting", 1, "hold_fire_share"),
    ):
        total = sum(v for k, v in counts.items() if k.startswith(f"{phase}:"))
        if total:
            shares[f"{prefix}{label}"] = counts.get(f"{phase}:{skip_option}", 0) / total
    return shares


class _JsonlLog:
    """Append-only metrics log: one JSON object per line, flushed per write."""

    def __init__(self, path: Path) -> None:
        self._path = path

    def write(self, record: dict[str, Any]) -> None:
        with self._path.open("a") as handle:
            handle.write(json.dumps(record) + "\n")


def train(
    env_config: Path,
    max_rounds: int,
    seed: int = 0,
    gamma: float = 0.9,
    gae_lambda: float = 0.95,
    rollout_rounds: int = 8,
    learning_rate: float = 3e-4,
    entropy_coef_selector: float = 0.01,
    entropy_coef_action: float = 0.01,
    minibatch_size: int = 64,
    n_update_epochs: int = 4,
    clip_epsilon: float = 0.2,
    embedding_size: int = 128,
    n_layers: int = 4,
    n_heads: int = 8,
    eval_every_rounds: int = 7680,
    n_eval_episodes: int = 20,
    eval_seed_base: int = 500000,
    checkpoint_every_rounds: int = 25600,
    device: str = "cuda",
    torch_threads: int = 2,
    run_suffix: str = "",
    wandb_group: str = "",
    use_wandb: bool = True,
    out_root: Path = Path("checkpoints/per_model"),
) -> Path:
    """Train a `SetNetwork` on the per-model facade with single-scalar PPO.

    Plain python so a launcher (or a test) can call it with real defaults;
    the CLI command below is a thin wrapper. Returns the run directory.
    """
    if eval_every_rounds % rollout_rounds or checkpoint_every_rounds % rollout_rounds:
        raise ValueError(
            "eval_every_rounds and checkpoint_every_rounds must be multiples "
            f"of rollout_rounds ({rollout_rounds}) — otherwise the eval "
            "cadence drifts with the overshoot and two grid cells with "
            "different rollout budgets eval at different round counts, "
            "making their curves comparable only at the final point."
        )
    torch.set_num_threads(torch_threads)
    torch.manual_seed(seed)
    revision = _git_revision()
    torch_device = torch.device(device)

    config = parse_yaml_raw_as(WargameEnvConfig, env_config.read_text())
    config.render_mode = None
    env = PerModelEnv(config, build_info=False)
    eval_env = PerModelEnv(config, build_info=False)

    network_config = SetNetworkConfig(
        embedding_size=embedding_size, n_layers=n_layers, n_heads=n_heads
    )
    ppo_config = PerModelPPOConfig(
        gamma=gamma,
        gae_lambda=gae_lambda,
        rollout_rounds=rollout_rounds,
        learning_rate=learning_rate,
        entropy_coef_selector=entropy_coef_selector,
        entropy_coef_action=entropy_coef_action,
        minibatch_size=minibatch_size,
        n_update_epochs=n_update_epochs,
        clip_epsilon=clip_epsilon,
    )
    network = SetNetwork.from_env(env, network_config).to(torch_device)
    for module in network.modules():
        if isinstance(module, torch.nn.Dropout) and module.p > 0:
            raise ValueError(
                "Nonzero dropout: rollouts sample with the network in train "
                "mode, so the sampled log-prob and evaluate_transitions' "
                "recompute would come from different dropout masks and every "
                "PPO ratio would be silently wrong. Wire eval/train mode "
                "handling before enabling dropout."
            )
    agent = SetAgent(network, device=torch_device)
    optimizer = torch.optim.Adam(network.parameters(), lr=ppo_config.learning_rate)
    sample_generator = torch.Generator().manual_seed(seed)
    update_generator = torch.Generator().manual_seed(seed + 1)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # The pid disambiguates two launches of one cell in the same second — a
    # shared run dir interleaves metrics.jsonl and clobbers last.pt silently.
    name = f"per_model_{env_config.stem}{run_suffix}_s{seed}_{stamp}_p{os.getpid()}"
    run_dir = out_root / name
    run_dir.mkdir(parents=True, exist_ok=True)
    log = _JsonlLog(run_dir / "metrics.jsonl")
    # The config as run, verbatim: the path in provenance can outlive the
    # file it named (a worktree launch), and a re-scored checkpoint must be
    # rebuilt against the scenario that trained it, not today's edit of it.
    (run_dir / "env_config.yaml").write_text(env_config.read_text())
    provenance = {
        "env_config": str(env_config.resolve()),
        "seed": seed,
        "revision": revision,
        "torch_threads": torch_threads,
        "device": device,
        "rollout_seed_base": ROLLOUT_SEED_BASE,
        "eval_seed_base": eval_seed_base,
        "max_rounds": max_rounds,
        "network_config": network_config.model_dump(),
        "ppo_config": ppo_config.model_dump(),
    }
    (run_dir / "provenance.json").write_text(json.dumps(provenance, indent=2))
    logger.info("Run dir: {}", run_dir)

    wandb_run = None
    if use_wandb:
        import wandb

        # Same project AND entity as every whole-phase run (the constants in
        # model/common/wandb.py, not imported to keep Lightning off this
        # path) — without the entity, runs land under the account default
        # and vanish from the dashboard everything else reports to.
        wandb_run = wandb.init(  # type: ignore[attr-defined]
            project="wargame_rl",
            entity="wargame_rl",
            group=wandb_group or None,
            name=name,
            config=provenance,
        )

    # The scripted bar, measured once at start exactly as the whole-phase
    # trainer's on_train_start does (same policies, same seed band, same
    # episode count) and logged under the same keys: without a floor and a
    # reference the eval numbers say nothing.
    baseline_record: dict[str, Any] = {}
    if use_wandb:
        baseline_env = WargameEnv(config, renderer=None, build_info=False)
        baseline_seeds = [BASELINE_SEED_BASE + i for i in range(BASELINE_EPISODES)]
        for baseline_name in BASELINE_POLICIES:
            bar = evaluate_baseline(
                build_baseline_policy(baseline_name), baseline_env, baseline_seeds
            )
            baseline_record[f"eval/baseline_{baseline_name}_win_rate"] = (
                bar.win_rate * 100
            )
            baseline_record[f"eval/baseline_{baseline_name}_vp_margin"] = bar.vp_margin
            baseline_record[f"eval/baseline_{baseline_name}_at_objectives"] = (
                bar.final_fraction_at_objectives
            )
            baseline_record[f"eval/baseline_{baseline_name}_fraction_alive"] = (
                bar.final_fraction_alive
            )
        log.write({"rounds": 0, **baseline_record})
        if wandb_run is not None:
            wandb_run.log(baseline_record, step=0)

    # Seed the rollout env's layout stream once; later resets continue it.
    env.reset(seed=ROLLOUT_SEED_BASE + seed)
    # First collect_rollout call resets into the stream.
    observation: PerModelObservation | None = None

    episode_margins: list[float] = []
    episode_breakdowns: list[dict[str, float]] = []

    def on_episode_end(finished: PerModelEnv) -> None:
        episode_margins.append(float(finished.player_vp - finished.opponent_vp))
        episode_breakdowns.append(dict(finished.episode_reward_breakdown))

    rounds_done = 0
    env_steps = 0
    next_eval = eval_every_rounds
    next_checkpoint = checkpoint_every_rounds
    started = time.perf_counter()
    overhead = 0.0  # eval + checkpoint wall time, excluded from rounds_per_s
    # Wandb gets one aggregated row per whole-phase-epoch-equivalent (1024
    # rounds) rather than one per 8-round rollout — the cadence every old
    # dashboard was built on; the local JSONL keeps the per-update rows.
    window: list[dict[str, Any]] = []
    next_wandb_flush = WANDB_LOG_EVERY_ROUNDS
    while rounds_done < max_rounds:
        budget = min(ppo_config.rollout_rounds, max_rounds - rounds_done)
        rollout_started = time.perf_counter()
        transitions, bootstrap, observation = collect_rollout(
            env,
            agent,
            budget,
            start_observation=observation,
            generator=sample_generator,
            on_episode_end=on_episode_end,
        )
        rollout_seconds = time.perf_counter() - rollout_started
        update_started = time.perf_counter()
        losses = ppo_update(
            network,
            optimizer,
            transitions,
            bootstrap,
            ppo_config,
            generator=update_generator,
        )
        update_seconds = time.perf_counter() - update_started
        rounds_done += budget
        env_steps += len(transitions)
        elapsed = time.perf_counter() - started - overhead
        n_minibatches = max(losses.pop("n_minibatches", 1.0), 1.0)
        record: dict[str, Any] = {
            "rounds": rounds_done,
            "env_steps": env_steps,
            "epoch_equivalent": rounds_done / WANDB_LOG_EVERY_ROUNDS,
            "steps": len(transitions),
            "rounds_per_s": rounds_done / elapsed,
            # The whole-phase trainer's names, same definitions.
            "loss/train_loss": losses["train_loss"],
            "loss/policy_loss": losses["policy_loss"],
            "loss/value_loss": losses["value_loss"],
            "loss/entropy_loss": losses["entropy"],
            "train/clip_fraction": losses["clip_fraction"],
            "train/approx_kl": losses["approx_kl"],
            "train/explained_variance": losses["explained_variance"],
            "train/grad_norm": losses["grad_norm"],
            "train/grad_clipped_fraction": losses["grad_clipped_fraction"],
            "perf/rollout_s": rollout_seconds,
            "perf/update_s": update_seconds,
            "perf/epoch_s": rollout_seconds + update_seconds,
            "perf/env_steps_per_s": len(transitions) / max(rollout_seconds, 1e-9),
            "perf/update_ms_per_minibatch": update_seconds * 1000.0 / n_minibatches,
            **_declaration_shares(agent.reset_declaration_counts(), "train/"),
        }
        if episode_margins:
            recent = episode_margins[-20:]
            record["train_vp_margin"] = sum(recent) / len(recent)
            record["train_episodes"] = len(episode_margins)
        for name, value in _mean_breakdown(episode_breakdowns[-20:]).items():
            record[f"reward/components/{name}"] = value
        log.write(record)
        window.append(record)

        if wandb_run is not None and (
            rounds_done >= next_wandb_flush or rounds_done >= max_rounds
        ):
            next_wandb_flush = rounds_done + WANDB_LOG_EVERY_ROUNDS
            wandb_run.log(_mean_records(window), step=rounds_done)
            window = []

        if rounds_done >= next_eval or rounds_done >= max_rounds:
            next_eval = rounds_done + eval_every_rounds
            eval_started = time.perf_counter()
            network.eval()
            eval_rewards: list[float] = []
            eval_steps: list[int] = []
            result = evaluate_per_model(
                eval_env,
                agent,
                seeds=[eval_seed_base + i for i in range(n_eval_episodes)],
                name="eval",
                episode_rewards=eval_rewards,
                episode_steps=eval_steps,
            )
            network.train()
            margins = np.array(result.vp_margin_per_episode, dtype=np.float64)
            eval_record: dict[str, Any] = {
                "rounds": rounds_done,
                # The whole-phase trainer's eval keys, same definitions —
                # win_rate in percent, elo the monotone margin transform
                # against this config's own opponent (not a fitted rating).
                "eval/vp_player": result.player_vp,
                "eval/vp_opponent": result.opponent_vp,
                "eval/vp_margin": result.vp_margin,
                "eval/win_rate": result.win_rate * 100.0,
                "eval/elo": rating_from_score(float(margin_score(margins).mean())),
                "eval/fraction_alive": result.final_fraction_alive,
                "eval/objectives_held": result.objectives_held,
                "reward/mean_episode_reward": float(np.mean(eval_rewards)),
                "reward/max_episode_reward": float(np.max(eval_rewards)),
                "reward/min_episode_reward": float(np.min(eval_rewards)),
                "mean_episode_steps": float(np.mean(eval_steps)),
                # The passive-attractor instrument: the skip declarations
                # gate five models through one greedy logit, and an eval
                # sitting at the do-nothing fingerprint shows up here first.
                **_declaration_shares(agent.reset_declaration_counts(), "eval/"),
            }
            if result.coherency_rate is not None:
                eval_record["eval/coherency_rate"] = result.coherency_rate
            if result.models_out_of_coherency is not None:
                eval_record["eval/models_out_of_coherency"] = (
                    result.models_out_of_coherency
                )
            if result.exposure_rate is not None:
                eval_record["eval/exposure_rate"] = result.exposure_rate
            if result.terrain_proximity is not None:
                eval_record["eval/terrain_proximity"] = result.terrain_proximity
            if result.firepower_ratio is not None:
                eval_record["eval/firepower_ratio"] = result.firepower_ratio
            log.write(eval_record)
            if wandb_run is not None:
                wandb_run.log(eval_record, step=rounds_done)
            logger.info(
                "rounds {} | eval vp_margin {:.1f} held {:.2f} coherent {} | {:.2f} rounds/s",
                rounds_done,
                result.vp_margin,
                result.objectives_held,
                result.coherency_rate,
                rounds_done / elapsed,
            )
            overhead += time.perf_counter() - eval_started

        if rounds_done >= next_checkpoint or rounds_done >= max_rounds:
            next_checkpoint = rounds_done + checkpoint_every_rounds
            checkpoint_started = time.perf_counter()
            for checkpoint_path in (
                run_dir / f"pm-{rounds_done:08d}.pt",
                run_dir / "last.pt",
            ):
                _save_checkpoint(
                    checkpoint_path,
                    network,
                    network_config,
                    ppo_config,
                    env,
                    env_config,
                    rounds_done,
                    seed,
                    revision,
                )
            overhead += time.perf_counter() - checkpoint_started

    if wandb_run is not None:
        wandb_run.finish()
    logger.info(
        "Done: {} rounds in {:.0f}s", rounds_done, time.perf_counter() - started
    )
    return run_dir


@app.command()
def train_command(
    env_config: Path = typer.Argument(..., help="Environment YAML config."),
    max_rounds: int = typer.Option(
        ..., help="Total rounds (turn cycles) of experience to train on."
    ),
    seed: int = typer.Option(0, help="Init and sampling seed."),
    gamma: float = typer.Option(0.9, help="Per-ROUND discount."),
    gae_lambda: float = typer.Option(0.95, help="Per-ROUND GAE decay."),
    rollout_rounds: int = typer.Option(8, help="Rollout budget per update, in rounds."),
    learning_rate: float = typer.Option(3e-4, "--lr"),
    entropy_coef_selector: float = typer.Option(0.01),
    entropy_coef_action: float = typer.Option(0.01),
    minibatch_size: int = typer.Option(64),
    n_update_epochs: int = typer.Option(4),
    clip_epsilon: float = typer.Option(0.2),
    embedding_size: int = typer.Option(128),
    n_layers: int = typer.Option(4),
    n_heads: int = typer.Option(8),
    eval_every_rounds: int = typer.Option(
        7680, help="Greedy evaluation cadence, in rounds of experience."
    ),
    n_eval_episodes: int = typer.Option(20),
    eval_seed_base: int = typer.Option(
        500000, help="Evaluation layout band (500000+ is the in-run band)."
    ),
    checkpoint_every_rounds: int = typer.Option(25600),
    device: str = typer.Option("cuda", help="cuda or cpu."),
    torch_threads: int = typer.Option(
        2, help="torch.set_num_threads — pinned and recorded per issue #306."
    ),
    run_suffix: str = typer.Option("", help="Disambiguates concurrent runs."),
    wandb_group: str = typer.Option("", help="Wandb group name."),
    use_wandb: bool = typer.Option(True, "--wandb/--no-wandb"),
    out_root: Path = typer.Option(Path("checkpoints/per_model")),
) -> None:
    """Train a `SetNetwork` on the per-model facade with single-scalar PPO."""
    train(
        env_config=env_config,
        max_rounds=max_rounds,
        seed=seed,
        gamma=gamma,
        gae_lambda=gae_lambda,
        rollout_rounds=rollout_rounds,
        learning_rate=learning_rate,
        entropy_coef_selector=entropy_coef_selector,
        entropy_coef_action=entropy_coef_action,
        minibatch_size=minibatch_size,
        n_update_epochs=n_update_epochs,
        clip_epsilon=clip_epsilon,
        embedding_size=embedding_size,
        n_layers=n_layers,
        n_heads=n_heads,
        eval_every_rounds=eval_every_rounds,
        n_eval_episodes=n_eval_episodes,
        eval_seed_base=eval_seed_base,
        checkpoint_every_rounds=checkpoint_every_rounds,
        device=device,
        torch_threads=torch_threads,
        run_suffix=run_suffix,
        wandb_group=wandb_group,
        use_wandb=use_wandb,
        out_root=out_root,
    )


if __name__ == "__main__":
    app()
