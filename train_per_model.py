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

import torch
import typer
from loguru import logger
from pydantic_yaml import parse_yaml_raw_as

from wargame_rl.wargame.envs.per_model import PerModelEnv
from wargame_rl.wargame.envs.per_model.types import PerModelObservation
from wargame_rl.wargame.envs.types import WargameEnvConfig
from wargame_rl.wargame.model.per_model import SetAgent, SetNetwork, SetNetworkConfig
from wargame_rl.wargame.model.per_model.evaluate import evaluate_per_model
from wargame_rl.wargame.model.per_model.ppo import (
    PerModelPPOConfig,
    collect_rollout,
    ppo_update,
)

app = typer.Typer(pretty_exceptions_enable=False)

# The whole-phase trainer's rollout band (model/ppo/lightning.py), reused so
# neither facade trains on the evaluation bands (500000+/700000+/900000+).
ROLLOUT_SEED_BASE = 0


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

        wandb_run = wandb.init(  # type: ignore[attr-defined]
            project="wargame_rl",
            group=wandb_group or None,
            name=name,
            config=provenance,
        )

    # Seed the rollout env's layout stream once; later resets continue it.
    env.reset(seed=ROLLOUT_SEED_BASE + seed)
    # First collect_rollout call resets into the stream.
    observation: PerModelObservation | None = None

    episode_margins: list[float] = []

    def on_episode_end(finished: PerModelEnv) -> None:
        episode_margins.append(float(finished.player_vp - finished.opponent_vp))

    rounds_done = 0
    next_eval = eval_every_rounds
    next_checkpoint = checkpoint_every_rounds
    started = time.perf_counter()
    overhead = 0.0  # eval + checkpoint wall time, excluded from rounds_per_s
    while rounds_done < max_rounds:
        budget = min(ppo_config.rollout_rounds, max_rounds - rounds_done)
        transitions, bootstrap, observation = collect_rollout(
            env,
            agent,
            budget,
            start_observation=observation,
            generator=sample_generator,
            on_episode_end=on_episode_end,
        )
        losses = ppo_update(
            network,
            optimizer,
            transitions,
            bootstrap,
            ppo_config,
            generator=update_generator,
        )
        rounds_done += budget
        elapsed = time.perf_counter() - started - overhead
        record: dict[str, Any] = {
            "rounds": rounds_done,
            "steps": len(transitions),
            "rounds_per_s": rounds_done / elapsed,
            **losses,
            **_declaration_shares(agent.reset_declaration_counts(), "train/"),
        }
        if episode_margins:
            recent = episode_margins[-20:]
            record["train_vp_margin"] = sum(recent) / len(recent)
            record["train_episodes"] = len(episode_margins)
        log.write(record)
        if wandb_run is not None:
            wandb_run.log(record, step=rounds_done)

        if rounds_done >= next_eval or rounds_done >= max_rounds:
            next_eval = rounds_done + eval_every_rounds
            eval_started = time.perf_counter()
            network.eval()
            result = evaluate_per_model(
                eval_env,
                agent,
                seeds=[eval_seed_base + i for i in range(n_eval_episodes)],
                name="eval",
            )
            network.train()
            eval_record = {
                "rounds": rounds_done,
                "eval/vp_margin": result.vp_margin,
                "eval/win_rate": result.win_rate,
                "eval/objectives_held": result.objectives_held,
                "eval/coherency_rate": result.coherency_rate,
                "eval/alive": result.final_fraction_alive,
                # The passive-attractor instrument: the skip declarations
                # gate five models through one greedy logit, and an eval
                # sitting at the do-nothing fingerprint shows up here first.
                **_declaration_shares(agent.reset_declaration_counts(), "eval/"),
            }
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
