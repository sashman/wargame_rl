"""Fit a per-model clone's value head to the teacher's returns, leaving the
policy bit-identical -- the instrument for D2's cold-critic question.

The whole-army record (`CLAUDE.md` § Settled): PPO from a behaviour clone
with a randomly initialised critic destroyed the clone at every entropy
coefficient, and the per-model D2 arm reproduced it in two thousand rounds.
This records the teacher's games again (the same seeds the clone was fitted
on), turns the re-timed rewards into discounted returns, and fits ONLY the
value head by MSE, so a PPO run from the output differs from one from the
input by nothing but the critic.

Usage: just fit-per-model-critic <clone.pt> <teacher> <env_config> [n_episodes]
       [epochs] [out] [seed]
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import torch
from loguru import logger

from scripts.scenario_overrides import load_env_config
from wargame_rl.wargame.envs.per_model.env import PerModelEnv
from wargame_rl.wargame.model.per_model.checkpoint import load_checkpoint
from wargame_rl.wargame.model.per_model.clone import (
    CLONE_SEED_BASE,
    explained_variance,
    fit_critic,
    record_demonstrations,
    save_clone,
    value_targets,
)
from wargame_rl.wargame.selectors import build_per_model_chooser


def _revision() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True
        )
        return out.stdout.strip() or "unknown"
    except OSError:
        return "unknown"


def main() -> None:
    """Record, fit the value head, report explained variance, save."""
    if len(sys.argv) < 4:
        print(__doc__)
        raise SystemExit(2)
    clone_path, teacher, config_path = sys.argv[1], sys.argv[2], sys.argv[3]
    n_episodes = int(sys.argv[4]) if len(sys.argv) > 4 else 300
    epochs = int(sys.argv[5]) if len(sys.argv) > 5 else 20
    out = (
        Path(sys.argv[6])
        if len(sys.argv) > 6
        else Path(clone_path).with_name(Path(clone_path).stem + "-critic.pt")
    )
    seed = int(sys.argv[7]) if len(sys.argv) > 7 else 0

    config = load_env_config(config_path)
    env = PerModelEnv(config)
    chooser = build_per_model_chooser(teacher, [env], seed=seed, greedy=True)
    loaded = load_checkpoint(Path(clone_path))
    network = loaded.network
    policy_before = {k: v.clone() for k, v in network.state_dict().items()}
    logger.info(
        "Recording {} episodes of {} (seeds {}+)",
        n_episodes,
        chooser.label,
        CLONE_SEED_BASE,
    )
    demos = record_demonstrations(env, chooser.choose, n_episodes)
    targets = value_targets(demos, gamma=loaded.ppo_config.gamma)
    held_out = max(1, n_episodes // 5)
    train_rows = [i for i, t in enumerate(demos) if t.env_index < n_episodes - held_out]
    test_rows = [i for i, t in enumerate(demos) if t.env_index >= n_episodes - held_out]
    train = [demos[i] for i in train_rows]
    test = [demos[i] for i in test_rows]
    before = explained_variance(network, test, targets[test_rows])
    logger.info("Explained variance before the fit (held-out): {:.3f}", before)
    losses = fit_critic(network, train, targets[train_rows], epochs=epochs, seed=seed)
    after = explained_variance(network, test, targets[test_rows])
    logger.info(
        "Explained variance after the fit (held-out): {:.3f}; loss {:.3f} -> {:.3f}",
        after,
        losses[0],
        losses[-1],
    )
    changed = [
        k
        for k, v in network.state_dict().items()
        if not k.startswith("value_head") and not torch.equal(v, policy_before[k])
    ]
    if changed:
        raise RuntimeError(f"the policy changed under a critic fit: {changed[:3]}")
    save_clone(
        out,
        network,
        env_config=config.model_dump(mode="json"),
        seed=seed,
        revision=_revision(),
        teacher=chooser.label,
        n_episodes=n_episodes,
        epochs=epochs,
        match={
            "critic_explained_variance_held_out": after,
            "critic_explained_variance_before": before,
        },
    )
    provenance_path = out.with_suffix(".clone.json")
    provenance = json.loads(provenance_path.read_text())
    provenance["critic_fitted_from"] = str(clone_path)
    provenance_path.write_text(json.dumps(provenance, indent=2))
    print(
        f"\n{out}  critic fitted on {n_episodes} episodes x {epochs} epochs; held-out explained variance {before:.3f} -> {after:.3f}"
    )


if __name__ == "__main__":
    main()
