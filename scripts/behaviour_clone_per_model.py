"""Clone a scripted policy into the set network, so the per-model PPO can start
from a policy that already carries a whole plan.

**Why this exists.** On the curriculum's sequencing rung (C3 / C3b, #340)
neither trainer learned the escort's plan from reward: the per-model arm
learned to shoot the blockers off the point and then parked, the from-scratch
run learned almost nothing, and the whole-army control walked in and died. The
D rungs ask the next question — can the set network *hold* an ordered plan
(D1), and can PPO improve a policy that starts with one (D2)? — and this is
the instrument: the script's decisions recorded as the transitions the update
reads, fitted by maximum likelihood over the selector and the three heads.

Usage: just behaviour-clone-per-model <policy> <env_config> [n_episodes] [epochs]
       [out] [seed]

The teacher is any name `build_per_model_chooser` resolves (a scripted
baseline, `random`, or a `.pt` played greedy). Demonstrations use seeds
`CLONE_SEED_BASE + episode` (800000+), disjoint from the evaluation band; the
last fifth of the episodes is held out and the per-head match is reported on
it. The output is a checkpoint `train_per_model.py --warm-start-from` accepts
and every `measure-*` recipe plays.

⚠ **The critic is not fitted.** The saved value head is at its random
initialisation; PPO from a clone with a cold critic destroyed a whole-army
clone at every entropy setting (`CLAUDE.md` § Settled). D2's pre-registration
owns that question.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import torch
from loguru import logger

from scripts.scenario_overrides import load_env_config
from wargame_rl.wargame.envs.per_model.env import PerModelEnv
from wargame_rl.wargame.model.per_model.clone import (
    CLONE_SEED_BASE,
    fit_clone,
    match_report,
    record_demonstrations,
    save_clone,
)
from wargame_rl.wargame.model.per_model.net import SetNetwork
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
    """Record the teacher, fit the clone, report the match, save the `.pt`."""
    if len(sys.argv) < 3:
        print(__doc__)
        raise SystemExit(2)
    policy, config_path = sys.argv[1], sys.argv[2]
    n_episodes = int(sys.argv[3]) if len(sys.argv) > 3 else 300
    epochs = int(sys.argv[4]) if len(sys.argv) > 4 else 40
    out = (
        Path(sys.argv[5])
        if len(sys.argv) > 5
        else Path("checkpoints/per_model/clone.pt")
    )
    seed = int(sys.argv[6]) if len(sys.argv) > 6 else 0

    config = load_env_config(config_path)
    env = PerModelEnv(config)
    chooser = build_per_model_chooser(policy, [env], seed=seed, greedy=True)
    logger.info(
        "Recording {} episodes of {} on {} (seeds {}+)",
        n_episodes,
        chooser.label,
        config_path,
        CLONE_SEED_BASE,
    )
    demos = record_demonstrations(env, chooser.choose, n_episodes)
    held_out = max(1, n_episodes // 5)
    train = [t for t in demos if t.env_index < n_episodes - held_out]
    test = [t for t in demos if t.env_index >= n_episodes - held_out]
    logger.info(
        "{} decision steps to fit, {} held out ({} episodes)",
        len(train),
        len(test),
        held_out,
    )
    torch.manual_seed(seed)
    network = SetNetwork.from_env(env)
    before = match_report(network, test)
    logger.info("Match before the fit: {}", before)
    fit_clone(network, train, epochs=epochs, seed=seed, held_out=test)
    after = match_report(network, test)
    logger.info("Match after the fit: {}", after)
    save_clone(
        out,
        network,
        env_config=config.model_dump(mode="json"),
        seed=seed,
        revision=_revision(),
        teacher=chooser.label,
        n_episodes=n_episodes,
        epochs=epochs,
        match=after,
    )
    logger.info("Saved {}", out)
    print(
        f"\n{out}  clone of {chooser.label} on {config_path}: "
        f"{n_episodes} episodes x {epochs} epochs, seed {seed}"
    )
    print("held-out match: " + ", ".join(f"{k} {v:.3f}" for k, v in after.items()))


if __name__ == "__main__":
    main()
