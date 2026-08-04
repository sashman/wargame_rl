"""Measure every scripted baseline on one env config over a fixed seed set.

Gives learned-policy numbers a floor and a reference bar. `squad_march` is the
bar to beat; the spread between the baselines says *what kind* of mistake a
policy is making rather than only that it is losing.

Usage: just measure-baselines <env_config> [n_episodes]
"""

from __future__ import annotations

import sys

from pydantic_yaml import parse_yaml_raw_as

from wargame_rl.wargame.envs.baseline.evaluate import evaluate_baseline
from wargame_rl.wargame.envs.baseline.registry import build_baseline_policy
from wargame_rl.wargame.envs.types.config import WargameEnvConfig
from wargame_rl.wargame.model.common.factory import create_environment

# Ordered weakest to strongest so the printed table reads as a scale.
BASELINES = ("random", "greedy_nearest", "split_evenly", "squad_march")

# Held out from the training seed space so baselines stay a genuine reference.
EVAL_SEED_BASE = 10_000


def main() -> None:
    """Print a table of baseline results for the given env config."""
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(1)

    config_path = sys.argv[1]
    n_episodes = int(sys.argv[2]) if len(sys.argv) > 2 else 25

    with open(config_path) as handle:
        env_config = parse_yaml_raw_as(WargameEnvConfig, handle.read())

    env = create_environment(env_config=env_config)
    seeds = [EVAL_SEED_BASE + i for i in range(n_episodes)]

    print(f"\n{config_path}  ({n_episodes} episodes, seeds {seeds[0]}-{seeds[-1]})\n")
    header = f"{'baseline':<16}{'on_obj':>9}{'win':>8}{'player_vp':>11}"
    print(f"{header}{'opp_vp':>9}{'cohesion_gap':>14}")
    print("-" * 67)

    for name in BASELINES:
        policy = build_baseline_policy(name)
        result = evaluate_baseline(policy, env, seeds)
        print(
            f"{name:<16}"
            f"{result.final_fraction_at_objectives:>9.3f}"
            f"{result.win_rate:>8.2f}"
            f"{result.player_vp:>11.1f}"
            f"{result.opponent_vp:>9.1f}"
            f"{result.worst_cohesion_gap:>14.1f}"
        )
    print()


if __name__ == "__main__":
    main()
