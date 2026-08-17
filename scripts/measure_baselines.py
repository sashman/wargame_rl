"""Measure every scripted baseline on one env config over a fixed seed set.

Gives learned-policy numbers a floor and a reference bar. `squad_march` is the
bar to beat; the spread between the baselines says *what kind* of mistake a
policy is making rather than only that it is losing.

Pass `record` as a third argument to also write one event log per baseline to
`recordings/`, giving `just analyze-compare` a reference trace to put beside an
agent's.

Usage: just measure-baselines <env_config> [n_episodes] [record]
"""

from __future__ import annotations

import sys
from pathlib import Path

from pydantic_yaml import parse_yaml_raw_as

from wargame_rl.wargame.envs.baseline.evaluate import (
    evaluate_baseline,
    format_optional_metric,
    record_baseline_episode,
)
from wargame_rl.wargame.envs.baseline.registry import build_baseline_policy
from wargame_rl.wargame.envs.types.config import WargameEnvConfig
from wargame_rl.wargame.model.common.factory import create_environment

# Ordered weakest to strongest so the printed table reads as a scale.
# `squad_march_shoot` and `contest_and_spread` are the only ones that fire, and
# therefore the only bars set by the policy class the learned agent is in.
#
# `contest_and_spread` is here because `squad_march_shoot` proved to be a weak
# ceiling, not a strong one: it allocates squads `k % n_objectives` and fires
# nearest-first, ignoring both an opponent that stops moving by round 9 and the
# fact that a second shot on a one-wound target is usually discarded. Quoting a
# learned policy only against the weaker bar overstates it.
BASELINES = (
    "random",
    "greedy_nearest",
    "split_evenly",
    "squad_march",
    "squad_march_shoot",
    "contest_and_spread",
)

# Held out from the training seed space so baselines stay a genuine reference.
EVAL_SEED_BASE = 10_000


def main() -> None:
    """Print a table of baseline results for the given env config."""
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(1)

    config_path = sys.argv[1]
    n_episodes = int(sys.argv[2]) if len(sys.argv) > 2 else 25
    record = len(sys.argv) > 3 and sys.argv[3].lower() in {"record", "true", "1"}
    # Pass the checkpoint script's base to put an agent and the baselines on
    # identical layouts; objective placement dominates episode variance, so an
    # unpaired comparison spends several VP of it on which maps each drew.
    # Empty means "not given" — the recipe passes every optional argument
    # through, so an omitted one arrives as "" rather than being absent.
    seed_base = (
        int(sys.argv[4]) if len(sys.argv) > 4 and sys.argv[4] else EVAL_SEED_BASE
    )

    with open(config_path) as handle:
        env_config = parse_yaml_raw_as(WargameEnvConfig, handle.read())

    env = create_environment(env_config=env_config)
    seeds = [seed_base + i for i in range(n_episodes)]

    print(f"\n{config_path}  ({n_episodes} episodes, seeds {seeds[0]}-{seeds[-1]})\n")
    header = (
        f"{'baseline':<18}{'on_obj':>9}{'win':>8}{'player_vp':>11}"
        f"{'opp_vp':>9}{'held':>7}{'alive':>8}{'coherent':>10}{'adrift':>8}"
        f"{'exposure':>10}{'terrain_d':>11}{'firepower':>12}"
    )
    print(header)
    print("-" * len(header))

    recorded: list[Path] = []
    for name in BASELINES:
        policy = build_baseline_policy(name)
        result = evaluate_baseline(policy, env, seeds)
        print(
            f"{name:<18}"
            f"{result.final_fraction_at_objectives:>9.3f}"
            f"{result.win_rate:>8.2f}"
            f"{result.player_vp:>11.1f}"
            f"{result.opponent_vp:>9.1f}"
            f"{result.objectives_held:>7.2f}"
            f"{result.final_fraction_alive:>8.3f}"
            f"{format_optional_metric(result.coherency_rate):>10}"
            f"{format_optional_metric(result.models_out_of_coherency, 2):>8}"
            f"{format_optional_metric(result.exposure_rate):>10}"
            f"{format_optional_metric(result.terrain_proximity, 1):>11}"
            f"{format_optional_metric(result.firepower_ratio, 2):>12}"
        )
        if record:
            recorded.append(
                record_baseline_episode(
                    build_baseline_policy(name),
                    env_config,
                    seeds[0],
                    Path("recordings") / f"baseline_{name}.jsonl",
                )
            )

    if recorded:
        print("\nreference traces:")
        for path in recorded:
            print(f"  {path}")
        print(f"\n  just analyze-compare {' '.join(str(p) for p in recorded)}")
    print()


if __name__ == "__main__":
    main()
