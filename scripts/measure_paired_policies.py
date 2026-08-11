"""Compare two scripted policies episode by episode on identical layouts.

`measure-baselines` prints one aggregate row per policy. On 25v25 the
per-episode `vp_margin` standard deviation is ~45-90, so two such rows cannot
resolve anything smaller than about 10-18 vp at n=100 — larger than most effects
measured in this project. Running both policies over the *same seed list* and
differencing per episode removes the layout variance that dominates those rows,
and the paired standard error is an order of magnitude smaller.

That is not a refinement, it is the difference between a result and an artefact.
The comparison this script was written for read **+8.0 vp_margin** as two
aggregate means over 60 episodes and **+1.7 +/- 5.7** once paired over 100.

Two rules this encodes, both learned the hard way here:

- **Append unconditionally.** Every episode contributes one entry to each arm,
  in seed order, so the arrays cannot drift out of alignment. A probe that
  `continue`d past an episode in one arm and not the other reported +21.1 where
  the paired truth was +10.6.
- **Report the win count beside the mean.** A positive mean with a losing win
  count is a heavy tail, not an improvement, and only one of the two numbers
  says so.

Usage: just measure-paired <policy_a> <policy_b> <env_config> [n] [seed_base]
"""

from __future__ import annotations

import sys

import numpy as np
from pydantic_yaml import parse_yaml_raw_as

from wargame_rl.wargame.envs.baseline.evaluate import selector_for
from wargame_rl.wargame.envs.baseline.registry import build_baseline_policy
from wargame_rl.wargame.envs.types import WargameEnvConfig
from wargame_rl.wargame.model.common.factory import create_environment

HELDOUT_SEED_BASE = 700000


def episode_margins(
    policy_name: str, env_config: WargameEnvConfig, seeds: list[int]
) -> np.ndarray:
    """Per-episode ``player_vp - opponent_vp``, one entry per seed, in order.

    Appending happens on every iteration and never behind a branch, so the
    returned array stays index-aligned with `seeds` and with the other arm's.
    """
    env = create_environment(env_config=env_config)
    select = selector_for(build_baseline_policy(policy_name))
    margins: list[float] = []
    for seed in seeds:
        observation, _info = env.reset(seed=seed)
        done = False
        while not done:
            observation, _reward, terminated, truncated, _step = env.step(
                select(observation, env)
            )
            done = terminated or truncated
        margins.append(float(env.player_vp - env.opponent_vp))
    return np.array(margins)


def main() -> None:
    """Print both arms' marginal means and the paired difference between them."""
    if len(sys.argv) < 4:
        print(__doc__)
        raise SystemExit(1)

    name_a, name_b, config_path = sys.argv[1], sys.argv[2], sys.argv[3]
    n_episodes = int(sys.argv[4]) if len(sys.argv) > 4 else 100
    seed_base = int(sys.argv[5]) if len(sys.argv) > 5 else HELDOUT_SEED_BASE
    seeds = [seed_base + index for index in range(n_episodes)]

    with open(config_path) as handle:
        env_config = parse_yaml_raw_as(WargameEnvConfig, handle.read())

    margins_a = episode_margins(name_a, env_config, seeds)
    margins_b = episode_margins(name_b, env_config, seeds)
    delta = margins_b - margins_a
    stderr = float(delta.std(ddof=1) / np.sqrt(len(delta))) or float("inf")
    decided = int((delta != 0).sum())

    print(f"\n{config_path}   {n_episodes} episodes, seeds {seed_base}+\n")
    print(f"{name_a:<24}{margins_a.mean():>9.1f}   (sd {margins_a.std():.1f})")
    print(f"{name_b:<24}{margins_b.mean():>9.1f}   (sd {margins_b.std():.1f})")
    print(f"\npaired difference       {delta.mean():>9.1f}   +/- {stderr:.1f} (1 se)")
    print(f"t = {delta.mean() / stderr:>.2f}")
    print(
        f"{name_b} ahead in {int((delta > 0).sum())} of {decided} episodes "
        f"that differed at all ({n_episodes - decided} identical)"
    )
    print(
        f"\nthe marginal sd is ~{margins_a.std():.0f} against a paired se of "
        f"{stderr:.1f} -- unpaired, an effect this size is invisible\n"
    )


if __name__ == "__main__":
    main()
