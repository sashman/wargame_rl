"""Compare two policies episode by episode on identical layouts.

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

Either argument may be a scripted baseline name **or a checkpoint path**, so a
trained policy can be paired against the bar directly.

Usage: just measure-paired <policy_a|ckpt> <policy_b|ckpt> <env_config> [n] [seed_base]
"""

from __future__ import annotations

import sys

import numpy as np

from scripts.measure_checkpoint import build_selector
from scripts.scenario_overrides import describe, load_env_config, parse_overrides
from wargame_rl.wargame.envs.baseline.evaluate import (
    ActionSelector,
    evaluate_selector,
    selector_for,
)
from wargame_rl.wargame.envs.baseline.registry import (
    build_baseline_policy,
    get_registry,
)
from wargame_rl.wargame.envs.types import WargameEnvConfig
from wargame_rl.wargame.envs.wargame import WargameEnv
from wargame_rl.wargame.model.common.factory import create_environment

HELDOUT_SEED_BASE = 700000


def _selector_for(name: str, env: WargameEnv) -> ActionSelector:
    """A scripted baseline by registry name, or a policy from a checkpoint path.

    Taking both is what lets the comparison that actually matters be paired --
    a trained checkpoint against the bar on identical layouts. Scoring each
    separately and differencing the aggregates cannot resolve it: at n=100 the
    standard error on each row is ~5-9 vp, so their difference carries ~7-13,
    which is the size of most effects worth measuring here.
    """
    if name in get_registry():
        return selector_for(build_baseline_policy(name))
    select, _net = build_selector(name, env)
    return select


def episode_margins(
    policy_name: str, env_config: WargameEnvConfig, seeds: list[int]
) -> np.ndarray:
    """Per-episode ``player_vp - opponent_vp``, one entry per seed, in order.

    Runs through `evaluate_selector` rather than its own loop so this script and
    the baseline table cannot answer the same question differently -- two
    implementations of "score a policy over seeds" drifting apart is exactly the
    class of defect the pairing here exists to guard against. The array it
    returns is built one entry per seed unconditionally, so it stays
    index-aligned with `seeds` and with the other arm's.
    """
    env = create_environment(env_config=env_config)
    select = _selector_for(policy_name, env)
    result = evaluate_selector(select, env, seeds, policy_name)
    return np.array(result.vp_margin_per_episode)


def main() -> None:
    """Print both arms' marginal means and the paired difference between them."""
    argv, overrides = parse_overrides(sys.argv)
    if len(argv) < 4:
        print(__doc__)
        raise SystemExit(1)

    name_a, name_b, config_path = argv[1], argv[2], argv[3]
    # A checkpoint path is far too long for a table column, and its basename is
    # what identifies the run anyway.
    label_a, label_b = (
        name.split("/")[-1][:30] if "/" in name else name for name in (name_a, name_b)
    )
    n_episodes = int(argv[4]) if len(argv) > 4 else 100
    seed_base = int(argv[5]) if len(argv) > 5 else HELDOUT_SEED_BASE
    seeds = [seed_base + index for index in range(n_episodes)]

    env_config = load_env_config(config_path, **overrides)

    margins_a = episode_margins(name_a, env_config, seeds)
    margins_b = episode_margins(name_b, env_config, seeds)
    delta = margins_b - margins_a
    stderr = float(delta.std(ddof=1) / np.sqrt(len(delta))) or float("inf")
    decided = int((delta != 0).sum())

    print(
        f"\n{config_path}{describe(overrides)}   {n_episodes} episodes, "
        f"seeds {seed_base}+\n"
    )
    print(f"{label_a:<32}{margins_a.mean():>9.1f}   (sd {margins_a.std():.1f})")
    print(f"{label_b:<32}{margins_b.mean():>9.1f}   (sd {margins_b.std():.1f})")
    print(f"\npaired difference       {delta.mean():>9.1f}   +/- {stderr:.1f} (1 se)")
    print(f"t = {delta.mean() / stderr:>.2f}")
    print(
        f"{label_b} ahead in {int((delta > 0).sum())} of {decided} episodes "
        f"that differed at all ({n_episodes - decided} identical)"
    )
    print(
        f"\nthe marginal sd is ~{margins_a.std():.0f} against a paired se of "
        f"{stderr:.1f} -- unpaired, an effect this size is invisible\n"
    )


if __name__ == "__main__":
    main()
