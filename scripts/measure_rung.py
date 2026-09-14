"""Read a curriculum rung: the scripted bar and every checkpoint, one table.

A rung's pre-registration is decided on three readouts the two facades share
-- the success rate, the turns-to-success and `held` -- measured at n=100 on
the held-out seeds the bar was measured on, so the turn difference against
the script pairs per episode. This prints the bar first, then one row per
checkpoint (a `.pt` plays the per-model facade, a `.ckpt` the whole-army
one, both through `evaluate_spec`), with the paired turn difference and the
passive pair on every per-model row.

Usage: just measure-rung <env_config> <n_episodes> <seed_base> <policy> <ckpt...>
"""

from __future__ import annotations

import sys

import numpy as np

from scripts.scenario_overrides import describe, load_env_config, parse_overrides
from wargame_rl.wargame.envs.evaluation import EvalResult, format_optional_metric
from wargame_rl.wargame.scoring import evaluate_spec
from wargame_rl.wargame.selectors import label_for

HELDOUT_SEED_BASE = 700_000


def format_row(label: str, result: EvalResult, bar: EvalResult | None) -> str:
    """One aligned row; the turn column pairs against `bar` when given."""
    turns = np.array(result.turns_per_episode, dtype=float)
    paired = ""
    if bar is not None and len(bar.turns_per_episode) == len(turns) and len(turns):
        delta = turns - np.array(bar.turns_per_episode, dtype=float)
        paired = (
            f"{delta.mean():>+7.2f} ± {delta.std(ddof=1) / np.sqrt(len(delta)):.2f}"
        )
    return (
        f"{label:<34}{format_optional_metric(result.success_rate, 3):>8}"
        f"{format_optional_metric(result.mean_turns, 2):>7}{paired:>16}"
        f"{result.objectives_held:>6.2f}{result.final_fraction_at_objectives:>8.3f}"
        f"{result.vp_margin:>9.1f}{format_optional_metric(result.vp_margin_se, 1):>7}"
        f"{format_optional_metric(result.coherency_rate):>9}"
        f"{format_optional_metric(result.stationary_share, 2):>6}"
        f"{format_optional_metric(result.hold_fire_share, 2):>6}"
    )


def main() -> None:
    """Print the bar and every checkpoint's row on the rung."""
    argv, overrides = parse_overrides(sys.argv)
    if len(argv) < 5:
        print(__doc__)
        raise SystemExit(1)
    config_path = argv[1]
    n_episodes = int(argv[2])
    seed_base = int(argv[3]) if argv[3] else HELDOUT_SEED_BASE
    policy_name = argv[4]
    checkpoints = argv[5:]

    env_config = load_env_config(config_path, **overrides)
    env_config.render_mode = None
    seeds = [seed_base + i for i in range(n_episodes)]

    print(
        f"\n{config_path}{describe(overrides)}  ({n_episodes} episodes, "
        f"seeds {seeds[0]}-{seeds[-1]}; turns are phase-clock turns)\n"
    )
    header = (
        f"{'policy':<34}{'success':>8}{'turns':>7}{'vs bar (paired)':>16}"
        f"{'held':>6}{'on obj':>8}{'vp':>9}{'±SE':>7}{'coherent':>9}{'stat':>6}{'hold':>6}"
    )
    print(header)
    print("-" * len(header))
    bar = evaluate_spec(policy_name, env_config, seeds, policy_name)
    print(format_row(policy_name, bar, None))
    for checkpoint in checkpoints:
        label = f"{label_for(checkpoint)}/{checkpoint.rsplit('/', 1)[-1]}"[:34]
        result = evaluate_spec(checkpoint, env_config, seeds, label)
        print(format_row(label, result, bar))
    print()


if __name__ == "__main__":
    main()
