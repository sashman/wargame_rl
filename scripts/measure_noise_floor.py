"""Measure how much of a config's outcome variance is dice rather than skill.

Every arm run so far has been a single seed with no error bar, and the 25v25
scenarios sit near a dead heat (vp_margin around zero), which is exactly where
the rolls decide. Without knowing the dice-only spread there is no way to say
whether a 2-point win-rate gap between two arms means anything.

Holds one scripted policy and a fixed set of layouts constant and varies only
the combat seed, then reports:

- **within-layout spread** — pure dice. The noise floor: any difference between
  two arms smaller than this is unreadable at one seed.
- **between-layout spread** — how much the scenario draw matters. Large values
  mean paired seeds (the same layouts for every policy) are essential, which is
  what `evaluate_selector` already does.

Usage: just measure-noise-floor <env_config> [n_layouts] [n_combat_seeds] [policy]
"""

from __future__ import annotations

import statistics
import sys

from pydantic_yaml import parse_yaml_raw_as

from wargame_rl.wargame.envs.baseline.evaluate import evaluate_baseline
from wargame_rl.wargame.envs.baseline.registry import build_baseline_policy
from wargame_rl.wargame.envs.types.config import WargameEnvConfig
from wargame_rl.wargame.model.common.factory import create_environment

# Held out from the training seed space, matching measure_baselines.
LAYOUT_SEED_BASE = 10_000
COMBAT_SEED_BASE = 900_000

# The bar, and the only baseline that shoots — the noise floor of a policy that
# never fires would understate the dice's contribution.
DEFAULT_POLICY = "squad_march_shoot"


def _argument(index: int, default: str) -> str:
    """Read a positional argument. Recipes pass omitted ones through as ''."""
    if len(sys.argv) > index and sys.argv[index]:
        return sys.argv[index]
    return default


def main() -> None:
    """Print per-layout and aggregate outcome spread for one config."""
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(1)

    config_path = sys.argv[1]
    n_layouts = int(_argument(2, "10"))
    n_combat_seeds = int(_argument(3, "10"))
    policy_name = _argument(4, DEFAULT_POLICY)

    with open(config_path) as handle:
        env_config = parse_yaml_raw_as(WargameEnvConfig, handle.read())

    env = create_environment(env_config=env_config)
    combat_seeds = [COMBAT_SEED_BASE + j for j in range(n_combat_seeds)]

    print(
        f"\n{config_path}  ({policy_name}, {n_layouts} layouts x "
        f"{n_combat_seeds} combat seeds = {n_layouts * n_combat_seeds} episodes)\n"
    )
    header = f"{'layout':<10}{'win_rate':>10}{'vp_margin':>12}{'vp_margin_sd':>15}"
    print(header)
    print("-" * len(header))

    layout_win_rates: list[float] = []
    layout_margins: list[float] = []
    within_layout_sds: list[float] = []

    for i in range(n_layouts):
        seed = LAYOUT_SEED_BASE + i
        # One call per (layout, combat seed) so each episode's outcome is
        # separable; a single call would average them away.
        margins: list[float] = []
        wins = 0
        for combat_seed in combat_seeds:
            result = evaluate_baseline(
                build_baseline_policy(policy_name),
                env,
                [seed],
                combat_seeds=[combat_seed],
            )
            margins.append(result.vp_margin)
            wins += int(result.win_rate > 0.5)

        win_rate = wins / n_combat_seeds
        mean_margin = statistics.fmean(margins)
        margin_sd = statistics.stdev(margins) if len(margins) > 1 else 0.0

        layout_win_rates.append(win_rate)
        layout_margins.append(mean_margin)
        within_layout_sds.append(margin_sd)
        print(f"{seed:<10}{win_rate:>10.2f}{mean_margin:>12.1f}{margin_sd:>15.1f}")

    print("-" * len(header))
    overall_win = statistics.fmean(layout_win_rates)
    between_sd = (
        statistics.stdev(layout_margins) if len(layout_margins) > 1 else float("nan")
    )
    within_sd = statistics.fmean(within_layout_sds)

    print(f"\n{policy_name} on {config_path}")
    print(f"  overall win rate                 {overall_win:.3f}")
    print(f"  vp_margin sd within a layout     {within_sd:.1f}   <- the dice")
    print(f"  vp_margin sd between layouts     {between_sd:.1f}   <- the scenario")
    print(
        "\n  Win-rate differences between arms smaller than the dice term are "
        "not\n  readable at one seed. If the scenario term dominates, paired "
        "seeds are\n  mandatory — which `evaluate_selector` already enforces.\n"
    )


if __name__ == "__main__":
    main()
