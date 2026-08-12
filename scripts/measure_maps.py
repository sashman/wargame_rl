"""Score a policy on the real table layouts, one row per map.

Final evaluation. Training uses `random_terrain`, which regenerates a layout
every episode — that is what makes a positioning result falsifiable, but it
never asks how the policy does on the *actual* boards the game is played on.
This runs the golden scenario unchanged and swaps only `terrain`, once per map
in `configs/evaluation/maps/`.

Overriding rather than shipping a config per map is deliberate. A 25v25 config
is ~13 KB of scenario, and copying it per map means every future reward change
has to be applied N times — the first one that is missed makes evaluation
measure a different game from training, silently.

Reports per map rather than a single mean, because the mean is the one number
that cannot answer the question you have: *which* layouts does the policy
handle badly. The spread across maps is printed for the same reason.

Accepts either a scripted baseline name or a checkpoint path, so the agent and
the bar are measured by one code path on identical layouts.

Usage: just measure-maps <policy|ckpt> <env_config> [n_episodes] [maps_dir]
"""

from __future__ import annotations

import statistics
import sys
from pathlib import Path
from typing import cast

from pydantic_yaml import parse_yaml_raw_as

from scripts.measure_checkpoint import HELDOUT_SEED_BASE, build_selector, label_for
from wargame_rl.wargame.envs.baseline.evaluate import (
    ActionSelector,
    BaselineResult,
    evaluate_selector,
    format_optional_metric,
    selector_for,
)
from wargame_rl.wargame.envs.baseline.registry import (
    build_baseline_policy,
    get_registry,
)
from wargame_rl.wargame.envs.types import TerrainMapConfig, WargameEnvConfig
from wargame_rl.wargame.model.common.factory import create_environment

DEFAULT_MAPS_DIR = Path("configs/evaluation/maps")


def load_maps(maps_dir: Path) -> list[TerrainMapConfig]:
    """Read every map in `maps_dir`, sorted by filename for a stable row order."""
    if not maps_dir.is_dir():
        raise SystemExit(f"maps directory not found: {maps_dir}")
    paths = sorted(maps_dir.glob("*.yaml"))
    if not paths:
        raise SystemExit(
            f"no maps in {maps_dir}. Each map is a file of the form:\n\n"
            "    name: table_01\n"
            "    terrain:\n"
            "      - { footprint: [12, 8, 18, 14] }\n\n"
            "See configs/README.md."
        )
    return [parse_yaml_raw_as(TerrainMapConfig, path.read_text()) for path in paths]


def config_for_map(
    base_config: WargameEnvConfig, terrain_map: TerrainMapConfig
) -> WargameEnvConfig:
    """Copy `base_config` with this map's terrain, and objectives, in place of its own.

    **Every other terrain mode is cleared, not just `terrain` set.** All three
    are mutually exclusive, and leaving either of the others on would replace
    the layout at reset and discard the map entirely — scoring the training
    distribution while printing a map's name. The failure is silent and it looks
    exactly like a real result: with `map_pool` left on, all 45 rows come back
    byte-identical, because every row scored the same drawn sequence.

    A map that carries objectives replaces the scenario's, including its
    `number_of_objectives`: the real layouts put six on the table, and a map's
    objectives are as much a part of it as its ruins. The placement constraints
    are cleared with them, since they govern the random draw this map replaces
    and the real layouts do not satisfy `objective_min_separation` anyway.
    """
    config = cast(WargameEnvConfig, base_config.model_copy(deep=True))
    config.terrain = list(terrain_map.terrain)
    config.random_terrain = None
    config.map_pool = None
    config.render_mode = None
    if terrain_map.objectives is not None:
        config.objectives = list(terrain_map.objectives)
        config.number_of_objectives = len(terrain_map.objectives)
        config.objectives_on_terrain = False
        config.objective_min_separation = None
        config.objective_terrain_clearance = None
    return config


def build_action_selector(
    policy_or_checkpoint: str, config: WargameEnvConfig
) -> tuple[ActionSelector, str]:
    """Resolve the first argument to a selector, accepting a policy or a path.

    Built per map because a checkpoint's network is sized from the env, and a
    scripted policy holds env state; neither is safe to carry across configs.
    """
    if Path(policy_or_checkpoint).exists():
        env = create_environment(env_config=config)
        select, _net = build_selector(policy_or_checkpoint, env)
        return select, label_for(policy_or_checkpoint)

    if policy_or_checkpoint not in get_registry():
        raise SystemExit(
            f"'{policy_or_checkpoint}' is neither a checkpoint path nor a "
            f"baseline. Known baselines: {', '.join(sorted(get_registry()))}"
        )
    policy = build_baseline_policy(policy_or_checkpoint)
    return selector_for(policy), policy_or_checkpoint


def format_row(label: str, result: BaselineResult) -> str:
    """One aligned row, the same columns `measure-checkpoint` prints."""
    return (
        f"{label:<20}{result.final_fraction_at_objectives:>9.3f}"
        f"{result.win_rate:>8.2f}{result.vp_margin:>12.1f}"
        f"{result.objectives_held:>7.2f}{result.final_fraction_alive:>8.3f}"
        f"{format_optional_metric(result.exposure_rate):>10}"
        f"{format_optional_metric(result.firepower_ratio, 2):>11}"
    )


def main() -> None:
    """Score one policy across every real map and print the per-map table."""
    if len(sys.argv) < 3:
        print(__doc__)
        raise SystemExit(1)

    policy_or_checkpoint = sys.argv[1]
    config_path = sys.argv[2]
    n_episodes = int(sys.argv[3]) if len(sys.argv) > 3 else 100
    maps_dir = Path(sys.argv[4]) if len(sys.argv) > 4 else DEFAULT_MAPS_DIR

    base_config = parse_yaml_raw_as(WargameEnvConfig, Path(config_path).read_text())
    maps = load_maps(maps_dir)
    # The same seeds on every map, so a difference between rows is the layout
    # and not the draw. Disjoint from training and model selection.
    seeds = [HELDOUT_SEED_BASE + i for i in range(n_episodes)]

    print(f"\n{policy_or_checkpoint}")
    print(
        f"{config_path}  ({len(maps)} maps x {n_episodes} episodes, "
        f"seeds {seeds[0]}-{seeds[-1]})\n"
    )
    header = (
        f"{'map':<20}{'on obj':>9}{'win':>8}{'VP margin':>12}"
        f"{'held':>7}{'alive':>8}{'exposure':>10}{'firepower':>11}"
    )
    print(header)
    print("-" * len(header))

    results: list[BaselineResult] = []
    for terrain_map in maps:
        config = config_for_map(base_config, terrain_map)
        select, _label = build_action_selector(policy_or_checkpoint, config)
        env = create_environment(env_config=config)
        result = evaluate_selector(select, env, seeds, terrain_map.name)
        results.append(result)
        print(format_row(terrain_map.name, result))
        env.close()

    print("-" * len(header))
    margins = [result.vp_margin for result in results]
    wins = [result.win_rate for result in results]
    print(
        f"{'mean':<20}{statistics.fmean(r.final_fraction_at_objectives for r in results):>9.3f}"
        f"{statistics.fmean(wins):>8.2f}{statistics.fmean(margins):>12.1f}"
        f"{statistics.fmean(r.objectives_held for r in results):>7.2f}"
        f"{statistics.fmean(r.final_fraction_alive for r in results):>8.3f}"
    )
    if len(results) > 1:
        # Printed because a mean over maps hides the case this evaluation exists
        # to find: strong on most tables, broken on one.
        print(
            f"{'spread (min..max)':<20}{'':>9}"
            f"{min(wins):>5.2f}..{max(wins):<2.2f}"
            f"{min(margins):>7.1f}..{max(margins):<.1f}"
        )
    print()


if __name__ == "__main__":
    main()
