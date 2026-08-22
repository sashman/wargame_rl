"""Do the agent's squads bunch, or spread out and arrive thick?

The agent stacks: **4.90 models on its top objective where `squad_march_take`
puts 2.73**, 55.3% of points empty, and exactly half the script's
`objective_hold` income from a pot it splits over half as many places. The
anti-stacking lever that would normally answer this -- `crowding_exponent: 1.0`,
which makes a point pay a fixed pot split between occupants, so spreading
strictly raises income -- is **already on and already ignored**, so adding more
of it is the wrong move.

That leaves a structural explanation. Models move in squads under a 2" chain and
`objective_hold` requires coherence, so **the squad, not the model, is the
allocation quantum**: you cannot send one model of three to a different point.
If the squads themselves converge, no reward weight fixes it and the stacking is
a movement problem.

This separates the two. For every step it records which squads are alive, which
objectives each squad stands on (`norms_offset <= radius`, the test VP scoring
uses), and how far apart the squad centroids are. The discriminating number is
**squads per occupied objective**: at 1.0 each squad has a point to itself, and
above it they are sharing.

⚠ Squads per occupied objective is bounded below by 1.0 and rises when a squad
is *destroyed* as well as when two converge, since a dead squad occupies
nothing. Read `squads alive` beside it.

Usage: just measure-squad-dispersion <policy|ckpt> <env_config> [n_episodes] [decode_topk]
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np

from scripts.measure_maps import build_action_selector, config_for_map, load_maps
from scripts.scenario_overrides import describe, load_env_config, parse_overrides
from wargame_rl.wargame.envs.domain.entities import alive_mask_for
from wargame_rl.wargame.envs.env_components.distance_cache import compute_distances
from wargame_rl.wargame.envs.types import WargameEnvConfig
from wargame_rl.wargame.model.common.factory import create_environment

HELDOUT_SEED_BASE = 700_000
DEFAULT_MAPS_DIR = Path("configs/evaluation/maps_heldout")


@dataclass
class Dispersion:
    """Squad-level occupancy and spread, accumulated over steps."""

    steps: int = 0
    squads_alive: int = 0
    squads_on_objective: int = 0
    objectives_occupied: int = 0
    squads_sharing: int = 0
    objectives_total: int = 0
    centroid_gap: float = 0.0
    centroid_samples: int = 0
    board_diagonal: float = 0.0

    def per_step(self, total: int) -> float:
        """A running total expressed per step."""
        return total / self.steps if self.steps else 0.0


def squad_occupancy(env: object) -> tuple[dict[int, set[int]], dict[int, np.ndarray]]:
    """Which objectives each alive squad stands on, and where its centre is."""
    models = env.player_models  # type: ignore[attr-defined]
    alive = np.asarray(alive_mask_for(models))
    cache = compute_distances(models, env.objectives, alive_mask=alive)  # type: ignore[attr-defined]
    inside = np.asarray(cache.model_obj_norms_offset <= cache.obj_radii)

    occupied: dict[int, set[int]] = {}
    centroids: dict[int, np.ndarray] = {}
    positions: dict[int, list[np.ndarray]] = {}
    for index, model in enumerate(models):
        if not alive[index]:
            continue
        group = int(model.group_id)
        occupied.setdefault(group, set()).update(np.flatnonzero(inside[index]).tolist())
        positions.setdefault(group, []).append(np.asarray(model.location, dtype=float))
    for group, locations in positions.items():
        centroids[group] = np.mean(np.stack(locations), axis=0)
    return occupied, centroids


def collect(
    policy: str, config: WargameEnvConfig, seeds: list[int], decode_topk: int
) -> Dispersion:
    """Walk every map and seed, tallying squad-level spread."""
    tally = Dispersion()
    for terrain_map in load_maps(DEFAULT_MAPS_DIR):
        per_map = config_for_map(config, terrain_map)
        select, _label = build_action_selector(policy, per_map, decode_topk, False)
        env = create_environment(env_config=per_map)
        tally.board_diagonal = float(
            np.hypot(per_map.board_width, per_map.board_height)
        )
        for seed in seeds:
            observation, _info = env.reset(seed=seed)
            done = False
            while not done:
                observation, _r, done, _t, _i = env.step(select(observation, env))
                occupied, centroids = squad_occupancy(env)
                standing = {g: o for g, o in occupied.items() if o}
                every_objective: set[int] = set()
                for objectives in standing.values():
                    every_objective |= objectives

                # A squad "shares" when some objective it stands on is also
                # stood on by another squad.
                shared = 0
                for group, objectives in standing.items():
                    others: set[int] = set()
                    for other, other_objectives in standing.items():
                        if other != group:
                            others |= other_objectives
                    if objectives & others:
                        shared += 1

                tally.steps += 1
                tally.squads_alive += len(centroids)
                tally.squads_on_objective += len(standing)
                tally.objectives_occupied += len(every_objective)
                tally.squads_sharing += shared
                tally.objectives_total += len(env.objectives)
                if len(centroids) > 1:
                    gaps = [
                        float(np.linalg.norm(a - b))
                        for a, b in combinations(centroids.values(), 2)
                    ]
                    tally.centroid_gap += float(np.mean(gaps))
                    tally.centroid_samples += 1
        env.close()
    return tally


def report(tally: Dispersion) -> None:
    """Print the bunching ratio and the spread it comes from."""
    alive = tally.per_step(tally.squads_alive)
    standing = tally.per_step(tally.squads_on_objective)
    occupied = tally.per_step(tally.objectives_occupied)
    sharing = tally.per_step(tally.squads_sharing)
    gap = tally.centroid_gap / tally.centroid_samples if tally.centroid_samples else 0.0
    share_of_diagonal = (
        100.0 * gap / tally.board_diagonal if tally.board_diagonal else 0.0
    )
    ratio = standing / occupied if occupied else float("nan")
    share_sharing = 100.0 * sharing / standing if standing else 0.0

    print(f"    squads alive                     {alive:>8.2f}")
    print(f"    squads standing on an objective  {standing:>8.2f}")
    print(f"    distinct objectives occupied     {occupied:>8.2f}")
    print(
        f"    objectives on the table          {tally.per_step(tally.objectives_total):>8.2f}"
    )
    print(f"    SQUADS PER OCCUPIED OBJECTIVE    {ratio:>8.2f}   <- 1.00 = one each")
    print(f"    squads sharing a point           {share_sharing:>7.1f}%")
    print(
        f'    mean gap between squad centres   {gap:>8.2f}"  ({share_of_diagonal:.1f}% of the diagonal)'
    )


def main() -> None:
    """Report whether squads converge or take separate objectives."""
    argv, overrides = parse_overrides(sys.argv)
    if len(argv) < 3:
        print(__doc__)
        raise SystemExit(1)

    policy = argv[1]
    config_path = argv[2]
    n_episodes = int(argv[3]) if len(argv) > 3 and argv[3] else 20
    decode_topk = int(argv[4]) if len(argv) > 4 and argv[4] else 1

    config = load_env_config(config_path, **overrides)
    seeds = [HELDOUT_SEED_BASE + i for i in range(n_episodes)]
    tally = collect(policy, config, seeds, decode_topk)

    print(f"\n{policy}")
    print(
        f"{config_path}{describe(overrides)}  ({n_episodes} episodes per map, "
        f"seeds {seeds[0]}-{seeds[-1]}, decode_topk={decode_topk})\n"
    )
    report(tally)
    print()


if __name__ == "__main__":
    main()
