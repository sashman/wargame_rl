"""Can the policy explore WHERE a squad goes, or only how fast?

The joint-feasibility geometry of a 3-model squad says a product policy pays
`p^k` for any *directional* disagreement inside a squad, but nothing for
disagreeing about speed. So the entropy that survives training is predicted to
be one shared angle with several speeds -- the policy may explore how fast a
squad moves and not where it moves. Reassigning a squad to a different objective
is exactly a coordinated change of angle, which such a policy cannot propose.

This tests that directly. It reads each model's ARGMAX action (`decode_topk=1`,
so the play-time joint decoder cannot launder the answer), converts it to its
angle bin, and per squad per step reports:

- **circular variance** of the squad's chosen angles: 0.0 = perfectly aligned,
  1.0 = uniformly spread. This is the rank-1 collapse statistic.
- **distinct angle bins** per squad, and the share of squads where every alive
  member picked the SAME bin.
- the **stay share**, because STAY has no angle and is trivially legal -- the
  scripts stand still on 38-57% of unit-moves and the agent on 0.4%, so a
  comparison that folds STAY into "aligned" would flatter the scripts.

⚠ Read `distinct angle bins` beside squad size: a one-model squad is aligned by
definition, and casualties shrink squads over an episode.

Usage: just measure-angle-collapse <policy|ckpt> <env_config> [n_episodes] [decode_topk]
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from scripts.measure_maps import config_for_map, load_maps
from scripts.scenario_overrides import describe, load_env_config, parse_overrides
from wargame_rl.wargame.envs.domain.entities import alive_mask_for
from wargame_rl.wargame.envs.env_components.actions import STAY_ACTION
from wargame_rl.wargame.envs.env_components.distance_cache import compute_distances
from wargame_rl.wargame.envs.types import WargameEnvConfig
from wargame_rl.wargame.envs.types.game_timing import BattlePhase
from wargame_rl.wargame.model.common.factory import create_environment
from wargame_rl.wargame.selectors import build_action_selector

HELDOUT_SEED_BASE = 700_000
DEFAULT_MAPS_DIR = Path("configs/evaluation/maps_heldout")


@dataclass
class Collapse:
    """Per-squad directional agreement, accumulated over steps."""

    squads: int = 0
    circular_variance: float = 0.0
    on_objective_variance: list[float] = field(default_factory=list)
    en_route_variance: list[float] = field(default_factory=list)
    distinct_bins: int = 0
    squad_members: int = 0
    fully_aligned: int = 0
    moves: int = 0
    stays: int = 0
    n_angles: int = 0
    # BETWEEN squads, which is the discriminating axis. Within-squad alignment
    # is what the winning script does BY DESIGN -- `squad_march` moves a squad
    # as a body -- so it cannot separate a good allocator from a bad one. What
    # can is whether the squads point at DIFFERENT places.
    steps_multi_squad: int = 0
    between_variance: float = 0.0
    distinct_squad_angles: int = 0
    squads_per_step: int = 0

    def mean(self, total: float) -> float:
        """A running total per squad observed."""
        return total / self.squads if self.squads else 0.0


def squad_angles(
    actions: np.ndarray, env: object, n_speed_bins: int, n_angles: int
) -> dict[int, list[int]]:
    """Angle bin chosen by each alive, moving model, grouped by squad."""
    models = env.player_models  # type: ignore[attr-defined]
    alive = np.asarray(alive_mask_for(models))
    by_squad: dict[int, list[int]] = {}
    for index, model in enumerate(models):
        if not alive[index]:
            continue
        action = int(actions[index])
        by_squad.setdefault(int(model.group_id), [])
        if action == STAY_ACTION:
            continue
        bin_index = (action - 1) // n_speed_bins
        if bin_index >= n_angles:
            raise ValueError(
                f"action {action} decodes to angle bin {bin_index} of {n_angles} "
                "-- a non-movement action reached the heading decode"
            )
        by_squad[int(model.group_id)].append(bin_index)
    return by_squad


def collect(
    policy: str, config: WargameEnvConfig, seeds: list[int], decode_topk: int
) -> Collapse:
    """Walk every map and seed, tallying directional agreement inside squads."""
    tally = Collapse(n_angles=config.n_movement_angles)
    for terrain_map in load_maps(DEFAULT_MAPS_DIR):
        per_map = config_for_map(config, terrain_map)
        env = create_environment(env_config=per_map)
        select = build_action_selector(policy, env, decode_topk).select
        n_speed_bins = per_map.n_speed_bins
        for seed in seeds:
            observation, _info = env.reset(seed=seed)
            done = False
            while not done:
                chosen = select(observation, env)
                actions = np.asarray(chosen.actions, dtype=int).reshape(-1)
                # Only the movement phase carries a heading. Without this the
                # shooting slice is decoded as `(action - 1) // n_speed_bins`
                # too, which lands on angle bin 16+ of a 16-bin wheel -- and
                # squadmates shoot the SAME target, so those rows read as
                # unanimous and dilute whichever policy shoots more.
                if env.game_clock_state.phase is not BattlePhase.movement:
                    observation, _r, done, _t, _i = env.step(chosen)
                    continue
                alive = np.asarray(alive_mask_for(env.player_models))  # type: ignore[attr-defined]
                tally.moves += int(np.sum(alive & (actions != STAY_ACTION)))
                tally.stays += int(np.sum(alive & (actions == STAY_ACTION)))

                headings: list[complex] = []
                modes: list[int] = []
                for _group, bins in squad_angles(
                    actions, env, n_speed_bins, tally.n_angles
                ).items():
                    if not bins:
                        continue
                    radians = 2.0 * np.pi * np.asarray(bins) / tally.n_angles
                    unit = np.exp(1j * radians)
                    resultant = float(np.abs(np.mean(unit)))
                    headings.append(complex(np.mean(unit)))
                    modes.append(int(np.bincount(bins).argmax()))
                    if len(bins) < 2:
                        continue  # A lone mover is aligned by definition.
                    tally.squads += 1
                    tally.circular_variance += 1.0 - resultant
                    # Split by whether the squad has ARRIVED. A squad standing
                    # where it means to stand can disagree about headings
                    # without that costing it anything; only a squad still
                    # travelling is testing the claim that disagreement stops
                    # movement.
                    members = [
                        m
                        for m in env.player_models  # type: ignore[attr-defined]
                        if int(m.group_id) == int(_group) and m.is_alive
                    ]
                    if members:
                        cache = compute_distances(
                            members,
                            env.objectives,  # type: ignore[attr-defined]
                            alive_mask=np.ones(len(members), dtype=bool),
                        )
                        arrived = bool(
                            (
                                cache.model_obj_norms_offset.min(axis=1)
                                <= cache.obj_radii[
                                    cache.model_obj_norms_offset.argmin(axis=1)
                                ]
                            ).mean()
                            > 0.5
                        )
                        target = (
                            tally.on_objective_variance
                            if arrived
                            else tally.en_route_variance
                        )
                        target.append(1.0 - resultant)
                    tally.distinct_bins += len(set(bins))
                    tally.squad_members += len(bins)
                    tally.fully_aligned += int(len(set(bins)) == 1)

                if len(headings) > 1:
                    vectors = np.asarray(headings)
                    vectors = vectors / np.maximum(np.abs(vectors), 1e-12)
                    tally.steps_multi_squad += 1
                    tally.between_variance += 1.0 - float(np.abs(np.mean(vectors)))
                    tally.distinct_squad_angles += len(set(modes))
                    tally.squads_per_step += len(modes)
                observation, _r, done, _t, _i = env.step(chosen)
        env.close()
    return tally


def report(tally: Collapse) -> None:
    """Print the collapse statistics."""
    total = tally.moves + tally.stays
    print(f"    moving squads sampled            {tally.squads:>10,}")
    print(
        f"    mean movers per squad            {tally.mean(tally.squad_members):>10.2f}"
    )
    print(
        f"    CIRCULAR VARIANCE of angles      {tally.mean(tally.circular_variance):>10.4f}"
        "   <- 0 = one shared direction"
    )
    print(
        f"    distinct angle bins per squad    {tally.mean(tally.distinct_bins):>10.2f}"
        f"   (of {tally.n_angles})"
    )
    print(
        f"    squads all on ONE angle          {100.0 * tally.fully_aligned / tally.squads if tally.squads else 0.0:>9.1f}%"
    )

    def _mean(sample: list[float]) -> str:
        return f"{float(np.mean(sample)):.4f} (n={len(sample):,})" if sample else "-"

    print(
        f"    variance -- squad ALREADY ON an objective  "
        f"{_mean(tally.on_objective_variance)}"
    )
    print(
        f"    variance -- squad EN ROUTE                 "
        f"{_mean(tally.en_route_variance)}"
    )
    print(
        f"    stay share of alive model-steps  {100.0 * tally.stays / total if total else 0.0:>9.1f}%"
    )
    steps = tally.steps_multi_squad
    if not steps:
        return
    print("\n    -- BETWEEN squads: the discriminating axis --")
    print(
        f"    moving squads per step           {tally.squads_per_step / steps:>10.2f}"
    )
    print(
        f"    CIRCULAR VARIANCE across squads  {tally.between_variance / steps:>10.4f}"
        "   <- 0 = the whole army marches one way"
    )
    print(
        f"    distinct squad headings per step {tally.distinct_squad_angles / steps:>10.2f}"
        f"   (of {tally.n_angles})"
    )


def main() -> None:
    """Report whether a squad's models ever choose different directions."""
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
