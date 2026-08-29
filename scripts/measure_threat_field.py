"""Where a policy stands, priced against where the opponent can shoot NEXT turn.

⚠ **THE OPPONENT MOVES BEFORE IT SHOOTS**, so the map of what bears *right now*
is not a cheaper version of this question -- it is a different and
systematically optimistic one. A cell behind a ruin from where an enemy stands
is shot from beside that ruin one move later. This census is taken against
`ThreatHorizon.next_turn`, which traces sight from every cell a shooter can
reach, and it prints the disagreement with the current-turn map as its own
number so the size of that error is on the record rather than assumed.

TWO THINGS IT MEASURES

**The calibration.** `false-safe` is the share of board cells the current-turn
map calls clear and the next-turn map calls dangerous. It is the quantitative
form of "do not read the current threat map".

**The exposure census.** For every living model at the end of the player's turn,
the expected casualties the opponent's next shooting phase costs a model
standing there -- split by whether the model is **on an objective** or **in
transit**. That split is the falsifier for the standing explanation of hoarding:
the agent finishes with far more of its army alive than the scripts while
holding fewer objectives, and "it is avoiding danger" predicts its exposure is
*lower* than theirs. If it is not, hoarding is a search failure and no risk-side
lever will touch it.

⚠ **COVER IS NOT APPLIED**, so every number is the expectation against a target
in the open. The three-state visibility predicate is not on `BattleView` and a
grid cell has no base radius, so cover at a cell is undefined rather than merely
expensive. The bias runs **against objectives specifically** -- every marker on
the real tables sits inside a terrain piece -- so this field paints the safest
ground in the game as dangerous. Read the on-objective column beside
`just measure-hold-hazard`, which prices the same trade with the real predicate.

⚠ **Reach is an upper bound.** Coherency binds the opponent's move, freezing eats
~8% of ordered inches, and an advance forfeits the shooting so it never extends
threat. This overstates danger everywhere, uniformly enough to compare policies.

Usage: just measure-threat-field <policy|ckpt> <env_config> [n_episodes] [maps_dir] [decode_topk] [key=value...]
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from scripts.measure_checkpoint import HELDOUT_SEED_BASE
from scripts.measure_maps import DEFAULT_MAPS_DIR, config_for_map, load_maps
from scripts.scenario_overrides import describe, load_env_config, parse_overrides
from wargame_rl.wargame.envs.board.threat import (
    ThreatHorizon,
    VisibilityCache,
    attacker_stat_rows,
    move_reach,
    reference_model,
    threat_field,
)
from wargame_rl.wargame.envs.domain.entities import alive_mask_for
from wargame_rl.wargame.envs.env_components.distance_cache import compute_distances
from wargame_rl.wargame.envs.types.config import TerrainMapConfig, WargameEnvConfig
from wargame_rl.wargame.envs.wargame import WargameEnv
from wargame_rl.wargame.model.common.factory import create_environment
from wargame_rl.wargame.selectors import build_action_selector

# Matches the shipped overlays' `THREAT_SPACING`, so a printed number and a
# drawn frame describe the same cells.
SPACING = 1.0


@dataclass
class Tally:
    """Everything the census accumulates, so one pass answers both questions."""

    on_objective: list[float] = field(default_factory=list)
    in_transit: list[float] = field(default_factory=list)
    board_cells: int = 0
    current_cells: int = 0
    next_turn_cells: int = 0
    false_safe_cells: int = 0
    current_mean: list[float] = field(default_factory=list)
    next_turn_mean: list[float] = field(default_factory=list)
    samples: int = 0

    @property
    def all_models(self) -> list[float]:
        return self.on_objective + self.in_transit


def _on_objective(env: WargameEnv) -> np.ndarray:
    """`(n_models,)` -- is each player model on an objective, by the SCORING rule.

    `norms_offset <= obj_radii` from the model's base edge, which is the one
    definition of "on an objective" this project keeps. Three implementations of
    it once disagreed on 7.6% of slots; there is no reason to write a fourth.
    """
    models = env.player_models
    if not env.objectives:
        return np.zeros(len(models), dtype=bool)
    cache = compute_distances(models, env.objectives)
    return np.asarray((cache.model_obj_norms_offset <= cache.obj_radii).any(axis=1))


def _sample(
    env: WargameEnv,
    config: WargameEnvConfig,
    cache: VisibilityCache,
    tally: Tally,
) -> None:
    """Price the board once, for one battle round."""
    shooters = env.opponent_models
    ranges = np.asarray(env.opponent_max_ranges)
    stats = attacker_stat_rows(config.opponent_models, len(shooters))
    reference = reference_model(env.player_models, config.models)
    moves = move_reach(config, config.opponent_models, len(shooters))

    now = threat_field(
        env,
        shooters,
        ranges,
        stats,
        reference,
        horizon=ThreatHorizon.current,
        spacing=SPACING,
    )
    later = threat_field(
        env,
        shooters,
        ranges,
        stats,
        reference,
        horizon=ThreatHorizon.next_turn,
        move=moves,
        spacing=SPACING,
        visibility=cache,
    )

    dangerous_now = now.casualties > 0.0
    dangerous_later = later.casualties > 0.0
    tally.board_cells += later.grid.n_cells
    tally.current_cells += int(dangerous_now.sum())
    tally.next_turn_cells += int(dangerous_later.sum())
    tally.false_safe_cells += int((dangerous_later & ~dangerous_now).sum())
    tally.current_mean.append(float(now.casualties.mean()))
    tally.next_turn_mean.append(float(later.casualties.mean()))
    tally.samples += 1

    alive = alive_mask_for(env.player_models)
    if not alive.any():
        return
    positions = np.array(
        [[float(m.location[0]), float(m.location[1])] for m in env.player_models]
    )[alive]
    exposure = later.at(positions)
    held = _on_objective(env)[alive]
    tally.on_objective.extend(exposure[held].tolist())
    tally.in_transit.extend(exposure[~held].tolist())


def _battle_round(env: WargameEnv) -> int | None:
    """The round the clock is on, or None outside the battle."""
    battle_round: int | None = env.game_clock_state.battle_round
    return battle_round


# Sampled once per battle ROUND rather than per step. Per step would price the
# board mid-turn, when half an army has moved and half has not, which is a
# position no player ever chooses -- and it would cost a full sweep per phase.
# Once a round is also seat-agnostic: "what does their next shooting phase cost
# a model standing here" is well defined whoever has the turn, and the cadence
# is then identical for every policy compared.


def collect(
    policy: str,
    config: WargameEnvConfig,
    maps: list[TerrainMapConfig],
    seeds: list[int],
    decode_topk: int,
) -> Tally:
    """One pass over every table, sampling once per player turn."""
    tally = Tally()
    for terrain_map in maps:
        per_map = config_for_map(config, terrain_map)
        env = create_environment(env_config=per_map)
        try:
            select = build_action_selector(policy, env, decode_topk).select
            cache: VisibilityCache | None = None
            for seed in seeds:
                observation, _info = env.reset(seed=seed)
                # Terrain is fixed per table, and sight depends on terrain
                # alone -- so this is built once and reused for every turn of
                # every episode here. It is the whole reason the two-hop is
                # affordable at all.
                if cache is None:
                    cache = VisibilityCache.build(
                        env,
                        spacing=SPACING,
                        max_range=float(np.max(env.opponent_max_ranges)),
                    )
                done = truncated = False
                sampled = _battle_round(env)
                while not (done or truncated):
                    observation, _r, done, truncated, _i = env.step(
                        select(observation, env)
                    )
                    current = _battle_round(env)
                    if not (done or truncated) and current != sampled:
                        sampled = current
                        _sample(env, per_map, cache, tally)
        finally:
            env.close()
    return tally


def _stat(values: list[float]) -> str:
    if not values:
        return f"{'-':>9} {'-':>9} {'-':>7}"
    array = np.array(values)
    standard_error = array.std(ddof=1) / np.sqrt(len(array)) if len(array) > 1 else 0.0
    return f"{array.mean():>9.4f} {standard_error:>9.4f} {len(array):>7}"


def report(policy: str, tally: Tally) -> None:
    """Print the calibration and the census, in that order."""
    print("\ncurrent-turn versus next-turn -- how much the current map misses")
    if tally.board_cells:
        print(
            f"  {'threatened now':>28} {tally.current_cells / tally.board_cells:>8.1%}"
        )
        print(
            f"  {'threatened next turn':>28} {tally.next_turn_cells / tally.board_cells:>8.1%}"
        )
        print(
            f"  {'FALSE-SAFE cells':>28} {tally.false_safe_cells / tally.board_cells:>8.1%}"
            "   <- the current map calls these clear and they are not"
        )
        print(
            f"  {'mean casualties/cell, now':>28} {np.mean(tally.current_mean):>8.4f}"
        )
        print(
            f"  {'mean casualties/cell, next':>28} {np.mean(tally.next_turn_mean):>8.4f}"
        )

    print(
        f"\nexposure census -- expected casualties to a model standing there, {policy}"
    )
    print(f"  {'where':>14} {'mean':>9} {'se':>9} {'n':>7}")
    print(f"  {'on objective':>14} {_stat(tally.on_objective)}")
    print(f"  {'in transit':>14} {_stat(tally.in_transit)}")
    print(f"  {'all models':>14} {_stat(tally.all_models)}")

    if tally.on_objective and tally.in_transit:
        difference = np.mean(tally.on_objective) - np.mean(tally.in_transit)
        print(
            f"\n-> standing on an objective is {abs(difference):.4f} casualties "
            f"{'MORE' if difference > 0 else 'LESS'} exposed than being in transit."
        )
        print(
            "   ⚠ Cover is not applied and every objective on the real tables is "
            "a ruin, so this OVERSTATES the on-objective column specifically. "
            "Read it beside `just measure-hold-hazard`."
        )
    print(f"\n   {tally.samples} board samples, one per battle round.")


def main() -> None:
    """Census one policy's positions against the next-turn threat field."""
    argv, overrides = parse_overrides(sys.argv)
    if len(argv) < 3:
        print(__doc__)
        raise SystemExit(1)
    policy = argv[1]
    config_path = argv[2]
    n_episodes = int(argv[3]) if len(argv) > 3 and argv[3] else 5
    maps_dir = Path(argv[4]) if len(argv) > 4 and argv[4] else DEFAULT_MAPS_DIR
    decode_topk = int(argv[5]) if len(argv) > 5 and argv[5] else 1

    config = load_env_config(config_path, **overrides)
    maps = load_maps(maps_dir)
    seeds = [HELDOUT_SEED_BASE + i for i in range(n_episodes)]

    print(f"\n{policy}")
    print(
        f"{config_path}{describe(overrides)}  ({len(maps)} tables from {maps_dir}, "
        f"{n_episodes} episodes each, seeds {seeds[0]}+, decode_topk={decode_topk}, "
        f'{SPACING}" grid)'
    )
    report(policy, collect(policy, config, maps, seeds, decode_topk))


if __name__ == "__main__":
    main()
