"""What a policy actually buys with the advance move, and what it pays.

The advance slice is 48 of 150 actions on `25v25_maps_advance.yaml` -- one move
type taking 32% of the action space -- and it is encoded in a way that admits
three distinct defects. This census separates them on a trained policy, so the
next change is aimed at the one that binds rather than at all three at once.

1. **Dominated actions.** An advance bin is `fraction x (M + roll)`, so the
   lowest bin is always within a normal move's reach and the middle bin is
   whenever the roll is low. Such an action costs the unit its whole turn of
   shooting and buys a distance a normal move already delivers. At `M = 6` and
   a uniform D6 that is 50% of the slice in expectation. ⚠ The reason on file
   for tolerating them -- that a unit which cannot stop short cannot advance
   and halt -- does not hold: a unit's other members keep the NORMAL slice, so
   one model can trigger the advance while four stop short. Verified against
   `env.step`.

2. **Who pulled the trigger.** A move type is a UNIT decision in the rules; the
   action space is per model and resolves upward, so ONE model choosing an
   advance forfeits the shooting of all five. At initialisation that fires on
   85.5% of unit-turns (`1 - (1 - 48/150)^5`). The trigger histogram says
   whether a trained policy still advances by accident.

3. **Advancing to nowhere.** An advance that starts inside an objective and
   ENDS inside the same one has paid the unit's whole turn of shooting and
   gained no ground. ⚠ An advance that starts inside and *leaves* is the
   opposite -- reallocating to another point is the behaviour the record says
   is missing -- so the two must be counted apart. Reporting "advances from
   inside an objective" alone conflates them and reads as waste.

It also reports the within-unit spread of intended distances, because the
implementation lets a unit make one move type at per-model distances -- a 12"
against a 6" inside a 2" chain is the divergence the gap map admits.

⚠ Read this beside `just measure-freezing`: an advance is the longest move in
the game and therefore the most likely to be stopped, so a high advance share
and a high freeze rate are not independent observations.

Usage: just measure-advance-use <policy|ckpt> <env_config> [n_episodes] [decode_topk]
"""

from __future__ import annotations

import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from scripts.measure_maps import config_for_map, load_maps
from scripts.scenario_overrides import describe, load_env_config, parse_overrides
from wargame_rl.wargame.envs.domain.entities import WargameModel, alive_mask_for
from wargame_rl.wargame.envs.env_components.actions import STAY_ACTION
from wargame_rl.wargame.envs.env_components.distance_cache import compute_distances
from wargame_rl.wargame.envs.types import WargameEnvConfig
from wargame_rl.wargame.envs.types.game_timing import BattlePhase
from wargame_rl.wargame.model.common.factory import create_environment
from wargame_rl.wargame.selectors import build_action_selector

HELDOUT_SEED_BASE = 700_000
DEFAULT_MAPS_DIR = Path("configs/evaluation/maps_heldout")

# A bin landing exactly on the model's Move is still reachable by the top
# normal bin, so it is dominated. The tolerance absorbs the float division in
# `fraction x (M + roll)` rather than admitting a genuinely longer move.
DOMINATED_TOLERANCE = 1e-9


@dataclass
class AdvanceCensus:
    """Everything the advance slice was used for, over every map and seed."""

    move_steps: int = 0
    stays: int = 0
    advance_actions: int = 0
    dominated_advances: int = 0
    advance_from_inside: int = 0
    advance_stayed_inside: int = 0
    advance_distance: list[float] = field(default_factory=list)

    unit_turns: int = 0
    advancing_unit_turns: int = 0
    trigger_counts: Counter[int] = field(default_factory=Counter)
    unit_size: Counter[int] = field(default_factory=Counter)
    advancing_unit_spread: list[float] = field(default_factory=list)
    walking_unit_spread: list[float] = field(default_factory=list)

    shooting_slots: int = 0
    shooting_forfeited: int = 0

    # D-14: threat is move plus range. An advance crosses twice the ground,
    # so it can end inside an enemy's reach a whole turn earlier -- having
    # already spent the shooting that would answer. Split by move type,
    # because the share of models under threat says nothing on its own.
    ended_exposed_advancing: int = 0
    ended_moves_advancing: int = 0
    ended_exposed_walking: int = 0
    ended_moves_walking: int = 0

    def share(self, count: int, total: int) -> float:
        """A percentage, or zero when nothing was observed."""
        return 100.0 * count / total if total else 0.0


def _nearest_objective(
    members: list, objectives: list
) -> tuple[np.ndarray, np.ndarray]:
    """Per member, its nearest objective's index and whether it stands inside."""
    cache = compute_distances(
        members, objectives, alive_mask=np.ones(len(members), dtype=bool)
    )
    offsets = cache.model_obj_norms_offset
    nearest: np.ndarray = offsets.argmin(axis=1)
    inside: np.ndarray = offsets.min(axis=1) <= cache.obj_radii[nearest]
    return nearest, inside


def _still_inside(model: WargameModel, objectives: list, objective_index: int) -> bool:
    """Whether one model stands inside a NAMED objective, after its move."""
    cache = compute_distances([model], objectives, alive_mask=np.ones(1, dtype=bool))
    return bool(
        cache.model_obj_norms_offset[0, objective_index]
        <= cache.obj_radii[objective_index]
    )


def _tally_movement(
    census: AdvanceCensus, actions: np.ndarray, env: object, handler: object
) -> tuple[list[tuple[int, int]], set[int]]:
    """Tally one movement phase, and return the advances started from inside.

    Each entry is `(model index, objective index)` for a model that advanced
    while standing inside that objective. The caller re-checks them after the
    move, because whether such an advance was waste or a reallocation is only
    decided by where it ends up.
    """
    advance_slice = handler.advance_slice  # type: ignore[attr-defined]
    models = env.player_models  # type: ignore[attr-defined]
    alive = np.asarray(alive_mask_for(models))
    move_speeds = handler.move_speeds  # type: ignore[attr-defined]

    started_inside: list[tuple[int, int]] = []
    advanced_indices: set[int] = set()
    by_group: dict[int, list[int]] = {}
    for index, model in enumerate(models):
        if not alive[index]:
            continue
        by_group.setdefault(int(model.group_id), []).append(index)

    for group, indices in by_group.items():
        census.unit_turns += 1
        census.unit_size[len(indices)] += 1
        members = [models[i] for i in indices]
        nearest, inside = _nearest_objective(members, env.objectives)  # type: ignore[attr-defined]

        triggers = 0
        distances: list[float] = []
        for slot, index in enumerate(indices):
            action = int(actions[index])
            census.move_steps += 1
            if action == STAY_ACTION:
                census.stays += 1
                distances.append(0.0)
                continue
            displacement = handler.decode_action(  # type: ignore[attr-defined]
                action, model_idx=index, advance_roll=models[index].advance_roll
            )
            distance = float(np.linalg.norm(displacement))
            distances.append(distance)
            if advance_slice is None:
                continue
            if not advance_slice.start <= action < advance_slice.end:
                continue
            triggers += 1
            advanced_indices.add(index)
            census.advance_actions += 1
            census.advance_distance.append(distance)
            if distance <= float(move_speeds[index]) + DOMINATED_TOLERANCE:
                census.dominated_advances += 1
            if bool(inside[slot]):
                census.advance_from_inside += 1
                started_inside.append((index, int(nearest[slot])))

        spread = max(distances) - min(distances)
        if triggers:
            census.advancing_unit_turns += 1
            census.trigger_counts[triggers] += 1
            census.advancing_unit_spread.append(spread)
        else:
            # The control the statistic is worthless without. A unit that
            # splits its distances just as far on an ordinary move is telling
            # you about the policy, not about the advance.
            census.walking_unit_spread.append(spread)

    return started_inside, advanced_indices


def _tally_exposure(
    census: AdvanceCensus, env: object, advanced_indices: set[int]
) -> None:
    """After the move, who ended inside an enemy's weapon reach.

    Split by move type so the number means something: the share of an army
    under threat is a property of the board, and only the difference between
    the models that ran and the models that walked speaks to the move type.
    """
    enemies = [m for m in env.opponent_models if m.is_alive]  # type: ignore[attr-defined]
    if not enemies:
        return
    enemy_positions = np.array([m.location for m in enemies], dtype=float)
    enemy_reach = np.asarray(env.opponent_max_ranges, dtype=np.float64)  # type: ignore[attr-defined]
    alive_mask = np.asarray(
        alive_mask_for(env.opponent_models),  # type: ignore[attr-defined]
        dtype=np.bool_,
    )
    alive_reach = enemy_reach[alive_mask]
    for index, model in enumerate(env.player_models):  # type: ignore[attr-defined]
        if not model.is_alive:
            continue
        distances = np.linalg.norm(
            enemy_positions - np.asarray(model.location, dtype=float), axis=1
        )
        exposed = bool(np.any(distances <= alive_reach))
        if index in advanced_indices:
            census.ended_moves_advancing += 1
            census.ended_exposed_advancing += int(exposed)
        else:
            census.ended_moves_walking += 1
            census.ended_exposed_walking += int(exposed)


def collect(
    policy: str, config: WargameEnvConfig, seeds: list[int], decode_topk: int
) -> AdvanceCensus:
    """Walk every held-out map and seed, censusing the advance slice."""
    census = AdvanceCensus()
    for terrain_map in load_maps(DEFAULT_MAPS_DIR):
        per_map = config_for_map(config, terrain_map)
        env = create_environment(env_config=per_map)
        handler = env.player_action_handler
        select = build_action_selector(policy, env, decode_topk).select
        for seed in seeds:
            observation, _info = env.reset(seed=seed)
            done = False
            while not done:
                chosen = select(observation, env)
                phase = env.game_clock_state.phase
                started_inside: list[tuple[int, int]] = []
                advanced_indices: set[int] = set()
                if phase is BattlePhase.movement:
                    actions = np.asarray(chosen.actions, dtype=int).reshape(-1)
                    started_inside, advanced_indices = _tally_movement(
                        census, actions, env, handler
                    )
                elif phase is BattlePhase.shooting:
                    # Counted at the shooting phase rather than at the advance
                    # itself: a model killed in between never had the shot to
                    # forfeit, so charging it one would overstate the cost.
                    alive = np.asarray(alive_mask_for(env.player_models))
                    spent = np.array(
                        [m.advanced_this_turn for m in env.player_models], dtype=bool
                    )
                    census.shooting_slots += int(np.sum(alive))
                    census.shooting_forfeited += int(np.sum(alive & spent))
                observation, _r, done, _t, _i = env.step(chosen)
                if phase is BattlePhase.movement:
                    _tally_exposure(census, env, advanced_indices)
                for model_index, objective_index in started_inside:
                    if _still_inside(
                        env.player_models[model_index],
                        env.objectives,
                        objective_index,
                    ):
                        census.advance_stayed_inside += 1
        env.close()
    return census


def report(census: AdvanceCensus, has_advance: bool) -> None:
    """Print the census, defect by defect."""
    print(f"  movement model-steps      {census.move_steps}")
    print(
        f"  stay share                {census.share(census.stays, census.move_steps):.1f}%"
    )
    if not has_advance:
        print("  advance slice             NOT REGISTERED (this config has 0 bins)")
        return

    advances = census.advance_actions
    print(
        f"  advance actions chosen    {advances} "
        f"({census.share(advances, census.move_steps):.1f}% of model-steps)"
    )
    print(
        f"    ...DOMINATED (<= Move)  {census.dominated_advances} "
        f"({census.share(census.dominated_advances, advances):.1f}% of advances) "
        "-- paid the unit's shooting for a distance a normal move reaches"
    )
    print(
        f"    ...started INSIDE an obj {census.advance_from_inside} "
        f"({census.share(census.advance_from_inside, advances):.1f}% of advances)"
    )
    print(
        f"      ...and ENDED inside it {census.advance_stayed_inside} "
        f"({census.share(census.advance_stayed_inside, advances):.1f}% of advances) "
        "-- WASTE: the unit's shooting spent, no ground gained"
    )
    print(
        f"      ...and LEFT it         "
        f"{census.advance_from_inside - census.advance_stayed_inside} "
        f"({census.share(census.advance_from_inside - census.advance_stayed_inside, advances):.1f}% of advances) "
        "-- reallocation, which is the wanted behaviour"
    )
    if census.advance_distance:
        distances = np.asarray(census.advance_distance)
        print(
            f'    distance                mean {distances.mean():.2f}"  '
            f'median {np.median(distances):.2f}"  max {distances.max():.2f}"'
        )

    print(
        f"  unit-turns                {census.unit_turns}, of which "
        f"{census.advancing_unit_turns} advanced "
        f"({census.share(census.advancing_unit_turns, census.unit_turns):.1f}%)"
    )
    if census.trigger_counts:
        total = sum(census.trigger_counts.values())
        histogram = "  ".join(
            f"{k}:{100.0 * census.trigger_counts[k] / total:.0f}%"
            for k in sorted(census.trigger_counts)
        )
        print(f"    models that triggered   {histogram}")
        one_of_many = census.trigger_counts[1]
        print(
            f"    ONE model dragged the unit on {census.share(one_of_many, total):.1f}% "
            "of advancing unit-turns"
        )
    for label, values in (
        ("advancing", census.advancing_unit_spread),
        ("walking  ", census.walking_unit_spread),
    ):
        if not values:
            continue
        spread = np.asarray(values)
        print(
            f"    within-unit distance spread, {label}  "
            f'mean {spread.mean():.2f}"  '
            f'p90 {np.percentile(spread, 90):.2f}"  '
            f"over {len(values)} unit-turns"
        )
    print(
        '      (the 2" chain is what a split breaks; read the two lines'
        " against each other)"
    )
    print(
        "  ended inside an enemy's reach: "
        f"advancing {census.share(census.ended_exposed_advancing, census.ended_moves_advancing):.1f}%"
        f" ({census.ended_moves_advancing} model-moves)  vs  "
        f"walking {census.share(census.ended_exposed_walking, census.ended_moves_walking):.1f}%"
        f" ({census.ended_moves_walking})"
    )
    print(
        f"  shooting slots forfeited  {census.shooting_forfeited} of "
        f"{census.shooting_slots} "
        f"({census.share(census.shooting_forfeited, census.shooting_slots):.1f}%)"
    )


def main() -> None:
    """Census one policy's use of the advance move on the held-out tables."""
    argv, overrides = parse_overrides(sys.argv)
    if len(argv) < 3:
        print(__doc__)
        raise SystemExit(1)

    policy = argv[1]
    config_path = argv[2]
    n_episodes = int(argv[3]) if len(argv) > 3 and argv[3] else 10
    decode_topk = int(argv[4]) if len(argv) > 4 and argv[4] else 1

    config = load_env_config(config_path, **overrides)
    seeds = [HELDOUT_SEED_BASE + i for i in range(n_episodes)]

    print(f"\n{policy} on {config_path}{describe(overrides)}")
    print(
        f"  ({n_episodes} episodes per held-out map, seeds {seeds[0]}-{seeds[-1]}, "
        f"decode_topk={decode_topk}, advance bins={config.n_advance_speed_bins})"
    )
    census = collect(policy, config, seeds, decode_topk)
    report(census, has_advance=config.n_advance_speed_bins > 0)
    print()


if __name__ == "__main__":
    main()
