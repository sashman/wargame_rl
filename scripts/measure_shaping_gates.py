"""Why the travel reward never points at the objectives nobody holds.

`closest_objective_v2` is the **only** calculator that pays a model to move
*between* objectives — `objective_hold` pays it for standing still on one, and
`objective_coverage` is global, so no individual model has a private reason to
be the one that walks. Two gates inside it decide where that pay points, and
neither has ever been measured:

**The candidate gate is single-arrival.** `_is_positive_transition` asks whether
*this one model* arriving flips the objective's control label. An objective the
opponent holds 2-0 goes `opponent -> opponent`, so it is not a candidate for
anybody at all.

The gate pays a **two-model window** and nothing else. With the opponent holding
`o`, the transitions it recognises are `opponent -> contested` at `p = o - 1` and
`contested -> player` at `p = o`, so arrivals 1..o-1 earn **nothing**, arrivals
`o` and `o+1` are paid, and later arrivals are refused as already-ours. The
deeper the enemy is dug in, the longer the unpaid opening: against four models
the first three to walk over earn zero for doing it. That is not a reward for a
coordinated assault; it is a reward for arriving last.

**The assignment can starve a unit.** `_compute_group_assignment` iterates over
*objectives* and gives each to the group of the nearest eligible model, so one
group may own several while others own none. A model in a group that owns none
falls through to `fallback_to_nearest`, and with `progress_scale: 6.0` that
fallback is *paid* — it earns the model for closing on whatever is nearest,
which is usually a point its own side already holds.

If that is what happens, the documented result that **abandonment is invariant
to reward weight across five weightings** needs no further explanation:
reweighting a term that returns zero for the far objective changes nothing.

This measures it rather than arguing it. Nothing in the shipped calculator is
modified — the two methods are wrapped in place for the run, so what is counted
is the code that actually executes, not a second implementation of it that could
disagree.

Usage: just measure-shaping-gates <policy|ckpt> <env_config> [n_episodes] [maps_dir] [decode_topk]
"""

from __future__ import annotations

import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from scripts.measure_maps import build_action_selector, config_for_map, load_maps
from scripts.scenario_overrides import describe, load_env_config, parse_overrides
from wargame_rl.wargame.envs.domain.battle_view import BattleView
from wargame_rl.wargame.envs.env_components.distance_cache import DistanceCache
from wargame_rl.wargame.envs.reward.calculators.closest_objective_v2 import (
    ClosestObjectiveV2Calculator,
)
from wargame_rl.wargame.envs.types import WargameEnvConfig
from wargame_rl.wargame.model.common.factory import create_environment

HELDOUT_SEED_BASE = 700_000
DEFAULT_MAPS_DIR = Path("configs/evaluation/maps_heldout")


@dataclass
class GateCounts:
    """What the two gates did, accumulated over model-steps and steps."""

    steps: int = 0
    objectives_total: int = 0
    objectives_candidate: int = 0
    units_total: int = 0
    units_assigned: int = 0
    units_owning_several: int = 0
    model_steps: int = 0
    model_steps_assigned: int = 0
    model_steps_fallback: int = 0
    model_steps_no_target: int = 0
    # Why an objective was not a candidate, by control state. The gate reads
    # only the two counts, so these are exhaustive.
    blocked_reason: Counter[str] = field(default_factory=Counter)

    def as_rows(self, n_units_hint: int) -> list[tuple[str, str]]:
        """Human-readable summary lines. Percentages, because counts do not compare."""

        def pct(a: int, b: int) -> str:
            return f"{100.0 * a / b:.1f}%" if b else "-"

        return [
            (
                "objectives the travel reward can point at",
                f"{pct(self.objectives_candidate, self.objectives_total)}"
                f"  ({self.objectives_candidate / max(self.steps, 1):.2f} of "
                f"{self.objectives_total / max(self.steps, 1):.2f} per step)",
            ),
            (
                "units given an objective of their own",
                f"{pct(self.units_assigned, self.units_total)}"
                f"  ({self.units_assigned / max(self.steps, 1):.2f} of {n_units_hint})",
            ),
            (
                "steps where one unit owns 2+ objectives",
                pct(self.units_owning_several, self.steps),
            ),
            (
                "model-steps paid toward an assigned target",
                pct(self.model_steps_assigned, self.model_steps),
            ),
            (
                "model-steps paid toward NEAREST (fallback)",
                pct(self.model_steps_fallback, self.model_steps),
            ),
            (
                "model-steps paid nothing to travel",
                pct(self.model_steps_no_target, self.model_steps),
            ),
        ]


def instrument(calculator: ClosestObjectiveV2Calculator, counts: GateCounts) -> None:
    """Wrap the two gates in place, counting what they decide.

    Wrapped rather than reimplemented on purpose. Seven tests once covered the
    joint decoder and none of them called `env.step`, so every one asserted the
    decoder against its own relaxation; a diagnostic that recomputes the gate it
    is measuring has exactly that failure mode available to it.
    """
    real_mask = calculator._candidate_mask
    real_choose = calculator._choose_target_objective

    state: dict[str, object] = {}

    def counted_mask(
        player_in_range: np.ndarray,
        player_counts: np.ndarray,
        opponent_counts: np.ndarray,
        step_key: tuple[int, int],
    ) -> np.ndarray:
        mask = real_mask(player_in_range, player_counts, opponent_counts, step_key)
        if state.get("mask_step") != step_key:
            state["mask_step"] = step_key
            # An objective is a candidate for *somebody* when any row is set.
            per_objective = np.asarray(mask.any(axis=0)).reshape(-1)
            counts.steps += 1
            counts.objectives_total += int(per_objective.size)
            counts.objectives_candidate += int(per_objective.sum())
            for index, is_candidate in enumerate(per_objective):
                if is_candidate:
                    continue
                player, opponent = (
                    int(player_counts[index]),
                    int(opponent_counts[index]),
                )
                if player > opponent:
                    reason = "we already hold it"
                elif opponent >= player + 2:
                    reason = "they hold it by 2+ (one arrival cannot flip it)"
                elif opponent == player and player > 0:
                    reason = "contested, but everyone is already inside"
                else:
                    reason = "other"
                counts.blocked_reason[reason] += 1
        return mask

    def counted_choose(
        model_idx: int,
        view: BattleView,
        step_key: tuple[int, int],
        cache: DistanceCache,
        player_in_range: np.ndarray,
        player_counts: np.ndarray,
        opponent_counts: np.ndarray,
    ) -> int | None:
        chosen = real_choose(
            model_idx=model_idx,
            view=view,
            step_key=step_key,
            cache=cache,
            player_in_range=player_in_range,
            player_counts=player_counts,
            opponent_counts=opponent_counts,
        )
        counts.model_steps += 1
        # The assignment is memoised per step, so its identity changing is the
        # cheapest signal that a new step's assignment has been built.
        assignment = calculator._cached_group_assignment or {}
        if state.get("assign_step") is not calculator._cached_group_assignment:
            state["assign_step"] = calculator._cached_group_assignment
            owners = Counter(assignment.values())
            counts.units_total += len({m.group_id for m in view.player_models})
            counts.units_assigned += len(owners)
            counts.units_owning_several += (
                1 if any(count >= 2 for count in owners.values()) else 0
            )
        if chosen is None:
            counts.model_steps_no_target += 1
            return chosen
        group = int(view.player_models[model_idx].group_id)
        if assignment.get(chosen) == group:
            counts.model_steps_assigned += 1
        else:
            counts.model_steps_fallback += 1
        return chosen

    calculator._candidate_mask = counted_mask  # type: ignore[method-assign]
    calculator._choose_target_objective = counted_choose  # type: ignore[method-assign]


def find_calculator(env: object) -> ClosestObjectiveV2Calculator | None:
    """The live `closest_objective_v2` in the env's current reward phase."""
    manager = getattr(env, "phase_manager", None)
    if manager is None:
        return None
    for phase in manager.phases:
        for _name, calc in phase.per_model_calculators:
            if isinstance(calc, ClosestObjectiveV2Calculator):
                return calc
    return None


def main() -> None:
    """Print what the two shaping gates decided, per map and overall."""
    argv, overrides = parse_overrides(sys.argv)
    if len(argv) < 3:
        print(__doc__)
        raise SystemExit(1)

    policy_or_checkpoint = argv[1]
    config_path = argv[2]
    n_episodes = int(argv[3]) if len(argv) > 3 and argv[3] else 30
    maps_dir = Path(argv[4]) if len(argv) > 4 and argv[4] else DEFAULT_MAPS_DIR
    decode_topk = int(argv[5]) if len(argv) > 5 and argv[5] else 1

    base_config: WargameEnvConfig = load_env_config(config_path, **overrides)
    maps = load_maps(maps_dir)
    seeds = [HELDOUT_SEED_BASE + i for i in range(n_episodes)]

    print(f"\n{policy_or_checkpoint}")
    print(
        f"{config_path}{describe(overrides)}  ({len(maps)} maps x {n_episodes} "
        f"episodes, seeds {seeds[0]}-{seeds[-1]}, decode_topk={decode_topk})\n"
    )

    counts = GateCounts()
    n_units_hint = 0
    for terrain_map in maps:
        config = config_for_map(base_config, terrain_map)
        select, _label = build_action_selector(
            policy_or_checkpoint, config, decode_topk, False
        )
        env = create_environment(env_config=config)
        calculator = find_calculator(env)
        if calculator is None:
            raise SystemExit(
                "this config has no closest_objective_v2 calculator, so there are "
                "no travel-shaping gates to measure"
            )
        instrument(calculator, counts)
        for seed in seeds:
            observation, _ = env.reset(seed=seed)
            n_units_hint = max(
                n_units_hint, len({m.group_id for m in env.player_models})
            )
            done = False
            while not done:
                action = select(observation, env)
                observation, _reward, done, _trunc, _info = env.step(action)
        env.close()

    width = 46
    for label, value in counts.as_rows(n_units_hint):
        print(f"  {label:<{width}}{value}")

    if counts.blocked_reason:
        print("\n  why an objective was not a candidate:")
        total = sum(counts.blocked_reason.values())
        for reason, n in counts.blocked_reason.most_common():
            print(f"    {reason:<{width - 2}}{100.0 * n / total:>6.1f}%")
    print()


if __name__ == "__main__":
    main()
