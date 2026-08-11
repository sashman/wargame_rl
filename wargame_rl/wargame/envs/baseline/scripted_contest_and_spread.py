"""Baseline: allocate squads against the opponent's actual deployment, spread fire.

PROTOTYPE (2026-08-06). Built to test whether `squad_march_shoot`, the bar every
result in `reports/` is quoted against, is a weak ceiling -- it allocates squads
by a fixed `k % n_objectives` and fires nearest-first, both of which ignore
information that is free to read.

**The hypothesis was refuted.** On seeds 700000-700029 this scores 0.60 win /
+18.8 vp_margin against the bar's 0.77 / +39.4. See
`reports/2026-08-06-beat-the-shooting-opponent.md` § Part 5. It is kept as an
extra reference point and as the record of a refuted hypothesis.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from wargame_rl.wargame.envs.baseline.policy import (
    BaselinePolicy,
    step_toward_objective,
)
from wargame_rl.wargame.envs.baseline.registry import register_baseline
from wargame_rl.wargame.envs.env_components.actions import STAY_ACTION
from wargame_rl.wargame.envs.types import WargameEnvAction

if TYPE_CHECKING:
    from wargame_rl.wargame.envs.domain.entities import WargameModel
    from wargame_rl.wargame.envs.wargame import WargameEnv


class ScriptedContestAndSpreadPolicy(BaselinePolicy):
    """Take what the opponent left empty, and never shoot a model twice over.

    Two changes from `squad_march_shoot`, each targeting a measured defect:

    **Allocation reads the opponent.** `scripted_advance_to_objective` returns
    `STAY` unconditionally once inside an objective radius, so its deployment is
    frozen from about round 9 -- measured total displacement is exactly 0.0 from
    then on -- and it abandons at least one objective in every episode. Control
    is a strict count comparison, so an abandoned objective costs one model to
    take and holds for the rest of the game. A fixed `k % n_objectives` split
    cannot exploit that; walking objectives cheapest-first can.

    **Fire is spread, not stacked.** Models have ``max_wounds: 1`` and
    ``_resolve_shooting_action`` silently drops a shot whose target died earlier
    in the same phase, so every shot after the first successful one on a target
    is wasted. Per shot ``p(kill) = (4/6)**3 = 0.2963``: five shots on one target
    expect 0.83 kills, five on five distinct targets expect 1.48. Nearest-first
    with no teammate awareness measures 32.9% of shots discarded; claiming
    targets brings that down.

    Deliberately simple -- centroid steering and greedy assignment, no search.

    **Measured outcome: it loses to `squad_march_shoot`** (0.60 / +18.8 against
    0.77 / +39.4 on seeds 700000-700029), with near-identical occupancy (0.977
    vs 1.000) and *better* survival (0.516 vs 0.409). Massing 10/10/5 and
    winning two objectives outright beats spreading to take the abandoned one
    and losing the contested ones. Concentration is the reason the bar wins, not
    a defect in it -- worth remembering before pricing stacking as a bad thing.
    """

    def _predicted_opponent_counts(self, env: WargameEnv) -> list[int]:
        """How many opponents will end up on each objective.

        Counts each opponent against its *nearest* objective rather than the one
        it currently stands on. `scripted_advance_to_objective` steers by
        `argmin` distance and then parks, so this is where they are going, and
        it is already accurate at deployment -- where current occupancy is all
        zeros and would send every squad at the same point.
        """
        objectives = env.objectives
        counts = [0] * len(objectives)
        centres = np.array([o.location for o in objectives], dtype=float)

        for model in env.opponent_models:
            if not model.is_alive:
                continue
            distances = np.linalg.norm(
                centres - np.asarray(model.location, dtype=float), axis=1
            )
            counts[int(np.argmin(distances))] += 1
        return counts

    def _objective_allocation(self, env: WargameEnv, n_squads: int) -> list[int]:
        """Assign each squad an objective index, cheapest winnable one first.

        Returns a list of length `n_squads`. An objective needs
        ``opponent_count + 1`` models to control, so squads are committed to the
        cheapest objective until it is won, then to the next.

        Objectives that cannot be won with the squads still uncommitted are
        **skipped rather than half-garrisoned**. Control is a strict count
        comparison, so ten models losing a point 10-to-15 score exactly as much
        as zero models do -- those squads are worth more reinforcing a point
        already held.
        """
        objectives = env.objectives
        opponent_counts = self._predicted_opponent_counts(env)

        n_models = env.config.number_of_wargame_models
        squad_size = max(1, n_models // max(1, n_squads))
        order = sorted(range(len(objectives)), key=lambda i: opponent_counts[i])

        allocation: list[int] = []
        for objective_index in order:
            needed = opponent_counts[objective_index] + 1
            squads_required = -(-needed // squad_size)  # ceil
            if squads_required > n_squads - len(allocation):
                continue
            allocation.extend([objective_index] * squads_required)

        # Leftovers reinforce the cheapest objective taken. Surplus on a held
        # point still counts toward control, and it cannot be lost to a
        # counter-attack that never comes -- the opponent parks permanently.
        fallback = allocation[0] if allocation else order[0]
        while len(allocation) < n_squads:
            allocation.append(fallback)
        return allocation[:n_squads]

    def select_movement(
        self, models: list[WargameModel], env: WargameEnv
    ) -> WargameEnvAction:
        """March each squad, as a body, to the objective it was allocated."""
        objectives = env.objectives
        actions = [STAY_ACTION] * len(models)
        if not objectives:
            return WargameEnvAction(actions=actions)

        max_step = float(env.config.max_move_speed)
        group_ids = sorted({model.group_id for model in models})
        allocation = self._objective_allocation(env, len(group_ids))

        for squad_index, group_id in enumerate(group_ids):
            member_indices = [
                i
                for i, model in enumerate(models)
                if model.group_id == group_id and model.is_alive
            ]
            if not member_indices:
                continue

            objective = objectives[allocation[squad_index] % len(objectives)]
            radius = float(objective.radius_size)
            centroid = np.mean(
                [models[i].location for i in member_indices], axis=0, dtype=float
            )
            lead = np.asarray(objective.location, dtype=float) - centroid
            lead_distance = float(np.linalg.norm(lead))

            for i in member_indices:
                if lead_distance <= radius:
                    actions[i] = step_toward_objective(
                        models[i], objective.location, radius, env
                    )
                else:
                    # One squad vector for every member keeps relative positions,
                    # and therefore coherency, intact.
                    actions[i] = env.player_action_handler.best_action_toward(
                        float(lead[0]),
                        float(lead[1]),
                        max_step_length=min(max_step, lead_distance),
                    )

        return WargameEnvAction(actions=actions)

    def select_shooting(
        self,
        models: list[WargameModel],
        env: WargameEnv,
        action_mask: np.ndarray | None = None,
    ) -> WargameEnvAction:
        """Fire at the nearest valid target no teammate has already claimed."""
        shooting_slice = env.player_action_handler.shooting_slice
        actions = [STAY_ACTION] * len(models)
        if shooting_slice is None or action_mask is None:
            return WargameEnvAction(actions=actions)

        opponents = env.opponent_models
        if not opponents:
            return WargameEnvAction(actions=actions)
        opponent_locations = np.array([m.location for m in opponents], dtype=float)

        # **The target-claiming half of this baseline is gone, deliberately.**
        # It ranked targets by fewest claims so two models would not both fire at
        # one enemy, because a second shot at an already-dead *model* was thrown
        # away -- 29.6% of the time, by its own note. Weapons now name a *unit*
        # and the defender allocates, so a second shot at the same unit is only
        # wasted once the whole unit is destroyed. Spreading fire across units
        # would now cost concentration and buy nothing.
        #
        # What made this policy distinct survives untouched: its objective
        # allocation, above. Only the shooting half was rules-dependent, and it
        # is now the same nearest-unit rule `squad_march_shoot` uses.
        opponent_groups = np.array([m.group_id for m in opponents], dtype=int)
        for index, model in enumerate(models):
            if not model.is_alive:
                continue
            # The env mask already encodes range, line of sight, target-alive and
            # engagement-range validity, so honouring it keeps this baseline
            # playing by exactly the rules the learned policy plays by.
            valid = np.flatnonzero(
                action_mask[index, shooting_slice.start : shooting_slice.end]
            )
            if valid.size == 0:
                continue
            here = np.asarray(model.location, dtype=float)
            model_distances = np.linalg.norm(opponent_locations - here, axis=1)
            unit_distances = [
                model_distances[opponent_groups == unit].min()
                if (opponent_groups == unit).any()
                else np.inf
                for unit in valid
            ]
            target = int(valid[int(np.argmin(unit_distances))])
            actions[index] = shooting_slice.start + target

        return WargameEnvAction(actions=actions)


register_baseline("contest_and_spread", ScriptedContestAndSpreadPolicy)
