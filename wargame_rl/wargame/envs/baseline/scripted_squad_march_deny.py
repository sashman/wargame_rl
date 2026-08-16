"""Baseline: hold enough objectives to cap, then spend the surplus denying."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from wargame_rl.wargame.envs.baseline.registry import register_baseline
from wargame_rl.wargame.envs.baseline.scripted_squad_march_shoot import (
    ScriptedSquadMarchShootPolicy,
)

if TYPE_CHECKING:
    from wargame_rl.wargame.envs.domain.entities import WargameModel, WargameObjective
    from wargame_rl.wargame.envs.wargame import WargameEnv


def occupants(locations: np.ndarray, objective: WargameObjective) -> int:
    """Models of one side controlling `objective`, under VP's own membership rule.

    An area objective has `radius_size` 0.0 by design, so a distance-to-centre
    test counts only models standing exactly on the centroid. Membership has to
    follow the objective's own shape, which is what `contains_points` does.
    """
    if locations.size == 0:
        return 0
    if objective.area is not None:
        return int(objective.area.contains_points(locations).sum())
    centre = np.asarray(objective.location, dtype=float)
    return int(
        (
            np.linalg.norm(locations - centre, axis=1) <= float(objective.radius_size)
        ).sum()
    )


class ScriptedSquadMarchDenyPolicy(ScriptedSquadMarchShootPolicy):
    """`squad_march_shoot`, but surplus squads contest what the opponent holds.

    **Why this baseline exists.** VP is ``min(cap_per_turn, controlled *
    vp_per_objective)``, which on the shipped mission is ``min(15, held * 5)``.
    The cap therefore binds at **three** objectives while the real tables carry
    five or six, so a fourth objective you control is worth *zero* additional
    VP. Above the cap the only remaining gradient in ``vp_margin`` is denial:
    an objective taken off the opponent removes 5 from *their* score every
    round.

    `squad_march_shoot` reaches 4.00 objectives held by marching to a fixed
    ``k mod n`` assignment it never revises -- so whatever denial it performs is
    incidental. This policy spends the surplus deliberately: it commits the
    squads nearest the cheapest objectives to banking the cap, then sends every
    remaining squad at whatever the opponent currently controls.

    It exists to price the denial hypothesis **before** a reward that teaches it
    is trained, which is this project's standing discipline for a cheap proxy.
    If deliberate denial does not beat `squad_march_shoot` here, no reward that
    teaches denial will either, and that is worth knowing in minutes rather than
    after a training batch.

    Assignment is recomputed every movement phase, because control changes: an
    objective flips the moment counts cross, and a squad already en route to a
    point the opponent has abandoned should be re-tasked. Movement and shooting
    are inherited unchanged, so squads still steer on their centroid and stay in
    coherency, and the only difference from the bar is *where they are sent*.
    """

    def squad_objectives(
        self, models: list[WargameModel], env: WargameEnv, group_ids: list[int]
    ) -> list[WargameObjective]:
        """Assign the cheapest objectives to holding, and the rest to denial."""
        objectives = env.objectives
        player_locations = np.array(
            [m.location for m in models if m.is_alive], dtype=float
        )
        opponent_locations = np.array(
            [m.location for m in env.opponent_models if m.is_alive], dtype=float
        )
        if opponent_locations.size == 0:
            opponent_locations = np.empty((0, 2), dtype=float)

        opponent_counts = [occupants(opponent_locations, o) for o in objectives]
        player_counts = [occupants(player_locations, o) for o in objectives]

        centroids = []
        for group_id in group_ids:
            members = [
                m.location for m in models if m.group_id == group_id and m.is_alive
            ]
            centroids.append(
                np.mean(members, axis=0, dtype=float)
                if members
                # A wiped squad still needs a slot so the returned list lines up
                # with `group_ids`; its target is never used.
                else np.zeros(2, dtype=float)
            )

        # How many objectives it takes to saturate our own VP. Read off the
        # mission config rather than hardcoded, so a config that changes either
        # number changes this policy with it. The defaults mirror
        # `DefaultVPCalculator`, which is what an omitted `mission:` block gets.
        params = env.config.mission.params
        cap = int(params.get("cap_per_turn", 15))
        per_objective = int(params.get("vp_per_objective", 5)) or 1
        needed = max(1, min(len(objectives), cap // per_objective))

        # Hold the objectives the opponent contests least: cheapest to take and
        # cheapest to keep. Ties break on index so the choice is deterministic.
        order = sorted(range(len(objectives)), key=lambda i: (opponent_counts[i], i))
        hold_targets = order[:needed]
        # Everything else, denial first -- an objective the opponent controls is
        # worth +5 margin a round, one nobody holds is worth 0 above the cap.
        rest = order[needed:]
        deny_targets = sorted(
            rest,
            key=lambda i: (
                opponent_counts[i] <= player_counts[i],  # contested ones first
                opponent_counts[i],  # then the cheapest to flip
                i,
            ),
        )

        # Nearest squad to each target in priority order: holding the cap comes
        # first, because a denial that costs us a held objective is a net loss.
        priority = hold_targets + deny_targets
        unassigned = list(range(len(group_ids)))
        targets: list[int | None] = [None] * len(group_ids)
        for objective_index in priority:
            if not unassigned:
                break
            location = np.asarray(objectives[objective_index].location, dtype=float)
            nearest = min(
                unassigned,
                key=lambda s: float(np.linalg.norm(centroids[s] - location)),
            )
            targets[nearest] = objective_index
            unassigned.remove(nearest)
        # More squads than objectives: the leftovers reinforce the cap.
        for squad in unassigned:
            targets[squad] = hold_targets[squad % len(hold_targets)]

        return [objectives[index] for index in targets if index is not None]


register_baseline("squad_march_deny", ScriptedSquadMarchDenyPolicy)
