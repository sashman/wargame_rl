"""A seat: one side as an actor over the per-model decision machinery.

The player and the opponent are the same kind of thing here -- a force, its
enemies, the action handler that decodes its actions and states its legality,
its weapons, and optionally an adapter that answers decisions for it from a
whole-phase script. Which of the two the clock calls `player_1` is decided per
episode by `turn_order`, so a seat knows whether it is *the player* and the env
maps clock sides onto seats.

`full_phase_mask` is the whole-phase legality the phase facade hands a policy
-- `build_observation`'s overlays for the player, `_opponent_action_mask`'s for
the opponent -- reproduced here because a script plans its phase against
exactly that mask, and the bridge holds only if it is given the same one.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from wargame_rl.wargame.envs.domain.entities import WargameModel, alive_mask_for
from wargame_rl.wargame.envs.domain.sight import CLEAR
from wargame_rl.wargame.envs.env_components.actions import ActionHandler
from wargame_rl.wargame.envs.env_components.shooting_masks import (
    compute_unit_shooting_masks,
)
from wargame_rl.wargame.envs.types.game_timing import BattlePhase

if TYPE_CHECKING:
    from wargame_rl.wargame.envs.per_model.env import PerModelEnv
    from wargame_rl.wargame.envs.per_model.scripted import ScriptedSeat


@dataclass
class Seat:
    """One side's static wiring for the episode."""

    is_player: bool
    models: list[WargameModel]
    enemies: list[WargameModel]
    handler: ActionHandler
    ranged_weapons: list[Any]
    melee_weapons: list[Any]
    max_ranges: np.ndarray
    enemy_max_ranges: np.ndarray
    adapter: ScriptedSeat | None = None
    # Per-window result carriers, cleared when a reward window opens.
    shooting_results: list[Any] = field(default_factory=list)
    fight_results: list[Any] = field(default_factory=list)

    @property
    def n_models(self) -> int:
        return len(self.models)

    @property
    def group_ids(self) -> np.ndarray:
        return np.array([int(m.group_id) for m in self.models], dtype=np.intp)

    def alive(self) -> np.ndarray:
        """`(n_models,)` bool, read live."""
        return alive_mask_for(self.models)

    def alive_enemies(self) -> list[WargameModel]:
        return [m for m in self.enemies if m.is_alive]

    def unit_members(self, group: int, *, alive_only: bool = True) -> list[int]:
        """Model indices of `group`, in index order."""
        return [
            index
            for index, model in enumerate(self.models)
            if int(model.group_id) == group and (model.is_alive or not alive_only)
        ]

    def living_units(self) -> list[int]:
        """Group ids with at least one living member, ascending."""
        return sorted({int(m.group_id) for m in self.models if m.is_alive})

    def units_first_appearance(self) -> list[int]:
        """Every group id, dead members included, in first-appearance order.

        The order the phase facade rolls a side's dice in.
        """
        seen: list[int] = []
        for model in self.models:
            group = int(model.group_id)
            if group not in seen:
                seen.append(group)
        return seen

    def n_enemy_units(self) -> int:
        """Width of the enemy-unit pointer: the shooting slice's, else the count."""
        shooting = self.handler.shooting_slice
        if shooting is not None:
            return shooting.size
        return len({int(m.group_id) for m in self.enemies})


def unit_shooting_masks(env: PerModelEnv, seat: Seat) -> np.ndarray:
    """`(n_models, n_enemy_units)` -- which enemy units each model may fire at.

    The same call the phase facade makes on both seats, so range, sight, the
    advance/fall-back forfeit and the engagement gates are one rule.
    """
    shooting = seat.handler.shooting_slice
    if shooting is None or not seat.enemies:
        return np.zeros((seat.n_models, 0), dtype=bool)
    masks: np.ndarray = compute_unit_shooting_masks(
        np.array([m.location for m in seat.models]),
        np.array([m.location for m in seat.enemies]),
        seat.alive(),
        alive_mask_for(seat.enemies),
        seat.max_ranges,
        env.line_of_sight_matrix,
        np.array([m.group_id for m in seat.enemies], dtype=int),
        shooting.size,
        player_advanced=np.array(
            [m.advanced_this_turn or m.fell_back_this_turn for m in seat.models]
        ),
        player_groups=np.array([m.group_id for m in seat.models], dtype=int),
        engagement_range=env.rules_quantities.engagement_range,
        base_diameter=2.0 * env.rules_quantities.base_radius,
        exclude_engaged_targets=env.config.melee.enabled
        and env.config.melee.shield_engaged_targets,
    )
    return masks


def full_phase_mask(
    env: PerModelEnv,
    seat: Seat,
    phase: BattlePhase,
    *,
    shooting_overlay: bool,
) -> np.ndarray:
    """The whole-phase `(n_models, n_actions)` legality a script plans against."""
    handler = seat.handler
    alive = seat.alive()
    mask: np.ndarray = handler.registry.get_model_action_masks(
        phase, seat.n_models, alive_mask=alive
    )
    advance = handler.advance_slice
    if advance is not None and phase is BattlePhase.movement:
        mask[:, advance.start : advance.end] &= handler.advance_legality(
            seat.models, seat.enemies
        )
    movement = handler.movement_slice
    if env.config.melee.enabled and phase is BattlePhase.charge:
        mask[:, movement.start : movement.end] &= handler.charge_legality(
            seat.models, seat.enemies
        )
    if phase in (BattlePhase.pile_in, BattlePhase.consolidate):
        mask[:, movement.start : movement.end] &= handler.short_move_legality(
            seat.models, seat.enemies, phase
        )
    move_type = handler.move_type_slice
    if move_type is not None and phase is BattlePhase.command:
        mask[:, move_type.start : move_type.end] &= handler.declaration_legality(
            seat.models, seat.enemies
        )
    shooting = handler.shooting_slice
    if (
        phase is BattlePhase.shooting
        and shooting is not None
        and seat.enemies
        and shooting_overlay
    ):
        mask[:, shooting.start : shooting.end] &= unit_shooting_masks(env, seat)
    return mask


def unit_cover_for_shot(
    env: PerModelEnv,
    attacker: WargameModel,
    target_members: Sequence[WargameModel],
) -> bool:
    """Does the target UNIT have cover against this one attacker.

    `13-terrain.md` § Cover, as `WargameEnv._cover_mask` reads it: every member
    must be not fully visible (any blockage, `!= CLEAR`), traced with both
    models' base radii. One pair traced alone samples the same segment the
    batched trace does, so the answer is the batch's cell.

    `target_members` is the unit's membership as the PHASE saw it -- the phase
    facade declares every shot before any resolves, so a member killed by an
    earlier shot in the same phase still counts toward "every model". The
    caller passes that list; this does not re-read `is_alive`.
    """
    members = list(target_members)
    if not members:
        return False
    visibility = env.visibility_between(
        np.array([attacker.location], dtype=float),
        np.array([m.location for m in members], dtype=float),
        np.ones((1, len(members)), dtype=bool),
        origin_models=[attacker],
        target_models=members,
    )
    return bool((visibility[0] != CLEAR).all())
