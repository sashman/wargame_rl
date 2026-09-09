"""The token and relation observation: every entity a token, every relation a pair.

The second presentation adapter over the per-model facade (the first,
`observation.py`, carries the decision and what the turn revealed). This one
builds what a size-independent network reads: a set of feature rows per
entity whose widths depend on nothing but the entity's own kind, and a
pairwise relation vector between every model of the acting seat and every
other token, so that "distance to objective 3" is never a column but a
relation read while attending to objective 3.

Principle 2 (issue #283): no row's width depends on how many other entities
exist. The count-sized columns of the phase facade's observation -- distances
to each objective, the unit one-hot, expected damage per opponent, the
declared-target one-hots -- become relations here. The coherency columns
(nearest-squadmate distance, spread, component, unit offset) are dropped as
well, a deliberate departure from the #283 audit table: the same-unit
relation and the coordinate offsets carry what they carried, and the standing
finding is that the coherency gap is not perceptual.

Two rules of its own, both presentation: the *reveal* (a unit's dice are shown
only once it has declared, on either side -- gated on the declaration flags,
never on the roll being non-zero, because a scripted seat is pre-rolled at
turn start), and the *reach gate* (sight and cover are traced only within the
longer of the pair's weapon reaches; beyond it the answer cannot change any
legal choice, and the trace is the facade's most expensive query).

Numpy only. `envs/` may not import torch; the tensor side is
`model/per_model/`.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from wargame_rl.wargame.envs.domain.battlefield.sight import COVER
from wargame_rl.wargame.envs.domain.movement.engagement import engaged_with_any
from wargame_rl.wargame.envs.domain.sequencing.activation import CHARGE_TARGET_DECLINE
from wargame_rl.wargame.envs.domain.shooting.expectation import expected_damage_matrix
from wargame_rl.wargame.envs.env_components.actions import STAY_ACTION
from wargame_rl.wargame.envs.env_components.distance_cache import (
    compute_distances,
    objective_counts_from_norms_offset,
)
from wargame_rl.wargame.envs.env_components.observation_builder import _terrain_to_obs
from wargame_rl.wargame.envs.per_model.types import (
    N_DECLARATIONS,
    DecisionPoint,
    PerModelObservation,
    StepKind,
)
from wargame_rl.wargame.envs.types.game_timing import BATTLE_PHASE_ORDER, BattlePhase
from wargame_rl.wargame.envs.types.geometry import polygons_contain_points
from wargame_rl.wargame.envs.types.terrain_observation import TERRAIN_VERTEX_BUDGET

if TYPE_CHECKING:
    from wargame_rl.wargame.envs.domain.kernel.entities import WargameModel
    from wargame_rl.wargame.envs.domain.kernel.rules_quantities import RulesQuantities
    from wargame_rl.wargame.envs.per_model.env import PerModelEnv
    from wargame_rl.wargame.envs.per_model.seat import Seat

# --------------------------------------------------------------------- widths

# Context kinds: the game token is row 0 of the context by construction.
CONTEXT_GAME = 0
CONTEXT_OWN_UNIT = 1
CONTEXT_ENEMY_MODEL = 2
CONTEXT_ENEMY_UNIT = 3
CONTEXT_OBJECTIVE = 4
CONTEXT_TERRAIN = 5
N_CONTEXT_KINDS = 6

N_PHASES = len(BATTLE_PHASE_ORDER)
N_STEP_KINDS = len(StepKind)
N_CONSOLIDATION_MODES = 4

# A model token: the same block for both sides. Named offsets for the columns
# tests read back; the rest are documented by `_model_row`.
MODEL_X = 0
MODEL_Y = 1
MODEL_ALIVE = 2
MODEL_ACTED = 20
MODEL_ADVANCE_ROLL = 22
MODEL_CHARGE_ROLL = 24
MODEL_ENGAGED = 28
MODEL_DIM = 35

UNIT_DIM = 11
OBJECTIVE_DIM = 6
TERRAIN_DIM = 2 * TERRAIN_VERTEX_BUDGET + 1
GAME_DIM = 8 + N_PHASES + N_STEP_KINDS
CONTEXT_DIM = max(GAME_DIM, UNIT_DIM, MODEL_DIM, OBJECTIVE_DIM, TERRAIN_DIM)

# The relation vector: one layout for the (model, model) and (model, context)
# tensors, zero-filled with a presence flag per group of slots.
REL_DX = 0
REL_DY = 1
REL_OFFSET_PRESENT = 2
REL_SAME_UNIT = 3
REL_SAME_UNIT_PRESENT = 4
REL_VISIBLE = 5
REL_IN_COVER = 6
REL_SIGHT_PRESENT = 7
REL_RANGED_OUT = 8
REL_RANGED_IN = 9
REL_MELEE_OUT = 10
REL_MELEE_IN = 11
REL_RANGED_PRESENT = 12
REL_MELEE_PRESENT = 13
REL_INSIDE = 14
REL_TERRAIN_PRESENT = 15
RELATION_DIM = 16

_STAT_NORMALISERS = np.array([10.0, 6.0, 10.0, 6.0, 10.0])


class Head(IntEnum):
    """Which network scores a step's value factor; a function of (kind, phase)."""

    none = 0
    declaration = 1
    displacement = 2
    unit_pointer = 3


def head_for(kind: StepKind, phase: BattlePhase | None) -> Head:
    """The head a decision of `kind` in `phase` is scored by."""
    if kind is StepKind.close_turn:
        return Head.none
    if kind is StepKind.open:
        return Head.declaration
    if kind is StepKind.target:
        return Head.unit_pointer
    if phase in (BattlePhase.shooting, BattlePhase.fight):
        return Head.unit_pointer
    return Head.displacement


# ------------------------------------------------------------------ scenario


@dataclass(frozen=True)
class TokenScenario:
    """What does not change within an episode, built once per `env.episode_id`.

    Stat rows are read off the models' resolved stats (toughness, save) and
    the seat's weapon lists (the first ranged weapon, the first melee weapon),
    never off the config: an auto-built army has no config rows, and a zero
    toughness would tell the network everyone is unshootable while the dice
    use the real value.
    """

    episode_id: int
    half_width: float
    half_height: float
    diagonal: float
    terrain_rows: np.ndarray  # (T, TERRAIN_DIM)
    terrain_outlines: np.ndarray  # (T, V_max, 2)
    terrain_vertex_counts: np.ndarray  # (T,)
    own_groups: np.ndarray  # (U_own,) sorted distinct group ids, dead included
    enemy_groups: np.ndarray  # (U_enemy,)
    own_initial: np.ndarray  # (U_own,) members per unit
    enemy_initial: np.ndarray  # (U_enemy,)
    own_ranged: np.ndarray  # (P, 5) int
    own_melee: np.ndarray  # (P, 5) int
    enemy_ranged: np.ndarray  # (O, 5) int
    enemy_melee: np.ndarray  # (O, 5) int
    own_defence: np.ndarray  # (P, 2) int
    enemy_defence: np.ndarray  # (O, 2) int
    ranged_out: np.ndarray  # (P, O) expected damage, open ground
    ranged_in: np.ndarray  # (P, O): what enemy o does to model p
    melee_out: np.ndarray  # (P, O)
    melee_in: np.ndarray  # (P, O)
    own_moves: np.ndarray  # (P,)
    enemy_moves: np.ndarray  # (O,)
    own_reach: np.ndarray  # (P,)
    enemy_reach: np.ndarray  # (O,)
    objective_extent: np.ndarray  # (K,) radius, or the area's reach from its centroid
    objective_is_area: np.ndarray  # (K,) bool

    @classmethod
    def for_episode(cls, env: PerModelEnv, seat: Seat) -> TokenScenario:
        """Build the static part for `seat`'s point of view of this episode."""
        terrain = env.terrain
        n_pieces = len(terrain.footprints)
        rows = np.array(
            [o.outline for o in _terrain_to_obs(cast(Any, env))[:n_pieces]],
            dtype=np.float32,
        ).reshape(n_pieces, TERRAIN_DIM)
        own_ranged = _attack_rows(seat.ranged_weapons, seat.n_models, "ballistic_skill")
        own_melee = _attack_rows(seat.melee_weapons, seat.n_models, "melee_skill")
        enemy_seat = env.opponent_seat if seat.is_player else env.player_seat
        enemy_ranged = _attack_rows(
            enemy_seat.ranged_weapons, enemy_seat.n_models, "ballistic_skill"
        )
        enemy_melee = _attack_rows(
            enemy_seat.melee_weapons, enemy_seat.n_models, "melee_skill"
        )
        own_defence = _defence_rows(seat.models)
        enemy_defence = _defence_rows(seat.enemies)
        own_groups = _sorted_groups(seat.models)
        enemy_groups = _sorted_groups(seat.enemies)
        objectives = env.objectives
        extent = np.zeros(len(objectives), dtype=float)
        is_area = np.zeros(len(objectives), dtype=bool)
        for k, objective in enumerate(objectives):
            if objective.area is not None:
                is_area[k] = True
                vertices = objective.area.vertices
                extent[k] = float(
                    np.max(np.linalg.norm(vertices - objective.area.centroid, axis=1))
                )
            else:
                extent[k] = float(objective.radius_size)
        return cls(
            episode_id=env.episode_id,
            half_width=env.board_width / 2.0,
            half_height=env.board_height / 2.0,
            diagonal=float(np.hypot(env.board_width, env.board_height)),
            terrain_rows=rows,
            terrain_outlines=terrain.outlines,
            terrain_vertex_counts=terrain.vertex_counts,
            own_groups=own_groups,
            enemy_groups=enemy_groups,
            own_initial=_members_per_unit(seat.models, own_groups),
            enemy_initial=_members_per_unit(seat.enemies, enemy_groups),
            own_ranged=own_ranged,
            own_melee=own_melee,
            enemy_ranged=enemy_ranged,
            enemy_melee=enemy_melee,
            own_defence=own_defence,
            enemy_defence=enemy_defence,
            ranged_out=expected_damage_matrix(own_ranged, enemy_defence),
            ranged_in=expected_damage_matrix(enemy_ranged, own_defence).T,
            melee_out=expected_damage_matrix(own_melee, enemy_defence),
            melee_in=expected_damage_matrix(enemy_melee, own_defence).T,
            own_moves=np.asarray(seat.handler.move_speeds, dtype=float),
            enemy_moves=np.asarray(enemy_seat.handler.move_speeds, dtype=float),
            own_reach=np.asarray(seat.max_ranges, dtype=float),
            enemy_reach=np.asarray(seat.enemy_max_ranges, dtype=float),
            objective_extent=extent,
            objective_is_area=is_area,
        )


def _attack_rows(weapon_lists: list[Any], n_models: int, skill: str) -> np.ndarray:
    """`(n_models, 5)` of (attacks, skill, strength, ap, damage); zero rows unarmed."""
    rows = np.zeros((n_models, 5), dtype=np.int64)
    for index in range(min(n_models, len(weapon_lists))):
        weapons = weapon_lists[index]
        if not weapons:
            continue
        weapon = weapons[0]
        rows[index] = (
            int(weapon.attacks),
            int(getattr(weapon, skill)),
            int(weapon.strength),
            int(weapon.ap),
            int(weapon.damage),
        )
    return rows


def _defence_rows(models: list[WargameModel]) -> np.ndarray:
    return np.array(
        [(int(m.stats["toughness"]), int(m.stats["save"])) for m in models],
        dtype=np.int64,
    ).reshape(len(models), 2)


def _sorted_groups(models: list[WargameModel]) -> np.ndarray:
    return np.array(sorted({int(m.group_id) for m in models}), dtype=np.intp)


def _members_per_unit(models: list[WargameModel], groups: np.ndarray) -> np.ndarray:
    ids = np.array([int(m.group_id) for m in models], dtype=np.intp)
    return np.array([int(np.sum(ids == g)) for g in groups], dtype=np.intp)


# --------------------------------------------------------------- observation


@dataclass(frozen=True)
class TokenObservation:
    """One decision point as tokens, relations and the masks each head needs.

    `players` are the acting seat's models, the encoder's queries; `context`
    is every other token in one right-padded array with a kind per row, the
    game token at row 0. `opponent_unit_rows` index the enemy-unit rows of
    the context in ascending group-id order, which is also the column order
    of `unit_mask` after its column 0 (the no-target option).
    """

    kind: StepKind
    phase: BattlePhase | None
    head: Head
    seat_is_player: bool
    players: np.ndarray  # (P, MODEL_DIM) float32
    player_alive: np.ndarray  # (P,) bool
    player_groups: np.ndarray  # (P,) intp
    context: np.ndarray  # (C, CONTEXT_DIM) float32
    context_kind: np.ndarray  # (C,) intp
    context_alive: np.ndarray  # (C,) bool
    self_relations: np.ndarray  # (P, P, RELATION_DIM) float32
    cross_relations: np.ndarray  # (P, C, RELATION_DIM) float32
    opponent_unit_rows: np.ndarray  # (U,) intp, rows into `context`
    opponent_unit_groups: np.ndarray  # (U,) intp, the group id each column names
    selector_mask: np.ndarray  # (P,) bool
    declaration_mask: np.ndarray  # (P, N_DECLARATIONS) bool
    displacement_mask: np.ndarray  # (P, 1 + n_move + n_advance) bool
    unit_mask: np.ndarray  # (P, 1 + U) bool

    @property
    def n_players(self) -> int:
        return int(self.players.shape[0])

    @property
    def n_context(self) -> int:
        return int(self.context.shape[0])

    @property
    def n_units(self) -> int:
        return int(self.opponent_unit_rows.shape[0])


def build_tokens(
    env: PerModelEnv,
    seat: Seat,
    observation: PerModelObservation,
    scenario: TokenScenario,
) -> TokenObservation:
    """Assemble the token observation for `seat` at `observation.decision`."""
    if scenario.episode_id != env.episode_id:
        raise ValueError(
            f"scenario built for episode {scenario.episode_id}, env is in "
            f"episode {env.episode_id}"
        )
    point = observation.decision
    own = seat.models
    enemies = seat.enemies
    own_alive = seat.alive()
    enemy_alive = np.array([m.is_alive for m in enemies], dtype=bool)
    own_locs = _locations(own)
    enemy_locs = _locations(enemies)
    own_groups = np.array([int(m.group_id) for m in own], dtype=np.intp)
    enemy_groups = np.array([int(m.group_id) for m in enemies], dtype=np.intp)
    quantities = env.rules_quantities
    base_diameter = 2.0 * quantities.base_radius

    own_engaged = _engaged(
        own_locs, enemy_locs, enemy_alive, own_alive, quantities, base_diameter
    )
    enemy_engaged = _engaged(
        enemy_locs, own_locs, own_alive, enemy_alive, quantities, base_diameter
    )
    modes = (
        np.asarray(observation.consolidate_mode, dtype=np.intp)
        if seat.is_player
        else np.zeros(len(own), dtype=np.intp)
    )
    players = np.stack(
        [
            _model_row(
                own[i],
                scenario,
                ranged=scenario.own_ranged[i],
                melee=scenario.own_melee[i],
                reach=scenario.own_reach[i],
                move=scenario.own_moves[i],
                acted=bool(point.acted[i]),
                in_open_unit=point.open_unit is not None
                and int(own[i].group_id) == point.open_unit,
                engaged=bool(own_engaged[i]),
                mode=int(modes[i]),
            )
            for i in range(len(own))
        ]
    ).astype(np.float32)

    # ---- context ----------------------------------------------------------
    rows: list[np.ndarray] = []
    kinds: list[int] = []
    alive: list[bool] = []
    positions: list[np.ndarray | None] = []

    rows.append(_game_row(env, observation, point, scenario))
    kinds.append(CONTEXT_GAME)
    alive.append(True)
    positions.append(None)

    own_unit_rows = _unit_rows(
        own,
        own_alive,
        own_groups,
        scenario.own_groups,
        scenario.own_initial,
        scenario,
        acted=np.asarray(point.acted, dtype=bool),
        open_unit=point.open_unit,
        side=1.0,
    )
    for g, row in zip(scenario.own_groups, own_unit_rows):
        rows.append(row)
        kinds.append(CONTEXT_OWN_UNIT)
        alive.append(bool(np.any(own_alive & (own_groups == g))))
        positions.append(_centroid(own_locs, own_alive & (own_groups == g)))

    enemy_model_start = len(rows)
    for o, model in enumerate(enemies):
        rows.append(
            _model_row(
                model,
                scenario,
                ranged=scenario.enemy_ranged[o],
                melee=scenario.enemy_melee[o],
                reach=scenario.enemy_reach[o],
                move=float(scenario.enemy_moves[o]),
                acted=False,
                in_open_unit=False,
                engaged=bool(enemy_engaged[o]),
                mode=0,
            )
        )
        kinds.append(CONTEXT_ENEMY_MODEL)
        alive.append(bool(enemy_alive[o]))
        positions.append(enemy_locs[o])

    enemy_unit_start = len(rows)
    enemy_unit_rows = _unit_rows(
        enemies,
        enemy_alive,
        enemy_groups,
        scenario.enemy_groups,
        scenario.enemy_initial,
        scenario,
        acted=None,
        open_unit=None,
        side=0.0,
    )
    for g, row in zip(scenario.enemy_groups, enemy_unit_rows):
        rows.append(row)
        kinds.append(CONTEXT_ENEMY_UNIT)
        alive.append(bool(np.any(enemy_alive & (enemy_groups == g))))
        positions.append(_centroid(enemy_locs, enemy_alive & (enemy_groups == g)))

    objective_rows, objective_positions = _objective_rows(env, seat, scenario)
    for row, position in zip(objective_rows, objective_positions):
        rows.append(row)
        kinds.append(CONTEXT_OBJECTIVE)
        alive.append(True)
        positions.append(position)

    terrain_start = len(rows)
    for t in range(scenario.terrain_rows.shape[0]):
        rows.append(scenario.terrain_rows[t])
        kinds.append(CONTEXT_TERRAIN)
        alive.append(True)
        positions.append(None)

    n_context = len(rows)
    context = np.zeros((n_context, CONTEXT_DIM), dtype=np.float32)
    for c, row in enumerate(rows):
        context[c, : row.shape[0]] = row
    context_kind = np.array(kinds, dtype=np.intp)
    context_alive = np.array(alive, dtype=bool)

    # ---- relations --------------------------------------------------------
    n_own = len(own)
    self_relations = np.zeros((n_own, n_own, RELATION_DIM), dtype=np.float32)
    _write_offsets(self_relations, own_locs, own_locs, scenario)
    same = own_groups[:, None] == own_groups[None, :]
    self_relations[:, :, REL_SAME_UNIT] = same
    self_relations[:, :, REL_SAME_UNIT_PRESENT] = 1.0

    cross = np.zeros((n_own, n_context, RELATION_DIM), dtype=np.float32)
    positioned = [c for c, p in enumerate(positions) if p is not None]
    if positioned:
        targets = np.stack([positions[c] for c in positioned])  # type: ignore[misc]
        block = np.zeros((n_own, len(positioned), RELATION_DIM), dtype=np.float32)
        _write_offsets(block, own_locs, targets, scenario)
        cross[:, positioned, :] = block
    for u, g in enumerate(scenario.own_groups):
        c = 1 + u
        cross[:, c, REL_SAME_UNIT] = own_groups == g
        cross[:, c, REL_SAME_UNIT_PRESENT] = 1.0
    if enemies:
        _write_enemy_relations(
            cross,
            env,
            own,
            enemies,
            own_alive,
            enemy_alive,
            own_locs,
            enemy_locs,
            scenario,
            base_diameter,
            enemy_model_start,
        )
        _write_enemy_unit_relations(
            cross, enemy_alive, enemy_groups, scenario, enemy_unit_start
        )
    if scenario.terrain_rows.shape[0]:
        inside = polygons_contain_points(
            own_locs, scenario.terrain_outlines, scenario.terrain_vertex_counts
        )
        span = slice(terrain_start, terrain_start + inside.shape[1])
        cross[:, span, REL_INSIDE] = inside
        cross[:, span, REL_TERRAIN_PRESENT] = 1.0

    # ---- decision-side masks ---------------------------------------------
    unit_rows = np.arange(
        enemy_unit_start, enemy_unit_start + len(scenario.enemy_groups), dtype=np.intp
    )
    return TokenObservation(
        kind=point.kind,
        phase=point.phase,
        head=head_for(point.kind, point.phase),
        seat_is_player=bool(point.seat_is_player),
        players=players,
        player_alive=own_alive,
        player_groups=own_groups,
        context=context,
        context_kind=context_kind,
        context_alive=context_alive,
        self_relations=self_relations,
        cross_relations=cross,
        opponent_unit_rows=unit_rows,
        opponent_unit_groups=scenario.enemy_groups.copy(),
        selector_mask=np.asarray(point.selector_mask, dtype=bool).copy(),
        declaration_mask=np.asarray(point.declaration_mask, dtype=bool).copy(),
        displacement_mask=displacement_mask_from(point, seat),
        unit_mask=unit_mask_from(point, seat, scenario.enemy_groups),
    )


# ------------------------------------------------------------ mask derivation


def displacement_mask_from(point: DecisionPoint, seat: Seat) -> np.ndarray:
    """`(P, 1 + n_move + n_advance)`: STAY, the movement slice, the advance slice.

    Read straight off the point's action mask, so whatever the phase's ladder
    gates (advance rungs after a declaration, the charge and short-move
    legality rows) is already applied.
    """
    handler = seat.handler
    movement = handler.movement_slice
    advance = handler.advance_slice
    n_models = point.action_mask.shape[0]
    width = 1 + movement.size + (advance.size if advance is not None else 0)
    mask = np.zeros((n_models, width), dtype=bool)
    if point.kind is not StepKind.act or head_for(point.kind, point.phase) is not (
        Head.displacement
    ):
        return mask
    mask[:, 0] = point.action_mask[:, STAY_ACTION]
    mask[:, 1 : 1 + movement.size] = point.action_mask[:, movement.start : movement.end]
    if advance is not None:
        mask[:, 1 + movement.size :] = point.action_mask[:, advance.start : advance.end]
    return mask


def unit_mask_from(
    point: DecisionPoint, seat: Seat, enemy_groups: np.ndarray
) -> np.ndarray:
    """`(P, 1 + U)`: the no-target column, then one column per enemy unit.

    On an `act` step (shooting, fight) column 0 is STAY -- legal for a shooter
    holding fire, never legal in the fight phase, where a selected striker
    must strike -- and column j is the shooting-slice column of unit j's group
    id. On a `target` step column 0 is the decline, always legal, and column
    j is the target mask's column for that group.
    """
    n_models = point.action_mask.shape[0]
    mask = np.zeros((n_models, 1 + len(enemy_groups)), dtype=bool)
    head = head_for(point.kind, point.phase)
    if head is not Head.unit_pointer:
        return mask
    if point.kind is StepKind.target:
        mask[:, 0] = point.selector_mask
        for j, g in enumerate(enemy_groups):
            if g < point.target_mask.shape[1]:
                mask[:, 1 + j] = point.target_mask[:, g]
        return mask
    shooting = seat.handler.shooting_slice
    mask[:, 0] = point.action_mask[:, STAY_ACTION]
    if shooting is not None:
        for j, g in enumerate(enemy_groups):
            if g < shooting.size:
                mask[:, 1 + j] = point.action_mask[:, shooting.start + g]
    return mask


def displacement_to_action(column: int, seat: Seat) -> int:
    """The action index a displacement column names."""
    if column == 0:
        return STAY_ACTION
    movement = seat.handler.movement_slice
    if column <= movement.size:
        return movement.start + column - 1
    advance = seat.handler.advance_slice
    if advance is None:
        raise ValueError(f"displacement column {column} names no action")
    return advance.start + column - 1 - movement.size


def unit_column_to_value(
    column: int, kind: StepKind, seat: Seat, enemy_groups: np.ndarray
) -> int:
    """The env value a unit-pointer column names, for an `act` or `target` step."""
    if column == 0:
        return STAY_ACTION if kind is StepKind.act else CHARGE_TARGET_DECLINE
    group = int(enemy_groups[column - 1])
    if kind is StepKind.target:
        return group
    shooting = seat.handler.shooting_slice
    if shooting is None:
        raise ValueError("no shooting slice to name a target in")
    return shooting.start + group


# ------------------------------------------------------------------- rows


def _locations(models: list[WargameModel]) -> np.ndarray:
    return np.array([m.location for m in models], dtype=float).reshape(len(models), 2)


def _normalise(points: np.ndarray, scenario: TokenScenario) -> np.ndarray:
    out = np.empty_like(points, dtype=float)
    out[..., 0] = (points[..., 0] - scenario.half_width) / scenario.half_width
    out[..., 1] = (points[..., 1] - scenario.half_height) / scenario.half_height
    return out


def _engaged(
    positions: np.ndarray,
    other: np.ndarray,
    other_alive: np.ndarray,
    subject_alive: np.ndarray,
    quantities: RulesQuantities,
    base_diameter: float,
) -> np.ndarray:
    engagement_range = float(quantities.engagement_range)
    if positions.shape[0] == 0 or other.shape[0] == 0 or engagement_range <= 0.0:
        return np.zeros(positions.shape[0], dtype=bool)
    return engaged_with_any(
        positions,
        other,
        other_alive,
        subject_alive,
        engagement_range=engagement_range,
        base_diameter=base_diameter,
    )


def _model_row(
    model: WargameModel,
    scenario: TokenScenario,
    *,
    ranged: np.ndarray,
    melee: np.ndarray,
    reach: float,
    move: float,
    acted: bool,
    in_open_unit: bool,
    engaged: bool,
    mode: int,
) -> np.ndarray:
    """One model token. The reveal rule gates the dice on the declaration flags."""
    row = np.zeros(MODEL_DIM, dtype=np.float32)
    x, y = _normalise(np.asarray(model.location, dtype=float), scenario)
    max_wounds = float(model.stats["max_wounds"])
    row[MODEL_X] = x
    row[MODEL_Y] = y
    row[MODEL_ALIVE] = float(model.is_alive)
    row[3] = float(model.stats["current_wounds"]) / max_wounds if max_wounds else 0.0
    row[4] = max_wounds / 10.0
    row[5:10] = ranged / _STAT_NORMALISERS
    row[10:15] = melee / _STAT_NORMALISERS
    row[15] = float(model.stats["toughness"]) / 10.0
    row[16] = float(model.stats["save"]) / 7.0
    row[17] = reach / scenario.diagonal
    row[18] = move / scenario.diagonal
    row[19] = float(model.base_radius)
    row[MODEL_ACTED] = float(acted)
    row[21] = float(in_open_unit)
    row[MODEL_ADVANCE_ROLL] = (
        (model.advance_roll / 6.0) if model.declared_advance else 0.0
    )
    row[23] = float(model.advanced_this_turn)
    row[MODEL_CHARGE_ROLL] = (
        (model.charge_roll / 12.0) if model.declared_charge else 0.0
    )
    row[25] = float(model.fell_back_this_turn)
    row[26] = float(model.declared_charge)
    row[27] = float(model.charged_this_turn)
    row[MODEL_ENGAGED] = float(engaged)
    row[29] = float(model.fight_priority) / 3.0
    row[30] = float(model.fought_this_phase)
    if 0 <= mode < N_CONSOLIDATION_MODES:
        row[31 + mode] = 1.0
    return row


def _centroid(locations: np.ndarray, members: np.ndarray) -> np.ndarray | None:
    if not np.any(members):
        return None
    return np.asarray(locations[members].mean(axis=0), dtype=float)


def _unit_rows(
    models: list[WargameModel],
    alive: np.ndarray,
    groups: np.ndarray,
    unit_ids: np.ndarray,
    initial: np.ndarray,
    scenario: TokenScenario,
    *,
    acted: np.ndarray | None,
    open_unit: int | None,
    side: float,
) -> list[np.ndarray]:
    locations = _locations(models)
    rows = []
    for u, g in enumerate(unit_ids):
        members = groups == g
        living = members & alive
        row = np.zeros(UNIT_DIM, dtype=np.float32)
        centroid = _centroid(locations, living)
        if centroid is not None:
            row[0:2] = _normalise(centroid, scenario)
        n_initial = float(initial[u])
        row[2] = float(np.sum(living)) / n_initial if n_initial else 0.0
        row[3] = n_initial / 10.0
        row[4] = float(
            any(models[i].advanced_this_turn for i in np.flatnonzero(living))
        )
        row[5] = float(
            any(models[i].fell_back_this_turn for i in np.flatnonzero(living))
        )
        row[6] = float(any(models[i].declared_charge for i in np.flatnonzero(living)))
        row[7] = float(any(models[i].fought_this_phase for i in np.flatnonzero(living)))
        row[8] = float(open_unit is not None and int(g) == open_unit)
        if acted is not None and np.any(living):
            row[9] = float(np.sum(acted & living)) / float(np.sum(living))
        row[10] = side
        rows.append(row)
    return rows


def _objective_rows(
    env: PerModelEnv, seat: Seat, scenario: TokenScenario
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    objectives = env.objectives
    if not objectives:
        return [], []
    own_cache = compute_distances(seat.models, objectives, alive_mask=seat.alive())
    own_counts = objective_counts_from_norms_offset(
        own_cache.model_obj_norms_offset, own_cache.obj_radii
    )
    if seat.enemies:
        enemy_alive = np.array([m.is_alive for m in seat.enemies], dtype=bool)
        enemy_cache = compute_distances(
            seat.enemies, objectives, alive_mask=enemy_alive
        )
        enemy_counts = objective_counts_from_norms_offset(
            enemy_cache.model_obj_norms_offset, enemy_cache.obj_radii
        )
    else:
        enemy_counts = np.zeros(len(objectives), dtype=np.intp)
    rows: list[np.ndarray] = []
    positions: list[np.ndarray] = []
    for k, objective in enumerate(objectives):
        row = np.zeros(OBJECTIVE_DIM, dtype=np.float32)
        position = np.asarray(objective.location, dtype=float)
        row[0:2] = _normalise(position, scenario)
        row[2] = float(own_counts[k]) / 10.0
        row[3] = float(enemy_counts[k]) / 10.0
        row[4] = float(scenario.objective_extent[k]) / scenario.diagonal
        row[5] = float(scenario.objective_is_area[k])
        rows.append(row)
        positions.append(position)
    return rows, positions


def _game_row(
    env: PerModelEnv,
    observation: PerModelObservation,
    point: DecisionPoint,
    scenario: TokenScenario,
) -> np.ndarray:
    row = np.zeros(GAME_DIM, dtype=np.float32)
    n_rounds = float(env.n_rounds)
    row[0] = observation.battle_round / n_rounds if n_rounds else 0.0
    if point.phase is not None:
        row[1 + BATTLE_PHASE_ORDER.index(point.phase)] = 1.0
    row[1 + N_PHASES + list(StepKind).index(point.kind)] = 1.0
    base = 1 + N_PHASES + N_STEP_KINDS
    row[base] = float(observation.active_seat_is_player)
    row[base + 1] = observation.player_vp / 100.0
    row[base + 2] = observation.opponent_vp / 100.0
    row[base + 3] = observation.player_vp_delta / 15.0
    row[base + 4] = observation.opponent_vp_delta / 15.0
    row[base + 5] = float(point.open_unit is not None)
    row[base + 6] = float(point.forced_model is not None)
    return row


# ---------------------------------------------------------------- relations


def _write_offsets(
    block: np.ndarray, sources: np.ndarray, targets: np.ndarray, scenario: TokenScenario
) -> None:
    """Target minus source, normalised by the board, on every pair."""
    if sources.shape[0] == 0 or targets.shape[0] == 0:
        return
    delta = targets[None, :, :] - sources[:, None, :]
    block[:, :, REL_DX] = delta[:, :, 0] / scenario.half_width
    block[:, :, REL_DY] = delta[:, :, 1] / scenario.half_height
    block[:, :, REL_OFFSET_PRESENT] = 1.0


def _write_enemy_relations(
    cross: np.ndarray,
    env: PerModelEnv,
    own: list[WargameModel],
    enemies: list[WargameModel],
    own_alive: np.ndarray,
    enemy_alive: np.ndarray,
    own_locs: np.ndarray,
    enemy_locs: np.ndarray,
    scenario: TokenScenario,
    base_diameter: float,
    start: int,
) -> None:
    """Sight and cover within reach, expected damage both ways, per enemy model."""
    n_own, n_enemy = len(own), len(enemies)
    span = slice(start, start + n_enemy)
    cross[:, span, REL_RANGED_OUT] = scenario.ranged_out
    cross[:, span, REL_RANGED_IN] = scenario.ranged_in
    cross[:, span, REL_RANGED_PRESENT] = 1.0
    has_melee = (scenario.own_melee[:, 0] > 0)[:, None] | (
        scenario.enemy_melee[:, 0] > 0
    )[None, :]
    cross[:, span, REL_MELEE_OUT] = scenario.melee_out
    cross[:, span, REL_MELEE_IN] = scenario.melee_in
    cross[:, span, REL_MELEE_PRESENT] = has_melee
    distances = np.linalg.norm(own_locs[:, None, :] - enemy_locs[None, :, :], axis=2)
    reach = np.maximum(scenario.own_reach[:, None], scenario.enemy_reach[None, :])
    candidates = (
        (distances <= reach + base_diameter) & own_alive[:, None] & enemy_alive[None, :]
    )
    if not np.any(candidates):
        return
    visibility = env.visibility_between(
        own_locs,
        enemy_locs,
        candidates,
        origin_models=own,
        target_models=enemies,
        edges=True,
    ).reshape(n_own, n_enemy)
    cross[:, span, REL_VISIBLE] = np.where(candidates, visibility >= COVER, 0.0)
    cross[:, span, REL_IN_COVER] = np.where(candidates, visibility == COVER, 0.0)
    cross[:, span, REL_SIGHT_PRESENT] = candidates


def _write_enemy_unit_relations(
    cross: np.ndarray,
    enemy_alive: np.ndarray,
    enemy_groups: np.ndarray,
    scenario: TokenScenario,
    start: int,
) -> None:
    """Per enemy unit: damage OUT against one living member, IN summed over them."""
    for u, g in enumerate(scenario.enemy_groups):
        living = np.flatnonzero(enemy_alive & (enemy_groups == g))
        if living.size == 0:
            continue
        c = start + u
        first = living[0]
        cross[:, c, REL_RANGED_OUT] = scenario.ranged_out[:, first]
        cross[:, c, REL_RANGED_IN] = scenario.ranged_in[:, living].sum(axis=1)
        cross[:, c, REL_RANGED_PRESENT] = 1.0
        cross[:, c, REL_MELEE_OUT] = scenario.melee_out[:, first]
        cross[:, c, REL_MELEE_IN] = scenario.melee_in[:, living].sum(axis=1)
        cross[:, c, REL_MELEE_PRESENT] = (scenario.own_melee[:, 0] > 0) | bool(
            np.any(scenario.enemy_melee[living, 0] > 0)
        )


__all__ = [
    "CONTEXT_DIM",
    "CONTEXT_ENEMY_MODEL",
    "CONTEXT_ENEMY_UNIT",
    "CONTEXT_GAME",
    "CONTEXT_OBJECTIVE",
    "CONTEXT_OWN_UNIT",
    "CONTEXT_TERRAIN",
    "GAME_DIM",
    "MODEL_DIM",
    "N_CONTEXT_KINDS",
    "N_DECLARATIONS",
    "OBJECTIVE_DIM",
    "RELATION_DIM",
    "TERRAIN_DIM",
    "UNIT_DIM",
    "Head",
    "TokenObservation",
    "TokenScenario",
    "build_tokens",
    "displacement_mask_from",
    "displacement_to_action",
    "head_for",
    "unit_column_to_value",
    "unit_mask_from",
]
