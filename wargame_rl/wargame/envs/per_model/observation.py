"""The token/relation observation for the per-model facade (issue #285).

Principle 2: nothing about a token's width may depend on how many other
entities exist. Every entity is a token in a set — player models, unit tokens
for **both** sides, objectives, terrain pieces, one game token — and every
relation between entities (the offset, sight and cover, unit membership,
expected damage) is expressed **between tokens** as one pairwise vector,
zero-filled with presence flags where a field does not apply. The count-sized
columns of the whole-phase observation (per-objective distance pairs, the
`max_groups` one-hot, per-opponent expected damage) do not exist here.

Padding is a batching convenience: this module emits ragged per-episode
arrays; the model-side collation pads to the **batch maximum**, never to a
config budget.

This is a NEW observation contract for the new facade only — nothing the
whole-phase pipeline reads is touched, and no golden governs these arrays.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from wargame_rl.wargame.envs.domain.entities import alive_mask_for
from wargame_rl.wargame.envs.domain.shooting import expected_damage_matrix
from wargame_rl.wargame.envs.domain.sight import COVER, HIDDEN
from wargame_rl.wargame.envs.env_components.actions import (
    ADVANCE_DIE_FACES,
    CHARGE_DICE_MAX,
    STAY_ACTION,
)
from wargame_rl.wargame.envs.per_model.types import PerModelObservation, StepKind
from wargame_rl.wargame.envs.types import BattlePhase
from wargame_rl.wargame.envs.types.config import ModelConfig
from wargame_rl.wargame.envs.types.game_timing import BATTLE_PHASE_ORDER
from wargame_rl.wargame.envs.types.geometry import polygons_contain_points

if TYPE_CHECKING:
    from wargame_rl.wargame.envs.per_model.facade import PerModelEnv
    from wargame_rl.wargame.envs.wargame_model import WargameModel

# Stat normalisers. Deliberately the same values the whole-phase tensor
# pipeline uses (`model/common/observation.py`), restated here because `envs/`
# may not import `model/` — these are part of this contract now and drift is
# acceptable, since no checkpoint spans the two observation worlds.
NORM_ATTACKS = 10.0
NORM_BALLISTIC_SKILL = 6.0
NORM_STRENGTH = 10.0
NORM_AP = 6.0
NORM_DAMAGE = 10.0
NORM_TOUGHNESS = 10.0
NORM_SAVE = 7.0
NORM_MAX_WOUNDS = 100.0
NORM_EXPECTED_DAMAGE = 10.0

# The one accepted cap: a piece is an 8-vertex silhouette (the real tables
# carry at most 8). A cap on a token's OWN width is fine under Principle 2 —
# what is forbidden is a width that depends on how many other entities exist.
TERRAIN_VERTEX_BUDGET = 8

# The game token.
GAME_DIM = 7
# Model tokens (both sides share the width — a model is a model).
MODEL_DIM = 25
# Unit tokens (both sides).
UNIT_DIM = 6
# Objective tokens: location, control counts, extent. Control is always
# observed here — the whole-phase flag `observe_objective_control` exists to
# keep old checkpoints loading, and no checkpoint spans the two worlds.
OBJECTIVE_DIM = 5
TERRAIN_DIM = 2 * TERRAIN_VERTEX_BUDGET + 1

# Context token type ids, embedded model-side.
CTX_GAME = 0
CTX_PLAYER_UNIT = 1
CTX_OPPONENT_MODEL = 2
CTX_OPPONENT_UNIT = 3
CTX_OBJECTIVE = 4
CTX_TERRAIN = 5
N_CONTEXT_KINDS = 6

# The pairwise relation vector: the union of every set's fields, zero-filled
# with a presence flag where a field does not apply.
REL_DX = 0
REL_DY = 1
REL_OFFSET_PRESENT = 2
REL_SAME_UNIT = 3
REL_SAME_UNIT_PRESENT = 4
REL_VISIBLE = 5
REL_IN_COVER = 6
REL_SIGHT_PRESENT = 7
REL_DMG_OUT = 8
REL_DMG_IN = 9
REL_DMG_PRESENT = 10
REL_INSIDE_TERRAIN = 11
REL_TERRAIN_PRESENT = 12
RELATION_DIM = 13

# The declaration head's fixed vocabulary; which entries are legal is
# phase-dependent (see `declaration_mask`). Movement: 0 normal, 1 remain
# stationary, 2 advance. Shooting: 0 shoot, 1 hold fire. Charge: 0 decline,
# 1 charge. Fight: 0..3 activation priority.
N_DECLARATION_OPTIONS = 4


@dataclass(slots=True)
class TokenObservation:
    """One state as tokens, relations and masks. All arrays are float32/bool.

    ``context_*`` arrays stack the union of context tokens in a fixed set
    order — game, player units, opponent models, opponent units, objectives,
    terrain — with each row's set named by ``context_kinds``. Rows within a
    set keep entity order, so ``opponent_unit_rows`` can address the target
    pointer's candidates.
    """

    kind: StepKind
    phase_index: int
    player_tokens: np.ndarray  # (n_p, MODEL_DIM)
    player_alive: np.ndarray  # (n_p,) bool — dead are masked, never dropped
    selection_mask: np.ndarray  # (n_p,) bool
    context_tokens: np.ndarray  # (n_ctx, CTX feature width = max of set dims)
    context_kinds: np.ndarray  # (n_ctx,) int
    context_mask: np.ndarray  # (n_ctx,) bool — False = dead/empty, drop as key
    self_relations: np.ndarray  # (n_p, n_p, RELATION_DIM)
    cross_relations: np.ndarray  # (n_p, n_ctx, RELATION_DIM)
    opponent_unit_rows: np.ndarray  # (n_target_units,) int — rows in context
    displacement_mask: np.ndarray  # (n_p, 1 + n_move_actions) bool
    advance_mask: np.ndarray  # (n_p, n_advance_actions) bool; width 0 if none
    target_mask: np.ndarray  # (n_p, 1 + n_target_units) bool; col 0 = hold
    declaration_mask: np.ndarray  # (n_p, N_DECLARATION_OPTIONS) bool


# The context feature width: every set's token is right-padded to this.
CONTEXT_DIM = max(GAME_DIM, UNIT_DIM, MODEL_DIM, OBJECTIVE_DIM, TERRAIN_DIM)


def build_token_observation(
    env: PerModelEnv, observation: PerModelObservation
) -> TokenObservation:
    """Extract the current state of `env` as tokens and relations.

    `observation` is the facade's light step observation, which carries the
    step kind and the selection state this builder must agree with.
    """
    board_w = float(env.board_width)
    board_h = float(env.board_height)
    diagonal = float(np.hypot(board_w, board_h)) or 1.0

    players = env.wargame_models
    opponents = env.opponent_models
    player_alive = alive_mask_for(players)
    opponent_alive = alive_mask_for(opponents) if opponents else np.zeros(0, dtype=bool)

    player_stats = _stat_rows(env.config.models, len(players))
    opponent_stats = _stat_rows(env.config.opponent_models, len(opponents))
    player_tokens = _model_tokens(
        players,
        player_stats,
        env.player_max_ranges,
        env.player_action_handler.move_speeds,
        acted=observation.acted_mask,
        board_w=board_w,
        board_h=board_h,
        diagonal=diagonal,
    )
    opponent_move = (
        env.opponent_action_handler.move_speeds
        if opponents
        else np.zeros(0, dtype=float)
    )
    opponent_tokens = _model_tokens(
        opponents,
        opponent_stats,
        env.opponent_max_ranges,
        opponent_move,
        acted=np.zeros(len(opponents), dtype=bool),
        board_w=board_w,
        board_h=board_h,
        diagonal=diagonal,
    )

    player_units, player_unit_ids, player_unit_centroids = _unit_tokens(
        players, board_w, board_h
    )
    opponent_units, opponent_unit_ids, opponent_unit_centroids = _unit_tokens(
        opponents, board_w, board_h
    )
    objectives = _objective_tokens(env, diagonal)
    terrain = _terrain_tokens(env, board_w, board_h)
    game = _game_token(env, observation)

    # -- context union, in fixed set order --
    sets: list[tuple[int, np.ndarray, np.ndarray]] = [
        (CTX_GAME, game[np.newaxis, :], np.ones(1, dtype=bool)),
        (
            CTX_PLAYER_UNIT,
            player_units,
            _unit_alive(players, player_unit_ids),
        ),
        (CTX_OPPONENT_MODEL, opponent_tokens, opponent_alive),
        (
            CTX_OPPONENT_UNIT,
            opponent_units,
            _unit_alive(opponents, opponent_unit_ids),
        ),
        (CTX_OBJECTIVE, objectives, np.ones(len(objectives), dtype=bool)),
        (CTX_TERRAIN, terrain, np.ones(len(terrain), dtype=bool)),
    ]
    n_ctx = sum(rows.shape[0] for _, rows, _m in sets)
    context_tokens = np.zeros((n_ctx, CONTEXT_DIM), dtype=np.float32)
    context_kinds = np.zeros(n_ctx, dtype=np.int64)
    context_mask = np.zeros(n_ctx, dtype=bool)
    context_positions = np.full((n_ctx, 2), np.nan, dtype=np.float64)
    row = 0
    set_start: dict[int, int] = {}
    for kind_id, rows, mask in sets:
        set_start[kind_id] = row
        count = rows.shape[0]
        if count:
            context_tokens[row : row + count, : rows.shape[1]] = rows
            context_kinds[row : row + count] = kind_id
            context_mask[row : row + count] = mask
        row += count

    # Board positions per context row, for the offset relation. The game token
    # has none (NaN → offset_present 0).
    _write_centroids(
        context_positions, set_start[CTX_PLAYER_UNIT], player_unit_centroids
    )
    if len(opponents):
        context_positions[
            set_start[CTX_OPPONENT_MODEL] : set_start[CTX_OPPONENT_MODEL]
            + len(opponents)
        ] = np.array([m.location for m in opponents], dtype=np.float64)
    _write_centroids(
        context_positions, set_start[CTX_OPPONENT_UNIT], opponent_unit_centroids
    )
    if len(objectives):
        context_positions[
            set_start[CTX_OBJECTIVE] : set_start[CTX_OBJECTIVE] + len(objectives)
        ] = np.array([o.location for o in env.objectives], dtype=np.float64)
    footprints = env.terrain.footprints
    if len(terrain):
        context_positions[
            set_start[CTX_TERRAIN] : set_start[CTX_TERRAIN] + len(terrain)
        ] = np.array(
            [
                [
                    (fp.polygon.bounds[0] + fp.polygon.bounds[2]) / 2.0,
                    (fp.polygon.bounds[1] + fp.polygon.bounds[3]) / 2.0,
                ]
                for fp in footprints
            ],
            dtype=np.float64,
        )

    player_positions = np.array([m.location for m in players], dtype=np.float64)
    group_ids = np.array([int(m.group_id) for m in players], dtype=np.intp)

    self_relations = _self_relations(player_positions, group_ids, diagonal)
    cross_relations = _cross_relations(
        env,
        player_positions,
        group_ids,
        player_stats,
        opponent_stats,
        context_positions,
        context_kinds,
        set_start,
        player_unit_ids,
        opponent_unit_ids,
        diagonal,
    )

    (
        displacement_mask,
        advance_mask,
        target_mask,
        declaration_mask,
    ) = _action_masks(env, observation, len(opponent_unit_ids))

    opponent_unit_rows = set_start[CTX_OPPONENT_UNIT] + np.arange(
        len(opponent_unit_ids), dtype=np.int64
    )

    return TokenObservation(
        kind=observation.kind,
        phase_index=(
            BATTLE_PHASE_ORDER.index(observation.phase)
            if observation.phase is not None
            else 0
        ),
        player_tokens=player_tokens,
        player_alive=player_alive,
        selection_mask=observation.selection_mask.copy(),
        context_tokens=context_tokens,
        context_kinds=context_kinds,
        context_mask=context_mask,
        self_relations=self_relations,
        cross_relations=cross_relations,
        opponent_unit_rows=opponent_unit_rows,
        displacement_mask=displacement_mask,
        advance_mask=advance_mask,
        target_mask=target_mask,
        declaration_mask=declaration_mask,
    )


# -- Tokens -------------------------------------------------------------------


def _stat_rows(configs: list[ModelConfig] | None, n_models: int) -> np.ndarray:
    """(n, 7) raw stats per model: attacks, bs, strength, ap, damage, T, Sv."""
    rows = np.zeros((n_models, 7), dtype=np.float32)
    for i in range(n_models):
        if configs is not None and i < len(configs):
            cfg = configs[i]
            rows[i, 5] = cfg.toughness
            rows[i, 6] = cfg.save
            if cfg.weapons:
                weapon = cfg.weapons[0]
                rows[i, 0] = weapon.attacks
                rows[i, 1] = weapon.ballistic_skill
                rows[i, 2] = weapon.strength
                rows[i, 3] = weapon.ap
                rows[i, 4] = weapon.damage
    return rows


def _model_tokens(
    models: list[WargameModel],
    stats: np.ndarray,
    max_ranges: np.ndarray,
    move_speeds: np.ndarray,
    *,
    acted: np.ndarray,
    board_w: float,
    board_h: float,
    diagonal: float,
) -> np.ndarray:
    """(n, MODEL_DIM) per-model tokens. Nothing here depends on any count."""
    n = len(models)
    tokens = np.zeros((n, MODEL_DIM), dtype=np.float32)
    for i, model in enumerate(models):
        wounds = float(model.stats["current_wounds"])
        max_wounds = float(model.stats["max_wounds"]) or 1.0
        tokens[i] = (
            2.0 * float(model.location[0]) / board_w - 1.0,
            2.0 * float(model.location[1]) / board_h - 1.0,
            1.0 if model.is_alive else 0.0,
            wounds / max_wounds,
            min(max_wounds / NORM_MAX_WOUNDS, 1.0),
            stats[i, 0] / NORM_ATTACKS,
            stats[i, 1] / NORM_BALLISTIC_SKILL,
            stats[i, 2] / NORM_STRENGTH,
            stats[i, 3] / NORM_AP,
            stats[i, 4] / NORM_DAMAGE,
            stats[i, 5] / NORM_TOUGHNESS,
            stats[i, 6] / NORM_SAVE,
            # The three inputs armies of varying profile need on the token
            # (the design's "missing today' rows): reach, speed, footprint.
            float(max_ranges[i]) / diagonal if len(max_ranges) > i else 0.0,
            float(move_speeds[i]) / diagonal if len(move_speeds) > i else 0.0,
            float(model.base_radius) / diagonal,
            1.0 if acted[i] else 0.0,
            float(model.advance_roll) / ADVANCE_DIE_FACES,
            1.0 if model.advanced_this_turn else 0.0,
            float(model.charge_roll) / CHARGE_DICE_MAX,
            1.0 if model.fell_back_this_turn else 0.0,
            1.0 if model.declared_advance else 0.0,
            1.0 if model.declared_charge else 0.0,
            1.0 if model.charged_this_turn else 0.0,
            float(model.fight_priority) / 3.0,
            1.0 if getattr(model, "fought_this_phase", False) else 0.0,
        )
    return tokens


def _unit_tokens(
    models: list[WargameModel], board_w: float, board_h: float
) -> tuple[np.ndarray, list[int], np.ndarray]:
    """Unit tokens, their ids (ascending) and raw live centroids (NaN if dead).

    A unit token is the live centroid, the alive fraction, a size feature and
    the unit's own turn flags. Units whose last member has fallen keep a row —
    masked, never dropped — so the target pointer's candidate list is stable
    within an episode.
    """
    units: dict[int, list[WargameModel]] = {}
    for model in models:
        units.setdefault(int(model.group_id), []).append(model)
    unit_ids = sorted(units)
    tokens = np.zeros((len(unit_ids), UNIT_DIM), dtype=np.float32)
    centroids = np.full((len(unit_ids), 2), np.nan, dtype=np.float64)
    for row, unit_id in enumerate(unit_ids):
        members = units[unit_id]
        living = [m for m in members if m.is_alive]
        if living:
            centroid = np.mean([m.location for m in living], axis=0)
            centroids[row] = centroid
            tokens[row, 0] = 2.0 * float(centroid[0]) / board_w - 1.0
            tokens[row, 1] = 2.0 * float(centroid[1]) / board_h - 1.0
        tokens[row, 2] = len(living) / len(members)
        tokens[row, 3] = min(len(members) / 20.0, 1.0)
        tokens[row, 4] = 1.0 if any(m.advanced_this_turn for m in living) else 0.0
        tokens[row, 5] = 1.0 if any(m.fell_back_this_turn for m in living) else 0.0
    return tokens, unit_ids, centroids


def _unit_alive(models: list[WargameModel], unit_ids: list[int]) -> np.ndarray:
    alive_units = {int(m.group_id) for m in models if m.is_alive}
    return np.array([uid in alive_units for uid in unit_ids], dtype=bool)


def _objective_tokens(env: PerModelEnv, diagonal: float) -> np.ndarray:
    from wargame_rl.wargame.envs.baseline.policy import objective_extent
    from wargame_rl.wargame.envs.env_components.distance_cache import (
        compute_distances,
        objective_counts_from_norms_offset,
    )

    objectives = env.objectives
    if not objectives:
        return np.zeros((0, OBJECTIVE_DIM), dtype=np.float32)
    # The control counts VP scores on — the one definition, shared with the
    # scoring path (`objective_counts_from_norms_offset`).
    obj_radii = np.array([o.radius_size for o in objectives], dtype=float)
    # `compute_distances` writes inf offsets for dead models under `alive_mask`,
    # which is exactly what keeps a corpse out of the count.
    player_counts = objective_counts_from_norms_offset(
        compute_distances(
            env.wargame_models,
            objectives,
            alive_mask=alive_mask_for(env.wargame_models),
        ).model_obj_norms_offset,
        obj_radii,
    )
    if env.opponent_models:
        opponent_counts = objective_counts_from_norms_offset(
            compute_distances(
                env.opponent_models,
                objectives,
                alive_mask=alive_mask_for(env.opponent_models),
            ).model_obj_norms_offset,
            obj_radii,
        )
    else:
        opponent_counts = np.zeros(len(objectives))
    establishment = float(max(len(env.wargame_models), len(env.opponent_models), 1))
    tokens = np.zeros((len(objectives), OBJECTIVE_DIM), dtype=np.float32)
    for i, objective in enumerate(objectives):
        tokens[i] = (
            2.0 * float(objective.location[0]) / env.board_width - 1.0,
            2.0 * float(objective.location[1]) / env.board_height - 1.0,
            float(player_counts[i]) / establishment,
            float(opponent_counts[i]) / establishment,
            objective_extent(objective) / diagonal,
        )
    return tokens


def _terrain_tokens(env: PerModelEnv, board_w: float, board_h: float) -> np.ndarray:
    footprints = env.terrain.footprints
    if not footprints:
        return np.zeros((0, TERRAIN_DIM), dtype=np.float32)
    half_w, half_h = board_w / 2.0, board_h / 2.0
    tokens = np.zeros((len(footprints), TERRAIN_DIM), dtype=np.float32)
    for i, footprint in enumerate(footprints):
        if footprint.n_vertices > TERRAIN_VERTEX_BUDGET:
            raise ValueError(
                f"terrain piece has {footprint.n_vertices} vertices, over the "
                f"{TERRAIN_VERTEX_BUDGET}-vertex cap"
            )
        padded = footprint.polygon.padded_to(TERRAIN_VERTEX_BUDGET)
        tokens[i, : 2 * TERRAIN_VERTEX_BUDGET : 2] = padded[:, 0] / half_w - 1.0
        tokens[i, 1 : 2 * TERRAIN_VERTEX_BUDGET : 2] = padded[:, 1] / half_h - 1.0
        tokens[i, -1] = footprint.n_vertices / float(TERRAIN_VERTEX_BUDGET)
    return tokens


def _game_token(env: PerModelEnv, observation: PerModelObservation) -> np.ndarray:
    state = env.game_clock_state
    n_rounds = float(env.n_rounds) or 1.0
    cap = 100.0
    return np.array(
        [
            (state.battle_round or 0) / n_rounds,
            min(env.player_vp / cap, 1.0),
            min(env.opponent_vp / cap, 1.0),
            env.player_vp_delta / 15.0,
            env.opponent_vp_delta / 15.0,
            1.0 if observation.kind is StepKind.turn_close else 0.0,
            observation.phase_model_steps / 100.0,
        ],
        dtype=np.float32,
    )


# -- Relations ----------------------------------------------------------------


def _offset_block(
    query_positions: np.ndarray, key_positions: np.ndarray, diagonal: float
) -> tuple[np.ndarray, np.ndarray]:
    """(Q, K, 2) normalised offsets plus a (Q, K) presence mask (NaN = none)."""
    deltas = key_positions[np.newaxis, :, :] - query_positions[:, np.newaxis, :]
    present = ~np.isnan(deltas).any(axis=2)
    deltas = np.nan_to_num(deltas, nan=0.0) / diagonal
    return deltas.astype(np.float32), present


def _self_relations(
    player_positions: np.ndarray, group_ids: np.ndarray, diagonal: float
) -> np.ndarray:
    n = len(player_positions)
    relations = np.zeros((n, n, RELATION_DIM), dtype=np.float32)
    if n == 0:
        return relations
    offsets, _present = _offset_block(player_positions, player_positions, diagonal)
    relations[:, :, REL_DX : REL_DY + 1] = offsets
    relations[:, :, REL_OFFSET_PRESENT] = 1.0
    relations[:, :, REL_SAME_UNIT] = (
        group_ids[:, np.newaxis] == group_ids[np.newaxis, :]
    ).astype(np.float32)
    relations[:, :, REL_SAME_UNIT_PRESENT] = 1.0
    return relations


def _cross_relations(
    env: PerModelEnv,
    player_positions: np.ndarray,
    group_ids: np.ndarray,
    player_stats: np.ndarray,
    opponent_stats: np.ndarray,
    context_positions: np.ndarray,
    context_kinds: np.ndarray,
    set_start: dict[int, int],
    player_unit_ids: list[int],
    opponent_unit_ids: list[int],
    diagonal: float,
) -> np.ndarray:
    n_p = len(player_positions)
    n_ctx = len(context_kinds)
    relations = np.zeros((n_p, n_ctx, RELATION_DIM), dtype=np.float32)
    if n_p == 0 or n_ctx == 0:
        return relations

    offsets, present = _offset_block(player_positions, context_positions, diagonal)
    relations[:, :, REL_DX : REL_DY + 1] = offsets
    relations[:, :, REL_OFFSET_PRESENT] = present.astype(np.float32)

    # Membership: a player model and its own unit token.
    unit_start = set_start[CTX_PLAYER_UNIT]
    for row, unit_id in enumerate(player_unit_ids):
        members = group_ids == unit_id
        relations[members, unit_start + row, REL_SAME_UNIT] = 1.0
    relations[
        :, unit_start : unit_start + len(player_unit_ids), REL_SAME_UNIT_PRESENT
    ] = 1.0

    opponents = env.opponent_models
    if opponents:
        opp_start = set_start[CTX_OPPONENT_MODEL]
        n_o = len(opponents)
        # Sight and cover, traced within weapon reach only: beyond both models'
        # reach the answer cannot change any legal choice, and the ungated
        # trace is the whole-phase facade's most expensive query.
        opponent_positions = np.array([m.location for m in opponents], dtype=float)
        gaps = np.linalg.norm(
            player_positions[:, np.newaxis, :] - opponent_positions[np.newaxis, :, :],
            axis=2,
        )
        reach = np.maximum(
            env.player_max_ranges[:, np.newaxis], env.opponent_max_ranges[np.newaxis, :]
        )
        alive_pairs = (
            alive_mask_for(env.wargame_models)[:, np.newaxis]
            & alive_mask_for(opponents)[np.newaxis, :]
        )
        candidates = (gaps <= reach) & alive_pairs
        if candidates.any():
            visibility = env.visibility_between(
                player_positions,
                opponent_positions,
                candidates,
                origin_models=env.wargame_models,
                target_models=list(opponents),
            )
            relations[:, opp_start : opp_start + n_o, REL_VISIBLE] = (
                visibility != HIDDEN
            ).astype(np.float32)
            relations[:, opp_start : opp_start + n_o, REL_IN_COVER] = (
                visibility == COVER
            ).astype(np.float32)
        relations[:, opp_start : opp_start + n_o, REL_SIGHT_PRESENT] = (
            candidates.astype(np.float32)
        )

        # Expected damage both ways — a dice function of the two stat lines,
        # memoised per distinct pair inside `expected_damage_matrix`.
        damage_out = expected_damage_matrix(
            player_stats[:, :5].astype(np.int64),
            opponent_stats[:, 5:7].astype(np.int64),
        )
        damage_in = expected_damage_matrix(
            opponent_stats[:, :5].astype(np.int64),
            player_stats[:, 5:7].astype(np.int64),
        ).T
        relations[:, opp_start : opp_start + n_o, REL_DMG_OUT] = np.clip(
            damage_out / NORM_EXPECTED_DAMAGE, 0.0, 1.0
        )
        relations[:, opp_start : opp_start + n_o, REL_DMG_IN] = np.clip(
            damage_in / NORM_EXPECTED_DAMAGE, 0.0, 1.0
        )
        relations[:, opp_start : opp_start + n_o, REL_DMG_PRESENT] = 1.0

    terrain = env.terrain
    if terrain.footprints:
        terrain_start = set_start[CTX_TERRAIN]
        # One (P, N) pass over the collection's prebuilt padded outlines —
        # identical semantics to Footprint.contains (edge-inclusive, padding
        # masked by vertex_counts), which was costing 25 x 16 scalar calls.
        inside = polygons_contain_points(
            player_positions,
            terrain.outlines,
            terrain.vertex_counts,
            include_boundary=True,
        )
        n_terrain = len(terrain.footprints)
        relations[:, terrain_start : terrain_start + n_terrain, REL_INSIDE_TERRAIN] = (
            inside.astype(np.float32)
        )
        relations[:, terrain_start : terrain_start + n_terrain, REL_TERRAIN_PRESENT] = (
            1.0
        )

    return relations


# -- Action masks -------------------------------------------------------------


def _action_masks(
    env: PerModelEnv,
    observation: PerModelObservation,
    n_target_units: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """The per-model legality the heads decode under, from the env's own mask.

    Sliced out of `compute_player_action_mask` — the same function the
    whole-phase facade puts on its observation — so the heads and the env
    cannot disagree about what is legal.
    """
    handler = env.player_action_handler
    n_p = len(env.wargame_models)
    n_move = handler.n_move_actions
    displacement = np.zeros((n_p, 1 + n_move), dtype=bool)
    advance_slice = handler.advance_slice
    n_advance = advance_slice.size if advance_slice is not None else 0
    advance = np.zeros((n_p, n_advance), dtype=bool)
    target = np.zeros((n_p, 1 + n_target_units), dtype=bool)
    declaration = np.zeros((n_p, N_DECLARATION_OPTIONS), dtype=bool)

    if observation.kind is StepKind.turn_close:
        return displacement, advance, target, declaration

    full = env.current_action_mask()
    movement = handler.movement_slice
    displacement[:, 0] = full[:, STAY_ACTION]
    displacement[:, 1:] = full[:, movement.start : movement.end]
    if advance_slice is not None:
        advance[:, :] = full[:, advance_slice.start : advance_slice.end]
    shooting_slice = handler.shooting_slice
    if shooting_slice is not None and observation.phase is BattlePhase.shooting:
        target[:, 0] = full[:, STAY_ACTION]
        width = min(n_target_units, shooting_slice.size)
        target[:, 1 : 1 + width] = full[
            :, shooting_slice.start : shooting_slice.start + width
        ]

    declaration[:, :] = _declaration_legality(env, observation)
    return displacement, advance, target, declaration


def _declaration_legality(
    env: PerModelEnv, observation: PerModelObservation
) -> np.ndarray:
    """Which declaration options a unit's opening step may take, per model.

    Coarse on purpose: the facade applies declarations as permissively as the
    engine does, and the fine-grained gates (an engaged unit may not advance,
    a charge needs an eligible unit) already live in the movement masks the
    displacement head decodes under.
    """
    n_p = len(env.wargame_models)
    legality = np.zeros((n_p, N_DECLARATION_OPTIONS), dtype=bool)
    phase = observation.phase
    if phase is BattlePhase.movement:
        legality[:, 0] = True
        legality[:, 1] = True
        legality[:, 2] = env.player_action_handler.advance_slice is not None
    elif phase is BattlePhase.shooting:
        legality[:, 0] = True
        legality[:, 1] = True
    elif phase is BattlePhase.charge:
        legality[:, 0] = True
        legality[:, 1] = bool(env.config.melee.enabled)
    elif phase is BattlePhase.fight:
        legality[:, :] = True
    else:
        legality[:, 0] = True
    return legality


def _write_centroids(
    context_positions: np.ndarray, start: int, centroids: np.ndarray
) -> None:
    """Unit live centroids into the context position table (NaN = no member).

    A unit with no living member keeps NaN, so its offset relation carries
    presence 0 — and its context row is masked anyway.
    """
    if centroids.shape[0]:
        context_positions[start : start + centroids.shape[0]] = centroids
