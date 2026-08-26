"""Build observations and info from battle state (BattleView).

Extracted so observation shape or content can be varied without touching step/reset.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from wargame_rl.wargame.envs.domain.battle_view import BattleView
from wargame_rl.wargame.envs.domain.coherency import (
    base_to_base_distances,
    evaluate_coherency,
)
from wargame_rl.wargame.envs.domain.entities import alive_mask_for
from wargame_rl.wargame.envs.domain.value_objects import POSITION_DTYPE
from wargame_rl.wargame.envs.env_components.actions import (
    ADVANCE_DIE_FACES,
    CHARGE_DICE_MAX,
    ActionRegistry,
)
from wargame_rl.wargame.envs.env_components.distance_cache import (
    compute_distances,
    objective_counts_from_norms_offset,
)
from wargame_rl.wargame.envs.env_components.shooting_masks import (
    compute_unit_shooting_masks,
)
from wargame_rl.wargame.envs.types import (
    WargameEnvInfo,
    WargameEnvObjectiveObservation,
    WargameEnvObservation,
    WargameModelObservation,
    WargameTerrainObservation,
)
from wargame_rl.wargame.envs.types.game_timing import BattlePhase
from wargame_rl.wargame.envs.types.terrain_observation import TERRAIN_VERTEX_BUDGET

if TYPE_CHECKING:
    from wargame_rl.wargame.envs.domain.entities import WargameObjective
    from wargame_rl.wargame.envs.env_components.distance_cache import DistanceCache
    from wargame_rl.wargame.envs.types.config import ModelConfig
    from wargame_rl.wargame.envs.wargame_model import WargameModel


def update_distances_to_objectives(
    wargame_models: list[WargameModel],
    objectives: list[WargameObjective],
    distance_cache: DistanceCache | None = None,
) -> None:
    """Update each model's distances_to_objectives from current locations. Mutates models."""
    # No `.astype(int)` here. The vector to the objective is the single most
    # informative feature the policy has, and truncating it to whole units threw
    # away sub-unit steering on a board where a move is now any real length.
    if distance_cache is not None:
        deltas = distance_cache.model_obj_deltas.astype(POSITION_DTYPE)
        for i, model in enumerate(wargame_models):
            model.distances_to_objectives = deltas[i]
        return

    for model in wargame_models:
        model.distances_to_objectives = np.array(
            [model.location - obj.location for obj in objectives],
            dtype=POSITION_DTYPE,
        )


def _unit_strengths(models: list[WargameModel]) -> dict[int, float]:
    """Fraction of each unit still alive, keyed by ``group_id``.

    A unit whose last model has fallen maps to 0.0 rather than being absent, so
    the lookup never has to guess. Numbering is per army, so this is only ever
    called with one army's models.
    """
    totals: dict[int, int] = {}
    alive: dict[int, int] = {}
    for model in models:
        totals[model.group_id] = totals.get(model.group_id, 0) + 1
        alive[model.group_id] = alive.get(model.group_id, 0) + int(model.is_alive)
    return {gid: alive[gid] / totals[gid] for gid in totals}


def _pad_distances(distances: np.ndarray, budget: int) -> np.ndarray:
    """Pad a model's (n_objectives, 2) delta block out to `budget` rows with zeros.

    The zeros are meaningless on their own — a zero delta is what standing on an
    objective looks like — and are read only alongside the presence flags built
    by `_objective_presence`.
    """
    padded = np.zeros((budget, 2), dtype=distances.dtype)
    padded[: len(distances)] = distances
    return padded


def _objective_presence(n_objectives: int, budget: int) -> np.ndarray:
    """Per-slot flags: 1.0 for the first `n_objectives` slots, 0.0 for padding."""
    presence = np.zeros(budget, dtype=np.float32)
    presence[:n_objectives] = 1.0
    return presence


@dataclass(frozen=True, slots=True)
class CoherencyDistances:
    """The two coherency distances, in board units, resolved once per step.

    A small carrier rather than two loose floats: the observation builder passes
    them through three layers, and a pair of bare floats in that signature is
    exactly how a nearest and a furthest end up swapped.
    """

    nearest: float
    furthest: float


def _unit_offsets(models: list[WargameModel], spread_cap: float) -> np.ndarray:
    """Per-model vector to its unit's live centroid, over the spread cap.

    This is the quantity `ScriptedSquadMarchPolicy.select_movement` computes and
    steers every member of a unit along; reproducing its *effect* from per-model
    inputs is what a behaviour clone has been failing to do.

    Dead models report ``(0, 0)`` — the same "nothing to correct" value a lone
    or already-centred model reports — because `phase_manager` and the coherency
    predicate both iterate the living, and a casualty must never read as a model
    that needs to move.
    """
    offsets = np.zeros((len(models), 2), dtype=np.float32)
    if not models:
        return offsets

    positions = np.array([m.location for m in models], dtype=float)
    group_ids = np.array([m.group_id for m in models], dtype=np.intp)
    alive = alive_mask_for(models)
    live = np.flatnonzero(alive)
    for group_id in np.unique(group_ids[live]):
        members = live[group_ids[live] == group_id]
        centroid = positions[members].mean(axis=0)
        # Clipped per axis: the sign is the signal, and how far is already
        # carried by `coherency_spread`.
        offsets[members] = np.clip(
            (centroid - positions[members]) / spread_cap, -1.0, 1.0
        )
    return offsets


def _coherency_features(
    models: list[WargameModel],
    nearest_distance: float,
    furthest_distance: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-model spread ratio and component fraction, for the coherency inputs.

    The spread ratio is the distance to the furthest live model in the unit over
    the spread cap, clipped at 1 so "legal" and "twice as far as legal" do not
    share a saturated value at the boundary the policy actually has to find.

    The component fraction is how much of the unit is in this model's own chain
    component. It is 1.0 for a unit in one piece, and a dead or lone model
    reports a compliant (0.0, 1.0) rather than a violation.
    """
    n = len(models)
    spread = np.zeros(n, dtype=np.float32)
    component = np.ones(n, dtype=np.float32)
    if n == 0:
        return spread, component

    report = evaluate_coherency(
        positions=np.array([m.location for m in models], dtype=float),
        group_ids=np.array([m.group_id for m in models], dtype=np.intp),
        alive_mask=alive_mask_for(models),
        base_radii=np.array([m.base_radius for m in models], dtype=float),
        nearest_distance=nearest_distance,
        furthest_distance=furthest_distance,
    )
    for unit in report.units:
        if unit.size <= 1:
            continue
        gaps = base_to_base_distances(
            np.array([models[i].location for i in unit.member_indices], dtype=float),
            np.array([models[i].base_radius for i in unit.member_indices], dtype=float),
        )
        furthest = gaps.max(axis=1)
        counts = np.bincount(unit.component, minlength=unit.n_components)
        spread[unit.member_indices] = np.clip(furthest / furthest_distance, 0.0, 1.0)
        component[unit.member_indices] = counts[unit.component] / unit.size
    return spread, component


# A D6. The roll is divided by it so the column is in [0, 1] like every
# other normalised feature, rather than being 6x the scale of its neighbours.


def _models_to_obs(
    models: list[WargameModel],
    max_groups: int,
    model_configs: list[ModelConfig] | None = None,
    observe_unit_strength: bool = False,
    objective_budget: int | None = None,
    n_objectives: int | None = None,
    coherency: CoherencyDistances | None = None,
    unit_centroid_cap: float | None = None,
    observe_advance: bool = False,
    advance_is_known: bool = True,
    observe_melee: bool = False,
) -> list[WargameModelObservation]:
    strengths = _unit_strengths(models) if observe_unit_strength else {}
    offsets = (
        None if unit_centroid_cap is None else _unit_offsets(models, unit_centroid_cap)
    )
    spread, component = (
        _coherency_features(models, coherency.nearest, coherency.furthest)
        if coherency is not None
        else (None, None)
    )
    presence = (
        None
        if objective_budget is None
        else _objective_presence(
            n_objectives if n_objectives is not None else 0, objective_budget
        )
    )
    result: list[WargameModelObservation] = []
    for i, m in enumerate(models):
        w_attacks = 0
        w_bs = 0
        w_str = 0
        w_ap = 0
        w_dmg = 0
        toughness = 0
        save = 0
        if model_configs is not None and i < len(model_configs):
            cfg = model_configs[i]
            toughness = cfg.toughness
            save = cfg.save
            if cfg.weapons:
                w = cfg.weapons[0]
                w_attacks = w.attacks
                w_bs = w.ballistic_skill
                w_str = w.strength
                w_ap = w.ap
                w_dmg = w.damage
        result.append(
            WargameModelObservation(
                location=m.location,
                distances_to_objectives=(
                    m.distances_to_objectives
                    if objective_budget is None
                    else _pad_distances(m.distances_to_objectives, objective_budget)
                ),
                objective_present=presence,
                group_id=m.group_id,
                max_groups=max_groups,
                alive=1.0 if m.is_alive else 0.0,
                current_wounds=int(m.stats["current_wounds"]),
                max_wounds=int(m.stats["max_wounds"]),
                weapon_attacks=w_attacks,
                weapon_ballistic_skill=w_bs,
                weapon_strength=w_str,
                weapon_ap=w_ap,
                weapon_damage=w_dmg,
                toughness=toughness,
                save_stat=save,
                unit_strength=(
                    strengths.get(m.group_id, 0.0) if observe_unit_strength else None
                ),
                advance_roll=(
                    None
                    if not observe_advance
                    else (
                        float(m.advance_roll) / ADVANCE_DIE_FACES
                        if advance_is_known
                        else 0.0
                    )
                ),
                advanced_this_turn=(
                    None
                    if not observe_advance
                    else (float(m.advanced_this_turn) if advance_is_known else 0.0)
                ),
                # The melee pair shares `advance_is_known`, not a flag of its
                # own: both sides' per-turn state goes stale for exactly the same
                # reason -- each rolls and spends at the start of its OWN turn.
                charge_roll=(
                    None
                    if not observe_melee
                    else (
                        float(m.charge_roll) / CHARGE_DICE_MAX
                        if advance_is_known
                        else 0.0
                    )
                ),
                fell_back_this_turn=(
                    None
                    if not observe_melee
                    else (float(m.fell_back_this_turn) if advance_is_known else 0.0)
                ),
                # ⚠ Gated on `advance_is_known` exactly like the two above,
                # and for the same reason: a declaration is spent by the end of
                # the turn it is made in, so on the side whose turn it is NOT it
                # would report a charge the unit is no longer under. (An earlier
                # comment here claimed the opposite of the code beside it; the
                # code was right. A rules-lawyer audit caught the contradiction.)
                declared_charge=(
                    None
                    if not observe_melee
                    else (float(m.declared_charge) if advance_is_known else 0.0)
                ),
                coherency_spread=(None if spread is None else float(spread[i])),
                coherency_component=(
                    None if component is None else float(component[i])
                ),
                unit_offset=(None if offsets is None else offsets[i]),
            )
        )
    return result


def _terrain_to_obs(
    view: BattleView,
) -> list[WargameTerrainObservation]:
    """Build terrain observations: padded outline vertices, plus a vertex count.

    When `terrain_budget` is set the token *sequence* is padded to it as well,
    with all-zero rows. No extra flag is needed to mark those: the vertex-count
    column is zero on them and no real piece has zero vertices, which is what
    the network keys on to drop them from attention.


    Vertices are normalised to [-1, 1] by the board half-dimensions and padded to
    `TERRAIN_VERTEX_BUDGET` by repeating the last one, so pieces with different
    vertex counts stack into one array — which observation batching requires.

    This is the first encoding that can tell an outline from its bounding box,
    and therefore the first honest test of whether the agent can use terrain: the
    whole cover line of work was run against four numbers that made an L-shaped
    ruin and a solid block identical.

    A piece with more vertices than the budget is a config error, not something
    to silently truncate — dropping vertices would quietly shrink a ruin the
    sight trace is still using at full size.
    """
    half_w = view.board_width / 2.0
    half_h = view.board_height / 2.0
    result: list[WargameTerrainObservation] = []
    for fp in view.terrain.footprints:
        if fp.n_vertices > TERRAIN_VERTEX_BUDGET:
            raise ValueError(
                f"terrain piece has {fp.n_vertices} vertices, over the "
                f"observation budget of {TERRAIN_VERTEX_BUDGET}. Raise "
                "TERRAIN_VERTEX_BUDGET (which changes the network's input "
                "width, so existing checkpoints will fail to load) or simplify "
                "the outline."
            )
        padded = fp.polygon.padded_to(TERRAIN_VERTEX_BUDGET)
        normalised = np.empty(2 * TERRAIN_VERTEX_BUDGET + 1, dtype=np.float32)
        normalised[0 : 2 * TERRAIN_VERTEX_BUDGET : 2] = (padded[:, 0] - half_w) / half_w
        normalised[1 : 2 * TERRAIN_VERTEX_BUDGET : 2] = (padded[:, 1] - half_h) / half_h
        normalised[-1] = fp.n_vertices / TERRAIN_VERTEX_BUDGET
        result.append(WargameTerrainObservation(outline=normalised))

    budget = view.config.terrain_budget
    if budget is not None:
        if len(result) > budget:
            raise ValueError(
                f"layout has {len(result)} terrain pieces, over the "
                f"terrain_budget of {budget}"
            )
        width = 2 * TERRAIN_VERTEX_BUDGET + 1
        result.extend(
            WargameTerrainObservation(outline=np.zeros(width, dtype=np.float32))
            for _ in range(budget - len(result))
        )
    return result


def _pad_objectives(
    observations: list[WargameEnvObjectiveObservation],
    view: BattleView,
    with_control: bool,
) -> list[WargameEnvObjectiveObservation]:
    """Mark the real objectives present and pad the list out to `objective_budget`.

    A padding slot sits at the board centre with zero control, so that once the
    tensor pipeline normalises location by the board half-dimensions its whole
    row is zero — which is what lets the network recognise padding without being
    told how many objectives this particular layout had. `present` is what keeps
    that test safe in the other direction: a real objective at the exact centre
    would otherwise produce an all-zero row too.
    """
    budget = view.config.objective_budget
    if budget is None:
        return observations
    if len(observations) > budget:
        raise ValueError(
            f"layout has {len(observations)} objectives, over the "
            f"objective_budget of {budget}"
        )
    for observation in observations:
        observation.present = 1.0
    centre = np.array([view.board_width / 2.0, view.board_height / 2.0], dtype=float)
    control = {"player_count": 0.0, "opponent_count": 0.0, "radius": 0.0}
    observations.extend(
        WargameEnvObjectiveObservation(
            location=centre.copy(),
            present=0.0,
            **(control if with_control else {}),
        )
        for _ in range(budget - len(observations))
    )
    return observations


def _in_range_counts(
    models: list[WargameModel], objectives: list[WargameObjective]
) -> np.ndarray:
    """Alive models of one side in range of each objective, per the scoring rule.

    Delegates to `compute_distances` and `objective_counts_from_norms_offset` so
    there is exactly one definition of "on an objective" in the codebase.
    """
    if not models or not objectives:
        return np.zeros(len(objectives), dtype=int)
    cache = compute_distances(models, objectives, alive_mask=alive_mask_for(models))
    return objective_counts_from_norms_offset(
        cache.model_obj_norms_offset, cache.obj_radii
    )


def _objectives_to_obs(
    view: BattleView, with_control: bool
) -> list[WargameEnvObjectiveObservation]:
    """Objective observations, optionally carrying per-objective control state.

    Counts are of *alive* models in range of each objective, normalised by a
    static army size so the feature stays O(1) rather than shrinking with force
    size.

    ⚠ **The count is the one the mission scores, and it did not used to be.**
    This built its own membership test -- `area.contains_points` on the model
    *centre* -- while VP, `objective_hold` and every control read use
    `norms_offset <= obj_radii`, which measures from the model's **base edge**.
    A model whose base overlaps the ruin while its centre sits outside scored
    for the mission and was invisible here. Measured on the held-out nine before
    the fix: **206 of 2,700 (objective, step) slots disagreed -- 7.6%**, 215
    models miscounted. Every objective-keyed reward term and every proposed
    mission primitive reads this feature, so the standing rule "check the agent
    can observe what the lever keys on" was quietly false for all of them.

    **Both sides share one divisor, and that is the point.** Control is decided
    by a raw count comparison (`player_count > opponent_count`), so the two
    columns are only comparable if they are on the same scale. Dividing each
    side by *its own* establishment made them incomparable the moment the armies
    were different sizes: at 25 v 18, ten of ours reads 0.40 and nine of theirs
    reads 0.50, so the observation says they are winning an objective we
    control. At parity the two divisors coincide, which is why every shipped
    config is unaffected.
    """
    if not with_control:
        return _pad_objectives(
            [
                WargameEnvObjectiveObservation(location=obj.location)
                for obj in view.objectives
            ],
            view,
            with_control,
        )

    # Built the same way `DefaultVPCalculator.compute_vp` builds them -- same
    # function, same alive masks -- so the two cannot drift apart again. Dead
    # models arrive as `inf` and fall out of the comparison.
    player_counts = _in_range_counts(view.player_models, view.objectives)
    opponent_counts = _in_range_counts(view.opponent_models, view.objectives)

    # The larger establishment, so neither side's column can exceed 1.0 and the
    # two stay directly comparable. Identical to either when the armies match.
    establishment = max(
        1,
        view.config.number_of_wargame_models,
        view.config.number_of_opponent_models,
    )
    board_diagonal = float(np.hypot(view.board_width, view.board_height)) or 1.0

    observations = []
    for index, objective in enumerate(view.objectives):
        # An area's "radius" is reported as the radius of a disc with the same
        # area, so the feature keeps meaning "how big is this objective" across
        # both kinds rather than collapsing to zero for one of them.
        if objective.area is not None:
            extent = float(np.sqrt(objective.area.area / np.pi))
        else:
            extent = float(objective.radius_size)
        observations.append(
            WargameEnvObjectiveObservation(
                location=objective.location,
                player_count=float(player_counts[index]) / establishment,
                opponent_count=float(opponent_counts[index]) / establishment,
                radius=extent / board_diagonal,
            )
        )
    return _pad_objectives(observations, view, with_control)


def build_observation(
    view: BattleView,
    distance_cache: DistanceCache | None = None,
    action_registry: ActionRegistry | None = None,
) -> WargameEnvObservation:
    """Build the Gym observation from battle state (BattleView)."""
    if distance_cache is not None:
        update_distances_to_objectives(
            view.player_models, view.objectives, distance_cache
        )
    if view.opponent_models:
        update_distances_to_objectives(view.opponent_models, view.objectives, None)

    action_mask: np.ndarray | None = None
    if action_registry is not None:
        phase = view.game_clock_state.phase or BattlePhase.movement
        player_alive = alive_mask_for(view.player_models)
        action_mask = action_registry.get_model_action_masks(
            phase, len(view.player_models), alive_mask=player_alive
        )
        if view.config.melee.enabled and phase == BattlePhase.charge:
            # A charge is an ordinary movement action in an extraordinary phase,
            # so the slice is already unmasked here -- what has to be added is
            # the rules' own gates: the unit must be eligible to declare, and
            # the 2D6 is the charge move's maximum.
            movement_slice = action_registry.slice_for("movement")
            action_mask[:, movement_slice.start : movement_slice.end] &= (
                view.player_charge_legality
            )
            # ⚠ **REMOVED 2026-08-26: a declared unit MAY decline.** This used to
            # strike STAY from every declared model that held a legal rung, on
            # the reasoning that a declaration should BIND rather than merely
            # permit. The rules say otherwise, explicitly:
            # `11-charge-phase.md` step 3 -- *"If a legal charge move is possible
            # AND THE CONTROLLING PLAYER STILL WANTS TO MAKE IT, make it.
            # Otherwise the unit does not move. Either way the charge is
            # resolved."* Declining after the roll is a right the game grants and
            # this took it away.
            #
            # It was not a harmless extra: a rules-lawyer audit measured it
            # binding on **30 of 31** declared units, and it compounds with the
            # declaration sitting two phases early -- a unit committed in the
            # COMMAND phase, before it has moved or shot, could not then back out
            # of a charge that the movement phase had made hopeless. It walked at
            # nothing and the referee reverted it.
            #
            # ⚠ It does not follow that half a unit may charge. That is enforced
            # where it belongs, in `_enforce_charge`: a stationary model does not
            # veto its unit's charge, but the unit must still end COHERENT and
            # engaged, so a squad that half-commits stretches and reverts anyway.
            # The rule was being enforced twice, once correctly.

        if action_registry.has_slice("move_type") and phase == BattlePhase.command:
            # ⚠ The declaration was UNMASKED until 2026-08-26. A unit the rules
            # make ineligible could still declare -- a charge declaration is a
            # bit-exact no-op, so the policy got a free action with nothing to
            # learn from, and an advance declaration spends the unit's shooting
            # the moment it is made. See `ActionHandler.declaration_legality`.
            move_type_slice = action_registry.slice_for("move_type")
            action_mask[:, move_type_slice.start : move_type_slice.end] &= (
                view.player_declaration_legality
            )

        if action_registry.has_slice("advance") and phase == BattlePhase.movement:
            # The advance rungs are ABSOLUTE distances above Move, so the turn's
            # D6 no longer changes what an action means -- it decides which
            # rungs are reachable. Without this mask a policy could pick a 12"
            # advance on a roll of 1 and silently receive a 7" one, which is the
            # non-stationary semantics the absolute ladder exists to remove.
            advance_slice = action_registry.slice_for("advance")
            action_mask[:, advance_slice.start : advance_slice.end] &= (
                view.player_advance_legality
            )
        if (
            action_registry.has_slice("shooting")
            and phase == BattlePhase.shooting
            and view.opponent_models
        ):
            shooting_slice = action_registry.slice_for("shooting")
            opponent_alive = alive_mask_for(view.opponent_models)
            player_positions = np.array([m.location for m in view.player_models])
            opponent_positions = np.array([m.location for m in view.opponent_models])
            player_ranges = view.player_max_ranges
            # Advancing and falling back both cost the turn's shooting, so the
            # mask takes their union -- `docs/rules/09-movement-phase.md`.
            player_advanced = np.array(
                [
                    m.advanced_this_turn or m.fell_back_this_turn
                    for m in view.player_models
                ]
            )
            shooting_validity = compute_unit_shooting_masks(
                player_positions,
                opponent_positions,
                player_alive,
                opponent_alive,
                player_ranges,
                view.line_of_sight_matrix,
                np.array([m.group_id for m in view.opponent_models], dtype=int),
                shooting_slice.end - shooting_slice.start,
                player_advanced=player_advanced,
                # The shooter's own unit, so that one model in contact silences
                # its squadmates -- the rule is per unit, not per model.
                player_groups=np.array(
                    [m.group_id for m in view.player_models], dtype=int
                ),
                engagement_range=view.rules_quantities.engagement_range,
                base_diameter=2.0 * view.rules_quantities.base_radius,
                exclude_engaged_targets=view.config.melee.enabled
                and view.config.melee.shield_engaged_targets,
            )
            action_mask[:, shooting_slice.start : shooting_slice.end] &= (
                shooting_validity
            )

    clock = view.game_clock_state
    phase = clock.phase or BattlePhase.movement
    battle_phase_index = list(BattlePhase).index(phase)
    battle_round = clock.battle_round if clock.battle_round is not None else 1
    max_groups = view.config.max_groups
    objectives_obs = _objectives_to_obs(view, view.config.observe_objective_control)
    terrain_obs = _terrain_to_obs(view)
    # Resolved to board units here, once, rather than per model -- the config
    # authors them in inches like every other rules distance.
    coherency = (
        CoherencyDistances(
            nearest=view.rules_quantities.scale.to_units(
                view.config.coherency.nearest_distance
            ),
            furthest=view.rules_quantities.scale.to_units(
                view.config.coherency.furthest_distance
            ),
        )
        if view.config.observe_coherency
        else None
    )
    # Resolved here for the same reason as the pair above: the config authors it
    # in inches, and this is the one place that knows the scale.
    unit_centroid_cap = (
        view.rules_quantities.scale.to_units(view.config.coherency.furthest_distance)
        if view.config.observe_unit_centroid
        else None
    )
    # ⚠ Gated on the CHARGE PHASE BEING STEPPED, not on `melee.enabled`, and the
    # difference is the whole point of the dark control. `25v25_maps_melee.yaml`
    # and `..._melee_dark.yaml` differ in exactly one scalar -- `melee.enabled`
    # -- so that the arm and its control share an init and the per-seed
    # difference is a PAIRED estimator. Gating the columns on that same scalar
    # would give the two configs different tensor widths, different input
    # projections and therefore different weights at step 0, destroying the only
    # thing the pair exists to provide.
    #
    # With melee off the roll is never taken, so both columns are constant zero:
    # informationally identical to not having them, exactly as for the
    # opponent's zeroed columns below. Every golden config skips `charge`, so
    # none of them is touched.
    observe_melee = (
        BattlePhase.charge not in view.config.skip_phases
        if view.config.melee.observe_charge is None
        else view.config.melee.observe_charge
    )
    return WargameEnvObservation(
        current_turn=view.current_turn,
        wargame_models=_models_to_obs(
            view.player_models,
            max_groups,
            model_configs=view.config.models,
            observe_unit_strength=view.config.observe_unit_strength,
            objective_budget=view.config.objective_budget,
            n_objectives=len(view.objectives),
            coherency=coherency,
            unit_centroid_cap=unit_centroid_cap,
            observe_advance=view.config.n_advance_speed_bins > 0,
            observe_melee=observe_melee,
        ),
        objectives=objectives_obs,
        board_width=view.board_width,
        board_height=view.board_height,
        opponent_models=_models_to_obs(
            view.opponent_models,
            max_groups,
            model_configs=view.config.opponent_models,
            observe_unit_strength=view.config.observe_unit_strength,
            objective_budget=view.config.objective_budget,
            n_objectives=len(view.objectives),
            coherency=coherency,
            unit_centroid_cap=unit_centroid_cap,
            observe_advance=view.config.n_advance_speed_bins > 0,
            observe_melee=observe_melee,
            # ⚠ The opponent's advance AND melee columns are ZEROED, not read.
            # Each side rolls at the start of its OWN turn, so the opponent's
            # values are zero in round 1 and one turn STALE thereafter -- they
            # record what it rolled and did on its last turn, which says nothing
            # about the turn it is about to take. A stale column is worse than no
            # column: the network has to learn to ignore it.
            #
            # Zeroed rather than dropped because the player and opponent tokens
            # share a feature width (`observation.py` asserts it), so removing
            # two columns from one side alone does not typecheck at the tensor.
            # A constant-zero column contributes nothing through the embedding,
            # so this is informationally identical to dropping it -- and unlike
            # dropping it, costs no shape change and orphans no checkpoint.
            advance_is_known=False,
        ),
        terrain=terrain_obs,
        action_mask=action_mask,
        battle_round=battle_round,
        battle_phase_index=battle_phase_index,
        n_rounds=view.n_rounds,
        player_vp=view.player_vp,
        opponent_vp=view.opponent_vp,
        player_vp_delta=view.player_vp_delta,
    )


def build_info(view: BattleView) -> WargameEnvInfo:
    """Build the Gym info dict from battle state (BattleView)."""
    dz = view.deployment_zone
    odz = view.opponent_deployment_zone
    deployment_zone = (int(dz[0]), int(dz[1]), int(dz[2]), int(dz[3]))
    opponent_deployment_zone = (int(odz[0]), int(odz[1]), int(odz[2]), int(odz[3]))
    max_groups = view.config.max_groups
    objectives_obs = [
        WargameEnvObjectiveObservation(location=obj.location) for obj in view.objectives
    ]
    return WargameEnvInfo(
        current_turn=view.current_turn,
        wargame_models=_models_to_obs(
            view.player_models, max_groups, model_configs=view.config.models
        ),
        objectives=objectives_obs,
        opponent_models=_models_to_obs(
            view.opponent_models, max_groups, model_configs=view.config.opponent_models
        ),
        deployment_zone=deployment_zone,
        opponent_deployment_zone=opponent_deployment_zone,
        player_vp=view.player_vp,
        opponent_vp=view.opponent_vp,
        player_vp_delta=view.player_vp_delta,
        opponent_vp_delta=view.opponent_vp_delta,
    )
