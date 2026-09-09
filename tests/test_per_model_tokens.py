"""The token and relation observation, pinned to the env it reads.

Every claim here is checked against the env's own answer -- the decision
point's masks decoded column by column, the sight trace, the damage
arithmetic, the scoring definition of "on an objective", the footprint's own
containment test -- never against a recorded file. The size-independence
claims are the receipts for Principle 2: the widths at six models in two
units equal the widths at ten in five, and the enemy-unit columns are the
sorted distinct group ids even when those ids have a gap.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.per_model_seats import (
    random_legal_action,
    shooting_charging_driver,
    small_config,
)
from wargame_rl.wargame.envs.domain.attacks.stats import DefenderStats
from wargame_rl.wargame.envs.domain.battlefield.sight import COVER
from wargame_rl.wargame.envs.domain.sequencing.activation import (
    CHARGE_TARGET_DECLINE,
    MoveDeclaration,
)
from wargame_rl.wargame.envs.domain.shooting.expectation import expected_damage
from wargame_rl.wargame.envs.env_components.actions import STAY_ACTION
from wargame_rl.wargame.envs.env_components.distance_cache import (
    compute_distances,
    objective_counts_from_norms_offset,
)
from wargame_rl.wargame.envs.per_model import PerModelEnv, PerModelObservation, StepKind
from wargame_rl.wargame.envs.per_model.tokens import (
    CONTEXT_DIM,
    CONTEXT_ENEMY_MODEL,
    CONTEXT_ENEMY_UNIT,
    CONTEXT_GAME,
    CONTEXT_OBJECTIVE,
    CONTEXT_OWN_UNIT,
    CONTEXT_TERRAIN,
    MODEL_ADVANCE_ROLL,
    MODEL_ALIVE,
    MODEL_DIM,
    MODEL_X,
    REL_DX,
    REL_DY,
    REL_IN_COVER,
    REL_INSIDE,
    REL_MELEE_OUT,
    REL_OFFSET_PRESENT,
    REL_RANGED_IN,
    REL_RANGED_OUT,
    REL_SAME_UNIT,
    REL_SIGHT_PRESENT,
    REL_TERRAIN_PRESENT,
    REL_VISIBLE,
    RELATION_DIM,
    Head,
    TokenObservation,
    TokenScenario,
    build_tokens,
    displacement_to_action,
    unit_column_to_value,
)
from wargame_rl.wargame.envs.types import WargameEnvConfig
from wargame_rl.wargame.envs.types.config.terrain import RandomTerrainConfig
from wargame_rl.wargame.envs.types.game_timing import BattlePhase


def _tokens(env: PerModelEnv, observation: object) -> TokenObservation:
    seat = env.player_seat
    return build_tokens(env, seat, observation, TokenScenario.for_episode(env, seat))  # type: ignore[arg-type]


def _walk(
    env: PerModelEnv, seed: int, driver: str = "random"
) -> list[tuple[TokenObservation, PerModelObservation]]:
    """Every (tokens, observation) pair of one episode, under a seat."""
    observation, _ = env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    scenario = TokenScenario.for_episode(env, env.player_seat)
    pairs = []
    done = False
    while not done:
        pairs.append(
            (build_tokens(env, env.player_seat, observation, scenario), observation)
        )
        point = observation.decision
        action = (
            random_legal_action(point, rng)
            if driver == "random"
            else shooting_charging_driver(env, point)
        )
        observation, _r, done, _t, _i = env.step(action)
    return pairs


def _with_advance(config: WargameEnvConfig) -> WargameEnvConfig:
    """Three advance rungs; the command phase has to be stepped for the declaration."""
    skipped = [p for p in config.skip_phases if p is not BattlePhase.command]
    return WargameEnvConfig(
        **{**config.model_dump(), "n_advance_speed_bins": 3, "skip_phases": skipped}
    )


def _with_terrain(config: WargameEnvConfig) -> WargameEnvConfig:
    return WargameEnvConfig(
        **{
            **config.model_dump(),
            "random_terrain": RandomTerrainConfig(
                count=3, min_size=4, max_size=6, mirror=False
            ).model_dump(),
        }
    )


# ------------------------------------------------------------- Principle 2


def test_widths_do_not_depend_on_the_scenario_size() -> None:
    """Six models in two units and ten in five give the same token widths."""
    small = _tokens(*_reset(PerModelEnv(small_config())))
    large = _tokens(
        *_reset(
            PerModelEnv(
                small_config(
                    n_models=10, group_ids=[i // 2 for i in range(10)], max_groups=5
                )
            )
        )
    )
    assert small.players.shape == (6, MODEL_DIM) and large.players.shape == (
        10,
        MODEL_DIM,
    )
    assert small.context.shape[1] == large.context.shape[1] == CONTEXT_DIM
    assert (
        small.self_relations.shape[2] == large.self_relations.shape[2] == RELATION_DIM
    )
    assert small.declaration_mask.shape[1] == large.declaration_mask.shape[1]
    assert small.displacement_mask.shape[1] == large.displacement_mask.shape[1]
    assert small.n_units == 2 and large.n_units == 5


def _reset(env: PerModelEnv, seed: int = 1) -> tuple[PerModelEnv, object]:
    observation, _ = env.reset(seed=seed)
    return env, observation


def test_enemy_models_and_own_models_share_one_row_width() -> None:
    env, observation = _reset(PerModelEnv(small_config()))
    tokens = _tokens(env, observation)
    enemy_rows = tokens.context[tokens.context_kind == CONTEXT_ENEMY_MODEL]
    assert enemy_rows.shape == (6, CONTEXT_DIM)
    # An enemy row is a model row right-padded: its x is where a model's x is.
    enemy_x = (np.array([m.location for m in env.opponent_models])[:, 0] - 30.0) / 30.0
    assert np.allclose(enemy_rows[:, MODEL_X], enemy_x, atol=1e-6)


def test_the_context_is_laid_out_game_first_then_by_kind() -> None:
    env, observation = _reset(PerModelEnv(small_config()))
    tokens = _tokens(env, observation)
    kinds = tokens.context_kind
    assert kinds[0] == CONTEXT_GAME
    assert list(kinds) == sorted(kinds), "kinds are contiguous and in order"
    assert set(kinds) == {
        CONTEXT_GAME,
        CONTEXT_OWN_UNIT,
        CONTEXT_ENEMY_MODEL,
        CONTEXT_ENEMY_UNIT,
        CONTEXT_OBJECTIVE,
    }
    assert np.all(kinds[tokens.opponent_unit_rows] == CONTEXT_ENEMY_UNIT)
    assert list(tokens.opponent_unit_groups) == [0, 1]


def test_enemy_unit_columns_are_the_sorted_distinct_group_ids_even_with_a_gap() -> None:
    """A {0, 2} army has two units and a hole in the shooting slice; the
    pointer has two columns and names group 2 in the second."""
    env, observation = _reset(
        PerModelEnv(small_config(group_ids=[0, 0, 0, 2, 2, 2], max_groups=3))
    )
    tokens = _tokens(env, observation)
    assert list(tokens.opponent_unit_groups) == [0, 2]
    assert tokens.n_units == 2
    assert env.player_seat.n_enemy_units() == 3, "the facade's slice keeps the hole"
    seat = env.player_seat
    shooting = seat.handler.shooting_slice
    assert shooting is not None
    assert unit_column_to_value(2, StepKind.act, seat, tokens.opponent_unit_groups) == (
        shooting.start + 2
    )
    assert (
        unit_column_to_value(2, StepKind.target, seat, tokens.opponent_unit_groups) == 2
    )


# ---------------------------------------------------------- the reveal rule


def test_the_advance_roll_is_hidden_until_the_unit_declares() -> None:
    env = PerModelEnv(_with_advance(small_config()))
    observation, _ = env.reset(seed=2)
    point = observation.decision
    assert point.kind is StepKind.open and point.phase is BattlePhase.movement
    before = _tokens(env, observation)
    assert not np.any(before.players[:, MODEL_ADVANCE_ROLL])
    model = int(np.flatnonzero(point.selector_mask)[0])
    assert point.declaration_mask[model][MoveDeclaration.advance]
    from wargame_rl.wargame.envs.per_model import PerModelAction

    observation, *_ = env.step(PerModelAction.open(model, MoveDeclaration.advance))
    after = _tokens(env, observation)
    unit = env.wargame_models[model].group_id
    for index, wargame_model in enumerate(env.wargame_models):
        expected = (
            wargame_model.advance_roll / 6.0 if wargame_model.group_id == unit else 0.0
        )
        assert after.players[index, MODEL_ADVANCE_ROLL] == pytest.approx(expected)
    assert after.players[model, MODEL_ADVANCE_ROLL] > 0


# ------------------------------------------------- the masks are the env's


@pytest.mark.parametrize("melee", [False, True])
def test_the_token_masks_decode_to_exactly_the_envs_masks(melee: bool) -> None:
    """Every displacement and unit column, decoded, is legal iff the point
    says so -- on a gapped-id army, so a positional column would be caught."""
    config = small_config(
        melee=melee,
        opponent_x=22 if melee else 56,
        rounds=2,
        group_ids=[0, 0, 0, 2, 2, 2],
        max_groups=3,
    )
    env = PerModelEnv(config)
    seat = env.player_seat
    kinds_seen: set[tuple[StepKind, BattlePhase | None]] = set()
    for tokens, observation in _walk(
        env, seed=4, driver="charging" if melee else "random"
    ):
        point = observation.decision  # type: ignore[attr-defined]
        kinds_seen.add((point.kind, point.phase))
        assert np.array_equal(tokens.selector_mask, point.selector_mask)
        assert np.array_equal(tokens.declaration_mask, point.declaration_mask)
        for model in range(tokens.n_players):
            if tokens.head is Head.displacement:
                for column in range(tokens.displacement_mask.shape[1]):
                    action = displacement_to_action(column, seat)
                    assert (
                        tokens.displacement_mask[model, column]
                        == point.action_mask[model, action]
                    )
            elif tokens.head is Head.unit_pointer:
                for column in range(tokens.unit_mask.shape[1]):
                    value = unit_column_to_value(
                        column, point.kind, seat, tokens.opponent_unit_groups
                    )
                    if point.kind is StepKind.target:
                        expected = (
                            point.selector_mask[model]
                            if value == CHARGE_TARGET_DECLINE
                            else point.target_mask[model, value]
                        )
                    else:
                        expected = point.action_mask[model, value]
                    assert tokens.unit_mask[model, column] == expected
            else:
                assert not tokens.displacement_mask.any()
                assert not tokens.unit_mask.any()
    assert (StepKind.act, BattlePhase.movement) in kinds_seen
    if melee:
        assert (StepKind.target, BattlePhase.charge) in kinds_seen


def test_column_zero_is_stay_for_a_shooter_and_never_for_a_striker() -> None:
    env = PerModelEnv(small_config(melee=True, opponent_x=22, rounds=2))
    seen = {"shooting": False, "fight": False}
    for tokens, observation in _walk(env, seed=5, driver="charging"):
        point = observation.decision  # type: ignore[attr-defined]
        if point.kind is not StepKind.act or tokens.head is not Head.unit_pointer:
            continue
        selectable = np.flatnonzero(point.selector_mask)
        if point.phase is BattlePhase.shooting:
            seen["shooting"] = True
            assert tokens.unit_mask[selectable, 0].all()
        if point.phase is BattlePhase.fight:
            seen["fight"] = True
            assert not tokens.unit_mask[:, 0].any()
            assert tokens.unit_mask[selectable, 1:].any(axis=1).all()
    assert seen["shooting"] and seen["fight"], "the walk reached both kinds of act"


def test_a_target_step_decodes_column_zero_to_the_decline() -> None:
    env = PerModelEnv(small_config(melee=True, opponent_x=22, rounds=1))
    seat = env.player_seat
    groups = np.array([0, 1])
    assert (
        unit_column_to_value(0, StepKind.target, seat, groups) == CHARGE_TARGET_DECLINE
    )
    assert unit_column_to_value(0, StepKind.act, seat, groups) == STAY_ACTION
    assert displacement_to_action(0, seat) == STAY_ACTION


# ---------------------------------------------------------------- relations


def test_offsets_are_antisymmetric_and_present_on_every_positioned_pair() -> None:
    env, observation = _reset(PerModelEnv(small_config()))
    tokens = _tokens(env, observation)
    dx = tokens.self_relations[:, :, REL_DX]
    dy = tokens.self_relations[:, :, REL_DY]
    assert np.allclose(dx, -dx.T) and np.allclose(dy, -dy.T)
    assert tokens.self_relations[:, :, REL_OFFSET_PRESENT].all()
    present = tokens.cross_relations[:, :, REL_OFFSET_PRESENT]
    assert not present[:, 0].any(), "the game token has no position"
    positioned = tokens.context_kind != CONTEXT_GAME
    assert present[:, positioned].all()


def test_same_unit_is_the_group_id_on_models_and_on_own_unit_tokens() -> None:
    env, observation = _reset(PerModelEnv(small_config()))
    tokens = _tokens(env, observation)
    groups = tokens.player_groups
    assert np.array_equal(
        tokens.self_relations[:, :, REL_SAME_UNIT] > 0.5,
        groups[:, None] == groups[None, :],
    )
    own_units = np.flatnonzero(tokens.context_kind == CONTEXT_OWN_UNIT)
    for column, unit_row in enumerate(own_units):
        assert np.array_equal(
            tokens.cross_relations[:, unit_row, REL_SAME_UNIT] > 0.5, groups == column
        )


def test_sight_and_cover_are_the_envs_trace_within_reach_and_absent_beyond() -> None:
    # 8" apart across the middle of the board: the near pairs are within the
    # 12" reach, the far ones (a unit twenty inches up the board) are not.
    env, observation = _reset(
        PerModelEnv(_with_terrain(small_config(player_x=26, opponent_x=34)))
    )
    tokens = _tokens(env, observation)
    own = env.wargame_models
    enemies = env.opponent_models
    rows = np.flatnonzero(tokens.context_kind == CONTEXT_ENEMY_MODEL)
    own_locs = np.array([m.location for m in own], dtype=float)
    enemy_locs = np.array([m.location for m in enemies], dtype=float)
    full = env.visibility_between(
        own_locs, enemy_locs, None, origin_models=own, target_models=enemies, edges=True
    )
    distances = np.linalg.norm(own_locs[:, None] - enemy_locs[None, :], axis=2)
    reach = np.maximum(env.player_max_ranges[:, None], env.opponent_max_ranges[None, :])
    within = distances <= reach + 2.0 * env.rules_quantities.base_radius
    present = tokens.cross_relations[:, rows, REL_SIGHT_PRESENT] > 0.5
    assert np.array_equal(present, within)
    assert within.any() and not within.all(), "the geometry exercises both sides"
    visible = tokens.cross_relations[:, rows, REL_VISIBLE] > 0.5
    in_cover = tokens.cross_relations[:, rows, REL_IN_COVER] > 0.5
    assert np.array_equal(visible[within], (full >= COVER)[within])
    assert np.array_equal(in_cover[within], (full == COVER)[within])
    assert not visible[~within].any() and not in_cover[~within].any()


@pytest.mark.parametrize("melee", [False, True])
def test_expected_damage_relations_are_the_scalar_expectation_per_pair(
    melee: bool,
) -> None:
    config = small_config(melee=melee)
    env, observation = _reset(PerModelEnv(config))
    tokens = _tokens(env, observation)
    rows = np.flatnonzero(tokens.context_kind == CONTEXT_ENEMY_MODEL)
    assert config.models is not None and config.opponent_models is not None
    for p, own in enumerate(env.wargame_models):
        for o, enemy in enumerate(env.opponent_models):
            defender = _defender(enemy)
            out = expected_damage(config.models[p].weapons[0], defender)
            back = expected_damage(config.opponent_models[o].weapons[0], _defender(own))
            assert tokens.cross_relations[p, rows[o], REL_RANGED_OUT] == pytest.approx(
                out
            )
            assert tokens.cross_relations[p, rows[o], REL_RANGED_IN] == pytest.approx(
                back
            )
            melee_out = tokens.cross_relations[p, rows[o], REL_MELEE_OUT]
            if melee:
                weapon = config.models[p].melee_weapons[0]
                expected = _melee_expectation(weapon, defender)
                assert melee_out == pytest.approx(expected)
            else:
                assert melee_out == 0.0


def _defender(model: object) -> DefenderStats:
    stats = model.stats  # type: ignore[attr-defined]
    return DefenderStats(toughness=int(stats["toughness"]), save=int(stats["save"]))


def _melee_expectation(weapon: object, defender: DefenderStats) -> float:
    from wargame_rl.wargame.envs.domain.attacks.expectation import (
        expected_attack_damage,
    )

    return expected_attack_damage(
        int(weapon.melee_skill),  # type: ignore[attr-defined]
        weapon,  # type: ignore[arg-type]
        defender,
    )


def test_objective_counts_are_the_scoring_definition() -> None:
    env = PerModelEnv(small_config(player_x=14, opponent_x=46))
    observation, _ = env.reset(seed=1)
    tokens = _tokens(env, observation)
    rows = tokens.context[tokens.context_kind == CONTEXT_OBJECTIVE]
    own = compute_distances(
        env.wargame_models, env.objectives, alive_mask=env.player_seat.alive()
    )
    enemy_alive = np.array([m.is_alive for m in env.opponent_models])
    theirs = compute_distances(
        env.opponent_models, env.objectives, alive_mask=enemy_alive
    )
    own_counts = objective_counts_from_norms_offset(
        own.model_obj_norms_offset, own.obj_radii
    )
    their_counts = objective_counts_from_norms_offset(
        theirs.model_obj_norms_offset, theirs.obj_radii
    )
    assert own_counts.sum() > 0, "the armies deploy onto their home objectives"
    assert np.allclose(rows[:, 2] * 10.0, own_counts)
    assert np.allclose(rows[:, 3] * 10.0, their_counts)


def test_inside_terrain_is_the_footprints_own_answer() -> None:
    env, observation = _reset(PerModelEnv(_with_terrain(small_config())))
    tokens = _tokens(env, observation)
    rows = np.flatnonzero(tokens.context_kind == CONTEXT_TERRAIN)
    assert rows.size == 3
    assert tokens.cross_relations[:, rows, REL_TERRAIN_PRESENT].all()
    for p, model in enumerate(env.wargame_models):
        x, y = (float(v) for v in model.location)
        for t, footprint in enumerate(env.terrain.footprints):
            assert bool(tokens.cross_relations[p, rows[t], REL_INSIDE] > 0.5) == (
                footprint.contains(x, y)
            )


# ------------------------------------------------------------ dead and cache


def test_a_dead_models_row_is_present_and_masked() -> None:
    env, observation = _reset(PerModelEnv(small_config()))
    victim = env.wargame_models[1]
    victim.stats["current_wounds"] = 0
    tokens = _tokens(env, observation)
    assert tokens.players.shape[0] == 6
    assert tokens.players[1, MODEL_ALIVE] == 0.0 and not tokens.player_alive[1]
    enemy = env.opponent_models[3]
    enemy.stats["current_wounds"] = 0
    tokens = _tokens(env, observation)
    enemy_rows = np.flatnonzero(tokens.context_kind == CONTEXT_ENEMY_MODEL)
    assert not tokens.context_alive[enemy_rows[3]]
    assert tokens.context_alive[enemy_rows[0]]


def test_a_stale_scenario_is_refused() -> None:
    env = PerModelEnv(small_config())
    observation, _ = env.reset(seed=1)
    scenario = TokenScenario.for_episode(env, env.player_seat)
    observation, _ = env.reset(seed=2)
    with pytest.raises(ValueError, match="episode"):
        build_tokens(env, env.player_seat, observation, scenario)


def test_the_opponent_seats_tokens_are_the_mirror() -> None:
    """Built for the other seat, its models are the queries and ours the context."""
    env, observation = _reset(PerModelEnv(small_config()))
    ours = _tokens(env, observation)
    seat = env.opponent_seat
    theirs = build_tokens(env, seat, observation, TokenScenario.for_episode(env, seat))  # type: ignore[arg-type]
    their_x = (np.array([m.location for m in env.opponent_models])[:, 0] - 30.0) / 30.0
    assert np.allclose(theirs.players[:, MODEL_X], their_x, atol=1e-6)
    our_rows = theirs.context[theirs.context_kind == CONTEXT_ENEMY_MODEL]
    assert np.allclose(our_rows[:, MODEL_X], ours.players[:, MODEL_X], atol=1e-6)
    assert theirs.n_units == ours.n_units
