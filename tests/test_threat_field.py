"""The threat field is a NEXT-turn quantity, and that is the whole design.

Three assertions carry this file.

**Equivalence.** At `ThreatHorizon.current` the field reproduces the shipped
`compute_threat_region` mask cell for cell. That pins the new sweep against
already-verified, pixel-asserted behaviour and is the only reason the `current`
horizon exists at all.

**Superset.** `next_turn` can only ever add threatened ground to `current`.
Moving before shooting cannot un-threaten a cell, so a single counterexample is
a bug in the reachable-origin set rather than a surprising board.

**The wound clip**, which every shipped config makes a no-op and which would
otherwise ship a threefold overstatement the first time a heavy weapon appears.
"""

from __future__ import annotations

import numpy as np
import pytest

from wargame_rl.wargame.envs.board.grid import board_grid, board_grid_for
from wargame_rl.wargame.envs.board.threat import (
    ReferenceModel,
    ThreatHorizon,
    VisibilityCache,
    attacker_stat_rows,
    move_reach,
    reachable_cells,
    reference_model,
    threat_field,
)
from wargame_rl.wargame.envs.domain.entities import alive_mask_for
from wargame_rl.wargame.envs.domain.shooting import DefenderStats
from wargame_rl.wargame.envs.env_components.actions import ActionHandler
from wargame_rl.wargame.envs.renders.v2.control import compute_threat_region
from wargame_rl.wargame.envs.types.config.battle import OpponentPolicyConfig
from wargame_rl.wargame.envs.types.config.entities import ModelConfig, WeaponProfile
from wargame_rl.wargame.envs.types.config.env import (
    RandomTerrainConfig,
    WargameEnvConfig,
)
from wargame_rl.wargame.envs.wargame import WargameEnv

# Coarse on purpose: these are correctness assertions, and the shipped 1" grid
# costs 16x as much for the same answers.
SPACING = 2.0


def _rifle(**overrides: int) -> WeaponProfile:
    stats = {"range": 12, "attacks": 1, "ballistic_skill": 3, "strength": 4, "ap": 1}
    return WeaponProfile(**{**stats, **overrides})


@pytest.fixture
def board() -> WargameEnv:
    """A small armed board with terrain.

    The default config carries **no opponent models and no weapons**, which
    would make every threat field trivially empty and every assertion here pass
    for the wrong reason. Terrain is required too: with none, sight is never
    blocked and `next_turn` could not differ from `current` by walking around
    anything.
    """
    env = WargameEnv(
        config=WargameEnvConfig(
            render_mode=None,
            number_of_battle_rounds=100,
            board_width=30,
            board_height=30,
            number_of_wargame_models=4,
            number_of_opponent_models=4,
            models=[ModelConfig(group_id=0, weapons=[_rifle()]) for _ in range(4)],
            opponent_models=[
                ModelConfig(group_id=0, weapons=[_rifle()]) for _ in range(4)
            ],
            random_terrain=RandomTerrainConfig(count=6, min_size=3, max_size=5),
            opponent_policy=OpponentPolicyConfig(type="random"),
        )
    )
    env.reset(seed=4242)
    return env


def _shooters(env: WargameEnv) -> tuple[list, np.ndarray, np.ndarray, ReferenceModel]:
    """The opponent as the firing side, as every caller here uses it."""
    models = env.opponent_models
    return (
        models,
        np.asarray(env.opponent_max_ranges),
        attacker_stat_rows(env.config.opponent_models, len(models)),
        reference_model(env.player_models, env.config.models),
    )


def _current_mask(env: WargameEnv) -> np.ndarray:
    """The boolean grid `compute_threat_region` rasterises into rings.

    Reconstructed here rather than read off the rings, because ring extraction
    and Chaikin smoothing are lossy on purpose -- comparing against the drawing
    would be comparing against a picture of the answer.
    """
    grid = board_grid_for(env, SPACING)
    models = env.opponent_models
    alive = alive_mask_for(models)
    origins = np.array(
        [[float(m.location[0]), float(m.location[1])] for m in models], dtype=float
    )[alive]
    ranges = np.asarray(env.opponent_max_ranges, dtype=float)[alive]
    distances = np.linalg.norm(
        origins[:, np.newaxis, :] - grid.centres[np.newaxis, :, :], axis=2
    )
    candidates = (distances <= ranges[:, np.newaxis]) & (ranges > 0)[:, np.newaxis]
    visible = np.asarray(env.line_of_sight_matrix(origins, grid.centres, candidates))
    return np.asarray((candidates & visible).any(axis=0))


class TestTheCurrentHorizonMatchesTheShippedOverlay:
    def test_the_threatened_cells_are_the_overlays_own(self, board: WargameEnv) -> None:
        """Arrange a live board, act by sweeping both ways, assert cell equality."""
        models, ranges, stats, reference = _shooters(board)

        field = threat_field(
            board,
            models,
            ranges,
            stats,
            reference,
            horizon=ThreatHorizon.current,
            spacing=SPACING,
        )

        assert np.array_equal(field.casualties > 0.0, _current_mask(board))

    def test_the_overlay_still_draws_something_for_that_mask(
        self, board: WargameEnv
    ) -> None:
        """Guards the equivalence above from passing on an all-clear board."""
        rings = compute_threat_region(
            board,
            board.opponent_models,
            np.asarray(board.opponent_max_ranges),
            spacing=SPACING,
            smooth=0,
        )

        assert rings
        assert _current_mask(board).any()


class TestTheNextTurnHorizon:
    def test_it_is_a_superset_of_the_current_turn(self, board: WargameEnv) -> None:
        """Moving first adds threatened ground and can never remove any."""
        models, ranges, stats, reference = _shooters(board)
        cache = VisibilityCache.build(
            board, spacing=SPACING, max_range=float(ranges.max())
        )
        moves = move_reach(board.config, board.config.opponent_models, len(models))

        now = threat_field(
            board,
            models,
            ranges,
            stats,
            reference,
            horizon=ThreatHorizon.current,
            spacing=SPACING,
        )
        later = threat_field(
            board,
            models,
            ranges,
            stats,
            reference,
            horizon=ThreatHorizon.next_turn,
            move=moves,
            spacing=SPACING,
            visibility=cache,
        )

        assert not (~(later.casualties > 0.0) & (now.casualties > 0.0)).any()

    def test_it_finds_ground_the_current_turn_map_calls_safe(
        self, board: WargameEnv
    ) -> None:
        """The false-safe cells are the reason this module exists.

        A `next_turn` field identical to `current` would mean the two-hop never
        fired -- which is exactly what a silently-skipped reachable half looks
        like, and is why this asserts strict growth rather than containment.
        """
        models, ranges, stats, reference = _shooters(board)
        cache = VisibilityCache.build(
            board, spacing=SPACING, max_range=float(ranges.max())
        )
        moves = move_reach(board.config, board.config.opponent_models, len(models))

        now = threat_field(
            board,
            models,
            ranges,
            stats,
            reference,
            horizon=ThreatHorizon.current,
            spacing=SPACING,
        )
        later = threat_field(
            board,
            models,
            ranges,
            stats,
            reference,
            horizon=ThreatHorizon.next_turn,
            move=moves,
            spacing=SPACING,
            visibility=cache,
        )

        assert (later.casualties > 0.0).sum() > (now.casualties > 0.0).sum()

    def test_it_refuses_to_run_without_a_cache_rather_than_downgrading(
        self, board: WargameEnv
    ) -> None:
        """Falling back to current-turn sight would answer a different question silently."""
        models, ranges, stats, reference = _shooters(board)
        moves = move_reach(board.config, board.config.opponent_models, len(models))

        with pytest.raises(ValueError, match="needs a VisibilityCache"):
            threat_field(
                board,
                models,
                ranges,
                stats,
                reference,
                horizon=ThreatHorizon.next_turn,
                move=moves,
                spacing=SPACING,
            )

    def test_a_cache_gated_below_the_longest_weapon_is_refused(
        self, board: WargameEnv
    ) -> None:
        """An under-gated cache reads False for pairs it simply never traced."""
        models, ranges, stats, reference = _shooters(board)
        moves = move_reach(board.config, board.config.opponent_models, len(models))
        short = VisibilityCache.build(board, spacing=SPACING, max_range=1.0)

        with pytest.raises(ValueError, match="rebuild it at the longer range"):
            threat_field(
                board,
                models,
                ranges,
                stats,
                reference,
                horizon=ThreatHorizon.next_turn,
                move=moves,
                spacing=SPACING,
                visibility=short,
            )

    def test_a_cache_built_at_another_spacing_is_refused(
        self, board: WargameEnv
    ) -> None:
        """Mismatched grids would index one board's cells into another's."""
        models, ranges, stats, reference = _shooters(board)
        moves = move_reach(board.config, board.config.opponent_models, len(models))
        other = VisibilityCache.build(
            board, spacing=SPACING * 2, max_range=float(ranges.max())
        )

        with pytest.raises(ValueError, match="both must be built at the same spacing"):
            threat_field(
                board,
                models,
                ranges,
                stats,
                reference,
                horizon=ThreatHorizon.next_turn,
                move=moves,
                spacing=SPACING,
                visibility=other,
            )


class TestTheScalar:
    def test_it_is_casualties_and_the_wound_clip_binds(self, board: WargameEnv) -> None:
        """Damage 3 against Wounds 1 removes one model per hit, not three.

        No shipped config exercises this -- every one is `damage: 1` against
        `max_wounds: 1` -- so it is asserted directly on the stat rows.
        """
        models, ranges, _stats, _reference = _shooters(board)
        heavy = attacker_stat_rows(
            [
                ModelConfig(group_id=0, weapons=[WeaponProfile(range=12, damage=3)])
                for _ in models
            ],
            len(models),
        )
        one_wound = ReferenceModel(DefenderStats(toughness=3, save=4), max_wounds=1)

        field = threat_field(
            board,
            models,
            ranges,
            heavy,
            one_wound,
            horizon=ThreatHorizon.current,
            spacing=SPACING,
        )

        threatened = field.casualties > 0.0
        assert np.allclose(field.wounds[threatened], 3.0 * field.casualties[threatened])

    def test_the_clip_is_a_no_op_on_the_profiles_every_config_ships(
        self, board: WargameEnv
    ) -> None:
        models, ranges, stats, reference = _shooters(board)

        field = threat_field(
            board,
            models,
            ranges,
            stats,
            reference,
            horizon=ThreatHorizon.current,
            spacing=SPACING,
        )

        assert reference.max_wounds == 1
        assert np.allclose(field.wounds, field.casualties)

    def test_an_unarmed_side_threatens_nothing(self, board: WargameEnv) -> None:
        """`range > 0` is load-bearing: `0 <= 0` would threaten the cell it stands on."""
        models, _ranges, stats, reference = _shooters(board)

        field = threat_field(
            board,
            models,
            np.zeros(len(models)),
            stats,
            reference,
            horizon=ThreatHorizon.current,
            spacing=SPACING,
        )

        assert not field.casualties.any()
        assert not field.shooter_count.any()

    def test_shooter_count_and_casualties_agree_on_which_cells_bear(
        self, board: WargameEnv
    ) -> None:
        models, ranges, stats, reference = _shooters(board)

        field = threat_field(
            board,
            models,
            ranges,
            stats,
            reference,
            horizon=ThreatHorizon.current,
            spacing=SPACING,
        )

        assert np.array_equal(field.casualties > 0.0, field.shooter_count > 0)

    def test_at_reads_the_field_at_a_models_own_position(
        self, board: WargameEnv
    ) -> None:
        models, ranges, stats, reference = _shooters(board)
        field = threat_field(
            board,
            models,
            ranges,
            stats,
            reference,
            horizon=ThreatHorizon.current,
            spacing=SPACING,
        )
        positions = np.array(
            [[float(m.location[0]), float(m.location[1])] for m in board.player_models]
        )

        values = field.at(positions)

        assert values.shape == (len(positions),)
        assert (values >= 0.0).all()


class TestBands:
    def test_they_are_disjoint(self) -> None:
        """Nested level sets would double their alpha wherever they overlap."""
        from wargame_rl.wargame.envs.board.grid import board_grid
        from wargame_rl.wargame.envs.board.threat import ThreatField

        grid = board_grid(10, 10, 1.0)
        values = np.linspace(0.0, 5.0, grid.n_cells, dtype=np.float32)
        field = ThreatField(
            grid=grid,
            casualties=values,
            wounds=values,
            shooter_count=(values > 0).astype(np.int32),
            horizon=ThreatHorizon.current,
            reference=ReferenceModel(DefenderStats(toughness=3, save=4), 1),
            reach=np.zeros(0),
        )

        bands = field.bands([0.5, 0.85])

        stacked: np.ndarray = sum(band.astype(int) for band in bands)  # type: ignore[assignment]
        assert stacked.max() <= 1
        # Bands cover the THREATENED cells and nothing else: safe ground belongs
        # to no band, so the renderer leaves it unpainted rather than washing the
        # whole table in the lowest colour.
        assert sum(int(band.sum()) for band in bands) == int((values > 0.0).sum())
        assert not any(band[values == 0.0].any() for band in bands)

    def test_an_untouched_board_yields_empty_bands_rather_than_raising(self) -> None:
        from wargame_rl.wargame.envs.board.grid import board_grid
        from wargame_rl.wargame.envs.board.threat import ThreatField

        grid = board_grid(4, 4, 1.0)
        field = ThreatField(
            grid=grid,
            casualties=np.zeros(grid.n_cells, dtype=np.float32),
            wounds=np.zeros(grid.n_cells, dtype=np.float32),
            shooter_count=np.zeros(grid.n_cells, dtype=np.int32),
            horizon=ThreatHorizon.next_turn,
            reference=ReferenceModel(DefenderStats(toughness=3, save=4), 1),
            reach=np.zeros(0),
        )

        assert not any(band.any() for band in field.bands([0.5]))


class TestTheOriginSet:
    """`reachable_cells` is the seam a charge layer plugs into.

    It is tested on its own because both threats start with the same question --
    where can the opponent be -- and only then diverge on what it can do from
    there. A charge shares this and must NOT share the visibility cache.
    """

    def test_a_stationary_model_reaches_nothing(self) -> None:
        """Move 0 is an empty origin set, not the cell it happens to stand on.

        The caller unions the exact position in separately, so returning a cell
        here would double-count it -- and a model that cannot move has no *new*
        firing position, which is the thing this set exists to enumerate.
        """
        grid = board_grid(20, 20, 1.0)

        assert reachable_cells(grid, np.array([10.0, 10.0]), 0.0).size == 0

    def test_it_grows_monotonically_with_the_move(self) -> None:
        grid = board_grid(40, 40, 1.0)

        def at(move: float) -> set[int]:
            return set(reachable_cells(grid, np.array([20.0, 20.0]), move).tolist())

        near, far = at(3.0), at(6.0)

        assert near < far

    def test_every_cell_returned_is_within_the_move(self) -> None:
        """The set is the contract; a cell outside it would invent a shot."""
        grid = board_grid(40, 40, 1.0)
        origin = np.array([12.0, 17.0])

        cells = reachable_cells(grid, origin, 5.0)

        assert cells.size > 0
        assert (
            np.linalg.norm(grid.centres[cells] - origin, axis=1) <= 5.0 + 1e-9
        ).all()

    def test_a_disc_of_radius_six_is_about_its_area(self) -> None:
        """~pi r^2 cells at 1 inch spacing -- catches an off-by-one on the radius."""
        grid = board_grid(60, 44, 1.0)

        cells = reachable_cells(grid, np.array([30.0, 22.0]), 6.0)

        assert 0.8 * np.pi * 36 < cells.size < 1.2 * np.pi * 36


def test_move_reach_agrees_with_the_action_handler(board: WargameEnv) -> None:
    """`ActionHandler` is the authority on how far a model moves.

    `board/` is a leaf and cannot import `env_components/`, so `move_reach`
    re-derives the resolution rather than reading `move_speeds`. This is what
    stops the two drifting -- the analysis layer must not price a move the
    action space cannot order.
    """
    config = board.config
    handler = ActionHandler(
        config,
        n_models=len(board.player_models),
        model_moves=[m.move for m in (config.models or ())] or None,
    )

    reach = move_reach(config, config.models, len(board.player_models))

    assert np.allclose(reach, handler.move_speeds)


def test_a_per_model_move_override_is_honoured() -> None:
    """The horde config's Move 12 must not silently read as the scenario's 6."""
    config = WargameEnvConfig(
        render_mode=None,
        number_of_wargame_models=2,
        models=[
            ModelConfig(group_id=0, move=12.0, weapons=[WeaponProfile(range=12)]),
            ModelConfig(group_id=0, weapons=[WeaponProfile(range=12)]),
        ],
    )

    reach = move_reach(config, config.models, 2)

    assert reach[0] == pytest.approx(12.0)
    assert reach[1] == pytest.approx(config.max_move_speed)
