"""Terrain as outlines, and objectives that *are* terrain.

The second is a rules change, not a reskin: an area objective is the ground
itself, so control is standing inside an outline rather than within a distance
of a point. It reaches the rest of the environment through the `norms_offset`
seam — give the area a radius of 0 and report distance to its edge — so no
reward, VP or criteria code learns that objectives have shapes.
"""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from wargame_rl.wargame.envs.domain.terrain_placement import generate_terrain
from wargame_rl.wargame.envs.domain.value_objects import BoardDimensions
from wargame_rl.wargame.envs.env_components.distance_cache import compute_distances
from wargame_rl.wargame.envs.types.config import (
    ObjectiveConfig,
    RandomTerrainConfig,
    TerrainPieceConfig,
    WargameEnvConfig,
)
from wargame_rl.wargame.envs.wargame import WargameEnv

BOARD = BoardDimensions(width=60, height=44)

# A square from (10, 10) to (16, 16), as an explicit outline.
SQUARE_AREA = [(10.0, 10.0), (16.0, 10.0), (16.0, 16.0), (10.0, 16.0)]


class TestTerrainPieceConfig:
    def test_a_piece_is_a_rectangle_or_an_outline_never_both(self) -> None:
        """Two conventions in one field is how a layout comes out a unit small."""
        with pytest.raises(ValidationError, match="exactly one"):
            TerrainPieceConfig(footprint=(1, 1, 3, 3), outline=SQUARE_AREA)
        with pytest.raises(ValidationError, match="exactly one"):
            TerrainPieceConfig()

    def test_the_two_forms_use_different_conventions(self) -> None:
        """A rectangle is inclusive *cells*; an outline is taken literally."""
        from_cells = TerrainPieceConfig(footprint=(10, 10, 15, 15)).to_polygon()
        from_outline = TerrainPieceConfig(outline=SQUARE_AREA).to_polygon()

        assert from_cells.bounds == (10.0, 10.0, 16.0, 16.0)
        assert from_outline.bounds == (10.0, 10.0, 16.0, 16.0)

    def test_a_concave_outline_loads(self) -> None:
        """Nothing in the geometry assumes convexity, and an L is the point."""
        ell = TerrainPieceConfig(
            outline=[
                (0.0, 0.0),
                (6.0, 0.0),
                (6.0, 2.0),
                (2.0, 2.0),
                (2.0, 6.0),
                (0.0, 6.0),
            ]
        ).to_polygon()

        assert ell.contains(1.0, 5.0) is True
        assert ell.contains(5.0, 5.0) is False

    def test_validation_runs_on_the_resolved_shape(self) -> None:
        """A rectangle and an outline are held to the same in-bounds rule."""
        with pytest.raises(ValidationError, match="outside the board"):
            WargameEnvConfig(
                board_width=10,
                board_height=10,
                terrain=[
                    TerrainPieceConfig(outline=[(8.0, 8.0), (12.0, 8.0), (12.0, 9.0)])
                ],
            )


class TestPolygonGeneration:
    def _layout(self, seed: int, **overrides: object) -> list:
        spec = RandomTerrainConfig(
            count=7,
            min_size=3,
            max_size=6,
            mirror=True,
            edge_margin=2,
            min_gap=1,
            n_vertices=6,
            **overrides,  # type: ignore[arg-type]
        )
        return generate_terrain(spec, BOARD, np.random.default_rng(seed)).footprints

    def test_the_default_is_still_rectangles(self) -> None:
        """Every terrain profile in the repo was tuned against rectangles.

        Turning outlines on has to be a deliberate config change, or a shipped
        profile silently starts hiding less board than it was measured at.
        """
        spec = RandomTerrainConfig(count=7, min_size=3, max_size=6)
        pieces = generate_terrain(spec, BOARD, np.random.default_rng(0)).footprints

        assert all(piece.n_vertices == 4 for piece in pieces)

    @pytest.mark.parametrize("seed", range(6))
    def test_generated_outlines_carry_the_configured_vertex_count(
        self, seed: int
    ) -> None:
        assert all(piece.n_vertices == 6 for piece in self._layout(seed))

    @pytest.mark.parametrize("seed", range(6))
    def test_a_mirrored_layout_is_symmetric_shape_for_shape(self, seed: int) -> None:
        """A mirrored *pair* is one outline and its reflection, not two draws.

        Inscribing twice would put two different shapes on ground that is meant
        to be equal, which is the exact bias `mirror` exists to remove — and it
        would be invisible in a bounding-box comparison, since both shapes sit in
        reflected boxes either way.
        """
        pieces = self._layout(seed)
        reflected = sorted(
            tuple(np.round(BOARD.width - p.polygon.vertices[:, 0], 6))
            + tuple(np.round(p.polygon.vertices[:, 1], 6))
            for p in pieces
        )
        original = sorted(
            tuple(np.round(p.polygon.vertices[:, 0], 6))
            + tuple(np.round(p.polygon.vertices[:, 1], 6))
            for p in pieces
        )
        # Compare as multisets of coordinates: the reflected layout must be the
        # same set of shapes, though each one's winding is reversed.
        assert sorted(sorted(r) for r in reflected) == sorted(
            sorted(o) for o in original
        )

    @pytest.mark.parametrize("seed", range(6))
    def test_the_odd_centre_piece_is_its_own_mirror(self, seed: int) -> None:
        """An odd count leaves one piece unpaired, so it has to be symmetric.

        Built by drawing half the vertices and reflecting them, rather than by
        hulling a shape with its own reflection — which would double the vertex
        count and overflow the observation's outline budget.
        """
        pieces = self._layout(seed)
        centre = min(
            pieces, key=lambda p: abs(float(p.polygon.centroid[0]) - BOARD.width / 2)
        )
        mirrored = centre.polygon.mirrored(float(BOARD.width))

        assert centre.n_vertices == mirrored.n_vertices
        assert sorted(np.round(centre.polygon.vertices[:, 0], 6)) == pytest.approx(
            sorted(np.round(mirrored.vertices[:, 0], 6))
        )


class TestAreaObjectives:
    def test_an_area_is_not_also_a_disc(self) -> None:
        """Two definitions of "in range" and no way to tell them apart."""
        with pytest.raises(ValidationError, match="drop x and y"):
            ObjectiveConfig(area=SQUARE_AREA, x=5, y=5)
        with pytest.raises(ValidationError, match="no radius"):
            ObjectiveConfig(area=SQUARE_AREA, radius_size=3)

    def test_an_area_objective_reports_its_centroid_as_its_location(self) -> None:
        """Anything steering toward an objective still needs a point to aim at."""
        env = WargameEnv(
            config=WargameEnvConfig(
                board_width=30,
                board_height=30,
                number_of_wargame_models=2,
                number_of_objectives=1,
                objectives=[ObjectiveConfig(area=SQUARE_AREA)],
            )
        )
        env.reset(seed=0)
        objective = env.objectives[0]

        assert objective.is_area is True
        assert objective.radius_size == 0.0
        assert objective.location == pytest.approx([13.0, 13.0])

    def test_range_is_measured_to_the_edge_so_downstream_tests_are_unchanged(
        self,
    ) -> None:
        """The `norms_offset` seam, which is the whole trick.

        Radius 0 plus distance-to-edge means `norms_offset <= obj_radii` still
        reads as "close enough to count" everywhere it appears — in the reward,
        in VP scoring, in the success criteria — with no branch anywhere.
        """
        env = WargameEnv(
            config=WargameEnvConfig(
                board_width=30,
                board_height=30,
                number_of_wargame_models=3,
                number_of_objectives=1,
                objectives=[ObjectiveConfig(area=SQUARE_AREA)],
            )
        )
        env.reset(seed=0)
        env.wargame_models[0].location = np.array([13.0, 13.0])  # inside
        env.wargame_models[1].location = np.array([16.0, 13.0])  # exactly on edge
        env.wargame_models[2].location = np.array([20.0, 13.0])  # 4 outside

        cache = compute_distances(env.wargame_models, env.objectives)

        assert cache.model_obj_norms_offset[:, 0] == pytest.approx([0.0, 0.0, 4.0])
        at_objective = cache.model_obj_norms_offset <= cache.obj_radii
        assert at_objective[:, 0].tolist() == [True, True, False]

    def test_a_marker_objective_is_untouched_by_any_of_this(self) -> None:
        """The common case must not pay for the new one, or change under it."""
        env = WargameEnv(
            config=WargameEnvConfig(
                board_width=30,
                board_height=30,
                number_of_wargame_models=2,
                number_of_objectives=1,
                objectives=[ObjectiveConfig(x=15, y=15, radius_size=3)],
            )
        )
        env.reset(seed=0)
        env.wargame_models[0].location = np.array([19.0, 15.0])

        cache = compute_distances(env.wargame_models, env.objectives)

        assert env.objectives[0].is_area is False
        assert cache.model_obj_norms_offset[0, 0] == pytest.approx(4.0)


class TestObjectivesOnTerrain:
    def _env(self, **overrides: object) -> WargameEnv:
        config: dict[str, object] = {
            "board_width": 60,
            "board_height": 44,
            "number_of_wargame_models": 4,
            "number_of_objectives": 3,
            "deployment_zone": (0, 0, 20, 44),
            "opponent_deployment_zone": (40, 0, 60, 44),
            "objectives_on_terrain": True,
            # Dense enough that the centre band reliably holds three pieces.
            # A sparser profile makes these tests fail on the occasional seed
            # for a legitimate reason -- see the loud-failure test below.
            "random_terrain": RandomTerrainConfig(
                count=29,
                min_size=3,
                max_size=5,
                mirror=True,
                edge_margin=2,
                min_gap=1,
                n_vertices=6,
            ),
        }
        config.update(overrides)
        return WargameEnv(config=WargameEnvConfig(**config))  # type: ignore[arg-type]

    @pytest.mark.parametrize("seed", range(6))
    def test_every_objective_is_one_of_the_terrain_pieces(self, seed: int) -> None:
        env = self._env()
        env.reset(seed=seed)

        outlines = [
            piece.polygon.vertices.tobytes() for piece in env.terrain.footprints
        ]
        for objective in env.objectives:
            assert objective.is_area
            assert objective.area is not None
            assert objective.area.vertices.tobytes() in outlines

    @pytest.mark.parametrize("seed", range(6))
    def test_objectives_are_clear_of_both_deployment_zones(self, seed: int) -> None:
        """Otherwise one side starts standing on the prize."""
        env = self._env()
        env.reset(seed=seed)

        for objective in env.objectives:
            assert objective.area is not None
            x0, _, x1, _ = objective.area.bounds
            assert x0 >= 20.0
            assert x1 <= 40.0

    @pytest.mark.parametrize("seed", range(6))
    def test_the_chosen_set_is_symmetric_on_a_mirrored_layout(self, seed: int) -> None:
        """Chosen by distance to the board centre, not at random.

        A mirrored pair sits at equal distance, so they are taken together and
        neither side gets the closer prize. Picking randomly would hand one side
        an advantage on some seeds and it would never show up in an aggregate.
        """
        env = self._env()
        env.reset(seed=seed)

        centres = sorted(float(o.location[0]) for o in env.objectives)
        reflected = sorted(60.0 - x for x in centres)
        assert centres == pytest.approx(reflected, abs=1e-6)

    def test_too_few_eligible_pieces_fails_loudly(self) -> None:
        """Reusing one piece for two objectives would quietly shrink the mission."""
        env = self._env(
            number_of_objectives=8,
            random_terrain=RandomTerrainConfig(
                count=4, min_size=3, max_size=4, mirror=True, edge_margin=2, min_gap=1
            ),
        )

        with pytest.raises(ValueError, match="terrain pieces clear"):
            env.reset(seed=0)


class TestSnapshotRoundTrip:
    """An area objective's outline is episode state, not configuration.

    With `objectives_on_terrain` the outline comes from the layout, so it varies
    every episode. Reconstructing a board from `location` alone would put a
    marker where the rules have a shape and score control against a radius of 0
    — a replay that looks right and scores wrong.
    """

    def _env(self) -> WargameEnv:
        return WargameEnv(
            config=WargameEnvConfig(
                board_width=60,
                board_height=44,
                number_of_wargame_models=4,
                number_of_objectives=3,
                number_of_battle_rounds=3,
                base_radius=0.63,
                deployment_zone=(0, 0, 20, 44),
                opponent_deployment_zone=(40, 0, 60, 44),
                objectives_on_terrain=True,
                random_terrain=RandomTerrainConfig(
                    count=29,
                    min_size=3,
                    max_size=5,
                    mirror=True,
                    edge_margin=2,
                    min_gap=1,
                    n_vertices=6,
                ),
            )
        )

    def test_an_area_objective_survives_a_snapshot_round_trip(self) -> None:
        env = self._env()
        env.reset(seed=4)
        before = [o.area.vertices.copy() for o in env.objectives if o.area is not None]
        assert len(before) == 3

        snapshot = env.to_snapshot()
        env.reset(seed=9)  # a different layout, so a stale outline would show
        env.load_state(snapshot)

        after = [o.area.vertices for o in env.objectives if o.area is not None]
        assert len(after) == 3
        for original, restored in zip(before, after):
            assert np.allclose(original, restored)

    def test_a_model_base_is_recorded(self) -> None:
        """The radius decides control range and collision, so a replay needs it."""
        env = self._env()
        env.reset(seed=4)

        snapshot = env.to_snapshot()

        assert snapshot.player_models[0].base_radius == pytest.approx(0.63)


class TestLayoutSurvivesTheTail:
    """A layout constraint has to hold on every reset, not on average.

    This is a regression test with a price attached. The shipped profile was
    checked over 200 layouts and had a *minimum* of five eligible pieces, which
    read as ample margin — but a training run resets tens of thousands of times,
    so a draw with a per-episode probability well under 1% is a certainty. It
    killed a 619-epoch run.

    Terrain is now redrawn until it can host the objectives, so what this pins
    is that a long run of resets never raises.
    """

    def test_thousands_of_resets_never_fail_to_place_objectives(self) -> None:
        config = WargameEnvConfig(
            board_width=60,
            board_height=44,
            number_of_wargame_models=2,
            number_of_objectives=3,
            number_of_battle_rounds=1,
            deployment_zone=(0, 0, 20, 44),
            opponent_deployment_zone=(40, 0, 60, 44),
            objectives_on_terrain=True,
            random_terrain=RandomTerrainConfig(
                count=37,
                min_size=3,
                max_size=6,
                mirror=True,
                edge_margin=2,
                min_gap=1,
                n_vertices=6,
            ),
        )
        env = WargameEnv(config=config)

        for seed in range(2000):
            env.reset(seed=seed)
            assert all(objective.is_area for objective in env.objectives)


def test_spread_selection_beats_centre_selection_on_separation() -> None:
    """`objectives_spread_on_terrain` picks the furthest-apart eligible pieces.

    Nearest-to-centre packs every objective into the middle of the board, which
    removes the travel trade-off between them. Pinned on the *outcome* -- the
    minimum pairwise separation -- rather than on which pieces are chosen, since
    the selection rule may change but "further apart" is the contract.
    """
    import itertools

    from pydantic_yaml import parse_yaml_raw_as

    from wargame_rl.wargame.envs.types.config import WargameEnvConfig
    from wargame_rl.wargame.model.common.factory import create_environment

    with open("configs/experiments/25v25_polygon_terrain_objectives.yaml") as handle:
        raw = handle.read()

    def min_separations(config_text: str) -> list[float]:
        config = parse_yaml_raw_as(WargameEnvConfig, config_text)
        env = create_environment(env_config=config)
        out = []
        for seed in range(700_000, 700_030):
            env.reset(seed=seed)
            centres = [np.asarray(o.location, dtype=float) for o in env.objectives]
            out.append(
                min(
                    float(np.linalg.norm(a - b))
                    for a, b in itertools.combinations(centres, 2)
                )
            )
        return out

    centre = min_separations(raw)
    spread = min_separations(
        # Anchored on the newline so the commented example of the same key
        # further up the file is not also rewritten into a duplicate key.
        raw.replace(
            "\nobjectives_on_terrain: true",
            "\nobjectives_on_terrain: true\nobjectives_spread_on_terrain: true",
        )
    )

    assert sum(spread) / len(spread) > sum(centre) / len(centre)
    # Never worse on any individual layout: the spread set is chosen by maximising
    # exactly this quantity, so a regression here means the flag is not wired.
    assert all(s >= c - 1e-9 for s, c in zip(spread, centre))


def test_spread_selection_keeps_objectives_mirror_symmetric() -> None:
    """Spread selection must not hand one side an extra objective.

    Maximising separation *unconstrained* looks fair -- a mirrored layout's
    mirrored sets score identically -- but that says the mirror image scores the
    same, not that the winner is its own mirror image. Two pieces on one side and
    one on the other can out-separate any balanced set, and did: 38 of 200
    layouts came out 2-1, up to 3.67" off centre.
    """
    from pydantic_yaml import parse_yaml_raw_as

    from wargame_rl.wargame.envs.types.config import WargameEnvConfig
    from wargame_rl.wargame.model.common.factory import create_environment

    with open("configs/experiments/25v25_spread_objectives.yaml") as handle:
        config = parse_yaml_raw_as(WargameEnvConfig, handle.read())
    assert config.objectives_spread_on_terrain
    env = create_environment(env_config=config)
    middle = config.board_width / 2.0

    for seed in range(700_000, 700_060):
        env.reset(seed=seed)
        xs = [float(np.asarray(o.location, dtype=float)[0]) for o in env.objectives]
        left = sum(1 for x in xs if x < middle - 1e-6)
        right = sum(1 for x in xs if x > middle + 1e-6)
        assert left == right, f"seed {seed}: {left} left vs {right} right ({xs})"
        # The set reflects onto itself, so the mean sits exactly on the centre.
        assert abs(sum(xs) / len(xs) - middle) < 1e-6, f"seed {seed}: {xs}"
