"""Arrival order, in whole rounds, and which zone the ground was always in.

The arithmetic is deliberately plain, so what these assert is the *decisions*:
whole rounds rather than fractions (half a unit on an objective controls
nothing), the unit centroid rather than its nearest model (coherency will not
let one body go alone), the unit's slowest member rather than its fastest (the
unit arrives when its last model does), and distance to the objective's range
surface rather than to its centre (which is what control is scored on).
"""

from __future__ import annotations

import numpy as np
import pytest

from wargame_rl.wargame.envs.board.reach import (
    ObjectiveReach,
    Ownership,
    objective_reach,
)
from wargame_rl.wargame.envs.domain.entities import WargameModel, WargameObjective
from wargame_rl.wargame.envs.domain.value_objects import position
from wargame_rl.wargame.envs.types.config.env import WargameEnvConfig
from wargame_rl.wargame.envs.types.geometry import Polygon


class _Board:
    """The slice of `BattleView` `objective_reach` reads. Nothing is mocked --
    these are the real entity types, just placed by hand instead of by the
    placement sampler, which is the only way to state an exact distance."""

    def __init__(
        self,
        player: list[WargameModel],
        opponent: list[WargameModel],
        objectives: list[WargameObjective],
        player_zone: Polygon | None = None,
        opponent_zone: Polygon | None = None,
    ) -> None:
        self.player_models = player
        self.opponent_models = opponent
        self.objectives = objectives
        self.deployment_outline = player_zone
        self.opponent_deployment_outline = opponent_zone
        self.config = WargameEnvConfig(render_mode=None)


def _model(x: float, y: float, group_id: int = 0, alive: bool = True) -> WargameModel:
    model = WargameModel(
        location=position(x, y),
        stats={"max_wounds": 1, "current_wounds": 1, "toughness": 3, "save": 4},
        distances_to_objectives=np.zeros(0, dtype=float),
        group_id=group_id,
    )
    if not alive:
        model.take_damage(model.stats["max_wounds"])
    return model


def _marker(x: float, y: float, radius: float = 0.0) -> WargameObjective:
    return WargameObjective(location=position(x, y), radius_size=radius)


def _reach(
    board: _Board, player_move: float = 6.0, opponent_move: float = 6.0
) -> tuple[ObjectiveReach, ...]:
    return objective_reach(
        board,  # type: ignore[arg-type]
        np.full(len(board.player_models), player_move),
        np.full(len(board.opponent_models), opponent_move),
    )


class TestArrivalRounds:
    @pytest.mark.parametrize(
        ("distance", "expected"), [(0.0, 0), (1.0, 1), (6.0, 1), (6.1, 2), (18.0, 3)]
    )
    def test_rounds_are_whole_because_control_is_scored_at_one(
        self, distance: float, expected: int
    ) -> None:
        """Half a unit on an objective controls nothing -- control is a headcount
        evaluated at a scoring moment, so a fractional round is not an answer."""
        board = _Board([_model(0.0, 0.0)], [], [_marker(distance, 0.0)])

        assert _reach(board)[0].player_rounds == expected

    def test_a_unit_travels_from_its_centroid_not_its_nearest_model(self) -> None:
        """The body closest to an objective cannot go there alone: a 2" chain and
        a 9" span bind the unit, so quoting its distance prices an illegal move."""
        spread = _Board([_model(0.0, 0.0), _model(12.0, 0.0)], [], [_marker(12.0, 0.0)])

        # Centroid is at x=6, so 6" away -- one round. From the nearest model it
        # would read 0, i.e. already there.
        assert _reach(spread)[0].player_rounds == 1

    def test_a_unit_arrives_when_its_slowest_member_does(self) -> None:
        """Coherency will not let the fast half go on ahead."""
        board = _Board([_model(0.0, 0.0), _model(0.0, 0.0)], [], [_marker(12.0, 0.0)])

        rounds = objective_reach(
            board,  # type: ignore[arg-type]
            np.array([12.0, 4.0]),
            np.zeros(0),
        )[0].player_rounds

        assert rounds == 3

    def test_a_unit_that_cannot_move_never_arrives(self) -> None:
        board = _Board([_model(0.0, 0.0)], [], [_marker(12.0, 0.0)])

        assert _reach(board, player_move=0.0)[0].player_rounds == float("inf")

    def test_the_dead_are_not_counted_in_a_units_centroid(self) -> None:
        """A destroyed model keeps its position forever, so an unfiltered
        centroid would be dragged toward a corpse for the rest of the episode."""
        board = _Board(
            [_model(0.0, 0.0), _model(60.0, 0.0, alive=False)], [], [_marker(6.0, 0.0)]
        )

        assert _reach(board)[0].player_rounds == 1

    def test_a_wiped_out_side_reaches_nothing(self) -> None:
        board = _Board([_model(0.0, 0.0, alive=False)], [], [_marker(6.0, 0.0)])

        result = _reach(board)[0]

        assert result.player_rounds == float("inf")
        assert result.player_unit is None


class TestDistanceIsToWhatControlScores:
    def test_a_marker_is_reached_at_its_radius_not_its_centre(self) -> None:
        """Control is `norms_offset <= obj_radii`, so the radius is arrival."""
        board = _Board([_model(0.0, 0.0)], [], [_marker(6.0, 0.0, radius=6.0)])

        assert _reach(board)[0].player_rounds == 0

    def test_an_area_is_reached_at_its_outline_and_is_zero_inside(self) -> None:
        """Every objective on the real tables is an area, whose distance is to
        its edge -- and zero within it."""
        square = Polygon.from_rect(10.0, 0.0, 20.0, 10.0)
        area = WargameObjective(
            location=position(15.0, 5.0), radius_size=0.0, area=square
        )
        outside = _Board([_model(0.0, 5.0)], [], [area])
        inside = _Board([_model(15.0, 5.0)], [], [area])

        assert _reach(outside)[0].player_rounds == 2
        assert _reach(inside)[0].player_rounds == 0


class TestOwnership:
    def test_an_objective_is_classified_by_the_zone_it_stands_in(self) -> None:
        """The rules define no other regions -- no half, no centre. 34 of the 45
        real tables have non-rectangular zones, so a board-half rule would mean a
        different thing on every table."""
        mine = Polygon.from_rect(0.0, 0.0, 10.0, 40.0)
        theirs = Polygon.from_rect(30.0, 0.0, 40.0, 40.0)
        board = _Board(
            [_model(0.0, 0.0)],
            [_model(40.0, 0.0)],
            [_marker(5.0, 20.0), _marker(20.0, 20.0), _marker(35.0, 20.0)],
            player_zone=mine,
            opponent_zone=theirs,
        )

        assert [r.ownership for r in _reach(board)] == [
            Ownership.own_zone,
            Ownership.contested,
            Ownership.hostile,
        ]

    def test_everything_is_contested_when_no_zone_outlines_exist(self) -> None:
        """Generated scenarios carry rectangles rather than outlines; saying
        `contested` is honest there, and inventing a zone would not be."""
        board = _Board([_model(0.0, 0.0)], [], [_marker(5.0, 5.0)])

        assert _reach(board)[0].ownership is Ownership.contested


class TestTheRace:
    def test_the_margin_is_how_many_rounds_the_player_arrives_ahead(self) -> None:
        board = _Board([_model(0.0, 0.0)], [_model(30.0, 0.0)], [_marker(6.0, 0.0)])

        result = _reach(board)[0]

        assert (result.player_rounds, result.opponent_rounds) == (1, 4)
        assert result.contested_margin == 3

    def test_a_tie_is_a_zero_margin_rather_than_a_win(self) -> None:
        """Control is a strict comparison, so arriving together is not arriving
        first -- a tie controls nothing."""
        board = _Board([_model(0.0, 0.0)], [_model(12.0, 0.0)], [_marker(6.0, 0.0)])

        assert _reach(board)[0].contested_margin == 0

    def test_the_fastest_unit_is_named_so_a_plan_can_use_it(self) -> None:
        board = _Board(
            [_model(30.0, 0.0, group_id=0), _model(2.0, 0.0, group_id=1)],
            [],
            [_marker(0.0, 0.0)],
        )

        result = _reach(board)[0]

        assert result.player_unit == 1
        assert result.player_rounds == 1

    def test_a_board_with_no_objectives_yields_nothing_rather_than_raising(
        self,
    ) -> None:
        assert _reach(_Board([_model(0.0, 0.0)], [], [])) == ()
