"""Models with bases cannot walk through the enemy or end on top of each other.

Three asymmetries, all deliberate and all measurable in the geometry rather than
in a downstream metric:

- an enemy base stops the move where contact happens;
- a friendly base may be crossed but not ended on;
- resolution is sequential in model index order, which is a documented
  right-of-way bias and the price of a deterministic board.
"""

from __future__ import annotations

import numpy as np
import pytest

from wargame_rl.wargame.envs.domain.movement import resolve_move
from wargame_rl.wargame.envs.env_components.actions import STAY_ACTION
from wargame_rl.wargame.envs.types import WargameEnvAction, WargameEnvConfig
from wargame_rl.wargame.envs.types.config import OpponentPolicyConfig
from wargame_rl.wargame.envs.wargame import WargameEnv

RADIUS = 0.63
NONE_2D = np.zeros((0, 2), dtype=float)
NONE_1D = np.zeros(0, dtype=float)


def _move(
    start: tuple[float, float],
    displacement: tuple[float, float],
    *,
    blockers: list[tuple[float, float]] | None = None,
    passable: list[tuple[float, float]] | None = None,
    radius: float = RADIUS,
) -> np.ndarray:
    def centres(points: list[tuple[float, float]] | None) -> np.ndarray:
        return np.array(points, dtype=float) if points else NONE_2D

    def radii(points: list[tuple[float, float]] | None) -> np.ndarray:
        return np.full(len(points), radius) if points else NONE_1D

    return resolve_move(
        np.array(start, dtype=float),
        np.array(displacement, dtype=float),
        radius,
        centres(blockers),
        radii(blockers),
        centres(passable),
        radii(passable),
    )


class TestResolveMove:
    def test_an_unobstructed_move_is_exact(self) -> None:
        assert _move((0.0, 0.0), (3.0, 4.0)) == pytest.approx([3.0, 4.0])

    def test_a_dimensionless_model_is_unaffected(self) -> None:
        """Radius 0 must reproduce the pre-base behaviour exactly.

        Every result measured before models had bases assumed models could
        occupy the same point, and the golden gates still pin those.
        """
        assert _move(
            (0.0, 0.0), (10.0, 0.0), blockers=[(5.0, 0.0)], radius=0.0
        ) == pytest.approx([10.0, 0.0])

    def test_an_enemy_base_stops_the_move_at_contact(self) -> None:
        end = _move((0.0, 0.0), (10.0, 0.0), blockers=[(5.0, 0.0)])

        # Contact is base to base, so it stops two radii short of the centre.
        assert end[0] == pytest.approx(5.0 - 2 * RADIUS, abs=1e-4)
        assert end[1] == pytest.approx(0.0)

    def test_a_friendly_base_may_be_crossed_but_not_ended_on(self) -> None:
        """The asymmetry that keeps a squad from gridlocking on its own front rank."""
        through = _move((0.0, 0.0), (10.0, 0.0), passable=[(5.0, 0.0)])
        assert through[0] == pytest.approx(10.0), "friendly should not block the path"

        onto = _move((0.0, 0.0), (5.0, 0.0), passable=[(5.0, 0.0)])
        assert onto[0] < 5.0 - RADIUS, "ended inside a friendly base"

    def test_a_blocked_move_never_ends_overlapping(self) -> None:
        """The property the whole module exists for, over a fan of directions."""
        obstacles = [(3.0, 0.0), (0.0, 3.0), (-3.0, 1.0), (2.0, 2.0)]
        for angle in np.linspace(0, 2 * np.pi, 32, endpoint=False):
            step = (6.0 * float(np.cos(angle)), 6.0 * float(np.sin(angle)))
            end = _move(
                (0.0, 0.0), step, blockers=obstacles[:2], passable=obstacles[2:]
            )
            gaps = np.linalg.norm(np.array(obstacles) - end, axis=1)
            assert gaps.min() >= 2 * RADIUS - 1e-4, f"overlap after moving {step}"

    def test_a_model_already_in_contact_can_still_move_away(self) -> None:
        """A negative entry time must not drag the move to zero.

        Models start a turn touching after a blocked approach, and a rule that
        froze them there would be a permanent trap rather than a collision.
        """
        touching = 2 * RADIUS
        end = _move((0.0, 0.0), (-4.0, 0.0), blockers=[(touching, 0.0)])
        assert end[0] == pytest.approx(-4.0)

    def test_the_move_never_grows(self) -> None:
        """Collision response must not buy distance.

        A tangential slide was tried here and let a model out-travel its own
        Move characteristic before it was budgeted; this pins the invariant
        whatever the response becomes.
        """
        for angle in np.linspace(0, 2 * np.pi, 24, endpoint=False):
            step = np.array([6.0 * np.cos(angle), 6.0 * np.sin(angle)])
            end = _move(
                (0.0, 0.0),
                (float(step[0]), float(step[1])),
                blockers=[(3.0, 0.0), (0.0, 3.0)],
                passable=[(-2.0, -2.0)],
            )
            assert float(np.linalg.norm(end)) <= 6.0 + 1e-6


class TestEnvIntegration:
    def _env(self, base_radius: float) -> WargameEnv:
        return WargameEnv(
            config=WargameEnvConfig(
                board_width=40,
                board_height=40,
                number_of_wargame_models=8,
                number_of_opponent_models=8,
                number_of_objectives=2,
                number_of_battle_rounds=4,
                base_radius=base_radius,
                opponent_policy=OpponentPolicyConfig(type="random"),
                render_mode=None,
            )
        )

    def test_no_two_bases_overlap_at_any_point_in_an_episode(self) -> None:
        """Placement guaranteeing this is not enough — models still have to move."""
        env = self._env(RADIUS)
        env.reset(seed=11)
        env.action_space.seed(11)

        for _ in range(20):
            env.step(WargameEnvAction(actions=env.action_space.sample()))
            everyone = [
                m
                for m in list(env.wargame_models) + list(env.opponent_models)
                if m.is_alive
            ]
            positions = np.array([m.location for m in everyone], dtype=float)
            gaps = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=2)
            np.fill_diagonal(gaps, np.inf)
            assert gaps.min() >= 2 * RADIUS - 1e-3

    def test_the_enemy_blocks_and_a_squadmate_does_not(self) -> None:
        """Both halves of the asymmetry, driven through the action handler.

        Positions are set by hand rather than played into: the point is the
        movement rule, and letting an opponent policy choose where the blocker
        stands would make the test about the policy instead.
        """
        env = self._env(RADIUS)
        env.reset(seed=3)
        handler = env.player_action_handler
        east = handler.best_action_toward(1.0, 0.0)
        stay = STAY_ACTION

        for model in env.wargame_models:
            model.location = np.array([0.0, 30.0])
        for model in env.opponent_models:
            model.location = np.array([0.0, 30.0])
        env.wargame_models[0].location = np.array([2.0, 10.0])
        env.opponent_models[0].location = np.array([6.0, 10.0])

        handler.apply(
            WargameEnvAction(actions=[east] + [stay] * 7),
            env.wargame_models,
            env.board_width,
            env.board_height,
            handler.action_space,
            enemy_models=env.opponent_models,
        )
        blocked_by_enemy = float(env.wargame_models[0].location[0])
        assert blocked_by_enemy < 6.0 - RADIUS

        # Same geometry with a squadmate in the way: passable, so it goes past.
        env.wargame_models[0].location = np.array([2.0, 10.0])
        env.wargame_models[1].location = np.array([6.0, 10.0])
        env.opponent_models[0].location = np.array([0.0, 30.0])
        handler.apply(
            WargameEnvAction(actions=[east] + [stay] * 7),
            env.wargame_models,
            env.board_width,
            env.board_height,
            handler.action_space,
            enemy_models=env.opponent_models,
        )
        assert float(env.wargame_models[0].location[0]) > blocked_by_enemy
