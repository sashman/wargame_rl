"""The batched arrival geometry equals the per-model loop EXACTLY.

`steps_toward_objective` exists purely for throughput — the scripted opponent
paid two polygon passes per model against the same outline for every member of
a squad (issue #270, `reports/2026-09-05-where-the-opponent-turn-goes.md`).
It must be caching and batching, never a rule change, so every comparison here
is `assert_array_equal`, not `allclose`: the opponent's moves feed reward, and
`tests/test_reward_golden.py` is verified sensitive to one ULP — a tolerance
here would hide exactly the float-reassociation regression a vectorisation
introduces, and it would surface later as a golden failure far from the cause.

The ULP-sensitivity cases are the proof the equality assertions can fail at
all: nudge one input coordinate by one ULP and the compared arrays must differ.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from wargame_rl.wargame.envs.baseline.policy import (
    step_toward_objective,
    steps_toward_objective,
)
from wargame_rl.wargame.envs.domain.kernel.entities import WargameObjective
from wargame_rl.wargame.envs.domain.kernel.value_objects import position
from wargame_rl.wargame.envs.types.config import (
    ModelConfig,
    OpponentPolicyConfig,
    WargameEnvConfig,
)
from wargame_rl.wargame.envs.types.geometry import Polygon
from wargame_rl.wargame.envs.wargame import WargameEnv


def _random_polygon(rng: np.random.Generator) -> Polygon:
    """A convex n-gon of 3..8 vertices, the shape terrain generation produces."""
    n = int(rng.integers(3, 9))
    angles = np.sort(rng.uniform(0.0, 2.0 * np.pi, size=n))
    radii = rng.uniform(2.0, 8.0, size=n)
    centre = rng.uniform(10.0, 30.0, size=2)
    vertices = centre + np.stack(
        [radii * np.cos(angles), radii * np.sin(angles)], axis=1
    )
    return Polygon(vertices)


class TestPolygonBatchedQueries:
    """The two batched `Polygon` methods against their scalar originals."""

    @pytest.mark.parametrize("seed", range(5))
    def test_distances_to_boundary_matches_the_scalar_bit_for_bit(
        self, seed: int
    ) -> None:
        rng = np.random.default_rng(seed)
        polygon = _random_polygon(rng)
        points = rng.uniform(0.0, 40.0, size=(40, 2))

        batched = polygon.distances_to_boundary(points)
        scalar = np.array(
            [polygon.distance_to_boundary(float(x), float(y)) for x, y in points]
        )

        assert_array_equal(batched, scalar)

    def test_the_distance_comparison_is_sensitive_to_one_ulp(self) -> None:
        """Proof the equality above can fail: one ULP on one coordinate.

        Deterministic geometry rather than a random polygon: the point sits
        straight out from a vertical edge, so the distance IS the nudged
        coordinate and a one-ULP change cannot vanish into the norm the way
        it can when the other component dominates the sum of squares.
        """
        square = Polygon(
            np.array([[10.0, 10.0], [20.0, 10.0], [20.0, 20.0], [10.0, 20.0]])
        )
        points = np.array([[25.0, 15.0]])
        nudged = points.copy()
        nudged[0, 0] = np.nextafter(nudged[0, 0], np.inf)

        assert not np.array_equal(
            square.distances_to_boundary(points),
            square.distances_to_boundary(nudged),
        )

    @pytest.mark.parametrize("seed", range(5))
    def test_boundary_inclusive_contains_matches_the_scalar(self, seed: int) -> None:
        rng = np.random.default_rng(seed)
        polygon = _random_polygon(rng)
        points = rng.uniform(0.0, 40.0, size=(40, 2))
        # Points exactly ON edges, where interior-only and boundary-inclusive
        # answers differ — the reason `include_boundary` had to be threaded
        # through rather than reusing the interior-only default.
        midpoints = (polygon.vertices + np.roll(polygon.vertices, -1, axis=0)) / 2.0
        all_points = np.vstack([points, polygon.vertices, midpoints])

        batched = polygon.contains_points(all_points, include_boundary=True)
        scalar = np.array([polygon.contains(float(x), float(y)) for x, y in all_points])

        assert_array_equal(batched, scalar)
        # And the vertices really are the discriminating rows: on the boundary,
        # inclusive says True where the interior-only default says False.
        assert batched[len(points) : len(points) + polygon.n_vertices].all()


def _env(n_models: int) -> WargameEnv:
    return WargameEnv(
        config=WargameEnvConfig(
            board_width=60,
            board_height=40,
            number_of_wargame_models=n_models,
            number_of_objectives=1,
            number_of_battle_rounds=4,
            models=[ModelConfig(x=5 + i, y=5 + i, group_id=0) for i in range(n_models)],
            opponent_policy=OpponentPolicyConfig(type="random"),
            render_mode=None,
        )
    )


class TestStepsTowardObjective:
    """The batched helper against a scalar loop, through a real env's handler.

    A squad of one is deliberately not enough: a batching bug that collapses
    per-model state returns one model's answer for everyone, which only a
    multi-model squad with distinct positions can catch. The `model_idx`
    passthrough is exercised the same way — members are enumerated in
    `member_indices` order and each row must be the scalar call for exactly
    that model.
    """

    @pytest.mark.parametrize("seed", range(8))
    def test_an_area_objective_matches_the_scalar_loop(self, seed: int) -> None:
        env = _env(6)
        env.reset(seed=0)
        rng = np.random.default_rng(seed)
        for model in env.wargame_models:
            x, y = rng.uniform(2.0, 38.0, size=2)
            model.location = position(float(x), float(y))
        objective = WargameObjective(position(0.0, 0.0), 0.0)
        objective.set_area(_random_polygon(rng))
        indices = list(range(len(env.wargame_models)))

        batched = steps_toward_objective(env.wargame_models, indices, objective, env)
        scalar = [
            step_toward_objective(env.wargame_models[i], objective, env, i)
            for i in indices
        ]

        assert batched == scalar

    @pytest.mark.parametrize("seed", range(8))
    def test_a_marker_objective_matches_the_scalar_loop(self, seed: int) -> None:
        env = _env(6)
        env.reset(seed=0)
        rng = np.random.default_rng(seed)
        for model in env.wargame_models:
            x, y = rng.uniform(2.0, 38.0, size=2)
            model.location = position(float(x), float(y))
        objective = WargameObjective(position(20.0, 20.0), float(rng.uniform(0.5, 4.0)))
        indices = list(range(len(env.wargame_models)))

        batched = steps_toward_objective(env.wargame_models, indices, objective, env)
        scalar = [
            step_toward_objective(env.wargame_models[i], objective, env, i)
            for i in indices
        ]

        assert batched == scalar

    def test_models_inside_and_on_the_objective_stay(self) -> None:
        """The arrival rule survives batching: inside means STAY, exactly."""
        env = _env(3)
        env.reset(seed=0)
        objective = WargameObjective(position(0.0, 0.0), 0.0)
        objective.set_area(Polygon(np.array([[15, 15], [25, 15], [25, 25], [15, 25]])))
        env.wargame_models[0].location = position(20.0, 20.0)  # inside
        env.wargame_models[1].location = position(15.0, 20.0)  # exactly on the edge
        env.wargame_models[2].location = position(5.0, 20.0)  # outside
        indices = [0, 1, 2]

        batched = steps_toward_objective(env.wargame_models, indices, objective, env)
        scalar = [
            step_toward_objective(env.wargame_models[i], objective, env, i)
            for i in indices
        ]

        assert batched == scalar
        assert batched[0] == scalar[0]
        assert batched[1] == scalar[1]

    def test_an_empty_squad_returns_an_empty_list(self) -> None:
        env = _env(2)
        env.reset(seed=0)
        objective = WargameObjective(position(20.0, 20.0), 1.0)

        assert steps_toward_objective(env.wargame_models, [], objective, env) == []
