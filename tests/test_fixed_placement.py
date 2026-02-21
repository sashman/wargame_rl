"""Tests for entity configuration via ModelConfig / ObjectiveConfig."""

import numpy as np
import pytest

from wargame_rl.wargame.envs.env_components.placement import (
    fixed_objective_placement,
    fixed_wargame_model_placement,
)
from wargame_rl.wargame.envs.types import WargameEnvAction, WargameEnvConfig
from wargame_rl.wargame.envs.types.config import ModelConfig, ObjectiveConfig
from wargame_rl.wargame.envs.wargame import WargameEnv

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(
    models: list[ModelConfig] | None = None,
    objectives: list[ObjectiveConfig] | None = None,
    **overrides: object,
) -> WargameEnvConfig:
    defaults: dict = dict(
        board_width=20,
        board_height=20,
        number_of_wargame_models=2,
        number_of_objectives=2,
        objective_radius_size=2,
        render_mode=None,
    )
    defaults.update(overrides)
    return WargameEnvConfig(models=models, objectives=objectives, **defaults)


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


class TestConfigValidation:
    def test_valid_fixed_config(self) -> None:
        cfg = _make_config(
            models=[
                ModelConfig(x=1, y=2, group_id=0),
                ModelConfig(x=3, y=4, group_id=1),
            ],
            objectives=[
                ObjectiveConfig(x=15, y=15),
                ObjectiveConfig(x=18, y=18),
            ],
        )
        assert cfg.models is not None
        assert cfg.objectives is not None

    def test_models_count_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="models has 1 entries"):
            _make_config(models=[ModelConfig(x=0, y=0)])

    def test_objectives_count_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="objectives has 1 entries"):
            _make_config(objectives=[ObjectiveConfig(x=5, y=5)])

    def test_model_out_of_bounds_raises(self) -> None:
        with pytest.raises(ValueError, match="outside"):
            _make_config(
                models=[
                    ModelConfig(x=0, y=0),
                    ModelConfig(x=99, y=0),
                ],
            )

    def test_objective_out_of_bounds_raises(self) -> None:
        with pytest.raises(ValueError, match="outside"):
            _make_config(
                objectives=[
                    ObjectiveConfig(x=0, y=0),
                    ObjectiveConfig(x=0, y=99),
                ],
            )

    def test_model_at_board_edge_is_invalid(self) -> None:
        """x == board_width is out of bounds (0-indexed)."""
        with pytest.raises(ValueError, match="outside"):
            _make_config(
                models=[
                    ModelConfig(x=0, y=0),
                    ModelConfig(x=20, y=0),
                ],
            )

    def test_model_config_negative_coords_rejected(self) -> None:
        with pytest.raises(ValueError):
            ModelConfig(x=-1, y=0)

    def test_objective_config_negative_coords_rejected(self) -> None:
        with pytest.raises(ValueError):
            ObjectiveConfig(x=0, y=-1)

    def test_model_config_x_without_y_rejected(self) -> None:
        with pytest.raises(ValueError, match="both be set or both be None"):
            ModelConfig(x=5)

    def test_objective_config_y_without_x_rejected(self) -> None:
        with pytest.raises(ValueError, match="both be set or both be None"):
            ObjectiveConfig(y=5)

    def test_mixed_coords_some_models_with_some_without_rejected(self) -> None:
        with pytest.raises(ValueError, match="all models must have x/y"):
            _make_config(
                models=[
                    ModelConfig(x=1, y=2),
                    ModelConfig(group_id=1),
                ],
            )

    def test_mixed_coords_some_objectives_with_some_without_rejected(self) -> None:
        with pytest.raises(ValueError, match="all objectives must have x/y"):
            _make_config(
                objectives=[
                    ObjectiveConfig(x=10, y=10),
                    ObjectiveConfig(),
                ],
            )

    def test_none_models_and_objectives_is_valid(self) -> None:
        cfg = _make_config()
        assert cfg.models is None
        assert cfg.objectives is None

    def test_models_without_coords_no_bounds_check(self) -> None:
        """Models with no x/y should not trigger bounds validation."""
        cfg = _make_config(
            models=[
                ModelConfig(group_id=0, max_wounds=50),
                ModelConfig(group_id=1, max_wounds=75),
            ],
        )
        assert cfg.models is not None
        assert not cfg.has_fixed_model_positions


# ---------------------------------------------------------------------------
# has_fixed_*_positions properties
# ---------------------------------------------------------------------------


class TestHasFixedPositionsProperties:
    def test_no_models_is_not_fixed(self) -> None:
        cfg = _make_config()
        assert not cfg.has_fixed_model_positions
        assert not cfg.has_fixed_objective_positions

    def test_models_with_coords_is_fixed(self) -> None:
        cfg = _make_config(
            models=[ModelConfig(x=0, y=0), ModelConfig(x=1, y=1)],
        )
        assert cfg.has_fixed_model_positions

    def test_models_without_coords_is_not_fixed(self) -> None:
        cfg = _make_config(
            models=[ModelConfig(group_id=0), ModelConfig(group_id=1)],
        )
        assert not cfg.has_fixed_model_positions

    def test_objectives_with_coords_is_fixed(self) -> None:
        cfg = _make_config(
            objectives=[ObjectiveConfig(x=10, y=10), ObjectiveConfig(x=15, y=15)],
        )
        assert cfg.has_fixed_objective_positions

    def test_objectives_without_coords_is_not_fixed(self) -> None:
        cfg = _make_config(
            objectives=[ObjectiveConfig(), ObjectiveConfig(radius_size=5)],
        )
        assert not cfg.has_fixed_objective_positions


# ---------------------------------------------------------------------------
# Placement functions (unit)
# ---------------------------------------------------------------------------


class TestFixedPlacementFunctions:
    def test_fixed_model_placement_sets_coordinates(self) -> None:
        cfg = _make_config(
            models=[
                ModelConfig(x=3, y=7, group_id=0),
                ModelConfig(x=10, y=12, group_id=1),
            ],
        )
        env = WargameEnv(config=cfg)

        fixed_wargame_model_placement(env.wargame_models, cfg.models)  # type: ignore[arg-type]

        np.testing.assert_array_equal(env.wargame_models[0].location, [3, 7])
        np.testing.assert_array_equal(env.wargame_models[1].location, [10, 12])

    def test_fixed_model_placement_resets_wounds(self) -> None:
        cfg = _make_config(
            models=[
                ModelConfig(x=0, y=0, max_wounds=50),
                ModelConfig(x=1, y=1, max_wounds=75),
            ],
        )
        env = WargameEnv(config=cfg)
        env.wargame_models[0].stats["current_wounds"] = 10

        fixed_wargame_model_placement(env.wargame_models, cfg.models)  # type: ignore[arg-type]

        assert env.wargame_models[0].stats["current_wounds"] == 50
        assert env.wargame_models[1].stats["current_wounds"] == 75

    def test_fixed_model_placement_clears_previous_location(self) -> None:
        cfg = _make_config(
            models=[
                ModelConfig(x=5, y=5),
                ModelConfig(x=6, y=6),
            ],
        )
        env = WargameEnv(config=cfg)
        env.wargame_models[0].previous_location = np.array([99, 99])

        fixed_wargame_model_placement(env.wargame_models, cfg.models)  # type: ignore[arg-type]

        assert env.wargame_models[0].previous_location is None

    def test_fixed_objective_placement_sets_coordinates(self) -> None:
        cfg = _make_config(
            objectives=[
                ObjectiveConfig(x=15, y=10),
                ObjectiveConfig(x=18, y=5),
            ],
        )
        env = WargameEnv(config=cfg)

        fixed_objective_placement(env.objectives, cfg.objectives)  # type: ignore[arg-type]

        np.testing.assert_array_equal(env.objectives[0].location, [15, 10])
        np.testing.assert_array_equal(env.objectives[1].location, [18, 5])


# ---------------------------------------------------------------------------
# Entity creation (create_wargame_models / create_objectives)
# ---------------------------------------------------------------------------


class TestEntityCreation:
    def test_create_models_uses_config_group_ids(self) -> None:
        cfg = _make_config(
            models=[
                ModelConfig(x=0, y=0, group_id=5),
                ModelConfig(x=1, y=1, group_id=7),
            ],
        )
        models = WargameEnv.create_wargame_models(cfg)
        assert models[0].group_id == 5
        assert models[1].group_id == 7

    def test_create_models_uses_config_max_wounds(self) -> None:
        cfg = _make_config(
            models=[
                ModelConfig(x=0, y=0, max_wounds=30),
                ModelConfig(x=1, y=1, max_wounds=200),
            ],
        )
        models = WargameEnv.create_wargame_models(cfg)
        assert models[0].stats["max_wounds"] == 30
        assert models[0].stats["current_wounds"] == 30
        assert models[1].stats["max_wounds"] == 200

    def test_create_models_without_coords_uses_attributes(self) -> None:
        """models with no x/y still provide group_id and max_wounds."""
        cfg = _make_config(
            models=[
                ModelConfig(group_id=3, max_wounds=42),
                ModelConfig(group_id=5, max_wounds=99),
            ],
        )
        models = WargameEnv.create_wargame_models(cfg)
        assert models[0].group_id == 3
        assert models[0].stats["max_wounds"] == 42
        assert models[1].group_id == 5
        assert models[1].stats["max_wounds"] == 99

    def test_create_objectives_per_objective_radius_override(self) -> None:
        cfg = _make_config(
            objectives=[
                ObjectiveConfig(x=10, y=10, radius_size=5),
                ObjectiveConfig(x=15, y=15),
            ],
        )
        objectives = WargameEnv.create_objectives(cfg)
        assert objectives[0].radius_size == 5
        assert objectives[1].radius_size == cfg.objective_radius_size

    def test_create_objectives_no_override_uses_global(self) -> None:
        cfg = _make_config(
            objectives=[
                ObjectiveConfig(x=10, y=10),
                ObjectiveConfig(x=15, y=15),
            ],
        )
        objectives = WargameEnv.create_objectives(cfg)
        assert all(o.radius_size == cfg.objective_radius_size for o in objectives)

    def test_create_objectives_without_coords_uses_radius(self) -> None:
        """objectives with no x/y still provide per-objective radius_size."""
        cfg = _make_config(
            objectives=[
                ObjectiveConfig(radius_size=7),
                ObjectiveConfig(),
            ],
        )
        objectives = WargameEnv.create_objectives(cfg)
        assert objectives[0].radius_size == 7
        assert objectives[1].radius_size == cfg.objective_radius_size


# ---------------------------------------------------------------------------
# Environment integration (reset / step)
# ---------------------------------------------------------------------------


class TestFixedPlacementIntegration:
    @pytest.fixture
    def fixed_env(self) -> WargameEnv:
        cfg = _make_config(
            models=[
                ModelConfig(x=2, y=3, group_id=0),
                ModelConfig(x=4, y=5, group_id=1),
            ],
            objectives=[
                ObjectiveConfig(x=15, y=10),
                ObjectiveConfig(x=18, y=12),
            ],
        )
        return WargameEnv(config=cfg)

    def test_reset_places_at_exact_positions(self, fixed_env: WargameEnv) -> None:
        obs, _ = fixed_env.reset(seed=42)

        np.testing.assert_array_equal(obs.wargame_models[0].location, [2, 3])
        np.testing.assert_array_equal(obs.wargame_models[1].location, [4, 5])
        np.testing.assert_array_equal(obs.objectives[0].location, [15, 10])
        np.testing.assert_array_equal(obs.objectives[1].location, [18, 12])

    def test_reset_deterministic_without_seed(self, fixed_env: WargameEnv) -> None:
        """Fixed placements should be identical regardless of seed."""
        obs1, _ = fixed_env.reset(seed=1)
        obs2, _ = fixed_env.reset(seed=9999)

        for m1, m2 in zip(obs1.wargame_models, obs2.wargame_models):
            np.testing.assert_array_equal(m1.location, m2.location)
        for o1, o2 in zip(obs1.objectives, obs2.objectives):
            np.testing.assert_array_equal(o1.location, o2.location)

    def test_step_works_with_fixed_placement(self, fixed_env: WargameEnv) -> None:
        fixed_env.reset(seed=42)
        for _ in range(5):
            action = WargameEnvAction(actions=list(fixed_env.action_space.sample()))
            obs, reward, terminated, truncated, info = fixed_env.step(action)
            assert obs is not None
            assert isinstance(reward, float)
            if terminated or truncated:
                break

    def test_mixed_mode_fixed_models_random_objectives(self) -> None:
        cfg = _make_config(
            models=[
                ModelConfig(x=2, y=3),
                ModelConfig(x=4, y=5),
            ],
            objectives=None,
        )
        env = WargameEnv(config=cfg)
        obs, _ = env.reset(seed=42)

        np.testing.assert_array_equal(obs.wargame_models[0].location, [2, 3])
        np.testing.assert_array_equal(obs.wargame_models[1].location, [4, 5])

    def test_mixed_mode_random_models_fixed_objectives(self) -> None:
        cfg = _make_config(
            models=None,
            objectives=[
                ObjectiveConfig(x=15, y=10),
                ObjectiveConfig(x=18, y=12),
            ],
        )
        env = WargameEnv(config=cfg)
        obs, _ = env.reset(seed=42)

        np.testing.assert_array_equal(obs.objectives[0].location, [15, 10])
        np.testing.assert_array_equal(obs.objectives[1].location, [18, 12])

    def test_multiple_resets_always_same_positions(self, fixed_env: WargameEnv) -> None:
        for seed in [None, 1, 42, 100]:
            obs, _ = fixed_env.reset(seed=seed)
            np.testing.assert_array_equal(obs.wargame_models[0].location, [2, 3])
            np.testing.assert_array_equal(obs.objectives[1].location, [18, 12])


class TestAttributesOnlyIntegration:
    """Models/objectives with attributes but no coordinates → random placement."""

    def test_models_attributes_only_random_placement(self) -> None:
        cfg = _make_config(
            models=[
                ModelConfig(group_id=0, max_wounds=50),
                ModelConfig(group_id=1, max_wounds=75),
            ],
        )
        env = WargameEnv(config=cfg)
        obs, _ = env.reset(seed=42)

        assert env.wargame_models[0].stats["max_wounds"] == 50
        assert env.wargame_models[1].stats["max_wounds"] == 75
        assert env.wargame_models[0].group_id == 0
        assert env.wargame_models[1].group_id == 1
        assert len(obs.wargame_models) == 2

    def test_objectives_attributes_only_random_placement(self) -> None:
        cfg = _make_config(
            objectives=[
                ObjectiveConfig(radius_size=5),
                ObjectiveConfig(radius_size=8),
            ],
        )
        env = WargameEnv(config=cfg)
        obs, _ = env.reset(seed=42)

        assert env.objectives[0].radius_size == 5
        assert env.objectives[1].radius_size == 8
        assert len(obs.objectives) == 2

    def test_attributes_only_models_positions_vary_with_seed(self) -> None:
        """Without coordinates, different seeds should produce different positions."""
        cfg = _make_config(
            models=[
                ModelConfig(group_id=0, max_wounds=50),
                ModelConfig(group_id=0, max_wounds=50),
            ],
        )
        env = WargameEnv(config=cfg)
        obs1, _ = env.reset(seed=1)
        locs1 = [m.location.copy() for m in obs1.wargame_models]
        obs2, _ = env.reset(seed=9999)
        locs2 = [m.location.copy() for m in obs2.wargame_models]

        differs = any(not np.array_equal(a, b) for a, b in zip(locs1, locs2))
        assert differs, "Random placement should differ across seeds"

    def test_step_works_with_attributes_only(self) -> None:
        cfg = _make_config(
            models=[
                ModelConfig(group_id=0, max_wounds=50),
                ModelConfig(group_id=1, max_wounds=75),
            ],
            objectives=[
                ObjectiveConfig(radius_size=5),
                ObjectiveConfig(),
            ],
        )
        env = WargameEnv(config=cfg)
        env.reset(seed=42)
        for _ in range(5):
            action = WargameEnvAction(actions=list(env.action_space.sample()))
            obs, reward, terminated, truncated, _ = env.step(action)
            assert obs is not None
            assert isinstance(reward, float)
            if terminated or truncated:
                break
