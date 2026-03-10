"""Tests for the reward phases system (calculators, criteria, phase manager)."""

from __future__ import annotations

import numpy as np
import pytest

from wargame_rl.wargame.envs.env_components.distance_cache import (
    DistanceCache,
    compute_distances,
)
from wargame_rl.wargame.envs.reward.calculators.closest_objective import (
    ClosestObjectiveCalculator,
)
from wargame_rl.wargame.envs.reward.calculators.group_cohesion import (
    GroupCohesionCalculator,
)
from wargame_rl.wargame.envs.reward.calculators.registry import (
    CALCULATOR_REGISTRY,
    build_calculator,
)
from wargame_rl.wargame.envs.reward.criteria.all_at_objectives import (
    AllAtObjectivesCriteria,
)
from wargame_rl.wargame.envs.reward.criteria.all_models_grouped import (
    AllModelsGroupedCriteria,
)
from wargame_rl.wargame.envs.reward.criteria.registry import (
    CRITERIA_REGISTRY,
    build_criteria,
)
from wargame_rl.wargame.envs.reward.phase import (
    RewardCalculatorConfig,
    RewardPhaseConfig,
    SuccessCriteriaConfig,
)
from wargame_rl.wargame.envs.reward.phase_manager import RewardPhaseManager
from wargame_rl.wargame.envs.reward.reward import Reward
from wargame_rl.wargame.envs.reward.step_context import StepContext
from wargame_rl.wargame.envs.types import WargameEnvAction, WargameEnvConfig
from wargame_rl.wargame.envs.wargame import WargameEnv

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_env() -> WargameEnv:
    """A small environment with no reward phases (legacy mode)."""
    config = WargameEnvConfig(
        render_mode=None,
        board_width=20,
        board_height=20,
        number_of_wargame_models=2,
        number_of_objectives=1,
        objective_radius_size=2,
    )
    return WargameEnv(config=config)


@pytest.fixture
def phased_env() -> WargameEnv:
    """An environment with two reward phases configured."""
    config = WargameEnvConfig(
        render_mode=None,
        board_width=20,
        board_height=20,
        number_of_wargame_models=2,
        number_of_objectives=1,
        objective_radius_size=2,
        number_of_battle_rounds=40,
        reward_phases=[
            RewardPhaseConfig(
                name="group_up",
                reward_calculators=[
                    RewardCalculatorConfig(
                        type="group_cohesion",
                        weight=1.0,
                        params={
                            "group_max_distance": 5.0,
                            "violation_penalty": -1.0,
                        },
                    ),
                ],
                success_criteria=SuccessCriteriaConfig(
                    type="all_models_grouped",
                    params={"max_distance": 5.0},
                ),
                success_threshold=0.5,
                min_epochs=0,
            ),
            RewardPhaseConfig(
                name="reach_objectives",
                reward_calculators=[
                    RewardCalculatorConfig(type="closest_objective", weight=1.0),
                ],
                success_criteria=SuccessCriteriaConfig(
                    type="all_at_objectives",
                ),
                success_threshold=0.8,
                min_epochs=0,
            ),
        ],
    )
    return WargameEnv(config=config)


def _make_step_context(env: WargameEnv, cache: DistanceCache) -> StepContext:
    return StepContext(
        distance_cache=cache,
        current_turn=env.current_turn,
        max_turns=env.max_turns,
        board_width=env.board_width,
        board_height=env.board_height,
    )


# ---------------------------------------------------------------------------
# Calculator tests
# ---------------------------------------------------------------------------


class TestClosestObjectiveCalculator:
    def test_first_step_returns_zero(self, simple_env: WargameEnv) -> None:
        simple_env.reset()
        calc = ClosestObjectiveCalculator(weight=1.0)
        cache = compute_distances(simple_env.wargame_models, simple_env.objectives)
        ctx = _make_step_context(simple_env, cache)

        model = simple_env.wargame_models[0]
        assert model.previous_closest_objective_distance is None
        reward = calc.calculate(0, model, simple_env, ctx)
        assert reward == 0.0
        assert model.previous_closest_objective_distance is not None

    def test_weight_is_stored(self) -> None:
        calc = ClosestObjectiveCalculator(weight=0.5)
        assert calc.weight == 0.5

    def test_no_model_model_distances_needed(self) -> None:
        calc = ClosestObjectiveCalculator()
        assert calc.needs_model_model_distances is False

    def test_penalty_when_distance_increases(self, simple_env: WargameEnv) -> None:
        simple_env.reset()
        calc = ClosestObjectiveCalculator(weight=1.0)

        obj_loc = np.array([10, 10])
        simple_env.objectives[0].location = obj_loc.copy()
        model = simple_env.wargame_models[0]
        model.location = np.array([5, 5])

        cache_prev = compute_distances(simple_env.wargame_models, simple_env.objectives)
        ctx_prev = _make_step_context(simple_env, cache_prev)
        first = calc.calculate(0, model, simple_env, ctx_prev)
        assert first == 0.0

        model.location = np.array([0, 0])
        cache_now = compute_distances(simple_env.wargame_models, simple_env.objectives)
        ctx_now = _make_step_context(simple_env, cache_now)

        assert model.previous_closest_objective_distance is not None
        previous = float(model.previous_closest_objective_distance)
        closest_obj_idx = int(cache_now.model_obj_norms[0].argmin())
        distance_to_closest = float(
            cache_now.model_obj_norms_offset[0, closest_obj_idx]
        )
        max_diagonal = float(
            (simple_env.board_width**2 + simple_env.board_height**2) ** 0.5
        )
        normalized_distance = distance_to_closest / max_diagonal
        distance_improvement = normalized_distance - previous
        expected = -(float(2) * abs(distance_improvement) + float(0.3))

        reward = calc.calculate(0, model, simple_env, ctx_now)
        assert reward == pytest.approx(expected)

    def test_no_penalty_when_distance_decreases(self, simple_env: WargameEnv) -> None:
        simple_env.reset()
        calc = ClosestObjectiveCalculator(weight=1.0)

        obj_loc = np.array([10, 10])
        simple_env.objectives[0].location = obj_loc.copy()
        model = simple_env.wargame_models[0]
        model.location = np.array([0, 0])

        cache_prev = compute_distances(simple_env.wargame_models, simple_env.objectives)
        ctx_prev = _make_step_context(simple_env, cache_prev)
        calc.calculate(0, model, simple_env, ctx_prev)

        model.location = np.array([9, 9])
        cache_now = compute_distances(simple_env.wargame_models, simple_env.objectives)
        ctx_now = _make_step_context(simple_env, cache_now)
        reward = calc.calculate(0, model, simple_env, ctx_now)
        assert reward == 0.0


class TestGroupCohesionCalculator:
    def test_no_penalty_when_within_distance(self, simple_env: WargameEnv) -> None:
        simple_env.reset()
        m0 = simple_env.wargame_models[0]
        m1 = simple_env.wargame_models[1]
        m0.location = np.array([5, 5])
        m1.location = np.array([7, 5])
        m0.group_id = 0
        m1.group_id = 0

        calc = GroupCohesionCalculator(
            weight=1.0, group_max_distance=10.0, violation_penalty=-10.0
        )
        cache = compute_distances(
            simple_env.wargame_models,
            simple_env.objectives,
            compute_model_model=True,
        )
        ctx = _make_step_context(simple_env, cache)
        reward = calc.calculate(0, m0, simple_env, ctx)
        assert reward == 0.0

    def test_penalty_when_exceeding_distance(self, simple_env: WargameEnv) -> None:
        simple_env.reset()
        m0 = simple_env.wargame_models[0]
        m1 = simple_env.wargame_models[1]
        m0.location = np.array([0, 0])
        m1.location = np.array([15, 0])
        m0.group_id = 0
        m1.group_id = 0

        calc = GroupCohesionCalculator(
            weight=1.0, group_max_distance=5.0, violation_penalty=-2.0
        )
        cache = compute_distances(
            simple_env.wargame_models,
            simple_env.objectives,
            compute_model_model=True,
        )
        ctx = _make_step_context(simple_env, cache)
        reward = calc.calculate(0, m0, simple_env, ctx)
        assert reward < 0.0
        expected = -2.0 * (15.0 - 5.0)
        assert reward == pytest.approx(expected)

    def test_needs_model_model_distances(self) -> None:
        calc = GroupCohesionCalculator()
        assert calc.needs_model_model_distances is True

    def test_zero_when_no_model_model_norms(self, simple_env: WargameEnv) -> None:
        simple_env.reset()
        calc = GroupCohesionCalculator()
        cache = compute_distances(
            simple_env.wargame_models,
            simple_env.objectives,
            compute_model_model=False,
        )
        ctx = _make_step_context(simple_env, cache)
        reward = calc.calculate(0, simple_env.wargame_models[0], simple_env, ctx)
        assert reward == 0.0


# ---------------------------------------------------------------------------
# Calculator registry tests
# ---------------------------------------------------------------------------


class TestCalculatorRegistry:
    def test_known_types(self) -> None:
        assert "closest_objective" in CALCULATOR_REGISTRY
        assert "group_cohesion" in CALCULATOR_REGISTRY

    def test_build_closest_objective(self) -> None:
        calc = build_calculator("closest_objective", weight=0.7, params={})
        assert isinstance(calc, ClosestObjectiveCalculator)
        assert calc.weight == 0.7

    def test_build_group_cohesion_with_params(self) -> None:
        calc = build_calculator(
            "group_cohesion",
            weight=1.0,
            params={"group_max_distance": 3.0, "violation_penalty": -5.0},
        )
        assert isinstance(calc, GroupCohesionCalculator)
        assert calc.group_max_distance == 3.0
        assert calc.violation_penalty == -5.0

    def test_unknown_type_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown reward calculator"):
            build_calculator("nonexistent", weight=1.0, params={})


# ---------------------------------------------------------------------------
# Criteria tests
# ---------------------------------------------------------------------------


class TestAllAtObjectivesCriteria:
    def test_not_successful_when_far(self, simple_env: WargameEnv) -> None:
        simple_env.reset()
        simple_env.wargame_models[0].location = np.array([0, 0])
        simple_env.wargame_models[1].location = np.array([1, 0])
        simple_env.objectives[0].location = np.array([19, 19])

        cache = compute_distances(simple_env.wargame_models, simple_env.objectives)
        ctx = _make_step_context(simple_env, cache)
        criteria = AllAtObjectivesCriteria()
        assert criteria.is_successful(simple_env, ctx) is False

    def test_successful_when_at_objective(self, simple_env: WargameEnv) -> None:
        simple_env.reset()
        obj_loc = simple_env.objectives[0].location.copy()
        simple_env.wargame_models[0].location = obj_loc.copy()
        simple_env.wargame_models[1].location = obj_loc.copy()

        cache = compute_distances(simple_env.wargame_models, simple_env.objectives)
        ctx = _make_step_context(simple_env, cache)
        criteria = AllAtObjectivesCriteria()
        assert criteria.is_successful(simple_env, ctx) is True


class TestAllModelsGroupedCriteria:
    def test_grouped_when_close(self, simple_env: WargameEnv) -> None:
        simple_env.reset()
        simple_env.wargame_models[0].location = np.array([5, 5])
        simple_env.wargame_models[1].location = np.array([7, 5])
        simple_env.wargame_models[0].group_id = 0
        simple_env.wargame_models[1].group_id = 0

        cache = compute_distances(
            simple_env.wargame_models,
            simple_env.objectives,
            compute_model_model=True,
        )
        ctx = _make_step_context(simple_env, cache)
        criteria = AllModelsGroupedCriteria(max_distance=10.0)
        assert criteria.is_successful(simple_env, ctx) is True

    def test_not_grouped_when_far(self, simple_env: WargameEnv) -> None:
        simple_env.reset()
        simple_env.wargame_models[0].location = np.array([0, 0])
        simple_env.wargame_models[1].location = np.array([19, 19])
        simple_env.wargame_models[0].group_id = 0
        simple_env.wargame_models[1].group_id = 0

        cache = compute_distances(
            simple_env.wargame_models,
            simple_env.objectives,
            compute_model_model=True,
        )
        ctx = _make_step_context(simple_env, cache)
        criteria = AllModelsGroupedCriteria(max_distance=5.0)
        assert criteria.is_successful(simple_env, ctx) is False

    def test_different_groups_dont_need_proximity(self, simple_env: WargameEnv) -> None:
        simple_env.reset()
        simple_env.wargame_models[0].location = np.array([0, 0])
        simple_env.wargame_models[1].location = np.array([19, 19])
        simple_env.wargame_models[0].group_id = 0
        simple_env.wargame_models[1].group_id = 1

        cache = compute_distances(
            simple_env.wargame_models,
            simple_env.objectives,
            compute_model_model=True,
        )
        ctx = _make_step_context(simple_env, cache)
        criteria = AllModelsGroupedCriteria(max_distance=5.0)
        assert criteria.is_successful(simple_env, ctx) is True


# ---------------------------------------------------------------------------
# Criteria registry tests
# ---------------------------------------------------------------------------


class TestCriteriaRegistry:
    def test_known_types(self) -> None:
        assert "all_at_objectives" in CRITERIA_REGISTRY
        assert "all_models_grouped" in CRITERIA_REGISTRY

    def test_build_all_at_objectives(self) -> None:
        c = build_criteria("all_at_objectives", {})
        assert isinstance(c, AllAtObjectivesCriteria)

    def test_build_all_models_grouped(self) -> None:
        c = build_criteria("all_models_grouped", {"max_distance": 7.0})
        assert isinstance(c, AllModelsGroupedCriteria)
        assert c.max_distance == 7.0

    def test_unknown_type_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown success criteria"):
            build_criteria("nonexistent", {})


# ---------------------------------------------------------------------------
# Phase config tests
# ---------------------------------------------------------------------------


class TestPhaseConfig:
    def test_serialization_round_trip(self) -> None:
        cfg = RewardPhaseConfig(
            name="test_phase",
            reward_calculators=[
                RewardCalculatorConfig(type="closest_objective", weight=0.5),
            ],
            success_criteria=SuccessCriteriaConfig(type="all_at_objectives"),
            success_threshold=0.7,
            min_epochs=5,
        )
        data = cfg.model_dump()
        restored = RewardPhaseConfig(**data)
        assert restored.name == "test_phase"
        assert restored.success_threshold == 0.7
        assert restored.min_epochs == 5
        assert len(restored.reward_calculators) == 1


# ---------------------------------------------------------------------------
# Phase manager tests
# ---------------------------------------------------------------------------


class TestRewardPhaseManager:
    @pytest.fixture
    def two_phase_manager(self) -> RewardPhaseManager:
        configs = [
            RewardPhaseConfig(
                name="phase_one",
                reward_calculators=[
                    RewardCalculatorConfig(
                        type="group_cohesion",
                        weight=1.0,
                        params={
                            "group_max_distance": 5.0,
                            "violation_penalty": -1.0,
                        },
                    ),
                ],
                success_criteria=SuccessCriteriaConfig(
                    type="all_models_grouped",
                    params={"max_distance": 5.0},
                ),
                success_threshold=0.8,
                min_epochs=2,
            ),
            RewardPhaseConfig(
                name="phase_two",
                reward_calculators=[
                    RewardCalculatorConfig(type="closest_objective", weight=1.0),
                ],
                success_criteria=SuccessCriteriaConfig(
                    type="all_at_objectives",
                ),
                success_threshold=0.9,
                min_epochs=0,
            ),
        ]
        return RewardPhaseManager.from_configs(configs)

    def test_initial_state(self, two_phase_manager: RewardPhaseManager) -> None:
        assert two_phase_manager.current_phase_name == "phase_one"
        assert two_phase_manager.current_phase_index == 0
        assert two_phase_manager.is_final_phase is False

    def test_needs_model_model_distances(
        self, two_phase_manager: RewardPhaseManager
    ) -> None:
        assert two_phase_manager.needs_model_model_distances is True

    def test_no_advance_before_min_epochs(
        self, two_phase_manager: RewardPhaseManager
    ) -> None:
        advanced = two_phase_manager.try_advance(success_rate=1.0, current_epoch=1)
        assert advanced is False
        assert two_phase_manager.current_phase_name == "phase_one"

    def test_no_advance_below_threshold(
        self, two_phase_manager: RewardPhaseManager
    ) -> None:
        advanced = two_phase_manager.try_advance(success_rate=0.5, current_epoch=10)
        assert advanced is False

    def test_advance_when_conditions_met(
        self, two_phase_manager: RewardPhaseManager
    ) -> None:
        advanced = two_phase_manager.try_advance(success_rate=0.9, current_epoch=5)
        assert advanced is True
        assert two_phase_manager.current_phase_name == "phase_two"
        assert two_phase_manager.is_final_phase is True

    def test_no_advance_past_final(self, two_phase_manager: RewardPhaseManager) -> None:
        two_phase_manager.try_advance(success_rate=0.9, current_epoch=5)
        advanced = two_phase_manager.try_advance(success_rate=1.0, current_epoch=100)
        assert advanced is False

    def test_calculate_reward(self, phased_env: WargameEnv) -> None:
        phased_env.reset()
        action = WargameEnvAction(actions=phased_env.action_space.sample())
        _, reward, _, _, _ = phased_env.step(action)
        assert isinstance(reward, float)

    def test_check_success(self, phased_env: WargameEnv) -> None:
        phased_env.reset()
        action = WargameEnvAction(actions=phased_env.action_space.sample())
        phased_env.step(action)
        assert phased_env.phase_manager is not None
        assert phased_env.last_step_context is not None
        result = phased_env.phase_manager.check_success(
            phased_env, phased_env.last_step_context
        )
        assert isinstance(result, bool)

    def test_empty_configs_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one phase"):
            RewardPhaseManager.from_configs([])


# ---------------------------------------------------------------------------
# Environment integration tests
# ---------------------------------------------------------------------------


class TestEnvIntegration:
    def test_legacy_env_has_no_phase_manager(self, simple_env: WargameEnv) -> None:
        assert simple_env.phase_manager is None
        assert simple_env.last_step_context is None

    def test_phased_env_has_phase_manager(self, phased_env: WargameEnv) -> None:
        assert phased_env.phase_manager is not None
        assert phased_env.phase_manager.current_phase_name == "group_up"

    def test_legacy_step_unchanged(self, simple_env: WargameEnv) -> None:
        simple_env.reset()
        action = WargameEnvAction(actions=simple_env.action_space.sample())
        obs, reward, terminated, truncated, info = simple_env.step(action)
        assert isinstance(reward, float)
        assert truncated is False
        assert simple_env.last_step_context is None

    def test_phased_step_stores_context(self, phased_env: WargameEnv) -> None:
        phased_env.reset()
        assert phased_env.last_step_context is None
        action = WargameEnvAction(actions=phased_env.action_space.sample())
        phased_env.step(action)
        assert phased_env.last_step_context is not None

    def test_reset_clears_context(self, phased_env: WargameEnv) -> None:
        phased_env.reset()
        action = WargameEnvAction(actions=phased_env.action_space.sample())
        phased_env.step(action)
        assert phased_env.last_step_context is not None
        phased_env.reset()
        assert phased_env.last_step_context is None

    def test_phased_env_runs_full_episode(self, phased_env: WargameEnv) -> None:
        """Ensure a phased env can run to termination without errors."""
        phased_env.reset()
        done = False
        steps = 0
        while not done:
            action = WargameEnvAction(actions=phased_env.action_space.sample())
            _, _, done, _, _ = phased_env.step(action)
            steps += 1
            if steps > phased_env.max_turns + 5:
                pytest.fail("Episode did not terminate")
        assert steps > 0

    def test_phased_reward_matches_legacy_reward(self) -> None:
        legacy_config = WargameEnvConfig(
            render_mode=None,
            board_width=20,
            board_height=20,
            number_of_wargame_models=2,
            number_of_objectives=1,
            objective_radius_size=2,
            group_cohesion_enabled=True,
            group_max_distance=3.0,
            group_violation_penalty=-0.1,
        )
        phased_config = WargameEnvConfig(
            render_mode=None,
            board_width=20,
            board_height=20,
            number_of_wargame_models=2,
            number_of_objectives=1,
            objective_radius_size=2,
            group_cohesion_enabled=True,
            group_max_distance=3.0,
            group_violation_penalty=-0.1,
            reward_phases=[
                RewardPhaseConfig(
                    name="legacy_equivalent",
                    reward_calculators=[
                        RewardCalculatorConfig(type="closest_objective", weight=1.0),
                        RewardCalculatorConfig(type="group_cohesion", weight=1.0),
                    ],
                    success_criteria=SuccessCriteriaConfig(type="all_at_objectives"),
                    success_threshold=1.0,
                    min_epochs=0,
                )
            ],
        )
        legacy_env = WargameEnv(config=legacy_config)
        phased_env = WargameEnv(config=phased_config)

        legacy_env.reset()
        phased_env.reset()

        obj_loc = np.array([10, 10])
        legacy_env.objectives[0].location = obj_loc.copy()
        phased_env.objectives[0].location = obj_loc.copy()

        positions = [
            (np.array([5, 5]), np.array([6, 5])),
            (np.array([0, 0]), np.array([10, 0])),
            (np.array([9, 9]), np.array([9, 10])),
        ]
        for p0, p1 in positions:
            legacy_env.wargame_models[0].location = p0.copy()
            legacy_env.wargame_models[1].location = p1.copy()
            phased_env.wargame_models[0].location = p0.copy()
            phased_env.wargame_models[1].location = p1.copy()

            for env in (legacy_env, phased_env):
                env.wargame_models[0].group_id = 0
                env.wargame_models[1].group_id = 0

            legacy_cache = compute_distances(
                legacy_env.wargame_models,
                legacy_env.objectives,
                compute_model_model=True,
            )
            phased_cache = compute_distances(
                phased_env.wargame_models,
                phased_env.objectives,
                compute_model_model=True,
            )

            legacy_reward = Reward().calculate_reward(legacy_env, legacy_cache)
            assert phased_env.phase_manager is not None
            phased_ctx = _make_step_context(phased_env, phased_cache)
            phased_reward = phased_env.phase_manager.calculate_reward(
                phased_env, phased_ctx
            )

            assert legacy_reward == pytest.approx(phased_reward)

    def test_phased_terminal_success_bonus_applied(self) -> None:
        config = WargameEnvConfig(
            render_mode=None,
            board_width=20,
            board_height=20,
            number_of_wargame_models=2,
            number_of_objectives=1,
            objective_radius_size=2,
            group_cohesion_enabled=True,
            group_max_distance=3.0,
            group_violation_penalty=-0.1,
            max_turns_override=1,
            terminal_success_bonus=25.0,
            reward_phases=[
                RewardPhaseConfig(
                    name="legacy_equivalent",
                    reward_calculators=[
                        RewardCalculatorConfig(type="closest_objective", weight=1.0),
                        RewardCalculatorConfig(type="group_cohesion", weight=1.0),
                    ],
                    success_criteria=SuccessCriteriaConfig(type="all_at_objectives"),
                    success_threshold=1.0,
                    min_epochs=0,
                )
            ],
        )
        env = WargameEnv(config=config)
        env.reset()

        obj_loc = np.array([10, 10])
        env.objectives[0].location = obj_loc.copy()
        for model in env.wargame_models:
            model.location = obj_loc.copy()
            model.group_id = 0

        stay_actions = np.zeros(len(env.action_space.spaces), dtype=int)
        action = WargameEnvAction(actions=stay_actions)
        _, reward, terminated, _, _ = env.step(action)

        assert terminated is True
        assert reward == pytest.approx(25.0)


# ---------------------------------------------------------------------------
# WargameEnvConfig backward compatibility
# ---------------------------------------------------------------------------


class TestConfigBackwardCompat:
    def test_no_reward_phases_field_default(self) -> None:
        config = WargameEnvConfig(render_mode=None)
        assert config.reward_phases is None

    def test_reward_phases_parsed_from_dict(self) -> None:
        data = {
            "render_mode": None,
            "reward_phases": [
                {
                    "name": "test",
                    "reward_calculators": [{"type": "closest_objective"}],
                    "success_criteria": {"type": "all_at_objectives"},
                }
            ],
        }
        config = WargameEnvConfig.model_validate(data)
        assert config.reward_phases is not None
        assert len(config.reward_phases) == 1
        assert config.reward_phases[0].name == "test"
