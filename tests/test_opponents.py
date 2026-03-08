"""Tests for the opponent system: config, policies, env integration, observation, backward compatibility."""

import numpy as np
import pytest

from wargame_rl.wargame.envs.env_components.actions import STAY_ACTION, ActionHandler
from wargame_rl.wargame.envs.opponent.random_policy import RandomPolicy
from wargame_rl.wargame.envs.opponent.registry import (
    build_opponent_policy,
    get_registry,
)
from wargame_rl.wargame.envs.opponent.scripted_advance_to_objective_policy import (
    ScriptedAdvanceToObjectivePolicy,
)
from wargame_rl.wargame.envs.types import (
    OpponentPolicyConfig,
    TurnOrder,
    WargameEnvAction,
    WargameEnvConfig,
)
from wargame_rl.wargame.envs.types.config import ModelConfig, ObjectiveConfig
from wargame_rl.wargame.envs.wargame import WargameEnv
from wargame_rl.wargame.model.common.observation import observation_to_tensor
from wargame_rl.wargame.model.net import MLPNetwork, TransformerNetwork


def _make_opponent_config(**overrides: object) -> WargameEnvConfig:
    defaults: dict = dict(
        board_width=20,
        board_height=20,
        number_of_wargame_models=2,
        number_of_objectives=2,
        number_of_opponent_models=2,
        objective_radius_size=2,
        render_mode=None,
        opponent_policy=OpponentPolicyConfig(type="random"),
        number_of_battle_rounds=40,
    )
    defaults.update(overrides)
    return WargameEnvConfig(**defaults)


def _make_no_opponent_config(**overrides: object) -> WargameEnvConfig:
    defaults: dict = dict(
        board_width=20,
        board_height=20,
        number_of_wargame_models=2,
        number_of_objectives=2,
        objective_radius_size=2,
        render_mode=None,
        number_of_battle_rounds=40,
    )
    defaults.update(overrides)
    return WargameEnvConfig(**defaults)


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


class TestConfigValidation:
    def test_opponent_policy_required_when_opponents_exist(self) -> None:
        with pytest.raises(ValueError, match="opponent_policy must be set"):
            WargameEnvConfig(
                number_of_opponent_models=2,
                objective_radius_size=2,
            )

    def test_opponent_policy_not_required_when_no_opponents(self) -> None:
        cfg = _make_no_opponent_config()
        assert cfg.number_of_opponent_models == 0
        assert cfg.opponent_policy is None

    def test_opponent_models_count_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="opponent_models has"):
            WargameEnvConfig(
                number_of_opponent_models=2,
                objective_radius_size=2,
                opponent_policy=OpponentPolicyConfig(type="random"),
                opponent_models=[ModelConfig(group_id=0)],
            )

    def test_opponent_models_mixed_coords_raises(self) -> None:
        with pytest.raises(ValueError, match="all opponent_models must have x/y"):
            WargameEnvConfig(
                board_width=20,
                board_height=20,
                number_of_opponent_models=2,
                objective_radius_size=2,
                opponent_policy=OpponentPolicyConfig(type="random"),
                opponent_models=[
                    ModelConfig(x=5, y=5, group_id=0),
                    ModelConfig(group_id=0),
                ],
            )

    def test_opponent_models_out_of_bounds_raises(self) -> None:
        with pytest.raises(ValueError, match="opponent_models.*outside"):
            WargameEnvConfig(
                board_width=10,
                board_height=10,
                number_of_opponent_models=1,
                objective_radius_size=2,
                opponent_policy=OpponentPolicyConfig(type="random"),
                opponent_models=[ModelConfig(x=15, y=5, group_id=0)],
            )

    def test_valid_opponent_config(self) -> None:
        cfg = _make_opponent_config()
        assert cfg.number_of_opponent_models == 2
        assert cfg.opponent_policy is not None
        assert cfg.opponent_policy.type == "random"

    def test_turn_order_enum_values(self) -> None:
        assert TurnOrder.player.value == "player"
        assert TurnOrder.opponent.value == "opponent"
        assert TurnOrder.random.value == "random"

    def test_turn_order_defaults_to_player(self) -> None:
        cfg = _make_opponent_config()
        assert cfg.turn_order == TurnOrder.player

    def test_has_fixed_opponent_positions_false_by_default(self) -> None:
        cfg = _make_opponent_config()
        assert not cfg.has_fixed_opponent_positions

    def test_has_fixed_opponent_positions_true(self) -> None:
        cfg = _make_opponent_config(
            opponent_models=[
                ModelConfig(x=15, y=5, group_id=0),
                ModelConfig(x=15, y=10, group_id=0),
            ],
        )
        assert cfg.has_fixed_opponent_positions


# ---------------------------------------------------------------------------
# Policy registry
# ---------------------------------------------------------------------------


class TestPolicyRegistry:
    def test_random_registered(self) -> None:
        registry = get_registry()
        assert "random" in registry
        assert registry["random"] is RandomPolicy

    def test_scripted_advance_registered(self) -> None:
        registry = get_registry()
        assert "scripted_advance_to_objective" in registry
        assert (
            registry["scripted_advance_to_objective"]
            is ScriptedAdvanceToObjectivePolicy
        )

    def test_unknown_policy_raises(self) -> None:
        cfg = OpponentPolicyConfig(type="nonexistent")
        env = WargameEnv(config=_make_opponent_config())
        with pytest.raises(ValueError, match="Unknown opponent policy"):
            build_opponent_policy(cfg, env)

    def test_build_random_policy(self) -> None:
        cfg = OpponentPolicyConfig(type="random")
        env = WargameEnv(config=_make_opponent_config())
        policy = build_opponent_policy(cfg, env)
        assert isinstance(policy, RandomPolicy)


# ---------------------------------------------------------------------------
# RandomPolicy
# ---------------------------------------------------------------------------


class TestRandomPolicy:
    def test_returns_valid_action(self) -> None:
        env = WargameEnv(config=_make_opponent_config())
        env.reset(seed=42)
        policy = RandomPolicy(env=env)
        action = policy.select_action(env.opponent_models, env)
        assert isinstance(action, WargameEnvAction)
        assert len(action.actions) == env.config.number_of_opponent_models


# ---------------------------------------------------------------------------
# ScriptedAdvanceToObjectivePolicy
# ---------------------------------------------------------------------------


class TestScriptedAdvanceToObjectivePolicy:
    def test_returns_valid_action(self) -> None:
        cfg = _make_opponent_config(
            opponent_policy=OpponentPolicyConfig(type="scripted_advance_to_objective"),
        )
        env = WargameEnv(config=cfg)
        env.reset(seed=42)
        policy = ScriptedAdvanceToObjectivePolicy(env=env)
        action = policy.select_action(env.opponent_models, env)
        assert isinstance(action, WargameEnvAction)
        assert len(action.actions) == env.config.number_of_opponent_models

    def test_models_move_toward_objectives(self) -> None:
        """After several steps, opponent models should be closer to objectives."""
        cfg = WargameEnvConfig(
            board_width=40,
            board_height=40,
            number_of_wargame_models=1,
            number_of_objectives=1,
            number_of_opponent_models=1,
            objective_radius_size=2,
            opponent_policy=OpponentPolicyConfig(type="scripted_advance_to_objective"),
            opponent_models=[ModelConfig(x=35, y=20, group_id=0)],
            models=[ModelConfig(x=5, y=20, group_id=0)],
            objectives=[ObjectiveConfig(x=20, y=20)],
        )
        env = WargameEnv(config=cfg)
        env.reset(seed=42)

        initial_dist = np.linalg.norm(
            env.opponent_models[0].location - env.objectives[0].location
        )
        for _ in range(5):
            action = WargameEnvAction(actions=[0])  # player stays
            env.step(action)

        final_dist = np.linalg.norm(
            env.opponent_models[0].location - env.objectives[0].location
        )
        assert final_dist < initial_dist

    def test_scripted_opponent_does_not_overshoot_objective_when_close(self) -> None:
        """When close to an objective, scripted opponent uses reduced speed and does not overshoot."""
        cfg = WargameEnvConfig(
            board_width=40,
            board_height=40,
            number_of_wargame_models=1,
            number_of_objectives=1,
            number_of_opponent_models=1,
            objective_radius_size=3,
            max_move_speed=6.0,
            n_speed_bins=6,
            opponent_policy=OpponentPolicyConfig(type="scripted_advance_to_objective"),
            opponent_models=[
                ModelConfig(x=24, y=20, group_id=0)
            ],  # 4 cells E; 1 cell outside radius 3
            models=[ModelConfig(x=5, y=20, group_id=0)],
            objectives=[ObjectiveConfig(x=20, y=20)],
        )
        env = WargameEnv(config=cfg)
        env.reset(seed=42)
        obj_loc = env.objectives[0].location
        radius = env.objectives[0].radius_size
        opponent = env.opponent_models[0]
        dist_before = np.linalg.norm(opponent.location - obj_loc)
        assert dist_before > radius, "opponent must start outside capture radius"
        # One step: player stays, opponent acts
        env.step(WargameEnvAction(actions=[0]))
        dist_after = np.linalg.norm(env.opponent_models[0].location - obj_loc)
        # Should not overshoot: either inside radius or closer than before
        assert dist_after <= radius or dist_after <= dist_before, (
            "scripted opponent overshot objective"
        )

    def test_scripted_opponent_lands_inside_objective_radius(self) -> None:
        """Scripted opponent advancing toward an objective eventually lands inside its radius."""
        cfg = WargameEnvConfig(
            board_width=40,
            board_height=40,
            number_of_wargame_models=1,
            number_of_objectives=1,
            number_of_opponent_models=1,
            objective_radius_size=3,
            opponent_policy=OpponentPolicyConfig(type="scripted_advance_to_objective"),
            opponent_models=[ModelConfig(x=30, y=20, group_id=0)],
            models=[ModelConfig(x=5, y=20, group_id=0)],
            objectives=[ObjectiveConfig(x=20, y=20)],
        )
        env = WargameEnv(config=cfg)
        env.reset(seed=42)
        obj_loc = env.objectives[0].location
        radius = env.objectives[0].radius_size
        for _ in range(30):
            dist = np.linalg.norm(env.opponent_models[0].location - obj_loc)
            if dist < radius:
                break
            env.step(WargameEnvAction(actions=[0]))
        dist_final = np.linalg.norm(env.opponent_models[0].location - obj_loc)
        assert dist_final < radius, (
            "scripted opponent should land inside objective radius"
        )


# ---------------------------------------------------------------------------
# ActionHandler extensions
# ---------------------------------------------------------------------------


class TestActionHandlerExtensions:
    def test_n_models_override(self) -> None:
        cfg = _make_no_opponent_config()
        handler = ActionHandler(cfg, n_models=3)
        assert len(handler.action_space) == 3

    def test_best_action_toward_east(self) -> None:
        cfg = _make_no_opponent_config()
        handler = ActionHandler(cfg)
        action = handler.best_action_toward(10.0, 0.0)
        assert action != 0  # not stay

    def test_best_action_toward_zero_is_stay(self) -> None:
        cfg = _make_no_opponent_config()
        handler = ActionHandler(cfg)
        action = handler.best_action_toward(0.0, 0.0)
        assert action == 0

    def test_encode_decode_roundtrip(self) -> None:
        cfg = _make_no_opponent_config()
        handler = ActionHandler(cfg)
        for a_idx in range(cfg.n_movement_angles):
            for s_idx in range(cfg.n_speed_bins):
                encoded = handler.encode_action(a_idx, s_idx)
                assert 1 <= encoded <= handler.n_actions - 1

    def test_best_action_toward_max_distance_returns_stay_when_no_speed_fits(
        self,
    ) -> None:
        cfg = _make_no_opponent_config()
        handler = ActionHandler(cfg)
        action = handler.best_action_toward(1.0, 0.0, max_distance=0.5)
        assert action == STAY_ACTION

    def test_best_action_toward_max_distance_caps_displacement(self) -> None:
        cfg = _make_no_opponent_config()
        handler = ActionHandler(cfg)
        action = handler.best_action_toward(1.0, 0.0, max_distance=2.0)
        displacement = handler._decode_action(action)
        magnitude = float(np.linalg.norm(displacement))
        assert magnitude <= 2.0 or action == STAY_ACTION


# ---------------------------------------------------------------------------
# Environment integration
# ---------------------------------------------------------------------------


class TestEnvIntegration:
    def test_reset_places_opponent_models(self) -> None:
        env = WargameEnv(config=_make_opponent_config())
        env.reset(seed=42)
        assert len(env.opponent_models) == 2
        for m in env.opponent_models:
            assert (m.location >= 0).all()

    def test_step_moves_opponent_models(self) -> None:
        env = WargameEnv(config=_make_opponent_config())
        env.reset(seed=42)
        initial_locs = [m.location.copy() for m in env.opponent_models]
        for _ in range(10):
            action = WargameEnvAction(actions=list(env.action_space.sample()))
            env.step(action)
        # At least one opponent should have moved (random policy, 10 steps)
        moved = any(
            not (m.location == loc).all()
            for m, loc in zip(env.opponent_models, initial_locs)
        )
        assert moved

    def test_observation_includes_opponent_models(self) -> None:
        env = WargameEnv(config=_make_opponent_config())
        obs, _ = env.reset(seed=42)
        assert hasattr(obs, "opponent_models")
        assert len(obs.opponent_models) == 2

    def test_info_includes_opponent_models(self) -> None:
        env = WargameEnv(config=_make_opponent_config())
        _, info = env.reset(seed=42)
        assert "opponent_models" in info

    def test_opponent_action_space_sized_correctly(self) -> None:
        env = WargameEnv(config=_make_opponent_config())
        assert len(env.opponent_action_space) == 2

    def test_turn_order_player(self) -> None:
        cfg = _make_opponent_config(turn_order=TurnOrder.player)
        assert cfg.turn_order == TurnOrder.player

    def test_turn_order_opponent(self) -> None:
        cfg = _make_opponent_config(turn_order=TurnOrder.opponent)
        assert cfg.turn_order == TurnOrder.opponent

    def test_turn_order_random(self) -> None:
        cfg = _make_opponent_config(turn_order=TurnOrder.random)
        assert cfg.turn_order == TurnOrder.random

    def test_fixed_opponent_placement(self) -> None:
        cfg = _make_opponent_config(
            opponent_models=[
                ModelConfig(x=15, y=5, group_id=0),
                ModelConfig(x=15, y=10, group_id=0),
            ],
        )
        env = WargameEnv(config=cfg)
        env.reset(seed=42)
        assert (env.opponent_models[0].location == np.array([15, 5])).all()
        assert (env.opponent_models[1].location == np.array([15, 10])).all()


# ---------------------------------------------------------------------------
# Backward compatibility (no opponents)
# ---------------------------------------------------------------------------


class TestBackwardCompatibility:
    def test_no_opponents_observation_has_empty_list(self) -> None:
        env = WargameEnv(config=_make_no_opponent_config())
        obs, _ = env.reset(seed=42)
        assert obs.opponent_models == []

    def test_no_opponents_step_works(self) -> None:
        env = WargameEnv(config=_make_no_opponent_config())
        env.reset(seed=42)
        action = WargameEnvAction(actions=list(env.action_space.sample()))
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs is not None
        assert isinstance(reward, float)

    def test_no_opponents_env_has_empty_opponent_list(self) -> None:
        env = WargameEnv(config=_make_no_opponent_config())
        assert env.opponent_models == []

    def test_no_opponents_opponent_action_space_empty(self) -> None:
        env = WargameEnv(config=_make_no_opponent_config())
        assert len(env.opponent_action_space) == 0


# ---------------------------------------------------------------------------
# Observation tensors
# ---------------------------------------------------------------------------


class TestObservationTensors:
    def test_tensor_list_has_five_elements_with_opponents(self) -> None:
        env = WargameEnv(config=_make_opponent_config())
        obs, _ = env.reset(seed=42)
        tensors = observation_to_tensor(obs)
        assert len(tensors) == 5

    def test_tensor_list_has_five_elements_without_opponents(self) -> None:
        env = WargameEnv(config=_make_no_opponent_config())
        obs, _ = env.reset(seed=42)
        tensors = observation_to_tensor(obs)
        assert len(tensors) == 5
        assert tensors[3].shape[0] == 0  # 0 opponent models

    def test_opponent_tensor_shape(self) -> None:
        env = WargameEnv(config=_make_opponent_config())
        obs, _ = env.reset(seed=42)
        tensors = observation_to_tensor(obs)
        opp_tensor = tensors[3]
        assert opp_tensor.shape[0] == 2  # 2 opponent models
        assert opp_tensor.shape[1] == tensors[2].shape[1]  # same feature dim


# ---------------------------------------------------------------------------
# DQN networks with opponents
# ---------------------------------------------------------------------------


class TestDQNWithOpponents:
    def test_mlp_forward_with_opponents(self) -> None:
        env = WargameEnv(config=_make_opponent_config())
        net = MLPNetwork.policy_from_env(env)
        obs, _ = env.reset(seed=42)
        tensors = observation_to_tensor(obs, net.device)
        q_values = net(tensors)
        assert q_values.shape[0] == 1  # batch
        assert q_values.shape[1] == env.config.number_of_wargame_models

    def test_transformer_forward_with_opponents(self) -> None:
        env = WargameEnv(config=_make_opponent_config())
        net = TransformerNetwork.policy_from_env(env)
        obs, _ = env.reset(seed=42)
        tensors = observation_to_tensor(obs, net.device)
        q_values = net(tensors)
        assert q_values.shape[0] == 1
        assert q_values.shape[1] == env.config.number_of_wargame_models

    def test_mlp_forward_without_opponents(self) -> None:
        env = WargameEnv(config=_make_no_opponent_config())
        net = MLPNetwork.policy_from_env(env)
        obs, _ = env.reset(seed=42)
        tensors = observation_to_tensor(obs, net.device)
        q_values = net(tensors)
        assert q_values.shape[0] == 1
        assert q_values.shape[1] == env.config.number_of_wargame_models

    def test_transformer_forward_without_opponents(self) -> None:
        env = WargameEnv(config=_make_no_opponent_config())
        net = TransformerNetwork.policy_from_env(env)
        obs, _ = env.reset(seed=42)
        tensors = observation_to_tensor(obs, net.device)
        q_values = net(tensors)
        assert q_values.shape[0] == 1
        assert q_values.shape[1] == env.config.number_of_wargame_models
