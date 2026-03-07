"""Tests for the union action space with phase-aware action masking."""

from __future__ import annotations

import numpy as np
import pytest

from wargame_rl.wargame.envs.env_components.actions import (
    ActionHandler,
    ActionRegistry,
    ActionSlice,
)
from wargame_rl.wargame.envs.types import WargameEnvAction, WargameEnvConfig
from wargame_rl.wargame.envs.types.game_timing import BattlePhase
from wargame_rl.wargame.envs.wargame import WargameEnv

# ---------------------------------------------------------------------------
# ActionRegistry unit tests
# ---------------------------------------------------------------------------


class TestActionRegistry:
    def test_register_single_slice(self) -> None:
        reg = ActionRegistry()
        s = reg.register("stay", 1, frozenset(BattlePhase))
        assert s == ActionSlice("stay", 0, 1, frozenset(BattlePhase))
        assert reg.n_actions == 1

    def test_register_multiple_slices(self) -> None:
        reg = ActionRegistry()
        s1 = reg.register("stay", 1, frozenset(BattlePhase))
        s2 = reg.register("movement", 96, frozenset({BattlePhase.movement}))
        assert s1.start == 0 and s1.end == 1
        assert s2.start == 1 and s2.end == 97
        assert reg.n_actions == 97

    def test_duplicate_name_raises(self) -> None:
        reg = ActionRegistry()
        reg.register("stay", 1, frozenset(BattlePhase))
        with pytest.raises(ValueError, match="already registered"):
            reg.register("stay", 1, frozenset(BattlePhase))

    def test_slice_for(self) -> None:
        reg = ActionRegistry()
        reg.register("stay", 1, frozenset(BattlePhase))
        reg.register("movement", 96, frozenset({BattlePhase.movement}))
        assert reg.slice_for("stay").size == 1
        assert reg.slice_for("movement").size == 96

    def test_slices_property(self) -> None:
        reg = ActionRegistry()
        reg.register("a", 5, frozenset(BattlePhase))
        reg.register("b", 10, frozenset({BattlePhase.shooting}))
        slices = reg.slices
        assert len(slices) == 2
        assert slices[0].name == "a"
        assert slices[1].name == "b"


class TestActionMask:
    @pytest.fixture
    def registry(self) -> ActionRegistry:
        reg = ActionRegistry()
        reg.register("stay", 1, frozenset(BattlePhase))
        reg.register("movement", 96, frozenset({BattlePhase.movement}))
        return reg

    def test_movement_phase_all_valid(self, registry: ActionRegistry) -> None:
        mask = registry.get_action_mask(BattlePhase.movement)
        assert mask.shape == (97,)
        assert mask.all()

    def test_non_movement_phase_only_stay(self, registry: ActionRegistry) -> None:
        for phase in [
            BattlePhase.command,
            BattlePhase.shooting,
            BattlePhase.charge,
            BattlePhase.fight,
        ]:
            mask = registry.get_action_mask(phase)
            assert mask[0] is np.bool_(True)
            assert not mask[1:].any(), f"Non-stay actions should be masked in {phase}"

    def test_model_action_masks_shape(self, registry: ActionRegistry) -> None:
        masks = registry.get_model_action_masks(BattlePhase.movement, 3)
        assert masks.shape == (3, 97)
        assert masks.all()

    def test_model_action_masks_non_movement(self, registry: ActionRegistry) -> None:
        masks = registry.get_model_action_masks(BattlePhase.command, 2)
        assert masks.shape == (2, 97)
        assert masks[:, 0].all()
        assert not masks[:, 1:].any()


# ---------------------------------------------------------------------------
# ActionHandler with registry
# ---------------------------------------------------------------------------


class TestActionHandlerRegistry:
    def test_handler_has_registry(self) -> None:
        config = WargameEnvConfig()
        handler = ActionHandler(config)
        reg = handler.registry
        assert reg.n_actions == handler.n_actions

    def test_stay_and_movement_slices(self) -> None:
        config = WargameEnvConfig()
        handler = ActionHandler(config)
        reg = handler.registry
        stay = reg.slice_for("stay")
        movement = reg.slice_for("movement")
        assert stay.start == 0
        assert stay.end == 1
        assert movement.start == 1
        assert movement.end == handler.n_actions

    def test_n_actions_unchanged(self) -> None:
        config = WargameEnvConfig(n_movement_angles=16, n_speed_bins=6)
        handler = ActionHandler(config)
        assert handler.n_actions == 1 + 16 * 6


# ---------------------------------------------------------------------------
# Env integration: mask on observation
# ---------------------------------------------------------------------------


class TestEnvActionMask:
    def test_reset_includes_mask(self) -> None:
        env = WargameEnv(config=WargameEnvConfig())
        obs, _ = env.reset()
        assert obs.action_mask is not None
        n_models = len(obs.wargame_models)
        n_actions = env.n_actions
        assert obs.action_mask.shape == (n_models, n_actions)

    def test_step_includes_mask(self) -> None:
        env = WargameEnv(config=WargameEnvConfig())
        env.reset()
        action = WargameEnvAction(actions=list(env.action_space.sample()))
        obs, _, _, _, _ = env.step(action)
        assert obs.action_mask is not None
        assert obs.action_mask.shape == (
            len(obs.wargame_models),
            env.n_actions,
        )

    def test_mask_only_stay_during_command(self) -> None:
        """With skip_phases=[], reset lands on command; only stay is valid."""
        env = WargameEnv(config=WargameEnvConfig(skip_phases=[]))
        obs, _ = env.reset()
        assert obs.action_mask is not None
        assert obs.action_mask[:, 0].all()
        assert not obs.action_mask[:, 1:].any()

    def test_mask_all_valid_during_movement(self) -> None:
        """Default skip lands on movement after reset; all actions are valid."""
        env = WargameEnv(config=WargameEnvConfig())
        obs, _ = env.reset()
        assert obs.action_mask is not None
        assert obs.action_mask.all()


# ---------------------------------------------------------------------------
# Tensor pipeline
# ---------------------------------------------------------------------------


class TestTensorPipeline:
    def test_observation_to_tensor_returns_5_tensors(self) -> None:
        from wargame_rl.wargame.model.common.observation import observation_to_tensor

        env = WargameEnv(config=WargameEnvConfig())
        obs, _ = env.reset()
        tensors = observation_to_tensor(obs)
        assert len(tensors) == 5
        mask_tensor = tensors[4]
        assert mask_tensor.dtype == np.bool_ or str(mask_tensor.dtype) == "torch.bool"
        n_models = len(obs.wargame_models)
        n_actions = env.n_actions
        assert mask_tensor.shape == (n_models, n_actions)

    def test_batch_returns_5_tensors(self) -> None:
        from wargame_rl.wargame.model.common.observation import (
            observations_to_tensor_batch,
        )

        env = WargameEnv(config=WargameEnvConfig())
        obs1, _ = env.reset()
        action = WargameEnvAction(actions=list(env.action_space.sample()))
        obs2, _, _, _, _ = env.step(action)
        batch = observations_to_tensor_batch([obs1, obs2])
        assert len(batch) == 5
        assert batch[4].shape[0] == 2  # batch dim


# ---------------------------------------------------------------------------
# Experience batch with masks
# ---------------------------------------------------------------------------


class TestExperienceBatch:
    def test_batch_has_next_state_masks(self) -> None:
        from wargame_rl.wargame.model.common.dataset import experience_list_to_batch
        from wargame_rl.wargame.types import Experience

        env = WargameEnv(config=WargameEnvConfig())
        obs, _ = env.reset()
        action = WargameEnvAction(actions=list(env.action_space.sample()))
        new_obs, reward, done, _, _ = env.step(action)
        exp = Experience(obs, action, reward, done, new_obs, log_prob=None)
        batch = experience_list_to_batch([exp])
        assert batch.next_state_masks is not None
        assert batch.next_state_masks.shape[0] == 1  # batch of 1
        assert len(batch.state_tensors) == 4
        assert len(batch.new_state_tensors) == 4
