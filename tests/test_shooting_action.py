"""Tests for shooting action space: config types, registry, masking, and apply dispatch."""

from __future__ import annotations

import numpy as np
import pytest
from gymnasium import spaces
from pydantic import ValidationError

from wargame_rl.wargame.envs.domain.entities import WargameModel
from wargame_rl.wargame.envs.env_components.actions import (
    STAY_ACTION,
    ActionHandler,
)
from wargame_rl.wargame.envs.types import WargameEnvAction, WargameEnvConfig
from wargame_rl.wargame.envs.types.config import ModelConfig, WeaponProfile
from wargame_rl.wargame.envs.types.game_timing import BattlePhase

N_ANGLES = 8
N_SPEEDS = 3
N_MOVE_ACTIONS = N_ANGLES * N_SPEEDS  # 24
N_OPPONENTS = 4


def _make_model(x: int, y: int, max_wounds: int = 1) -> WargameModel:
    """Create a WargameModel at (x, y) with empty objective distances."""
    return WargameModel(
        location=np.array([x, y], dtype=np.int32),
        stats={"max_wounds": max_wounds, "current_wounds": max_wounds},
        distances_to_objectives=np.zeros((0, 2), dtype=np.int32),
        group_id=0,
    )


def _small_config(*, n_opponents: int = 0) -> WargameEnvConfig:
    """Config with small action space for faster tests."""
    kwargs: dict = {
        "n_movement_angles": N_ANGLES,
        "n_speed_bins": N_SPEEDS,
    }
    if n_opponents > 0:
        kwargs["number_of_opponent_models"] = n_opponents
        kwargs["opponent_policy"] = {"type": "random"}
    return WargameEnvConfig(**kwargs)


# ---------------------------------------------------------------------------
# WeaponProfile
# ---------------------------------------------------------------------------


class TestWeaponProfile:
    def test_valid_range(self) -> None:
        wp = WeaponProfile(range=24)
        assert wp.range == 24

    def test_zero_range_rejected(self) -> None:
        with pytest.raises(ValidationError):
            WeaponProfile(range=0)

    def test_negative_range_rejected(self) -> None:
        with pytest.raises(ValidationError):
            WeaponProfile(range=-1)


# ---------------------------------------------------------------------------
# ModelConfig.weapons
# ---------------------------------------------------------------------------


class TestModelConfigWeapons:
    def test_default_empty(self) -> None:
        assert ModelConfig().weapons == []

    def test_with_weapons(self) -> None:
        m = ModelConfig(weapons=[WeaponProfile(range=24)])
        assert m.weapons[0].range == 24

    def test_backward_compat_no_weapons_key(self) -> None:
        m = ModelConfig.model_validate({"group_id": 1})
        assert m.weapons == []


# ---------------------------------------------------------------------------
# Shooting slice registration
# ---------------------------------------------------------------------------


class TestShootingSliceRegistration:
    def test_shooting_slice_registered(self) -> None:
        cfg = _small_config(n_opponents=N_OPPONENTS)
        h = ActionHandler(cfg, n_shoot_targets=N_OPPONENTS)
        s = h.registry.slice_for("shooting")
        assert s.size == N_OPPONENTS

    def test_shooting_slice_phase_gating(self) -> None:
        cfg = _small_config(n_opponents=N_OPPONENTS)
        h = ActionHandler(cfg, n_shoot_targets=N_OPPONENTS)
        s = h.registry.slice_for("shooting")
        assert s.valid_phases == frozenset({BattlePhase.shooting})

    def test_no_shooting_slice_when_zero(self) -> None:
        cfg = _small_config()
        h = ActionHandler(cfg, n_shoot_targets=0)
        assert h.shooting_slice is None

    def test_n_actions_grows(self) -> None:
        cfg_no = _small_config()
        cfg_yes = _small_config(n_opponents=N_OPPONENTS)
        h_no = ActionHandler(cfg_no)
        h_yes = ActionHandler(cfg_yes, n_shoot_targets=N_OPPONENTS)
        assert h_yes.n_actions - h_no.n_actions == N_OPPONENTS

    def test_shooting_slice_contiguous_after_movement(self) -> None:
        cfg = _small_config(n_opponents=N_OPPONENTS)
        h = ActionHandler(cfg, n_shoot_targets=N_OPPONENTS)
        assert h.shooting_slice is not None
        assert h.shooting_slice.start == 1 + N_MOVE_ACTIONS


# ---------------------------------------------------------------------------
# Shooting action masks
# ---------------------------------------------------------------------------


class TestShootingActionMask:
    @pytest.fixture
    def handler(self) -> ActionHandler:
        cfg = _small_config(n_opponents=N_OPPONENTS)
        return ActionHandler(cfg, n_shoot_targets=N_OPPONENTS)

    def test_movement_phase_shooting_masked(self, handler: ActionHandler) -> None:
        mask = handler.registry.get_action_mask(BattlePhase.movement)
        s = handler.shooting_slice
        assert s is not None
        assert not mask[s.start : s.end].any()

    def test_shooting_phase_movement_masked(self, handler: ActionHandler) -> None:
        mask = handler.registry.get_action_mask(BattlePhase.shooting)
        move = handler.registry.slice_for("movement")
        s = handler.shooting_slice
        assert s is not None
        assert not mask[move.start : move.end].any()
        assert mask[s.start : s.end].all()

    def test_stay_valid_in_shooting_phase(self, handler: ActionHandler) -> None:
        mask = handler.registry.get_action_mask(BattlePhase.shooting)
        assert mask[STAY_ACTION]

    def test_dead_model_stay_only_in_shooting(self, handler: ActionHandler) -> None:
        alive = np.array([True, False], dtype=bool)
        masks = handler.registry.get_model_action_masks(
            BattlePhase.shooting, 2, alive_mask=alive
        )
        # Dead model: only STAY
        assert masks[1, STAY_ACTION]
        assert not masks[1, 1:].any()
        # Alive model: stay + shooting
        assert masks[0, STAY_ACTION]
        s = handler.shooting_slice
        assert s is not None
        assert masks[0, s.start : s.end].all()

    def test_all_phases_movement_only_when_no_shooting_slice(self) -> None:
        cfg = _small_config()
        h = ActionHandler(cfg)
        for phase in [BattlePhase.command, BattlePhase.shooting, BattlePhase.charge]:
            mask = h.registry.get_action_mask(phase)
            assert mask[STAY_ACTION]
            assert not mask[1:].any()


# ---------------------------------------------------------------------------
# Phase-aware apply
# ---------------------------------------------------------------------------


class TestPhaseAwareApply:
    def _action_space(self, handler: ActionHandler, n: int) -> spaces.Tuple:
        return spaces.Tuple(
            [spaces.Discrete(handler.n_actions) for _ in range(n)]
        )

    def test_shooting_action_noop(self) -> None:
        cfg = _small_config(n_opponents=N_OPPONENTS)
        h = ActionHandler(cfg, n_shoot_targets=N_OPPONENTS)
        model = _make_model(10, 10)
        original = model.location.copy()
        s = h.shooting_slice
        assert s is not None
        action = WargameEnvAction(actions=[s.start])
        h.apply(
            action, [model], 20, 20, self._action_space(h, 1),
            phase=BattlePhase.shooting,
        )
        np.testing.assert_array_equal(model.location, original)

    def test_movement_action_still_works(self) -> None:
        cfg = _small_config(n_opponents=N_OPPONENTS)
        h = ActionHandler(cfg, n_shoot_targets=N_OPPONENTS)
        model = _make_model(10, 10)
        original = model.location.copy()
        action = WargameEnvAction(actions=[1])  # first movement action
        h.apply(
            action, [model], 20, 20, self._action_space(h, 1),
            phase=BattlePhase.movement,
        )
        assert not np.array_equal(model.location, original)

    def test_stay_action_noop_in_shooting(self) -> None:
        cfg = _small_config(n_opponents=N_OPPONENTS)
        h = ActionHandler(cfg, n_shoot_targets=N_OPPONENTS)
        model = _make_model(10, 10)
        original = model.location.copy()
        action = WargameEnvAction(actions=[STAY_ACTION])
        h.apply(
            action, [model], 20, 20, self._action_space(h, 1),
            phase=BattlePhase.shooting,
        )
        np.testing.assert_array_equal(model.location, original)
