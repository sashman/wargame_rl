"""Tests for shooting action space: config types, registry, masking, and apply dispatch."""

from __future__ import annotations

import numpy as np
import pytest
from gymnasium import spaces
from pydantic import ValidationError

from wargame_rl.wargame.envs.domain.entities import WargameModel
from wargame_rl.wargame.envs.env_components.actions import STAY_ACTION, ActionHandler
from wargame_rl.wargame.envs.env_components.shooting_masks import (
    compute_shooting_masks,
    max_weapon_ranges,
)
from wargame_rl.wargame.envs.types import WargameEnvAction, WargameEnvConfig
from wargame_rl.wargame.envs.types.config import (
    ModelConfig,
    OpponentPolicyConfig,
    WeaponProfile,
)
from wargame_rl.wargame.envs.types.game_timing import BattlePhase
from wargame_rl.wargame.envs.wargame import WargameEnv

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
        return spaces.Tuple([spaces.Discrete(handler.n_actions) for _ in range(n)])

    def test_shooting_action_noop(self) -> None:
        cfg = _small_config(n_opponents=N_OPPONENTS)
        h = ActionHandler(cfg, n_shoot_targets=N_OPPONENTS)
        model = _make_model(10, 10)
        original = model.location.copy()
        s = h.shooting_slice
        assert s is not None
        action = WargameEnvAction(actions=[s.start])
        h.apply(
            action,
            [model],
            20,
            20,
            self._action_space(h, 1),
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
            action,
            [model],
            20,
            20,
            self._action_space(h, 1),
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
            action,
            [model],
            20,
            20,
            self._action_space(h, 1),
            phase=BattlePhase.shooting,
        )
        np.testing.assert_array_equal(model.location, original)


# ---------------------------------------------------------------------------
# compute_shooting_masks unit tests
# ---------------------------------------------------------------------------


class TestShootingMaskFunction:
    def test_all_valid_clear_los(self) -> None:
        pp = np.array([[0, 0], [5, 5]])
        op = np.array([[10, 0], [15, 5]])
        pa = np.array([True, True])
        oa = np.array([True, True])
        pr = np.array([20.0, 20.0])
        m = compute_shooting_masks(pp, op, pa, oa, pr, lambda *_: True)
        assert m.shape == (2, 2)
        assert m.all()

    def test_dead_opponent_masked(self) -> None:
        pp = np.array([[0, 0]])
        op = np.array([[5, 0], [10, 0]])
        pa = np.array([True])
        oa = np.array([True, False])
        pr = np.array([20.0])
        m = compute_shooting_masks(pp, op, pa, oa, pr, lambda *_: True)
        assert m[0, 0]
        assert not m[0, 1]

    def test_out_of_range_masked(self) -> None:
        pp = np.array([[0, 0]])
        op = np.array([[100, 0]])
        pa = np.array([True])
        oa = np.array([True])
        pr = np.array([5.0])
        m = compute_shooting_masks(pp, op, pa, oa, pr, lambda *_: True)
        assert not m[0, 0]

    def test_no_los_masked(self) -> None:
        pp = np.array([[0, 0]])
        op = np.array([[5, 0]])
        pa = np.array([True])
        oa = np.array([True])
        pr = np.array([20.0])
        m = compute_shooting_masks(pp, op, pa, oa, pr, lambda *_: False)
        assert not m[0, 0]

    def test_no_weapons_masked(self) -> None:
        pp = np.array([[0, 0]])
        op = np.array([[5, 0]])
        pa = np.array([True])
        oa = np.array([True])
        pr = np.array([0.0])
        m = compute_shooting_masks(pp, op, pa, oa, pr, lambda *_: True)
        assert not m[0, 0]

    def test_empty_opponents(self) -> None:
        pp = np.array([[0, 0], [5, 5]])
        op = np.zeros((0, 2))
        pa = np.array([True, True])
        oa = np.array([], dtype=bool)
        pr = np.array([20.0, 20.0])
        m = compute_shooting_masks(pp, op, pa, oa, pr, lambda *_: True)
        assert m.shape == (2, 0)


# ---------------------------------------------------------------------------
# max_weapon_ranges unit tests
# ---------------------------------------------------------------------------


class TestMaxWeaponRanges:
    def test_with_weapons(self) -> None:
        mc = ModelConfig(weapons=[WeaponProfile(range=24), WeaponProfile(range=12)])
        r = max_weapon_ranges([mc], 1)
        assert r[0] == 24.0

    def test_no_weapons(self) -> None:
        mc = ModelConfig()
        r = max_weapon_ranges([mc], 1)
        assert r[0] == 0.0

    def test_none_configs(self) -> None:
        r = max_weapon_ranges(None, 3)
        np.testing.assert_array_equal(r, [0.0, 0.0, 0.0])


# ---------------------------------------------------------------------------
# Env integration: shooting wiring
# ---------------------------------------------------------------------------


def _shooting_env_config() -> WargameEnvConfig:
    """Config with shooting enabled — movement + shooting phases only."""
    return WargameEnvConfig(
        number_of_wargame_models=2,
        number_of_opponent_models=2,
        board_width=30,
        board_height=30,
        models=[
            ModelConfig(x=5, y=5, weapons=[WeaponProfile(range=24)]),
            ModelConfig(x=10, y=10, weapons=[WeaponProfile(range=24)]),
        ],
        opponent_models=[
            ModelConfig(x=20, y=5),
            ModelConfig(x=25, y=10),
        ],
        skip_phases=[BattlePhase.command, BattlePhase.charge, BattlePhase.fight],
        opponent_policy=OpponentPolicyConfig(type="random"),
        n_movement_angles=N_ANGLES,
        n_speed_bins=N_SPEEDS,
    )


class TestEnvShootingIntegration:
    def test_n_actions_includes_shooting(self) -> None:
        env = WargameEnv(config=_shooting_env_config())
        assert env.n_actions == 1 + N_MOVE_ACTIONS + 2

    def test_reset_movement_phase_shooting_masked(self) -> None:
        """After reset (movement phase), shooting columns should all be False."""
        env = WargameEnv(config=_shooting_env_config())
        obs, _ = env.reset(seed=42)
        assert obs.action_mask is not None
        shooting_slice = env._action_handler.shooting_slice
        assert shooting_slice is not None
        assert not obs.action_mask[:, shooting_slice.start : shooting_slice.end].any()

    def test_step_into_shooting_phase(self) -> None:
        """Step with STAY during movement → lands on shooting → movement masked, some shooting valid."""
        env = WargameEnv(config=_shooting_env_config())
        env.reset(seed=42)
        stay_action = WargameEnvAction(actions=[STAY_ACTION, STAY_ACTION])
        obs, _, _, _, _ = env.step(stay_action)
        assert obs.action_mask is not None

        shooting_slice = env._action_handler.shooting_slice
        assert shooting_slice is not None
        move_slice = env._action_handler.registry.slice_for("movement")
        assert not obs.action_mask[:, move_slice.start : move_slice.end].any()

        for i in range(len(obs.wargame_models)):
            if obs.wargame_models[i].alive > 0:
                assert obs.action_mask[i, STAY_ACTION]

    def test_stay_always_valid(self) -> None:
        """STAY action is valid in both movement and shooting phases."""
        env = WargameEnv(config=_shooting_env_config())
        obs, _ = env.reset(seed=42)
        assert obs.action_mask is not None
        assert obs.action_mask[:, STAY_ACTION].all()

        stay_action = WargameEnvAction(actions=[STAY_ACTION, STAY_ACTION])
        obs2, _, _, _, _ = env.step(stay_action)
        assert obs2.action_mask is not None
        assert obs2.action_mask[:, STAY_ACTION].all()

    def test_backward_compat_no_opponents(self) -> None:
        cfg = WargameEnvConfig(n_movement_angles=N_ANGLES, n_speed_bins=N_SPEEDS)
        env = WargameEnv(config=cfg)
        assert env.n_actions == 1 + N_MOVE_ACTIONS
        assert env._action_handler.shooting_slice is None


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------


class TestBackwardCompat:
    def test_existing_configs_unchanged_nactions(self) -> None:
        cfg = WargameEnvConfig(n_movement_angles=16, n_speed_bins=6)
        env = WargameEnv(config=cfg)
        assert env.n_actions == 1 + 16 * 6

    def test_skip_phases_default_includes_shooting(self) -> None:
        cfg = WargameEnvConfig()
        assert BattlePhase.shooting in cfg.skip_phases

    def test_full_step_loop_no_opponents(self) -> None:
        cfg = WargameEnvConfig()
        env = WargameEnv(config=cfg)
        env.reset(seed=0)
        for _ in range(10):
            action = WargameEnvAction(actions=list(env.action_space.sample()))
            env.step(action)

    def test_tensor_pipeline_with_shooting(self) -> None:
        from wargame_rl.wargame.model.common.observation import observation_to_tensor

        env = WargameEnv(config=_shooting_env_config())
        obs, _ = env.reset(seed=42)
        tensors = observation_to_tensor(obs)
        assert len(tensors) == 5
        mask_tensor = tensors[4]
        assert mask_tensor.shape == (2, env.n_actions)
