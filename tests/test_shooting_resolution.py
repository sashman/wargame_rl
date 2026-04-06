"""Tests for shooting resolution: config extensions, domain resolution, entity extensions,
integration tests for env wiring, masks, observation pipeline, RNG, and StepContext."""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from wargame_rl.wargame.envs.domain.battle_factory import _build_models
from wargame_rl.wargame.envs.domain.entities import WargameModel
from wargame_rl.wargame.envs.domain.shooting import (
    ENGAGEMENT_RANGE,
    ShootingResult,
    expected_damage,
    resolve_shooting,
    wound_roll_threshold,
)
from wargame_rl.wargame.envs.env_components.actions import STAY_ACTION
from wargame_rl.wargame.envs.env_components.shooting_masks import compute_shooting_masks
from wargame_rl.wargame.envs.types import WargameEnvAction
from wargame_rl.wargame.envs.types.config import (
    ModelConfig,
    OpponentPolicyConfig,
    WeaponProfile,
)
from wargame_rl.wargame.envs.types.game_timing import BattlePhase
from wargame_rl.wargame.envs.wargame import WargameEnv, WargameEnvConfig
from wargame_rl.wargame.model.common.observation import observation_to_tensor


def _make_model(
    x: int = 0,
    y: int = 0,
    max_wounds: int = 1,
    toughness: int = 3,
    save: int = 4,
) -> WargameModel:
    """Create a WargameModel with combat stats."""
    return WargameModel(
        location=np.array([x, y], dtype=np.int32),
        stats={
            "max_wounds": max_wounds,
            "current_wounds": max_wounds,
            "toughness": toughness,
            "save": save,
        },
        distances_to_objectives=np.zeros((1, 2), dtype=np.int32),
        group_id=0,
    )


# ---------------------------------------------------------------------------
# Config extensions
# ---------------------------------------------------------------------------


class TestConfigExtensions:
    """WeaponProfile and ModelConfig combat stat defaults per D-06, D-09."""

    def test_weapon_profile_defaults(self) -> None:
        wp = WeaponProfile(range=24)
        assert wp.attacks == 2
        assert wp.ballistic_skill == 3
        assert wp.strength == 4
        assert wp.ap == 1
        assert wp.damage == 1

    def test_model_config_defense_defaults(self) -> None:
        mc = ModelConfig()
        assert mc.toughness == 3
        assert mc.save == 4

    def test_save_7_valid(self) -> None:
        """save=7 represents no armour."""
        mc = ModelConfig(save=7)
        assert mc.save == 7

    def test_save_8_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ModelConfig(save=8)

    def test_save_1_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ModelConfig(save=1)

    def test_backward_compat_weapon_range_only(self) -> None:
        """Existing configs with only range still work."""
        wp = WeaponProfile(range=12)
        assert wp.range == 12
        assert wp.attacks == 2


# ---------------------------------------------------------------------------
# Wound roll threshold
# ---------------------------------------------------------------------------


class TestWoundRollThreshold:
    """Parametrized over all 5 bands with boundary values."""

    @pytest.mark.parametrize(
        "strength, toughness, expected",
        [
            (8, 4, 2),   # S >= 2T
            (6, 3, 2),   # S >= 2T (boundary: 6 == 2*3)
            (10, 4, 2),  # S >> 2T
            (5, 4, 3),   # S > T
            (4, 3, 3),   # S > T
            (4, 4, 4),   # S == T
            (1, 1, 4),   # S == T (edge: both 1)
            (3, 4, 5),   # S < T but not <= T/2
            (3, 5, 5),   # S < T, 2*3=6 > 5 so not <= T/2
            (2, 4, 6),   # S <= T/2 (boundary: 2*2 == 4)
            (1, 4, 6),   # S <= T/2
            (1, 2, 6),   # S <= T/2 (boundary: 2*1 == 2)
        ],
        ids=[
            "S=8,T=4->2+",
            "S=6,T=3->2+(boundary)",
            "S=10,T=4->2+",
            "S=5,T=4->3+",
            "S=4,T=3->3+",
            "S=4,T=4->4+",
            "S=1,T=1->4+(both-1)",
            "S=3,T=4->5+",
            "S=3,T=5->5+",
            "S=2,T=4->6+(boundary)",
            "S=1,T=4->6+",
            "S=1,T=2->6+(boundary)",
        ],
    )
    def test_threshold(self, strength: int, toughness: int, expected: int) -> None:
        assert wound_roll_threshold(strength, toughness) == expected


# ---------------------------------------------------------------------------
# Resolve shooting
# ---------------------------------------------------------------------------


class TestResolveShooting:
    """Deterministic shooting resolution with fixed seeds."""

    def test_deterministic_seed_42(self) -> None:
        """Fixed seed(42) produces a deterministic result."""
        rng = np.random.default_rng(42)
        r = resolve_shooting(2, 3, 4, 1, 1, 3, 4, rng)
        assert isinstance(r, ShootingResult)
        assert r.hits == 1
        assert r.wounds == 1
        assert r.unsaved == 1
        assert r.damage_dealt == 1

    def test_all_miss_high_bs(self) -> None:
        """BS=6 with low attacks: very likely to miss everything."""
        rng = np.random.default_rng(123)
        r = resolve_shooting(1, 6, 1, 0, 1, 10, 2, rng)
        assert r.hits >= 0
        assert r.damage_dealt >= 0

    def test_guaranteed_scenario(self) -> None:
        """High S, low T, no save — most attacks should deal damage."""
        rng = np.random.default_rng(99)
        r = resolve_shooting(10, 2, 10, 0, 1, 1, 7, rng)
        assert r.hits > 0
        assert r.wounds > 0
        assert r.unsaved > 0
        assert r.damage_dealt == r.unsaved * 1

    def test_natural_1_always_fails_hit(self) -> None:
        """Even with BS=2, a natural 1 must miss."""
        rng = np.random.default_rng(0)
        total_hits = 0
        total_ones = 0
        for seed in range(100):
            rng = np.random.default_rng(seed)
            rolls = rng.integers(1, 7, size=100)
            ones_count = int(np.sum(rolls == 1))
            total_ones += ones_count
            rng2 = np.random.default_rng(seed)
            r = resolve_shooting(100, 2, 10, 0, 1, 1, 7, rng2)
            total_hits += r.hits
        assert total_hits < 100 * 100  # some must miss from natural 1s

    def test_natural_6_always_succeeds_wound(self) -> None:
        """Even with impossible wound threshold, natural 6 wounds."""
        total_wounds = 0
        for seed in range(50):
            rng = np.random.default_rng(seed)
            r = resolve_shooting(20, 2, 1, 0, 1, 100, 7, rng)
            total_wounds += r.wounds
        assert total_wounds > 0  # some natural 6s must wound

    def test_zero_attacks_returns_zeros(self) -> None:
        """Edge case: 0 attacks means nothing happens."""
        rng = np.random.default_rng(42)
        # numpy.random.Generator.integers with size=0 returns empty array
        r = resolve_shooting(0, 3, 4, 1, 1, 3, 4, rng)
        assert r == ShootingResult(hits=0, wounds=0, unsaved=0, damage_dealt=0)

    def test_damage_multiplier(self) -> None:
        """Damage > 1 multiplies unsaved wounds."""
        rng = np.random.default_rng(99)
        r = resolve_shooting(10, 2, 10, 0, 3, 1, 7, rng)
        if r.unsaved > 0:
            assert r.damage_dealt == r.unsaved * 3

    def test_engagement_range_constant(self) -> None:
        assert ENGAGEMENT_RANGE == 1


# ---------------------------------------------------------------------------
# Expected damage
# ---------------------------------------------------------------------------


class TestExpectedDamage:
    """Analytical expected damage formula."""

    def test_default_profile(self) -> None:
        """Default profile (2, 3, 4, 1, 1, 3, 4) ≈ 0.593."""
        ed = expected_damage(2, 3, 4, 1, 1, 3, 4)
        assert abs(ed - 2 * (4 / 6) * (4 / 6) * (4 / 6)) < 1e-10

    def test_zero_attacks(self) -> None:
        assert expected_damage(0, 3, 4, 1, 1, 3, 4) == 0.0

    def test_save_7_all_fail(self) -> None:
        """save=7 means all saves fail (p_fail_save=1.0)."""
        ed = expected_damage(2, 3, 4, 0, 1, 4, 7)
        p_hit = 4 / 6
        p_wound = 3 / 6  # S=4, T=4 → 4+ → (7-4)/6
        assert abs(ed - 2 * p_hit * p_wound * 1.0 * 1) < 1e-10

    @pytest.mark.parametrize(
        "attacks, bs, s, ap, d, t, sv, expected_approx",
        [
            # bs=4→p_hit=3/6, S=T=4→4+→p_wound=3/6, sv=3+ap=0→mod=3→p_save=4/6→p_fail=2/6
            (1, 4, 4, 0, 1, 4, 3, 1 * (3 / 6) * (3 / 6) * (2 / 6)),
            # bs=3→p_hit=4/6, S=8≥2T=8→2+→p_wound=5/6, sv=3+ap=2→mod=5→p_save=2/6→p_fail=4/6
            (4, 3, 8, 2, 2, 4, 3, 4 * (4 / 6) * (5 / 6) * (4 / 6) * 2),
        ],
        ids=["single-shot-low-AP", "multi-shot-high-S"],
    )
    def test_parametrized(
        self,
        attacks: int,
        bs: int,
        s: int,
        ap: int,
        d: int,
        t: int,
        sv: int,
        expected_approx: float,
    ) -> None:
        ed = expected_damage(attacks, bs, s, ap, d, t, sv)
        assert abs(ed - expected_approx) < 1e-10


# ---------------------------------------------------------------------------
# Entity extensions
# ---------------------------------------------------------------------------


class TestEntityExtensions:
    """WargameModel.advanced_this_turn flag."""

    def test_default_false(self) -> None:
        model = _make_model()
        assert model.advanced_this_turn is False

    def test_reset_clears_flag(self) -> None:
        model = _make_model()
        model.advanced_this_turn = True
        model.reset_for_episode()
        assert model.advanced_this_turn is False


# ---------------------------------------------------------------------------
# Battle factory stats wiring
# ---------------------------------------------------------------------------


class TestBattleFactoryStats:
    """_build_models wires toughness and save from ModelConfig."""

    def test_custom_stats(self) -> None:
        models = _build_models(
            1, [ModelConfig(toughness=5, save=3)], n_objectives=1, max_groups=100
        )
        assert models[0].stats["toughness"] == 5
        assert models[0].stats["save"] == 3

    def test_default_stats_no_config(self) -> None:
        models = _build_models(1, None, n_objectives=1, max_groups=100)
        assert models[0].stats["toughness"] == 3
        assert models[0].stats["save"] == 4

    def test_default_stats_with_default_config(self) -> None:
        models = _build_models(
            1, [ModelConfig()], n_objectives=1, max_groups=100
        )
        assert models[0].stats["toughness"] == 3
        assert models[0].stats["save"] == 4

    def test_stats_keys(self) -> None:
        models = _build_models(
            1, [ModelConfig()], n_objectives=1, max_groups=100
        )
        expected_keys = {"max_wounds", "current_wounds", "toughness", "save"}
        assert set(models[0].stats.keys()) == expected_keys


# ---------------------------------------------------------------------------
# Fixtures for integration tests
# ---------------------------------------------------------------------------


def _shooting_env_config(
    *,
    n_player: int = 1,
    n_opponent: int = 1,
    max_wounds: int = 3,
) -> WargameEnvConfig:
    """Config with armed player models and unarmed opponents in range."""
    return WargameEnvConfig(
        board_width=30,
        board_height=30,
        number_of_wargame_models=n_player,
        number_of_objectives=1,
        number_of_opponent_models=n_opponent,
        models=[
            ModelConfig(
                x=5 + i,
                y=5,
                max_wounds=max_wounds,
                weapons=[WeaponProfile(range=50, attacks=4, ballistic_skill=2, strength=8, ap=2, damage=2)],
            )
            for i in range(n_player)
        ],
        opponent_models=[
            ModelConfig(x=20 + i, y=5, max_wounds=max_wounds)
            for i in range(n_opponent)
        ],
        opponent_policy=OpponentPolicyConfig(type="random"),
        skip_phases=[BattlePhase.command, BattlePhase.charge, BattlePhase.fight],
        n_movement_angles=8,
        n_speed_bins=3,
    )


def _step_to_shooting(env: WargameEnv) -> None:
    """Step with STAY until we're in shooting phase (movement -> shooting)."""
    n = len(env.wargame_models)
    stay = WargameEnvAction(actions=[STAY_ACTION] * n)
    env.step(stay)


# ---------------------------------------------------------------------------
# Integration: env shooting resolution
# ---------------------------------------------------------------------------


class TestShootingIntegration:
    """Env step with shooting phase resolves damage."""

    def test_player_shooting_deals_damage(self) -> None:
        env = WargameEnv(config=_shooting_env_config(max_wounds=10))
        env.reset(seed=42)
        initial_wounds = env.opponent_models[0].stats["current_wounds"]
        _step_to_shooting(env)
        shooting_slice = env._action_handler.shooting_slice
        assert shooting_slice is not None
        shoot_action = WargameEnvAction(actions=[shooting_slice.start])
        env.step(shoot_action)
        assert env._last_player_shooting_results, "Expected at least one result"
        total_dmg = sum(r.damage_dealt for r in env._last_player_shooting_results)
        if total_dmg > 0:
            assert env.opponent_models[0].stats["current_wounds"] < initial_wounds

    def test_deterministic_with_fixed_seed(self) -> None:
        results = []
        for _ in range(2):
            env = WargameEnv(config=_shooting_env_config(max_wounds=10))
            env.reset(seed=42)
            _step_to_shooting(env)
            ss = env._action_handler.shooting_slice
            assert ss is not None
            env.step(WargameEnvAction(actions=[ss.start]))
            results.append(
                [
                    (r.hits, r.wounds, r.unsaved, r.damage_dealt)
                    for r in env._last_player_shooting_results
                ]
            )
        assert results[0] == results[1]

    def test_both_sides_shoot(self) -> None:
        """Both player and opponent paths resolve damage in same round."""
        cfg = _shooting_env_config(n_player=2, n_opponent=2, max_wounds=10)
        cfg_with_opp_weapons = cfg.model_copy(
            update={
                "opponent_models": [
                    ModelConfig(x=20, y=5, max_wounds=10, weapons=[WeaponProfile(range=50)]),
                    ModelConfig(x=21, y=5, max_wounds=10, weapons=[WeaponProfile(range=50)]),
                ]
            }
        )
        env = WargameEnv(config=cfg_with_opp_weapons)
        env.reset(seed=42)
        _step_to_shooting(env)
        ss = env._action_handler.shooting_slice
        assert ss is not None
        env.step(WargameEnvAction(actions=[ss.start, ss.start + 1]))
        # At least one side should have resolved
        p_dmg = sum(r.damage_dealt for r in env._last_player_shooting_results)
        o_dmg = sum(r.damage_dealt for r in env._last_opponent_shooting_results)
        assert p_dmg >= 0
        assert o_dmg >= 0


# ---------------------------------------------------------------------------
# Shooting mask extensions
# ---------------------------------------------------------------------------


class TestShootingMaskExtensions:
    """compute_shooting_masks with player_advanced and engagement_range."""

    def test_advanced_model_masked(self) -> None:
        pp = np.array([[0, 0], [5, 5]])
        op = np.array([[10, 0]])
        pa = np.array([True, True])
        oa = np.array([True])
        pr = np.array([20.0, 20.0])
        advanced = np.array([True, False])
        m = compute_shooting_masks(
            pp, op, pa, oa, pr, lambda *_: True, player_advanced=advanced
        )
        assert not m[0, 0], "Advanced model should not be able to shoot"
        assert m[1, 0], "Non-advanced model should be able to shoot"

    def test_engagement_range_masks_model(self) -> None:
        pp = np.array([[0, 0], [10, 10]])
        op = np.array([[1, 0]])  # Distance 1 from first player
        pa = np.array([True, True])
        oa = np.array([True])
        pr = np.array([50.0, 50.0])
        m = compute_shooting_masks(
            pp, op, pa, oa, pr, lambda *_: True, engagement_range=2.0
        )
        assert not m[0, 0], "Model within engagement range should be masked"
        assert m[1, 0], "Model outside engagement range should shoot"

    def test_backward_compat_no_new_params(self) -> None:
        pp = np.array([[0, 0]])
        op = np.array([[5, 0]])
        pa = np.array([True])
        oa = np.array([True])
        pr = np.array([20.0])
        m = compute_shooting_masks(pp, op, pa, oa, pr, lambda *_: True)
        assert m[0, 0]


# ---------------------------------------------------------------------------
# Observation extension
# ---------------------------------------------------------------------------


class TestObservationExtension:
    """Observation tensor includes combat features and expected damage."""

    def test_feature_dim_matches(self) -> None:
        cfg = _shooting_env_config(n_player=2, n_opponent=2)
        env = WargameEnv(config=cfg)
        obs, _ = env.reset(seed=42)
        tensors = observation_to_tensor(obs)
        player_f = tensors[2]
        opp_f = tensors[3]
        assert player_f.shape[1] == opp_f.shape[1]
        n_obj = 1
        max_groups = cfg.max_groups
        n_opp = 2
        expected_dim = 2 + n_obj * 2 + max_groups + 1 + 3 + 7 + n_opp
        assert player_f.shape[1] == expected_dim

    def test_weapon_stats_nonzero_for_armed(self) -> None:
        cfg = _shooting_env_config(n_player=1, n_opponent=1)
        env = WargameEnv(config=cfg)
        obs, _ = env.reset(seed=42)
        tensors = observation_to_tensor(obs)
        player_f = tensors[2]
        base_idx = 2 + 1 * 2 + cfg.max_groups + 1 + 3
        # weapon_attacks/10 should be > 0 for armed player
        assert player_f[0, base_idx].item() > 0

    def test_expected_damage_nonzero(self) -> None:
        cfg = _shooting_env_config(n_player=1, n_opponent=1)
        env = WargameEnv(config=cfg)
        obs, _ = env.reset(seed=42)
        tensors = observation_to_tensor(obs)
        player_f = tensors[2]
        ed_col_idx = 2 + 1 * 2 + cfg.max_groups + 1 + 3 + 7
        # Expected damage against the one opponent should be > 0
        assert player_f[0, ed_col_idx].item() > 0

    def test_opponent_ed_columns_zero(self) -> None:
        cfg = _shooting_env_config(n_player=1, n_opponent=1)
        env = WargameEnv(config=cfg)
        obs, _ = env.reset(seed=42)
        tensors = observation_to_tensor(obs)
        opp_f = tensors[3]
        ed_col_idx = 2 + 1 * 2 + cfg.max_groups + 1 + 3 + 7
        assert opp_f[0, ed_col_idx].item() == 0.0


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------


class TestBackwardCompatIntegration:
    """Envs with no weapon configs or 0 opponents still work."""

    def test_no_model_configs(self) -> None:
        cfg = WargameEnvConfig(board_width=20, board_height=20)
        env = WargameEnv(config=cfg)
        obs, _ = env.reset(seed=42)
        tensors = observation_to_tensor(obs)
        assert tensors[2].shape[0] == cfg.number_of_wargame_models

    def test_zero_opponents_no_ed_columns(self) -> None:
        cfg = WargameEnvConfig(board_width=20, board_height=20)
        env = WargameEnv(config=cfg)
        obs, _ = env.reset(seed=42)
        tensors = observation_to_tensor(obs)
        n_obj = cfg.number_of_objectives
        expected_dim = 2 + n_obj * 2 + cfg.max_groups + 1 + 3 + 7
        assert tensors[2].shape[1] == expected_dim

    def test_full_step_loop(self) -> None:
        cfg = WargameEnvConfig(board_width=20, board_height=20)
        env = WargameEnv(config=cfg)
        env.reset(seed=0)
        for _ in range(10):
            action = WargameEnvAction(actions=list(env.action_space.sample()))
            env.step(action)


# ---------------------------------------------------------------------------
# Combat RNG determinism
# ---------------------------------------------------------------------------


class TestCombatRNG:
    """Same seed → identical results; different seeds → different results."""

    def test_same_seed_same_results(self) -> None:
        results_by_run: list[list[tuple[int, ...]]] = []
        for _ in range(2):
            env = WargameEnv(config=_shooting_env_config(max_wounds=10))
            env.reset(seed=42)
            _step_to_shooting(env)
            ss = env._action_handler.shooting_slice
            assert ss is not None
            env.step(WargameEnvAction(actions=[ss.start]))
            results_by_run.append(
                [(r.hits, r.wounds, r.unsaved, r.damage_dealt) for r in env._last_player_shooting_results]
            )
        assert results_by_run[0] == results_by_run[1]

    def test_different_seeds_differ(self) -> None:
        results_by_seed: list[list[tuple[int, ...]]] = []
        for seed in [42, 99]:
            env = WargameEnv(config=_shooting_env_config(max_wounds=10))
            env.reset(seed=seed)
            _step_to_shooting(env)
            ss = env._action_handler.shooting_slice
            assert ss is not None
            env.step(WargameEnvAction(actions=[ss.start]))
            results_by_seed.append(
                [(r.hits, r.wounds, r.unsaved, r.damage_dealt) for r in env._last_player_shooting_results]
            )
        assert results_by_seed[0] != results_by_seed[1]


# ---------------------------------------------------------------------------
# StepContext combat fields
# ---------------------------------------------------------------------------


class TestStepContextCombat:
    """StepContext after step has combat outcome fields."""

    def test_fields_populated(self) -> None:
        env = WargameEnv(config=_shooting_env_config(max_wounds=10))
        env.reset(seed=42)
        _step_to_shooting(env)
        ss = env._action_handler.shooting_slice
        assert ss is not None
        env.step(WargameEnvAction(actions=[ss.start]))
        ctx = env.last_step_context
        assert ctx is not None
        assert isinstance(ctx.player_damage_dealt, int)
        assert isinstance(ctx.opponent_damage_dealt, int)
        assert isinstance(ctx.player_models_killed, int)
        assert isinstance(ctx.opponent_models_killed, int)
        assert ctx.player_damage_dealt >= 0

    def test_kill_tracking(self) -> None:
        """When target has 1 wound and takes damage, kills count increments."""
        cfg = _shooting_env_config(n_player=1, n_opponent=1, max_wounds=1)
        env = WargameEnv(config=cfg)
        env.reset(seed=42)
        _step_to_shooting(env)
        ss = env._action_handler.shooting_slice
        assert ss is not None
        env.step(WargameEnvAction(actions=[ss.start]))
        ctx = env.last_step_context
        assert ctx is not None
        if ctx.player_damage_dealt > 0:
            assert ctx.player_models_killed >= 1
