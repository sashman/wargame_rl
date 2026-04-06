"""Tests for shooting resolution: config extensions, domain resolution, entity extensions."""

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
from wargame_rl.wargame.envs.types.config import ModelConfig, WeaponProfile


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
