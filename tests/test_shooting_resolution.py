"""Tests for shooting resolution: config extensions, domain resolution, entity extensions,
integration tests for env wiring, masks, observation pipeline, RNG, and StepContext."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
from pydantic import ValidationError

from wargame_rl.wargame.envs.domain import rules_constants
from wargame_rl.wargame.envs.domain.battle_factory import _build_models
from wargame_rl.wargame.envs.domain.entities import WargameModel
from wargame_rl.wargame.envs.domain.rules_quantities import resolve_rules_quantities
from wargame_rl.wargame.envs.domain.shooting import (
    DefenderStats,
    ShootingResult,
    expected_damage,
    expected_damage_matrix,
    hit_probability,
    resolve_shooting,
    resolve_shooting_phase,
    wound_roll_threshold,
)
from wargame_rl.wargame.envs.env_components.actions import STAY_ACTION
from wargame_rl.wargame.envs.env_components.shooting_masks import compute_shooting_masks
from wargame_rl.wargame.envs.types import TerrainPieceConfig, WargameEnvAction
from wargame_rl.wargame.envs.types.config import (
    ModelConfig,
    OpponentPolicyConfig,
    WeaponProfile,
)
from wargame_rl.wargame.envs.types.game_timing import BattlePhase
from wargame_rl.wargame.envs.wargame import WargameEnv, WargameEnvConfig
from wargame_rl.wargame.model.common.observation import observation_to_tensor


def _all_visible(
    _origins: np.ndarray, _targets: np.ndarray, candidates: np.ndarray
) -> np.ndarray:
    """Sight stub: nothing blocks, so every candidate pair is visible."""
    return candidates


def _none_visible(
    _origins: np.ndarray, _targets: np.ndarray, candidates: np.ndarray
) -> np.ndarray:
    """Sight stub: everything blocks."""
    return np.zeros_like(candidates)


@dataclass(frozen=True, slots=True)
class _TestWeapon:
    """Lightweight weapon satisfying WeaponStats protocol (no Pydantic validation)."""

    attacks: int = 2
    ballistic_skill: int = 3
    strength: int = 4
    ap: int = 1
    damage: int = 1


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
            (8, 4, 2),  # S >= 2T
            (6, 3, 2),  # S >= 2T (boundary: 6 == 2*3)
            (10, 4, 2),  # S >> 2T
            (5, 4, 3),  # S > T
            (4, 3, 3),  # S > T
            (4, 4, 4),  # S == T
            (1, 1, 4),  # S == T (edge: both 1)
            (3, 4, 5),  # S < T but not <= T/2
            (3, 5, 5),  # S < T, 2*3=6 > 5 so not <= T/2
            (2, 4, 6),  # S <= T/2 (boundary: 2*2 == 4)
            (1, 4, 6),  # S <= T/2
            (1, 2, 6),  # S <= T/2 (boundary: 2*1 == 2)
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


def _wp(
    attacks: int = 2,
    bs: int = 3,
    strength: int = 4,
    ap: int = 1,
    damage: int = 1,
) -> _TestWeapon:
    """Shorthand to build a weapon satisfying WeaponStats protocol."""
    return _TestWeapon(
        attacks=attacks,
        ballistic_skill=bs,
        strength=strength,
        ap=ap,
        damage=damage,
    )


def _ds(toughness: int = 3, save: int = 4) -> DefenderStats:
    return DefenderStats(toughness=toughness, save=save)


class TestResolveShooting:
    """Deterministic shooting resolution with fixed seeds."""

    def test_deterministic_seed_42(self) -> None:
        """Fixed seed(42) produces a deterministic result."""
        rng = np.random.default_rng(42)
        r = resolve_shooting(_wp(), _ds(), rng)
        assert isinstance(r, ShootingResult)
        assert r.hits == 1
        assert r.wounds == 1
        assert r.unsaved == 1
        assert r.damage_dealt == 1

    def test_all_miss_high_bs(self) -> None:
        """BS=6 with low attacks: very likely to miss everything."""
        rng = np.random.default_rng(123)
        r = resolve_shooting(
            _wp(attacks=1, bs=6, strength=1, ap=0), _ds(toughness=10, save=2), rng
        )
        assert r.hits >= 0
        assert r.damage_dealt >= 0

    def test_guaranteed_scenario(self) -> None:
        """High S, low T, no save — most attacks should deal damage."""
        rng = np.random.default_rng(99)
        r = resolve_shooting(
            _wp(attacks=10, bs=2, strength=10, ap=0), _ds(toughness=1, save=7), rng
        )
        assert r.hits > 0
        assert r.wounds > 0
        assert r.unsaved > 0
        assert r.damage_dealt == r.unsaved * 1

    def test_natural_1_always_fails_hit(self) -> None:
        """Even with BS=2, a natural 1 must miss."""
        rng = np.random.default_rng(0)
        total_hits = 0
        total_ones = 0
        wp = _wp(attacks=100, bs=2, strength=10, ap=0)
        ds = _ds(toughness=1, save=7)
        for seed in range(100):
            rng = np.random.default_rng(seed)
            rolls = rng.integers(1, 7, size=100)
            ones_count = int(np.sum(rolls == 1))
            total_ones += ones_count
            rng2 = np.random.default_rng(seed)
            r = resolve_shooting(wp, ds, rng2)
            total_hits += r.hits
        assert total_hits < 100 * 100  # some must miss from natural 1s

    def test_natural_6_always_succeeds_wound(self) -> None:
        """Even with impossible wound threshold, natural 6 wounds."""
        total_wounds = 0
        wp = _wp(attacks=20, bs=2, strength=1, ap=0)
        ds = _ds(toughness=100, save=7)
        for seed in range(50):
            rng = np.random.default_rng(seed)
            r = resolve_shooting(wp, ds, rng)
            total_wounds += r.wounds
        assert total_wounds > 0  # some natural 6s must wound

    def test_zero_attacks_returns_zeros(self) -> None:
        """Edge case: 0 attacks means nothing happens."""
        rng = np.random.default_rng(42)
        r = resolve_shooting(_wp(attacks=0), _ds(), rng)
        assert r == ShootingResult(hits=0, wounds=0, unsaved=0, damage_dealt=0)

    def test_damage_multiplier(self) -> None:
        """Damage > 1 multiplies unsaved wounds."""
        rng = np.random.default_rng(99)
        r = resolve_shooting(
            _wp(attacks=10, bs=2, strength=10, ap=0, damage=3),
            _ds(toughness=1, save=7),
            rng,
        )
        if r.unsaved > 0:
            assert r.damage_dealt == r.unsaved * 3

    def test_engagement_range_default_is_the_env_value_not_the_rules_value(
        self,
    ) -> None:
        """Engagement range moved from a constant to config, at its old value.

        The rules say 2" (`docs/rules/constants.yaml`, `engagement.horizontal_in`)
        and the environment uses 1. Keeping 1 as the default is what makes the
        scale mechanism a no-op: every baseline and trained result in the repo
        was measured at 1, and raising it changes which shots are legal. Adopting
        the rules value is a scenario change to be measured, not a tidy-up.
        """
        config = WargameEnvConfig()
        assert config.engagement_range == 1.0
        assert rules_constants.ENGAGEMENT_RANGE_IN == 2.0

        quantities = resolve_rules_quantities(config)
        assert quantities.engagement_range == 1.0


# ---------------------------------------------------------------------------
# Whole-phase resolution
# ---------------------------------------------------------------------------


class TestResolveShootingPhase:
    """The phase-level service, exercised without building a Gym env.

    That it can be tested this way at all is the point of it living in the
    domain: the attack sequence used to be a private method on `WargameEnv`,
    so reaching it meant constructing a board, a clock and an action space.
    """

    def _weapons(self, count: int) -> list[list[_TestWeapon]]:
        return [[_TestWeapon()] for _ in range(count)]

    def test_a_dead_attacker_does_not_fire(self) -> None:
        attackers = [_make_model(), _make_model()]
        targets = [_make_model(max_wounds=10)]
        attackers[0].take_damage(1)

        results = resolve_shooting_phase(
            shots=[(0, 0), (1, 0)],
            attackers=attackers,
            targets=targets,
            attacker_weapons=self._weapons(2),
            rng=np.random.default_rng(0),
        )

        assert [r.attacker_idx for r in results] == [1]

    def test_a_dead_target_cannot_be_fired_at(self) -> None:
        attackers = [_make_model()]
        targets = [_make_model()]
        targets[0].take_damage(1)

        results = resolve_shooting_phase(
            shots=[(0, 0)],
            attackers=attackers,
            targets=targets,
            attacker_weapons=self._weapons(1),
            rng=np.random.default_rng(0),
        )

        assert results == []

    def test_an_unarmed_attacker_does_not_consume_dice(self) -> None:
        """No weapon means no roll, so the next shooter gets the first dice.

        Order of RNG consumption is the property that makes a refactor of this
        function safe to compare bit-for-bit against the previous one.
        """
        attackers = [_make_model(), _make_model()]
        targets = [_make_model(max_wounds=10)]

        armed_second = resolve_shooting_phase(
            shots=[(0, 0), (1, 0)],
            attackers=attackers,
            targets=targets,
            attacker_weapons=[[], [_TestWeapon()]],
            rng=np.random.default_rng(7),
        )

        targets = [_make_model(max_wounds=10)]
        alone = resolve_shooting_phase(
            shots=[(1, 0)],
            attackers=attackers,
            targets=targets,
            attacker_weapons=[[], [_TestWeapon()]],
            rng=np.random.default_rng(7),
        )

        assert [r.result for r in armed_second] == [r.result for r in alone]

    def test_kill_is_attributed_to_the_shot_that_landed_it(self) -> None:
        """`killed` cannot be recovered afterwards when several shots share a target."""
        attackers = [_make_model() for _ in range(6)]
        targets = [_make_model(max_wounds=1, save=7)]

        results = resolve_shooting_phase(
            shots=[(i, 0) for i in range(6)],
            attackers=attackers,
            targets=targets,
            attacker_weapons=self._weapons(6),
            rng=np.random.default_rng(3),
        )

        assert not targets[0].is_alive
        # Exactly one shot may claim the kill, and it is the last one resolved:
        # every later shot is filtered out by the dead-target check.
        assert sum(r.killed for r in results) == 1
        assert results[-1].killed


# ---------------------------------------------------------------------------
# Expected damage
# ---------------------------------------------------------------------------


class TestExpectedDamage:
    """Analytical expected damage formula."""

    def test_default_profile(self) -> None:
        """Default profile (2, 3, 4, 1, 1, 3, 4) ≈ 0.593."""
        ed = expected_damage(_wp(), _ds())
        assert abs(ed - 2 * (4 / 6) * (4 / 6) * (4 / 6)) < 1e-10

    def test_zero_attacks(self) -> None:
        assert expected_damage(_wp(attacks=0), _ds()) == 0.0

    def test_save_7_all_fail(self) -> None:
        """save=7 means all saves fail (p_fail_save=1.0)."""
        ed = expected_damage(_wp(ap=0), _ds(toughness=4, save=7))
        p_hit = 4 / 6
        p_wound = 3 / 6  # S=4, T=4 → 4+ → (7-4)/6
        assert abs(ed - 2 * p_hit * p_wound * 1.0 * 1) < 1e-10

    @pytest.mark.parametrize(
        "weapon, defender, expected_approx",
        [
            # bs=4→p_hit=3/6, S=T=4→4+→p_wound=3/6, sv=3+ap=0→mod=3→p_save=4/6→p_fail=2/6
            (
                _wp(attacks=1, bs=4, strength=4, ap=0),
                _ds(toughness=4, save=3),
                1 * (3 / 6) * (3 / 6) * (2 / 6),
            ),
            # bs=3→p_hit=4/6, S=8≥2T=8→2+→p_wound=5/6, sv=3+ap=2→mod=5→p_save=2/6→p_fail=4/6
            (
                _wp(attacks=4, bs=3, strength=8, ap=2, damage=2),
                _ds(toughness=4, save=3),
                4 * (4 / 6) * (5 / 6) * (4 / 6) * 2,
            ),
        ],
        ids=["single-shot-low-AP", "multi-shot-high-S"],
    )
    def test_parametrized(
        self,
        weapon: _TestWeapon,
        defender: DefenderStats,
        expected_approx: float,
    ) -> None:
        ed = expected_damage(weapon, defender)
        assert abs(ed - expected_approx) < 1e-10


class TestExpectedDamageUnderCover:
    """Cover worsens Ranged Skill by 1, and the closed form must say so.

    The formula was written before cover existed and read `ballistic_skill`
    straight off the weapon, so every expectation quoted beside a shot into
    terrain was the expectation for a target standing in the open.
    """

    @pytest.mark.parametrize("ballistic_skill", [2, 3, 4, 5, 6])
    def test_cover_costs_exactly_one_point_of_skill(self, ballistic_skill: int) -> None:
        """The in-cover number equals the open-ground number one skill worse."""
        in_cover = expected_damage(_wp(bs=ballistic_skill), _ds(), in_cover=True)
        worse_skill = expected_damage(_wp(bs=ballistic_skill + 1), _ds())

        assert abs(in_cover - worse_skill) < 1e-12

    def test_cover_never_reduces_a_shot_to_nothing(self) -> None:
        """RS 6 in cover resolves at 7 -- unreachable, and still hits on a 6.

        The naive `(7 - skill) / 6` returns 0.0 here, which would report the
        best-shielded target on the board as unhittable while the dice keep
        killing it. This is the case that makes cover a modifier rather than an
        absolute shield.
        """
        assert hit_probability(6, in_cover=True) == pytest.approx(1 / 6)
        assert expected_damage(_wp(bs=6), _ds(), in_cover=True) > 0.0

    @pytest.mark.parametrize("ballistic_skill", [2, 3, 4, 5, 6])
    def test_the_default_is_bit_identical_to_the_pre_cover_formula(
        self, ballistic_skill: int
    ) -> None:
        """Omitting the flag must reproduce the old number exactly, not nearly.

        The observation's expected-damage block passes no cover, so this is what
        keeps `test_observation_golden` and every trained checkpoint valid — a
        tolerance here would hide the float reassociation that would void them.
        """
        weapon, defender = _wp(bs=ballistic_skill), _ds()

        p_hit = (7 - weapon.ballistic_skill) / 6.0
        p_wound = (7 - wound_roll_threshold(weapon.strength, defender.toughness)) / 6.0
        p_save = (7 - (defender.save + weapon.ap)) / 6.0
        before = weapon.attacks * p_hit * p_wound * (1.0 - p_save) * weapon.damage

        assert expected_damage(weapon, defender) == before

    @pytest.mark.parametrize("ballistic_skill", [2, 4, 6])
    @pytest.mark.parametrize("in_cover", [False, True])
    def test_it_predicts_what_the_dice_actually_do(
        self, ballistic_skill: int, in_cover: bool
    ) -> None:
        """Monte Carlo `resolve_shooting` against the closed form.

        The two are only worth having separately if they agree; this is the
        assertion that would have caught the gap when cover landed, since it
        fails on the resolution path's own rules rather than on a restatement of
        the formula.
        """
        weapon = _wp(bs=ballistic_skill, attacks=4)
        defender = _ds()
        rng = np.random.default_rng(20260816)

        trials = 20_000
        total = sum(
            resolve_shooting(weapon, defender, rng, in_cover=in_cover).damage_dealt
            for _ in range(trials)
        )

        predicted = expected_damage(weapon, defender, in_cover=in_cover)
        assert abs(total / trials - predicted) < 0.02


class TestExpectedDamageMatrix:
    """The batched form must equal the scalar one, entry for entry.

    The observation pipeline built this matrix with a Python double loop over
    every player x opponent pair, constructing two dataclasses per pair — 625
    `expected_damage` calls per observation on a 25v25 config, roughly 1.5M per
    training epoch, all recomputing a matrix that is constant for the whole run
    because every input comes from static YAML.
    """

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_matches_the_scalar_loop(self, seed: int) -> None:
        """Bit-identical against a per-pair evaluation, mixed stat lines."""
        rng = np.random.default_rng(seed)
        attackers = np.column_stack(
            [
                rng.integers(0, 4, size=7),  # attacks, including 0 = "no weapon"
                rng.integers(2, 7, size=7),  # ballistic skill
                rng.integers(1, 11, size=7),  # strength
                rng.integers(0, 4, size=7),  # ap
                rng.integers(1, 4, size=7),  # damage
            ]
        )
        defenders = np.column_stack(
            [
                rng.integers(0, 9, size=5),  # toughness, including 0 = "no model"
                rng.integers(2, 8, size=5),  # save
            ]
        )

        expected = np.zeros((7, 5), dtype=np.float32)
        for i, attacker in enumerate(attackers):
            if attacker[0] == 0:
                continue
            for j, defender in enumerate(defenders):
                if defender[0] == 0:
                    continue
                expected[i, j] = expected_damage(
                    _wp(
                        attacks=int(attacker[0]),
                        bs=int(attacker[1]),
                        strength=int(attacker[2]),
                        ap=int(attacker[3]),
                        damage=int(attacker[4]),
                    ),
                    _ds(toughness=int(defender[0]), save=int(defender[1])),
                )

        actual = expected_damage_matrix(attackers, defenders)

        np.testing.assert_array_equal(actual, expected)

    def test_zero_toughness_scores_zero_rather_than_wounding_on_two(self) -> None:
        """A toughness of 0 means "no such model", not an infinitely soft one.

        Dropping the guard would be silent: `wound_roll_threshold` takes the
        `2 * toughness <= strength` branch and returns 2, so padding rows would
        report the *highest* expected damage on the board.
        """
        attackers = np.array([[2, 3, 4, 1, 1]])
        defenders = np.array([[0, 4]])

        assert expected_damage_matrix(attackers, defenders)[0, 0] == 0.0

    def test_empty_sides_give_an_empty_matrix(self) -> None:
        """Configs with no opponents must still produce a known-width block."""
        attackers = np.zeros((0, 5), dtype=np.int64)
        defenders = np.array([[4, 3]])

        assert expected_damage_matrix(attackers, defenders).shape == (0, 1)


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
        models = _build_models(1, [ModelConfig()], n_objectives=1, max_groups=100)
        assert models[0].stats["toughness"] == 3
        assert models[0].stats["save"] == 4

    def test_stats_keys(self) -> None:
        models = _build_models(1, [ModelConfig()], n_objectives=1, max_groups=100)
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
                weapons=[
                    WeaponProfile(
                        range=50,
                        attacks=4,
                        ballistic_skill=2,
                        strength=8,
                        ap=2,
                        damage=2,
                    )
                ],
            )
            for i in range(n_player)
        ],
        opponent_models=[
            ModelConfig(x=20 + i, y=5, max_wounds=max_wounds) for i in range(n_opponent)
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
        total_dmg = sum(
            r.result.damage_dealt for r in env._last_player_shooting_results
        )
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
                    (
                        r.result.hits,
                        r.result.wounds,
                        r.result.unsaved,
                        r.result.damage_dealt,
                    )
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
                    ModelConfig(
                        x=20, y=5, max_wounds=10, weapons=[WeaponProfile(range=50)]
                    ),
                    ModelConfig(
                        x=21, y=5, max_wounds=10, weapons=[WeaponProfile(range=50)]
                    ),
                ]
            }
        )
        env = WargameEnv(config=cfg_with_opp_weapons)
        env.reset(seed=42)
        _step_to_shooting(env)
        ss = env._action_handler.shooting_slice
        assert ss is not None
        env.step(WargameEnvAction(actions=[ss.start, ss.start + 1]))
        p_dmg = sum(r.result.damage_dealt for r in env._last_player_shooting_results)
        o_dmg = sum(r.result.damage_dealt for r in env._last_opponent_shooting_results)
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
            pp, op, pa, oa, pr, _all_visible, player_advanced=advanced
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
            pp, op, pa, oa, pr, _all_visible, engagement_range=2.0
        )
        assert not m[0, 0], "Model within engagement range should be masked"
        assert m[1, 0], "Model outside engagement range should shoot"

    def test_a_dead_enemy_does_not_keep_a_model_engaged(self) -> None:
        """A corpse must not lock a model out of shooting.

        The gate took `distances.min()` over *every* opponent and applied
        `opponent_alive` only afterwards, so a casualty lying next to a model
        went on pinning it for the rest of the episode. The rule is about being
        engaged by an enemy, and a dead model engages nobody.
        """
        pp = np.array([[0.0, 0.0]])
        op = np.array([[1.0, 0.0], [10.0, 0.0]])  # corpse adjacent, live one far
        pa = np.array([True])
        oa = np.array([False, True])
        pr = np.array([50.0])
        m = compute_shooting_masks(
            pp, op, pa, oa, pr, _all_visible, engagement_range=2.0
        )
        assert m[0, 1], "a corpse within engagement range must not pin the shooter"

    def test_a_live_enemy_still_engages(self) -> None:
        """The companion case: the gate must still fire on a living enemy."""
        pp = np.array([[0.0, 0.0]])
        op = np.array([[1.0, 0.0], [10.0, 0.0]])
        pa = np.array([True])
        oa = np.array([True, True])
        pr = np.array([50.0])
        m = compute_shooting_masks(
            pp, op, pa, oa, pr, _all_visible, engagement_range=2.0
        )
        assert not m[0, 1], "a living adjacent enemy must still pin the shooter"

    def test_every_enemy_dead_leaves_nobody_engaged(self) -> None:
        """The empty-set edge case the fix introduces.

        Masking the dead out before the minimum means a model with no living
        enemy at all reduces over an empty set. It must read as "not engaged",
        not as a crash or as engaged-by-default.
        """
        pp = np.array([[0.0, 0.0]])
        op = np.array([[1.0, 0.0]])
        pa = np.array([True])
        oa = np.array([False])
        pr = np.array([50.0])
        m = compute_shooting_masks(
            pp, op, pa, oa, pr, _all_visible, engagement_range=2.0
        )
        assert not m.any(), "no living target, so no shot -- but no crash either"

    def test_backward_compat_no_new_params(self) -> None:
        pp = np.array([[0, 0]])
        op = np.array([[5, 0]])
        pa = np.array([True])
        oa = np.array([True])
        pr = np.array([20.0])
        m = compute_shooting_masks(pp, op, pa, oa, pr, _all_visible)
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
                [
                    (
                        r.result.hits,
                        r.result.wounds,
                        r.result.unsaved,
                        r.result.damage_dealt,
                    )
                    for r in env._last_player_shooting_results
                ]
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
                [
                    (
                        r.result.hits,
                        r.result.wounds,
                        r.result.unsaved,
                        r.result.damage_dealt,
                    )
                    for r in env._last_player_shooting_results
                ]
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


# ---------------------------------------------------------------------------
# Terrain + shooting mask integration
# ---------------------------------------------------------------------------


def test_terrain_shooting_mask_blocks_through_footprint() -> None:
    """Shooter and target on opposite sides of footprint -> mask forbids."""
    cfg = WargameEnvConfig(
        board_width=30,
        board_height=30,
        number_of_wargame_models=1,
        number_of_objectives=1,
        number_of_opponent_models=1,
        terrain=[TerrainPieceConfig(footprint=(10, 4, 14, 6))],
        models=[
            ModelConfig(
                x=5,
                y=5,
                weapons=[WeaponProfile(range=50)],
            )
        ],
        opponent_models=[ModelConfig(x=20, y=5)],
        opponent_policy=OpponentPolicyConfig(type="random"),
        skip_phases=[BattlePhase.command, BattlePhase.charge, BattlePhase.fight],
    )
    env = WargameEnv(config=cfg)
    env.reset(seed=42)
    pp = np.array([[5, 5]])
    op = np.array([[20, 5]])
    pa = np.array([True])
    oa = np.array([True])
    pr = np.array([50.0])
    mask = compute_shooting_masks(pp, op, pa, oa, pr, env.line_of_sight_matrix)
    assert not mask[0, 0], "LOS through footprint should block shooting"
