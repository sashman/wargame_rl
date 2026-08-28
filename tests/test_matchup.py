"""Unit-level matchups are a reduction of the shipped per-model matrix.

Two things here are worth more than the rest. The **reduction axes**: all
shooters in a unit fire so the attacker axis sums, one representative defender
takes the maths so the defender axis does not -- getting that backwards reads
`n * m` times the truth and looks entirely plausible. And the **wound clip**,
which is identically 1.0 on every config in the repo (`damage: 1`,
`max_wounds: 1`) and so has no natural coverage at all: the first heavy weapon
anyone adds would inherit a threefold overstatement silently.
"""

from __future__ import annotations

import numpy as np
import pytest

from wargame_rl.wargame.envs.board.matchup import (
    UnitProfile,
    exchange_ratio,
    matchup_matrix,
    matchup_table,
    unit_profiles,
)
from wargame_rl.wargame.envs.domain.shooting import DefenderStats, expected_damage
from wargame_rl.wargame.envs.types.config.entities import ModelConfig, WeaponProfile


def _rifle(**overrides: int) -> WeaponProfile:
    stats = {"range": 12, "attacks": 1, "ballistic_skill": 3, "strength": 4, "ap": 1}
    return WeaponProfile(**{**stats, **overrides})


def _unit(
    group_id: int, n: int, weapon: WeaponProfile, move: float | None = None
) -> list[ModelConfig]:
    return [
        ModelConfig(group_id=group_id, weapons=[weapon], move=move) for _ in range(n)
    ]


def _profile(**overrides: object) -> UnitProfile:
    stats: dict[str, object] = {
        "group_id": 0,
        "n_models": 5,
        "move": 6.0,
        "weapon_range": 12.0,
        "attacks": 1,
        "ballistic_skill": 3,
        "strength": 4,
        "ap": 1,
        "damage": 1,
        "toughness": 3,
        "save": 4,
        "max_wounds": 1,
    }
    return UnitProfile(**{**stats, **overrides})  # type: ignore[arg-type]


class TestReductionAxes:
    def test_a_unit_of_five_does_five_times_one_models_damage(self) -> None:
        """Sum on the attacker axis. NOT twenty-five times -- see the class docstring."""
        one = _profile(n_models=1)
        five = _profile(n_models=5)
        single = expected_damage(one, DefenderStats(toughness=3, save=4))

        against_five = matchup_matrix((five,), (five,))[0, 0]

        assert against_five == pytest.approx(5.0 * single, rel=1e-6)

    def test_the_defender_axis_does_not_multiply(self) -> None:
        """A bigger target unit does not take more fire, only more of it to kill.

        Both targets are far larger than the volley so the destroyed-unit cap
        cannot bind -- otherwise this would be measuring the cap and passing for
        the wrong reason.
        """
        attacker = _profile(n_models=5)

        small = matchup_matrix((attacker,), (_profile(n_models=20),))[0, 0]
        large = matchup_matrix((attacker,), (_profile(n_models=40),))[0, 0]

        assert small == pytest.approx(large, rel=1e-6)
        assert 0.0 < float(small) < 20.0

    def test_casualties_cannot_exceed_the_target_units_model_count(self) -> None:
        """Fire that would remove more models than exist has removed the unit."""
        overwhelming = _profile(n_models=50, attacks=6, strength=10, ap=6)

        taken = matchup_matrix((overwhelming,), (_profile(n_models=3),))[0, 0]

        assert taken == pytest.approx(3.0)

    def test_overkill_is_reported_rather_than_hidden_by_the_cap(self) -> None:
        """A bound cap is fire being wasted, which is the thing worth seeing."""
        overwhelming = _profile(n_models=50, attacks=6, strength=10, ap=6)

        matchup = matchup_table((overwhelming,), (_profile(n_models=3),))[0][0]

        assert matchup.overkill_share > 0.5
        assert (
            matchup_table((_profile(n_models=1),), (_profile(n_models=20),))[0][
                0
            ].overkill_share
            == 0.0
        )


class TestTheWoundClip:
    """`expected_damage` returns wounds and does not clip them; casualties must."""

    def test_a_damage_three_weapon_does_not_remove_three_one_wound_models(self) -> None:
        """The case no shipped config exercises. Damage does not spill between models."""
        heavy = _profile(n_models=1, damage=3)
        light = _profile(n_models=1, damage=1)
        target = _profile(n_models=20, max_wounds=1)

        heavy_casualties = matchup_matrix((heavy,), (target,))[0, 0]
        light_casualties = matchup_matrix((light,), (target,))[0, 0]

        assert heavy_casualties == pytest.approx(light_casualties, rel=1e-6)

    def test_wounds_are_reported_unclipped_beside_the_casualties(self) -> None:
        """Both numbers are wanted: the clip is where the two diverge."""
        matchup = matchup_table(
            (_profile(n_models=1, damage=3),), (_profile(n_models=20, max_wounds=1),)
        )[0][0]

        assert matchup.wounds_per_round == pytest.approx(
            3.0 * matchup.casualties_per_round, rel=1e-6
        )

    def test_damage_three_against_wounds_two_still_removes_one_model_per_hit(
        self,
    ) -> None:
        """One hit kills a Wounds-2 model outright and wastes a point doing it.

        ⚠ REPLACES a test asserting `wounds * 2/3`, i.e. that a single Damage-3
        hit removes **two** Wounds-2 models. That came from reading
        `min(damage, max_wounds) / damage` as a wound-clip when what the caller
        needs is a damage-points-to-models conversion; the two coincide only at
        `max_wounds: 1`, which is every shipped config.
        """
        matchup = matchup_table(
            (_profile(n_models=1, damage=3),), (_profile(n_models=20, max_wounds=2),)
        )[0][0]

        assert matchup.casualties_per_round == pytest.approx(
            matchup.wounds_per_round / 3.0, rel=1e-6
        )

    def test_a_multi_wound_model_absorbs_several_light_hits(self) -> None:
        """The direction the old scale missed entirely: it returned 1.0 here.

        Damage 1 against Wounds 3 takes three hits to remove one model, so the
        volley removes a third of the models its damage points suggest.
        """
        matchup = matchup_table(
            (_profile(n_models=1, damage=1),), (_profile(n_models=20, max_wounds=3),)
        )[0][0]

        assert matchup.casualties_per_round == pytest.approx(
            matchup.wounds_per_round / 3.0, rel=1e-6
        )

    def test_the_clip_is_unchanged_at_every_one_wound_profile(self) -> None:
        """No measured number moves: the fix is a no-op wherever `max_wounds` is 1."""
        for damage in (1, 2, 3, 4, 5, 6):
            matchup = matchup_table(
                (_profile(n_models=1, damage=damage),),
                (_profile(n_models=99, max_wounds=1),),
            )[0][0]

            assert matchup.casualties_per_round == pytest.approx(
                matchup.wounds_per_round / damage, rel=1e-6
            )

    def test_the_clip_is_a_no_op_on_the_profiles_every_config_ships(self) -> None:
        """Damage 1 against Wounds 1: casualties and wounds are the same number."""
        matchup = matchup_table((_profile(),), (_profile(n_models=99),))[0][0]

        assert matchup.casualties_per_round == pytest.approx(matchup.wounds_per_round)


class TestReach:
    def test_the_elite_buys_two_thirds_of_a_round_over_the_fast_horde(self) -> None:
        """The `30v15` numbers: (24 - 12) / (12 + 6). Small, and correctly so."""
        elite = _profile(weapon_range=24.0, move=6.0)
        horde = _profile(weapon_range=12.0, move=12.0)

        matchup = matchup_table((elite,), (horde,))[0][0]

        assert matchup.reach_margin == pytest.approx(12.0)
        assert matchup.free_rounds == pytest.approx(2.0 / 3.0, rel=1e-6)

    def test_the_shorter_ranged_unit_gets_no_free_rounds(self) -> None:
        """Reach is not symmetric: only the longer gun is unanswered."""
        elite = _profile(weapon_range=24.0, move=6.0)
        horde = _profile(weapon_range=12.0, move=12.0)

        assert matchup_table((horde,), (elite,))[0][0].free_rounds == 0.0

    def test_range_never_enters_the_damage_scalar(self) -> None:
        """A pre-game tool has no positions; folding reach in would fake one."""
        near = _profile(weapon_range=12.0)
        far = _profile(weapon_range=48.0)

        assert matchup_matrix((near,), (near,))[0, 0] == pytest.approx(
            matchup_matrix((far,), (near,))[0, 0]
        )

    def test_the_exchange_ratio_is_infinite_where_only_one_side_can_answer(
        self,
    ) -> None:
        """`inf` is a real state of this game, not a division accident."""
        elite = _profile(weapon_range=24.0)
        horde = _profile(weapon_range=12.0)

        long_range, short_range = exchange_ratio(elite, horde)

        assert long_range == float("inf")
        assert short_range == pytest.approx(1.0)


class TestUnitProfiles:
    def test_units_come_back_in_ascending_group_order_with_their_sizes(self) -> None:
        configs = _unit(1, 3, _rifle()) + _unit(0, 5, _rifle())

        profiles = unit_profiles(configs, len(configs), default_move=6.0)

        assert [p.group_id for p in profiles] == [0, 1]
        assert [p.n_models for p in profiles] == [5, 3]

    def test_a_model_without_its_own_move_takes_the_scenarios(self) -> None:
        configs = _unit(0, 2, _rifle())

        assert unit_profiles(configs, 2, default_move=6.0)[0].move == 6.0
        assert (
            unit_profiles(_unit(0, 2, _rifle(), move=12.0), 2, default_move=6.0)[0].move
            == 12.0
        )

    def test_a_unit_of_mixed_profiles_is_refused_rather_than_averaged(self) -> None:
        """An averaged stat line describes a model that is not on the board."""
        configs = _unit(0, 2, _rifle()) + _unit(0, 1, _rifle(strength=9))

        with pytest.raises(ValueError, match="mixes model profiles"):
            unit_profiles(configs, len(configs), default_move=6.0)

    def test_an_unarmed_unit_threatens_nobody(self) -> None:
        """An empty `weapons` list means the model cannot shoot at all."""
        configs = [ModelConfig(group_id=0, weapons=[]) for _ in range(3)]

        profiles = unit_profiles(configs, 3, default_move=6.0)

        assert matchup_matrix(profiles, profiles)[0, 0] == 0.0

    def test_a_config_of_the_wrong_length_is_refused(self) -> None:
        """Positional alignment with the model list is the whole contract."""
        with pytest.raises(ValueError, match="for 9 models"):
            unit_profiles(_unit(0, 2, _rifle()), 9, default_move=6.0)


def test_an_empty_side_produces_an_empty_matrix_rather_than_raising() -> None:
    """A wiped-out army is a state the report still has to print."""
    assert matchup_matrix((), (_profile(),)).shape == (0, 1)
    assert np.array_equal(matchup_matrix((_profile(),), ()), np.zeros((1, 0)))
