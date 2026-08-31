"""Shooting names a unit, and the defender allocates.

The rule this file pins is one substitution with three consequences
(`docs/rules/04-making-attacks.md`, `05-attack-sequence.md`): a weapon selects an
enemy **unit**, the defender picks which of its models takes each attack, and an
attack is discarded only when that whole unit is destroyed.

Under the previous per-model targeting a shot at an already-dead model silently
evaporated, measured at a 36-40% discard rate -- a squad concentrating fire
killed its target with the first attacks and threw the rest away.
"""

from __future__ import annotations

import numpy as np
import pytest

from wargame_rl.wargame.envs.domain.battle_factory import (
    group_span,
    n_groups_for,
    unit_count,
)
from wargame_rl.wargame.envs.domain.entities import WargameModel
from wargame_rl.wargame.envs.domain.shooting import resolve_shooting_phase
from wargame_rl.wargame.envs.types.config import ModelConfig, WeaponProfile


def _model(group_id: int, wounds: int = 1, toughness: int = 3) -> WargameModel:
    """A bare target model in a given unit."""
    return WargameModel(
        location=np.array([0.0, 0.0]),
        stats={
            "max_wounds": wounds,
            "current_wounds": wounds,
            "toughness": toughness,
            "save": 7,  # never saves, so every wound lands and counts stay legible
        },
        distances_to_objectives=np.zeros((1, 2)),
        group_id=group_id,
    )


def _lethal_weapon() -> WeaponProfile:
    """A weapon that hits, wounds and kills on any roll."""
    return WeaponProfile(range=48, attacks=1, ballistic_skill=2, strength=10, damage=1)


class TestAttacksAreLostOnlyWhenTheUnitDies:
    def test_a_second_shot_at_a_unit_hits_a_survivor(self) -> None:
        """Two attacks on a two-model unit kill two models, not one.

        This is the whole defect in one assertion. Aimed at a model, the second
        attack was discarded because its target was already dead.
        """
        # Arrange — five attackers, one target unit of five.
        attackers = [_model(group_id=0) for _ in range(5)]
        targets = [_model(group_id=0) for _ in range(5)]
        weapons = [[_lethal_weapon()] for _ in attackers]

        # Act — every attacker declares against unit 0.
        results = resolve_shooting_phase(
            [(i, 0) for i in range(5)],
            attackers,
            targets,
            weapons,
            np.random.default_rng(0),
        )

        # Assert — dice-independent properties only. An unmodified 1 always
        # fails, so no stat line makes five attacks land five kills.
        assert len(results) == 5, "no attack may be discarded while the unit lives"
        kills = sum(r.killed for r in results)
        assert kills >= 2, "precondition: this seed must kill more than once"
        # Every kill is a *different* model. Under per-model targeting the
        # second attack on a dead model was thrown away, so kills could never
        # exceed the number of distinct models declared against.
        assert sum(not m.is_alive for m in targets) == kills
        assert len({r.target_idx for r in results if r.killed}) == kills

    def test_excess_attacks_against_a_wiped_unit_are_lost(self) -> None:
        """Once the unit is destroyed, the rest of the attacks go nowhere.

        The one discard the rules do allow, and the boundary of the fix: without
        it, attacks would spill onto a unit nobody aimed at.
        """
        # Arrange — five attackers against a unit of two.
        attackers = [_model(group_id=0) for _ in range(5)]
        targets = [_model(group_id=0) for _ in range(2)]
        weapons = [[_lethal_weapon()] for _ in attackers]

        # Act
        results = resolve_shooting_phase(
            [(i, 0) for i in range(5)],
            attackers,
            targets,
            weapons,
            np.random.default_rng(0),
        )

        # Assert
        assert len(results) == 2
        assert all(not m.is_alive for m in targets)

    def test_attacks_never_spill_onto_an_undeclared_unit(self) -> None:
        """A wiped target unit does not redirect fire to its neighbours."""
        # Arrange
        attackers = [_model(group_id=0) for _ in range(4)]
        targets = [_model(group_id=0), _model(group_id=1), _model(group_id=1)]
        weapons = [[_lethal_weapon()] for _ in attackers]

        # Act — all four declare against unit 0, which holds one model.
        results = resolve_shooting_phase(
            [(i, 0) for i in range(4)],
            attackers,
            targets,
            weapons,
            np.random.default_rng(0),
        )

        # Assert
        assert len(results) == 1
        assert not targets[0].is_alive
        assert all(m.is_alive for m in targets[1:]), "unit 1 was never declared against"


class TestTheDefenderAllocates:
    def test_a_wounded_model_takes_the_next_attack(self) -> None:
        """*"a model that has already lost Wounds if one is available"*.

        Concentrating damage is what stops a unit fielding a rank of survivors
        each one wound from death. Vacuous at `max_wounds: 1`, so this needs a
        multi-wound profile to say anything.
        """
        # Arrange — model 1 is already hurt.
        attackers = [_model(group_id=0)]
        targets = [_model(group_id=0, wounds=3) for _ in range(3)]
        targets[1].take_damage(1)
        weapons = [[_lethal_weapon()]]

        # Act
        results = resolve_shooting_phase(
            [(0, 0)], attackers, targets, weapons, np.random.default_rng(0)
        )

        # Assert
        assert len(results) == 1
        assert results[0].target_idx == 1, "the wounded model should absorb it"

    def test_the_declared_unit_is_recorded_alongside_the_bleeding_model(self) -> None:
        """`target_idx` is who bled; `target_group` is what was aimed at."""
        # Arrange
        attackers = [_model(group_id=0)]
        targets = [_model(group_id=0), _model(group_id=1)]
        weapons = [[_lethal_weapon()]]

        # Act
        results = resolve_shooting_phase(
            [(0, 1)], attackers, targets, weapons, np.random.default_rng(0)
        )

        # Assert
        assert results[0].target_group == 1
        assert results[0].target_idx == 1
        assert targets[0].is_alive, "unit 0 was not the declared target"


class TestUnitCountSizesTheActionSpace:
    @pytest.mark.parametrize(
        ("n_models", "max_groups", "expected"),
        [(25, 5, 5), (2, 5, 2), (7, 5, 4), (1, 5, 1), (0, 5, 0)],
    )
    def test_the_split_is_derived_not_assumed(
        self, n_models: int, max_groups: int, expected: int
    ) -> None:
        """`max_groups` is a cap, not the answer.

        The split is ``group_id = i // group_span(n, max_groups)``, and an
        uneven span leaves a short final unit: 7 models capped at 5 have a span
        of 2 and land in *four* units. An action space sized from `max_groups`
        would name one that does not exist.

        ⚠ **Replaces a case asserting ``(7, 5) -> 7``**, which pinned the
        aliasing bug rather than the intent: seven units exceed the cap, and
        `_group_ids_to_one_hot` clips to ``max_groups - 1``, so three of them
        shared one observation column. See `TestGroupIdsNeverAliasOntoOneColumn`.
        """
        assert n_groups_for(n_models, max_groups) == expected


class TestGroupIdsNeverAliasOntoOneColumn:
    """Regression for the group-id aliasing bug, fixed by rounding `group_span` up.

    `_group_ids_to_one_hot` **clips** ids to ``max_groups - 1`` rather than
    raising, so an army splitting into more units than the cap encoded two
    distinct units as one column -- silently, with no exception, while
    `unit_count` sized the shooting slice at the true count. The network could
    name units its observation could not tell apart.

    Measured live before the fix on `configs/experiments/30v15_fast_horde_vs_elite.yaml`:
    15 elites at ``max_groups: 6`` split into 8 units, ids 0..7, clipped to 0..5.
    """

    @pytest.mark.parametrize("max_groups", [2, 3, 5, 6, 8, 10])
    @pytest.mark.parametrize("n_models", list(range(1, 200)))
    def test_no_army_splits_into_more_units_than_the_cap(
        self, n_models: int, max_groups: int
    ) -> None:
        """The property that makes the one-hot's clip unreachable."""
        assert n_groups_for(n_models, max_groups) <= max_groups

    @pytest.mark.parametrize("max_groups", [2, 3, 5, 6, 8, 10])
    @pytest.mark.parametrize("n_models", list(range(1, 200)))
    def test_no_model_carries_a_group_id_the_one_hot_would_clip(
        self, n_models: int, max_groups: int
    ) -> None:
        """`_build_models` assigns ``group_id = i // span``; the highest must fit.

        Asserted against the split itself rather than against `n_groups_for`, so
        the two cannot drift apart and both be wrong.
        """
        span = group_span(n_models, max_groups)
        highest_id = (n_models - 1) // span

        assert highest_id <= max_groups - 1

    @pytest.mark.parametrize(
        ("n_models", "max_groups"),
        [(25, 5), (24, 8), (30, 6), (20, 5), (4, 2), (2, 1), (25, 25)],
    )
    def test_the_fix_is_bit_identical_where_the_cap_divides_the_army(
        self, n_models: int, max_groups: int
    ) -> None:
        """Every golden, evaluation and dev config is one of these shapes.

        Rounding up changes the span only when `max_groups` divides `n_models`
        unevenly, so the reward and observation goldens are untouched and no
        checkpoint is orphaned. The seven configs that *do* move are all under
        `configs/experiments/`, and all of them were aliasing.
        """
        assert group_span(n_models, max_groups) == n_models // max_groups

    def test_explicit_group_ids_win_over_the_count_split(self) -> None:
        """The action index *is* the group id, so the highest id sets the width.

        Two models both declared `group_id: 0` are one unit, where the
        count-based split would have made two.
        """
        configs = [ModelConfig(x=1, y=1, group_id=0), ModelConfig(x=2, y=2, group_id=0)]

        assert unit_count(2, 5, configs) == 1
        assert unit_count(2, 5, None) == 2
