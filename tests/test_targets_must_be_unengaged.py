"""A unit locked in melee cannot be shot at.

`docs/rules/04-making-attacks.md` § Select targets requires a target be
"visible, in range and **unengaged**". The engagement term gated only the
SHOOTER: nothing ever reduced the target axis, so a unit in contact could be
shot freely by every enemy not itself engaged.

⚠ It has been invisible rather than wrong. `back_off_to_unengaged` runs on every
mover on both seats, so engagement is 0.0000% of model-pairs and the clause has
never had an opportunity to bite. These tests therefore place models in contact
by hand rather than by playing, because no sequence of legal moves reaches the
state the rule is about.
"""

from __future__ import annotations

import numpy as np

from wargame_rl.wargame.envs.domain.engagement import engaged_units, engaged_with_any
from wargame_rl.wargame.envs.env_components.shooting_masks import (
    compute_unit_shooting_masks,
)


def _all_visible(
    origins: np.ndarray, targets: np.ndarray, candidates: np.ndarray
) -> np.ndarray:
    """No terrain: sight is clear wherever the cheap gates already allowed it."""
    return np.ones_like(candidates, dtype=bool)


def _scenario() -> dict[str, np.ndarray]:
    """Two player models; two enemy units. Player 1 is in contact with unit 1."""
    return dict(
        player_positions=np.array([[0.0, 0.0], [10.0, 0.0]]),
        opponent_positions=np.array([[5.0, 0.0], [10.5, 0.0]]),
        player_alive=np.array([True, True]),
        opponent_alive=np.array([True, True]),
        player_max_ranges=np.array([50.0, 50.0]),
        target_groups=np.array([0, 1], dtype=int),
    )


def _mask(exclude: bool) -> np.ndarray:
    s = _scenario()
    return compute_unit_shooting_masks(
        s["player_positions"],
        s["opponent_positions"],
        s["player_alive"],
        s["opponent_alive"],
        s["player_max_ranges"],
        _all_visible,
        s["target_groups"],
        2,
        engagement_range=1.0,
        base_diameter=0.0,
        exclude_engaged_targets=exclude,
    )


def test_the_engaged_unit_may_not_be_targeted_when_the_rule_is_on() -> None:
    """Enemy unit 1 is 0.5 from player 1, inside the 1.0 engagement range."""
    mask = _mask(exclude=True)
    # Player 0 is unengaged and may shoot the far unit 0 ...
    assert mask[0, 0]
    # ... but unit 1 is locked in melee and is off the table for everyone.
    assert not mask[0, 1]
    assert not mask[1, 1]


def test_without_the_rule_the_engaged_unit_is_freely_shootable() -> None:
    """The behaviour that shipped: only the shooter's engagement was checked."""
    mask = _mask(exclude=False)
    assert mask[0, 1], "the defect: an engaged unit was a legal target"


def test_the_shooter_side_gate_is_unchanged_either_way() -> None:
    """Player 1 is itself engaged, so it may not shoot at all — both ways."""
    for exclude in (True, False):
        assert not _mask(exclude)[1, 0]


def test_a_dead_enemy_engages_nobody() -> None:
    """A casualty keeps its position forever; it must not pin or shield anyone."""
    positions = np.array([[0.0, 0.0]])
    corpse = np.array([[0.5, 0.0]])
    assert not engaged_with_any(
        positions, corpse, np.array([False]), engagement_range=1.0
    )[0]


def test_a_unit_is_engaged_when_any_one_model_is() -> None:
    """The rule is unit-level: one model in contact shields the whole unit."""
    per_model = np.array([False, True, False])
    groups = np.array([0, 0, 1], dtype=int)
    assert list(engaged_units(per_model, groups, 2)) == [True, False]
