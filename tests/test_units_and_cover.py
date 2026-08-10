"""Units as a real entity, and the sight rules that depend on them.

A model with a base occludes. It ignores others in its own unit and in its
target's — otherwise a squad shields itself with its own front rank, and no
model can shoot past the man in front of it. A target only partly visible is
**in cover**, which worsens the attack by one.

Every one of these needs `base_radius > 0`. At radius 0 a model occludes
nothing and the two edge rays coincide with the centre one, so the whole feature
is a no-op — which is exactly why an earlier version that never applied the unit
rule at all kept the entire suite green.
"""

from __future__ import annotations

import numpy as np
import pytest

from wargame_rl.wargame.envs.domain import rules_constants
from wargame_rl.wargame.envs.domain.shooting import DefenderStats, resolve_shooting
from wargame_rl.wargame.envs.domain.sight import CLEAR, COVER, HIDDEN
from wargame_rl.wargame.envs.types import WargameEnvConfig
from wargame_rl.wargame.envs.types.config import (
    ModelConfig,
    OpponentPolicyConfig,
    WeaponProfile,
)
from wargame_rl.wargame.envs.wargame import WargameEnv

RADIUS = 1.0


class _Weapon:
    """Minimal WeaponStats, so the roll can be pinned without config validation."""

    def __init__(self, ballistic_skill: int) -> None:
        self.attacks = 200
        self.ballistic_skill = ballistic_skill
        self.strength = 4
        self.ap = 0
        self.damage = 1


class TestUnitIdentity:
    def test_an_unset_unit_makes_each_model_its_own(self) -> None:
        """Not `group_id`, deliberately.

        `group_id` defaults to 0 for every model, so falling back to it would
        turn a config that never sets groups into one 25-model unit — silently
        switching the sight rule off entirely. Own-unit is the
        maximally-occluding reading and the safe default.
        """
        env = WargameEnv(
            config=WargameEnvConfig(
                number_of_wargame_models=4,
                number_of_objectives=1,
                board_width=30,
                board_height=30,
                # Explicit configs, because that is what a real scenario has and
                # it is where `group_id` collapses to 0 for everyone.
                models=[ModelConfig() for _ in range(4)],
            )
        )

        assert {m.group_id for m in env.wargame_models} == {0}
        assert [m.unit_id for m in env.wargame_models] == [0, 1, 2, 3]

    def test_a_declared_unit_is_honoured_and_is_not_the_group(self) -> None:
        env = WargameEnv(
            config=WargameEnvConfig(
                number_of_wargame_models=4,
                number_of_objectives=1,
                board_width=30,
                board_height=30,
                models=[
                    ModelConfig(group_id=0, unit_id=7),
                    ModelConfig(group_id=0, unit_id=7),
                    ModelConfig(group_id=1, unit_id=9),
                    ModelConfig(group_id=1),
                ],
            )
        )

        assert [m.unit_id for m in env.wargame_models] == [7, 7, 9, 3]


def _sight_env(
    player: list[tuple[float, float, int]],
    opponent: list[tuple[float, float, int]],
) -> WargameEnv:
    """Board with models at fixed spots, each carrying a declared unit."""
    return WargameEnv(
        config=WargameEnvConfig(
            board_width=60,
            board_height=40,
            number_of_wargame_models=len(player),
            number_of_opponent_models=len(opponent),
            number_of_objectives=1,
            number_of_battle_rounds=2,
            base_radius=RADIUS,
            models=[
                ModelConfig(
                    x=int(x), y=int(y), unit_id=u, weapons=[WeaponProfile(range=50)]
                )
                for x, y, u in player
            ],
            opponent_models=[
                ModelConfig(x=int(x), y=int(y), unit_id=u) for x, y, u in opponent
            ],
            opponent_policy=OpponentPolicyConfig(type="random"),
            render_mode=None,
        )
    )


def _visibility(env: WargameEnv) -> np.ndarray:
    return env.visibility_between(
        np.array([m.location for m in env.wargame_models], dtype=float),
        np.array([m.location for m in env.opponent_models], dtype=float),
        origin_models=env.wargame_models,
        target_models=env.opponent_models,
    )


class TestModelOcclusion:
    def test_a_model_does_not_block_its_own_sight(self) -> None:
        """The observer's base sits on the start of every ray it casts."""
        env = _sight_env([(5, 20, 0)], [(40, 20, 1)])
        env.reset(seed=0)

        assert _visibility(env)[0, 0] == CLEAR

    def test_one_enemy_on_the_line_gives_cover_not_concealment(self) -> None:
        """A single same-size model cannot hide a target, and that is correct.

        The corridor is as wide as the pair's bases, so one blocker of the same
        radius covers the centre line and leaves both edges clear. Hiding takes
        a screen, not a man.
        """
        env = _sight_env([(5, 20, 0)], [(20, 20, 1), (40, 20, 2)])
        env.reset(seed=0)

        assert _visibility(env)[0, 1] == COVER

    def test_a_screen_of_enemies_hides_the_target(self) -> None:
        env = _sight_env(
            [(5, 20, 0)],
            [(20, 19, 1), (20, 20, 1), (20, 21, 1), (40, 20, 2)],
        )
        env.reset(seed=0)

        assert _visibility(env)[0, 3] == HIDDEN

    def test_a_squadmate_screen_does_not_block(self) -> None:
        """Without this a unit shields itself, and the man at the back never fires."""
        other_unit = _sight_env(
            [(5, 20, 0), (20, 19, 1), (20, 20, 1), (20, 21, 1)], [(40, 20, 9)]
        )
        other_unit.reset(seed=0)
        assert _visibility(other_unit)[0, 0] == HIDDEN

        same_unit = _sight_env(
            [(5, 20, 0), (20, 19, 0), (20, 20, 0), (20, 21, 0)], [(40, 20, 9)]
        )
        same_unit.reset(seed=0)
        assert _visibility(same_unit)[0, 0] == CLEAR

    def test_the_targets_own_unit_does_not_block(self) -> None:
        """A unit cannot hide behind its own front rank either."""
        other_unit = _sight_env(
            [(5, 20, 0)], [(20, 19, 1), (20, 20, 1), (20, 21, 1), (40, 20, 2)]
        )
        other_unit.reset(seed=0)
        assert _visibility(other_unit)[0, 3] == HIDDEN

        same_unit = _sight_env(
            [(5, 20, 0)], [(20, 19, 2), (20, 20, 2), (20, 21, 2), (40, 20, 2)]
        )
        same_unit.reset(seed=0)
        assert _visibility(same_unit)[0, 3] == CLEAR

    def test_the_shooting_mask_applies_the_unit_rule(self) -> None:
        """The regression this file exists for.

        The unit rule was implemented in the sight layer but never reached the
        shooting mask, which received positions and no models. Every config in
        the suite has `base_radius: 0`, where nothing occludes, so the whole
        suite stayed green while a squad could not shoot past its own front rank.
        """
        env = _sight_env(
            [(5, 20, 0), (20, 19, 0), (20, 20, 0), (20, 21, 0)], [(40, 20, 9)]
        )
        env.reset(seed=0)

        mask = env.sight_between(env.wargame_models, env.opponent_models)(
            np.array([m.location for m in env.wargame_models], dtype=float),
            np.array([m.location for m in env.opponent_models], dtype=float),
            np.ones((4, 1), dtype=bool),
        )

        assert bool(mask[0, 0]), "a squadmate must not block the shot"


class TestCover:
    def test_a_partly_screened_target_is_in_cover(self) -> None:
        """Some of the corridor blocked, some clear, is cover."""
        env = _sight_env([(5, 20, 0)], [(20, 21, 1), (40, 20, 2)])
        env.reset(seed=0)
        env.opponent_models[0].location = np.array([20.0, 20.8])

        assert _visibility(env)[0, 1] == COVER

    def test_cover_worsens_the_hit_roll_by_one(self) -> None:
        """And the unmodified 6 still hits, so cover is never an absolute shield."""
        defender = DefenderStats(toughness=3, save=7)
        weapon = _Weapon(ballistic_skill=4)

        open_ground = resolve_shooting(
            weapon, defender, np.random.default_rng(1), in_cover=False
        )
        in_cover = resolve_shooting(
            weapon, defender, np.random.default_rng(1), in_cover=True
        )

        assert rules_constants.COVER_RANGED_SKILL_PENALTY == 1
        assert in_cover.hits < open_ground.hits
        assert in_cover.hits > 0

    def test_no_base_means_no_cover_ever(self) -> None:
        """The property that makes all of this a no-op for older configs.

        With no base the two edge rays coincide with the centre ray, so a pair
        is CLEAR or HIDDEN and never in between — which is why the golden gates
        still pass unchanged on every config that predates model bases.
        """
        env = WargameEnv(
            config=WargameEnvConfig(
                board_width=60,
                board_height=40,
                number_of_wargame_models=2,
                number_of_opponent_models=2,
                number_of_objectives=1,
                number_of_battle_rounds=2,
                base_radius=0.0,
                opponent_policy=OpponentPolicyConfig(type="random"),
            )
        )
        env.reset(seed=3)

        assert not (_visibility(env) == COVER).any()


@pytest.mark.parametrize("seed", range(4))
def test_visibility_is_symmetric(seed: int) -> None:
    """A sees B exactly as well as B sees A.

    Occlusion is symmetric because the unit exemption is symmetric in the pair
    and the geometry is a segment. `firepower_ratio` reads an exposed model as
    one that can also fire, so an asymmetry here would make that metric count
    two different populations.
    """
    env = WargameEnv(
        config=WargameEnvConfig(
            board_width=60,
            board_height=40,
            number_of_wargame_models=6,
            number_of_opponent_models=6,
            number_of_objectives=2,
            number_of_battle_rounds=3,
            base_radius=RADIUS,
            opponent_policy=OpponentPolicyConfig(type="random"),
        )
    )
    env.reset(seed=seed)
    players = np.array([m.location for m in env.wargame_models], dtype=float)
    opponents = np.array([m.location for m in env.opponent_models], dtype=float)

    forward = env.visibility_between(
        players,
        opponents,
        origin_models=env.wargame_models,
        target_models=env.opponent_models,
    )
    backward = env.visibility_between(
        opponents,
        players,
        origin_models=env.opponent_models,
        target_models=env.wargame_models,
    )

    assert (forward == backward.T).all()
