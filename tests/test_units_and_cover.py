"""Model occlusion and cover, and the group rule they depend on.

`group_id` is this project's name for the rules' *unit*, so it is the rules
concept here too: a model with a base occludes, but ignores others in its own
group and in its target's group — otherwise a squad shields itself with its own
front rank and nobody can shoot past the man in front. A target only partly
visible is **in cover**, which worsens the attack by one.

Every one of these needs `base_radius > 0`. At radius 0 a model occludes
nothing and the edge rays coincide with the centre one, so the whole feature is
a no-op — which is exactly why an earlier version that never applied the group
rule on the shooting path kept the entire suite green.
"""

from __future__ import annotations

import numpy as np
import pytest

from wargame_rl.wargame.envs.domain import rules_constants
from wargame_rl.wargame.envs.domain.shooting import DefenderStats, resolve_shooting
from wargame_rl.wargame.envs.domain.sight import CLEAR, COVER, HIDDEN, group_keys
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


class TestGroupKeys:
    def test_the_two_armies_do_not_share_a_group_numbering(self) -> None:
        """Both armies have a group 0, and they are different groups.

        Comparing raw ids across sides would make each army's first squad ignore
        the other's when tracing sight — visible only as slightly too much
        shooting, on one squad pairing, in a metric nobody reads per-group.
        """
        player = [ModelConfig(group_id=0), ModelConfig(group_id=1)]
        env = WargameEnv(
            config=WargameEnvConfig(
                board_width=30,
                board_height=30,
                number_of_wargame_models=2,
                number_of_opponent_models=2,
                number_of_objectives=1,
                models=player,
                opponent_models=[ModelConfig(group_id=0), ModelConfig(group_id=1)],
                opponent_policy=OpponentPolicyConfig(type="random"),
            )
        )

        ours = group_keys(env.wargame_models, 0)
        theirs = group_keys(env.opponent_models, 1)

        assert not set(ours.tolist()) & set(theirs.tolist())


def _sight_env(
    player: list[tuple[float, float, int]],
    opponent: list[tuple[float, float, int]],
) -> WargameEnv:
    """Board with models at fixed spots, each carrying a declared group."""
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
                    x=int(x), y=int(y), group_id=u, weapons=[WeaponProfile(range=50)]
                )
                for x, y, u in player
            ],
            opponent_models=[
                ModelConfig(x=int(x), y=int(y), group_id=u) for x, y, u in opponent
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

    def test_a_same_group_screen_does_not_block(self) -> None:
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

    def test_the_targets_own_group_does_not_block(self) -> None:
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

    def test_the_shooting_mask_applies_the_group_rule(self) -> None:
        """The regression this file exists for.

        The group rule was implemented in the sight layer but never reached the
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

    Occlusion is symmetric because the group exemption is symmetric in the pair
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
