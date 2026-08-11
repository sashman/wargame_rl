"""Start-state augmentation: a squad begins the episode on an objective.

The augmentation exists to attack an *optimisation* failure rather than a
pricing one, so the property that matters most is not that it works but that it
cannot leak into a measurement. An evaluation that silently ran augmented starts
would score a different scenario against a bar measured on the real one, and it
would look entirely plausible doing it -- so the no-op guarantee is pinned on
positions, bit-for-bit, not on a flag.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from wargame_rl.wargame.envs.domain.battle_factory import from_config
from wargame_rl.wargame.envs.domain.placement import place_for_episode
from wargame_rl.wargame.envs.types.config import OpponentPolicyConfig, WargameEnvConfig
from wargame_rl.wargame.envs.wargame import WargameEnv

AUGMENT = {"augment_start": True}
BASE_RADIUS = 0.63

# sha256 of player start positions at seeds 3, 7 and 11 on `_config`, recorded
# from a git worktree at the commit *before* the augmentation existed. Do not
# regenerate it from this build -- that would pin the bug rather than the
# behaviour. Regenerate only from a worktree predating the feature.
PRE_FEATURE_START_DIGEST = (
    "0c3e1e218105a3076336d75a5322477af7390481b7c0e64648303155e8019a36"
)
# Same, but for a seeded reset followed by five *unseeded* ones -- the pattern
# the training rollout uses, and the only one a stray draw is visible in.
PRE_FEATURE_STREAM_DIGEST = (
    "d51e1b1474343e4c634f228bd4d49d7c17f67b76c55fb27862f7c1f5d6b733a5"
)


def _config(probability: float) -> WargameEnvConfig:
    """A small two-group scenario with objectives big enough to stand on."""
    return WargameEnvConfig(
        render_mode=None,
        board_width=30,
        board_height=20,
        number_of_wargame_models=6,
        number_of_opponent_models=4,
        number_of_objectives=2,
        objective_radius_size=3,
        max_groups=2,
        opponent_policy=OpponentPolicyConfig(type="random"),
        start_on_objective_probability=probability,
    )


def _positions(env: WargameEnv) -> np.ndarray:
    """Every player model's position, in model order."""
    return np.array([np.asarray(m.location, dtype=float) for m in env.wargame_models])


def _on_objective(env: WargameEnv) -> np.ndarray:
    """``(n_models,)`` mask of player models standing on any objective."""
    mask = []
    for model in env.wargame_models:
        location = np.asarray(model.location, dtype=float)
        mask.append(
            any(
                float(np.linalg.norm(location - np.asarray(o.location, dtype=float)))
                <= o.radius_size
                for o in env.objectives
            )
        )
    return np.array(mask, dtype=bool)


@pytest.mark.parametrize("probability", [0.0, 1.0])
def test_evaluation_start_matches_the_pre_feature_code(probability: float) -> None:
    """Without the option, the start is bit-identical to the code before this feature.

    This is the guarantee the whole design rests on: `just measure-checkpoint`
    and the in-training baselines never pass `augment_start`, so an augmented
    run must still be scored on exactly the layouts the bar was measured on. A
    failure here means every number quoted against that bar is void.

    **The digest is recorded from a worktree at the parent commit, and it has to
    be.** The obvious version of this test -- reset two envs of this build at
    the same seed and compare -- is *vacuous*: both run the new code, so a stray
    unconditional draw shifts both streams equally and cancels. That version was
    written first, and it passed while the augmentation drew on every reset.
    Verified sensitive: making the draw unconditional changes this digest.
    """
    # Arrange
    env = WargameEnv(config=_config(probability))
    digest = hashlib.sha256()

    # Act
    for seed in (3, 7, 11):
        env.reset(seed=seed)
        digest.update(np.ascontiguousarray(_positions(env), dtype=np.float64).tobytes())

    # Assert
    assert digest.hexdigest() == PRE_FEATURE_START_DIGEST


def test_augmentation_puts_a_whole_group_on_an_objective() -> None:
    """At probability 1.0 with the option passed, a squad starts on the point."""
    # Arrange
    env = WargameEnv(config=_config(1.0))

    # Act
    env.reset(seed=11, options=AUGMENT)

    # Assert
    on_objective = _on_objective(env)
    assert on_objective.any(), "no model was moved onto an objective"

    groups = np.array([m.group_id for m in env.wargame_models])
    moved_groups = set(groups[on_objective].tolist())
    assert len(moved_groups) == 1, "the augmentation should move exactly one group"
    group = moved_groups.pop()
    assert on_objective[groups == group].all(), "it should move the whole group"


def test_augmentation_never_overlaps_bases() -> None:
    """Teleported bases must clear each other and everyone already on the board.

    A squad dropped onto a point it shares with the enemy is exactly where
    overlap would happen, and overlapping bases are illegal everywhere else in
    the game -- an augmentation that produced them would be training the policy
    on states it can never encounter.
    """
    # Arrange
    config = _config(1.0)
    config.base_radius = BASE_RADIUS
    env = WargameEnv(config=config)

    # Act / Assert
    for seed in range(12):
        env.reset(seed=seed, options=AUGMENT)
        points = np.array(
            [
                np.asarray(m.location, dtype=float)
                for m in list(env.wargame_models) + list(env.opponent_models)
            ]
        )
        deltas = points[:, None, :] - points[None, :, :]
        distances = np.linalg.norm(deltas, axis=-1)
        np.fill_diagonal(distances, np.inf)
        assert distances.min() >= 2.0 * BASE_RADIUS - 1e-9, (
            f"seed {seed}: bases overlap at {distances.min():.4f}"
        )


def test_disabled_augmentation_draws_nothing_from_the_layout_rng() -> None:
    """A config carrying the field must not shift the unseeded layout stream.

    This is the test with teeth, and it took three attempts to find. The seeded
    digest above cannot see a stray draw at all: the augmentation runs *last* in
    placement and every seeded `reset` re-seeds, so a trailing draw cannot reach
    that episode's positions. It only surfaces on the **unseeded** resets the
    training rollout actually does, where `reset()` continues the stream and one
    extra draw reshuffles every episode after it.

    Verified sensitive: making the draw unconditional changes this digest
    (d51e1b14... -> 2dc8a13d...), while leaving the seeded one untouched.
    """
    # Arrange
    env = WargameEnv(config=_config(0.0))
    digest = hashlib.sha256()

    # Act
    env.reset(seed=3)
    for _ in range(5):
        env.reset()
        digest.update(np.ascontiguousarray(_positions(env), dtype=np.float64).tobytes())

    # Assert
    assert digest.hexdigest() == PRE_FEATURE_STREAM_DIGEST


def test_probability_zero_is_a_no_op_even_when_requested() -> None:
    """Asking for augmentation a config does not enable must draw nothing.

    The draw is conditional on both the option *and* a positive probability, so
    that the control arm of a screen -- same code path, probability 0 -- keeps
    the layout stream of every run that predates the feature.
    """
    # Arrange
    plain = WargameEnv(config=_config(0.0))
    asked = WargameEnv(config=_config(0.0))

    # Act
    plain.reset(seed=3)
    asked.reset(seed=3, options=AUGMENT)

    # Assert
    np.testing.assert_array_equal(_positions(plain), _positions(asked))


def test_augmentation_is_deterministic_for_a_seed() -> None:
    """Same seed and same option give the same start, so runs stay reproducible."""
    # Arrange
    first = WargameEnv(config=_config(1.0))
    second = WargameEnv(config=_config(1.0))

    # Act
    first.reset(seed=5, options=AUGMENT)
    second.reset(seed=5, options=AUGMENT)

    # Assert
    np.testing.assert_array_equal(_positions(first), _positions(second))


def test_partial_probability_produces_both_kinds_of_start() -> None:
    """At 0.5 the augmentation fires on some episodes and not others.

    Pins that the probability is read at all: a bug that treated any positive
    value as "always" would still pass every test above.
    """
    # Arrange
    env = WargameEnv(config=_config(0.5))
    outcomes = []

    # Act
    for seed in range(24):
        env.reset(seed=seed, options=AUGMENT)
        outcomes.append(bool(_on_objective(env).sum() >= 3))

    # Assert
    assert any(outcomes), "never fired at probability 0.5"
    assert not all(outcomes), "always fired at probability 0.5"


def test_teleported_squad_is_set_up_unengaged() -> None:
    """No teleported model is placed in base contact or inside engagement range.

    The rules spec requires a unit that is *set up* to be unengaged, and a model
    within engagement range cannot shoot at all -- deploying one on top of the
    enemy makes it a free kill that cannot fire back.

    Pinned at placement, which is the only state this function controls. `reset`
    afterwards resolves the opponent's whole turn before the agent's first
    observation, and they close back to base contact; that is the opponent's
    free turn, not this placement, and it is documented on the function.
    """
    # Arrange
    config = _config(1.0)
    config.base_radius = BASE_RADIUS
    engagement_range = 1.0
    required = 2.0 * BASE_RADIUS + engagement_range

    # Act / Assert
    for seed in range(20):
        battle = from_config(config)
        place_for_episode(
            battle, config, np.random.default_rng(seed), augment_start=True
        )
        mine = np.array(
            [np.asarray(m.location, dtype=float) for m in battle.player_models]
        )
        theirs = np.array(
            [np.asarray(m.location, dtype=float) for m in battle.opponent_models]
        )
        closest = float(np.linalg.norm(mine[:, None] - theirs[None], axis=-1).min())
        assert closest >= required - 1e-9, (
            f"seed {seed}: set up engaged at {closest:.4f}, need {required:.4f}"
        )
