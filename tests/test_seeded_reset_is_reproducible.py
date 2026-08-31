"""A seeded episode must not depend on what the env did before it.

Regression, 2026-08-26. `_ensure_advance_rolls` memoises on
`(battle_round, active_player)` so a side's dice are rolled once per turn, and
`reset()` did not clear the memo. `TransformerNetwork.from_env` sizes the
network from an **unseeded** `env.reset()`, which left the key at `(1, <side>)`
-- so the first *seeded* episode of every measurement that builds a network
skipped round 1's advance and charge rolls for one side, while a scripted
measurement (which never calls `from_env`) was unaffected. A one-sided bias
between exactly the two columns of every comparison table, and the direction was
set by an unseeded player-side draw, so agent rows did not reproduce at all.
"""

from __future__ import annotations

import numpy as np

from wargame_rl.wargame.envs.types import WargameEnvConfig
from wargame_rl.wargame.envs.types.config import MeleeConfig, MeleeWeaponProfile
from wargame_rl.wargame.envs.types.config.battle import OpponentPolicyConfig
from wargame_rl.wargame.envs.types.config.entities import ModelConfig
from wargame_rl.wargame.envs.wargame import WargameEnv
from wargame_rl.wargame.model.common.factory import create_environment

SEED = 700_000


def _env() -> WargameEnv:
    config = WargameEnvConfig(
        number_of_wargame_models=4,
        number_of_opponent_models=4,
        number_of_objectives=1,
        opponent_policy=OpponentPolicyConfig(
            type="scripted_baseline", params={"baseline": "hold_deployment"}
        ),
        models=[
            ModelConfig(group_id=index // 2, melee_weapons=[MeleeWeaponProfile()])
            for index in range(4)
        ],
        opponent_models=[
            ModelConfig(group_id=index // 2, melee_weapons=[MeleeWeaponProfile()])
            for index in range(4)
        ],
        melee=MeleeConfig(enabled=True),
        skip_phases=[],
    )
    return create_environment(config)


def _rolls(env: WargameEnv) -> tuple[tuple[float, ...], tuple[float, ...]]:
    return (
        tuple(float(m.charge_roll) for m in env.wargame_models),
        tuple(float(m.charge_roll) for m in env.opponent_models),
    )


def test_a_throwaway_reset_does_not_change_the_next_SEEDED_episode() -> None:
    """`from_env` resets to size the network. That must cost the caller nothing."""
    # Arrange
    clean = _env()
    try:
        clean.reset(seed=SEED)
        expected = _rolls(clean)
    finally:
        clean.close()

    # Act — exactly what `TransformerNetwork.from_env` does before the caller
    # ever asks for a seeded episode.
    warmed = _env()
    try:
        warmed.reset()
        warmed.reset(seed=SEED)

        # Assert
        assert _rolls(warmed) == expected, (
            "an unseeded reset changed the dice of the seeded episode after it"
        )
    finally:
        warmed.close()


def test_the_roll_memo_is_cleared_by_reset() -> None:
    """The mechanism, pinned so a future refactor cannot restore it.

    ⚠ Only the ACTIVE side rolls at the start of its own turn, so the assertion
    is that the memo names the CURRENT episode's round -- never a key carried in
    from the previous one. Asserting both sides have dice at reset would pin a
    promise the design does not make.
    """
    # Arrange
    env = _env()
    try:
        env.reset(seed=SEED)
        assert env._rolled_for is not None, "round 1 never rolled"

        # Act — a second episode, whose round 1 must roll again.
        env.reset(seed=SEED + 1)

        # Assert
        assert env._rolled_for is not None
        assert env._rolled_for[0] == 1, (
            f"reset carried a stale roll memo: {env._rolled_for}"
        )
        rolled_side = env._rolled_for[1]
        rolled = env._models_for(rolled_side)
        assert all(m.charge_roll > 0.0 for m in rolled), (
            "the side whose turn it is entered round 1 with no charge dice"
        )
    finally:
        env.close()


def test_repeated_seeded_resets_are_identical() -> None:
    """The same seed, on the same env instance, twice."""
    # Arrange
    env = _env()
    try:
        env.reset(seed=SEED)
        first = _rolls(env)
        positions_first = np.array(
            [m.location for m in env.wargame_models], dtype=float
        )

        # Act
        env.reset(seed=SEED)

        # Assert
        assert _rolls(env) == first
        assert np.array_equal(
            np.array([m.location for m in env.wargame_models], dtype=float),
            positions_first,
        )
    finally:
        env.close()
