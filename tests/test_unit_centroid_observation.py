"""`observe_unit_centroid` puts the direction to a unit's centroid on its models.

`observe_coherency` already carries *magnitudes* — how stretched a unit is, and
whether it is in one piece. Neither says which way to move: the spread ratio
reads identically from either end of a strung-out unit. This input is the
direction, and it is the quantity the scripted demonstrators actually steer on
(`ScriptedSquadMarchPolicy` moves every model of a unit along one shared
centroid vector, which is why their formation holds by construction).

It exists because a behaviour clone at 98.6% action match reproduces that only
to 0.665 unit coherency against the demonstrator's 0.884.
"""

from __future__ import annotations

import numpy as np
import torch

from wargame_rl.wargame.envs.types import WargameEnvConfig
from wargame_rl.wargame.envs.wargame import WargameEnv
from wargame_rl.wargame.model.common.observation import observation_to_tensor


def _config(**overrides: object) -> WargameEnvConfig:
    base: dict[str, object] = {
        "render_mode": None,
        "number_of_wargame_models": 4,
        "number_of_objectives": 2,
        "number_of_battle_rounds": 3,
        "max_groups": 2,
    }
    base.update(overrides)
    return WargameEnvConfig(**base)  # type: ignore[arg-type]


def test_off_by_default_and_byte_identical() -> None:
    """The flag defaults off and changes nothing when it is.

    The guarantee every observation flag in this env carries: a config that does
    not set it produces the same tensor as one that cannot.
    """
    assert _config().observe_unit_centroid is False

    env = WargameEnv(config=_config())
    observation, _ = env.reset(seed=7)
    tensors = observation_to_tensor(observation)
    env.close()

    assert all(m.unit_offset is None for m in observation.wargame_models)
    reference = [t.clone() for t in tensors[:5]]

    env2 = WargameEnv(config=_config())
    observation2, _ = env2.reset(seed=7)
    tensors2 = observation_to_tensor(observation2)
    env2.close()
    for a, b in zip(reference, tensors2[:5], strict=True):
        assert torch.equal(a.cpu(), b.cpu())


def test_it_points_from_the_model_toward_its_units_centre() -> None:
    """The vector points *inward*, and a model at the centre reports zero."""
    env = WargameEnv(config=_config(observe_unit_centroid=True))
    observation, _ = env.reset(seed=11)

    models = observation.wargame_models
    assert all(m.unit_offset is not None for m in models)

    for group_id in {m.group_id for m in models}:
        members = [m for m in models if m.group_id == group_id]
        positions = np.array([m.location for m in members], dtype=float)
        centroid = positions.mean(axis=0)
        for member, position in zip(members, positions, strict=True):
            offset = member.unit_offset
            assert offset is not None
            expected_direction = centroid - position
            # Same sign per axis: the offset is that vector, scaled and clipped.
            for axis in range(2):
                if abs(expected_direction[axis]) > 1e-6:
                    assert np.sign(offset[axis]) == np.sign(expected_direction[axis])
    env.close()


def test_the_widened_token_reaches_the_tensor() -> None:
    """Turning it on widens the per-model block by exactly two columns.

    Asserted against the flag-off tensor rather than a hardcoded width, so this
    keeps holding as other optional inputs come and go.
    """
    env_off = WargameEnv(config=_config())
    off, _ = env_off.reset(seed=3)
    width_off = observation_to_tensor(off)[2].shape[-1]
    env_off.close()

    env_on = WargameEnv(config=_config(observe_unit_centroid=True))
    on, _ = env_on.reset(seed=3)
    width_on = observation_to_tensor(on)[2].shape[-1]
    env_on.close()

    assert width_on == width_off + 2


def test_a_casualty_reports_nothing_to_correct() -> None:
    """Dead models report (0, 0), like a lone or already-centred one.

    The corpse trap this repo has hit before: `phase_manager` and the coherency
    predicate both iterate the living, so a casualty must never read as a model
    that still needs to close up.
    """
    env = WargameEnv(config=_config(observe_unit_centroid=True))
    env.reset(seed=5)
    env.wargame_models[0].stats["current_wounds"] = 0
    observation = env.observation

    offset = observation.wargame_models[0].unit_offset
    assert offset is not None
    assert np.array_equal(offset, np.zeros(2, dtype=np.float32))
    env.close()


def test_both_sides_carry_it_and_stay_the_same_width() -> None:
    """The opponent block must widen with the player's, or the tensor is ragged.

    Regression: the flag was wired into the player's `_models_to_obs` call and
    not the opponent's, so player rows came out 38 wide against the opponent's
    36 and the feature-dim assertion fired mid-collection. Both blocks share one
    `feature_dim`, computed from a single probe, so a per-side difference cannot
    be right.
    """
    env = WargameEnv(
        config=_config(
            observe_unit_centroid=True,
            number_of_opponent_models=4,
            opponent_policy={"type": "scripted_advance_to_objective"},
        )
    )
    observation, _ = env.reset(seed=13)

    assert observation.opponent_models, "need opponents for this to mean anything"
    assert all(m.unit_offset is not None for m in observation.opponent_models)

    tensors = observation_to_tensor(observation)
    player_width = tensors[2].shape[-1]
    opponent_width = tensors[3].shape[-1]
    assert player_width == opponent_width
    env.close()
