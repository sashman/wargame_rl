"""Every config model rejects keys it does not define.

Pydantic's default is to *ignore* an unknown key. For a config that selects a
scenario, that turns a typo into a silent no-op: the run starts, trains, and
reports a number for an environment nobody asked for. This project has already
spent GPU-hours on arms that were not measuring what their config claimed, so
the failure mode is not hypothetical.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel, ValidationError

from wargame_rl.wargame.envs.reward.phase import (
    RewardCalculatorConfig,
    RewardPhaseConfig,
    SuccessCriteriaConfig,
)
from wargame_rl.wargame.envs.types.config import (
    MissionConfig,
    ModelConfig,
    ObjectiveConfig,
    OpponentPolicyConfig,
    RandomTerrainConfig,
    TerrainMapConfig,
    TerrainPieceConfig,
    WargameEnvConfig,
    WeaponProfile,
)

# Every config model a YAML file can reach. A new one added without a
# `model_config` entry fails here rather than silently swallowing typos.
CONFIG_MODELS: list[type[BaseModel]] = [
    WargameEnvConfig,
    ModelConfig,
    ObjectiveConfig,
    WeaponProfile,
    TerrainPieceConfig,
    TerrainMapConfig,
    RandomTerrainConfig,
    MissionConfig,
    OpponentPolicyConfig,
    RewardPhaseConfig,
    RewardCalculatorConfig,
    SuccessCriteriaConfig,
]


@pytest.mark.parametrize("model", CONFIG_MODELS, ids=lambda m: m.__name__)
def test_model_forbids_unknown_keys(model: type[BaseModel]) -> None:
    """Declared on the class, so it survives someone adding fields later."""
    assert model.model_config.get("extra") == "forbid"


def test_a_misspelled_field_is_an_error_not_a_default() -> None:
    """The case that motivated this.

    `objective_radius_sze=99` used to give you an objective of radius 1 and no
    warning at all -- a config that looks like it is testing something and is
    not. The error names the offending key so the fix is obvious.
    """
    with pytest.raises(ValidationError, match="objective_radius_sze"):
        WargameEnvConfig(objective_radius_sze=99)  # type: ignore[call-arg]


def test_a_misspelled_key_inside_a_nested_model_is_caught_too() -> None:
    """Nested models are where the interesting settings live.

    A weapon profile or a reward calculator is exactly the kind of thing an
    experiment arm edits, and a typo there is the most expensive place to be
    silently ignored.
    """
    with pytest.raises(ValidationError, match="rnge"):
        WeaponProfile(rnge=24)  # type: ignore[call-arg]

    with pytest.raises(ValidationError, match="wieght"):
        RewardCalculatorConfig(type="vp_gain", wieght=1.0)  # type: ignore[call-arg]


@pytest.mark.parametrize(
    "legacy, expected_width, expected_height",
    [
        ({"size": 30}, 30, 30),
        ({"width": 25, "height": 40}, 25, 40),
    ],
)
def test_legacy_board_size_aliases_still_load(
    legacy: dict[str, int], expected_width: int, expected_height: int
) -> None:
    """`size` / `width` / `height` are aliases, and must be consumed not copied.

    They are translated by a `mode="before"` validator. It used to leave the
    original key in the dict, which was harmless only because unknown keys were
    ignored -- the very behaviour this module removes.
    """
    config = WargameEnvConfig.model_validate(legacy)

    assert config.board_width == expected_width
    assert config.board_height == expected_height


def test_free_form_params_dicts_are_still_open() -> None:
    """Forbidding extras must not reach *inside* a params dict.

    Calculator and policy parameters are `dict[str, Any]` on purpose -- each
    registry entry defines its own -- so the strictness stops at the model
    boundary. Their contents are validated by the calculator that reads them.
    """
    calculator = RewardCalculatorConfig(
        type="objective_hold", params={"crowding_exponent": 1.0, "weight_hint": "any"}
    )

    assert calculator.params["crowding_exponent"] == 1.0
