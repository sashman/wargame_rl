"""Scenario scalars set from the command line, without copying a config.

Two questions want the same scenario at several settings of one number — the
tabletop's five battle rounds, and a longer weapon range — and neither could be
asked, because every measurement script parsed argv positionally and handed the
YAML straight to the parser.

The failure this guards against is not an exception. It is a comparison that
silently measures a different game: a copied 13 KB config that missed a later
reward change, or an override token swallowed as a positional so the run scores
the *unmodified* scenario and prints a plausible table. Hence the two properties
pinned hardest here — no overrides is exactly a plain parse, and an override is
recognised wherever it appears in argv.
"""

from __future__ import annotations

import pytest
from pydantic_yaml import parse_yaml_raw_as

from scripts.scenario_overrides import (
    apply_overrides,
    describe,
    load_env_config,
    parse_overrides,
)
from wargame_rl.wargame.envs.types.config import TurnOrder, WargameEnvConfig

GOLDEN = "configs/golden/25v25_maps_two_mode.yaml"


class TestParsing:
    @pytest.mark.parametrize(
        "argv",
        [
            ["prog", "cfg.yaml", "100", "rounds=5"],
            ["prog", "rounds=5", "cfg.yaml", "100"],
            ["prog", "cfg.yaml", "rounds=5", "100"],
        ],
        ids=["trailing", "leading", "middle"],
    )
    def test_an_override_is_found_wherever_it_sits(self, argv: list[str]) -> None:
        """Position must not matter — that is the whole reason it is not one."""
        positional, overrides = parse_overrides(argv)
        assert positional == ["prog", "cfg.yaml", "100"]
        assert overrides == {"rounds": "5"}

    def test_argv_without_overrides_is_returned_unchanged(self) -> None:
        argv = ["prog", "cfg.yaml", "100", "", "700000"]
        positional, overrides = parse_overrides(argv)
        assert positional == argv
        assert overrides == {}

    def test_an_unknown_key_is_refused_rather_than_ignored(self) -> None:
        """Silently ignoring it would score the unmodified scenario."""
        with pytest.raises(SystemExit, match="unknown override 'round'"):
            parse_overrides(["prog", "cfg.yaml", "round=5"])

    def test_a_bad_turn_order_is_refused_at_the_boundary(self) -> None:
        config = parse_yaml_raw_as(WargameEnvConfig, open(GOLDEN).read())
        with pytest.raises(ValueError):
            apply_overrides(config, turn_order="whoever")


class TestApplying:
    def test_no_overrides_is_exactly_a_plain_parse(self) -> None:
        """The property every existing invocation of every script relies on."""
        plain = parse_yaml_raw_as(WargameEnvConfig, open(GOLDEN).read())
        assert load_env_config(GOLDEN).model_dump() == plain.model_dump()

    def test_rounds_changes_the_round_count_and_nothing_else(self) -> None:
        plain = parse_yaml_raw_as(WargameEnvConfig, open(GOLDEN).read()).model_dump()
        varied = load_env_config(GOLDEN, rounds="5").model_dump()
        differing = {key for key in plain if plain[key] != varied[key]}
        assert differing == {"number_of_battle_rounds"}
        assert varied["number_of_battle_rounds"] == 5

    def test_weapon_range_reaches_both_armies(self) -> None:
        """Reaching only `models` would set up an asymmetric firefight."""
        varied = load_env_config(GOLDEN, weapon_range="24")
        assert varied.models is not None and varied.opponent_models is not None
        for model in list(varied.models) + list(varied.opponent_models):
            assert model.weapons, (
                "fixture must arm its models for this to mean anything"
            )
            assert all(weapon.range == 24 for weapon in model.weapons)

    def test_turn_order_can_be_pinned(self) -> None:
        """measure-noise-floor books the random draw under 'the scenario'."""
        assert (
            load_env_config(GOLDEN, turn_order="player").turn_order is TurnOrder.player
        )

    def test_the_caller_s_config_is_not_mutated(self) -> None:
        """A script loads one config and scores several policies against it."""
        config = parse_yaml_raw_as(WargameEnvConfig, open(GOLDEN).read())
        before = config.model_dump()
        apply_overrides(config, rounds="5", weapon_range="24")
        assert config.model_dump() == before


def test_the_header_names_the_scenario_it_measured() -> None:
    """A table that does not say which scenario it ran gets compared to another."""
    assert describe({}) == ""
    assert (
        describe({"rounds": "5", "weapon_range": "24"})
        == "  [rounds=5, weapon_range=24]"
    )


class TestMissionOverrides:
    """The two mission knobs, which are the cheapest test of "is the mission the
    constraint" -- and the only overrides that are NOT self-contained scalars.
    """

    def test_cap_per_turn_reaches_the_mission_params(self) -> None:
        """The scoring cap must land where `DefaultVPCalculator` reads it."""
        config = load_env_config(GOLDEN, cap_per_turn="30")

        assert config.mission.params["cap_per_turn"] == 30

    def test_vp_per_objective_reaches_the_mission_params(self) -> None:
        """Per-objective value likewise."""
        config = load_env_config(GOLDEN, vp_per_objective="3")

        assert config.mission.params["vp_per_objective"] == 3

    def test_the_override_actually_changes_scored_vp(self) -> None:
        """SENSITIVITY. A knob that parses but does not change scoring is worse
        than no knob, because it reads as a measured null.
        """
        from wargame_rl.wargame.envs.mission.vp_calculator import DefaultVPCalculator

        base = load_env_config(GOLDEN)
        raised = load_env_config(GOLDEN, cap_per_turn="30")

        # The concrete class, not the ABC -- the cap is what this override moves,
        # and the ABC deliberately does not promise one.
        default_cap = DefaultVPCalculator(**base.mission.params).cap_per_turn
        raised_cap = DefaultVPCalculator(**raised.mission.params).cap_per_turn

        assert default_cap == 15
        assert raised_cap == 30

    def test_overriding_one_mission_knob_leaves_the_other_alone(self) -> None:
        """A deep copy per override, so scoring two settings cannot cross-talk."""
        raised = load_env_config(GOLDEN, cap_per_turn="30")
        plain = load_env_config(GOLDEN)

        assert "vp_per_objective" not in raised.mission.params
        assert plain.mission.params == {}
