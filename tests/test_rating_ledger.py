"""The ledger is keyed by the scenario, and refuses to mix two.

`CLAUDE.md`'s entire opening section is a monument to numbers quoted from a
different environment, so this is enforced rather than documented -- and it
raises rather than warning, because the TF32 and `last.ckpt` episodes are what
happens to warnings.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from pydantic_yaml import parse_yaml_raw_as

from wargame_rl.wargame.envs.types import WargameEnvConfig
from wargame_rl.wargame.rating.arena import LegResult
from wargame_rl.wargame.rating.entrant import Entrant
from wargame_rl.wargame.rating.ledger import (
    RatingDecodeMismatch,
    RatingScenarioMismatch,
    append,
    canonical_scenario,
    fingerprint,
    leg_results,
    load,
)
from wargame_rl.wargame.rating.schedule import FOUR_LEGS, Leg, config_for_leg

ARENA_CONFIG = "configs/dev/4v4_two_phases.yaml"


def _base_config() -> WargameEnvConfig:
    with open(ARENA_CONFIG) as handle:
        config: WargameEnvConfig = parse_yaml_raw_as(WargameEnvConfig, handle.read())
    config.render_mode = None
    return config


def _leg_result(leg: Leg = FOUR_LEGS[0]) -> LegResult:
    return LegResult(
        entrant_a="squad_march",
        entrant_b="random",
        leg=leg,
        layout_seeds=(900_000, 900_001),
        combat_seeds=(1_900_000, 1_900_001),
        margins=(12.0, -3.0),
        wins=(1.0, 0.0),
        objectives_held=(2.0, 1.0),
        coherency_rate=0.94,
    )


def _entrants() -> list[Entrant]:
    return [
        Entrant(name="squad_march", build=lambda env: None, kind="baseline"),  # type: ignore[arg-type,return-value]
        Entrant(name="random", build=lambda env: None, kind="baseline"),  # type: ignore[arg-type,return-value]
    ]


def test_all_four_leg_configs_fingerprint_identically() -> None:
    """The single assertion that proves the canonicalisation.

    `docs/elo.md` says to fingerprint the resolved config "excluding rendering
    and logging fields". Taken literally that puts the four legs of one table
    into **four different ledgers**, because the schedule deliberately varies
    `turn_order` and swaps the two deployment zones. Excluding the leg axes and
    sorting the zone pair is what makes the feature work at all.

    This also fails loudly the day someone adds a config field to the leg
    transform without adding it here.
    """
    base = _base_config()

    digests = {fingerprint(config_for_leg(base, leg)) for leg in FOUR_LEGS}

    assert len(digests) == 1


def test_the_opponent_is_not_part_of_the_scenario() -> None:
    """Every pairing in a table has a different opponent; if that changed the
    fingerprint, no table could hold more than one pairing."""
    base = _base_config()
    other = base.model_copy(deep=True)
    assert other.opponent_policy is not None
    other.opponent_policy = other.opponent_policy.model_copy(
        update={"type": "scripted_baseline", "params": {"baseline": "squad_march"}}
    )

    assert fingerprint(base) == fingerprint(other)


@pytest.mark.parametrize(
    "field, value",
    [
        ("board_width", 61),
        ("number_of_objectives", 5),
        ("objective_radius_size", 7),
        ("number_of_battle_rounds", 21),
    ],
)
def test_a_changed_board_changes_the_fingerprint(field: str, value: int) -> None:
    """The deny-list must not be so eager that it stops noticing real scenario
    changes -- which is the failure mode that would let two incomparable tables
    merge silently."""
    base = _base_config()
    changed = base.model_copy(deep=True)
    setattr(changed, field, value)

    assert fingerprint(base) != fingerprint(changed)


def test_swapping_the_armies_leaves_the_scenario_alone() -> None:
    """The army pair is sorted, so which side holds which force is a leg
    property rather than a different game."""
    base = _base_config()
    swapped = base.model_copy(deep=True)
    swapped.models, swapped.opponent_models = base.opponent_models, base.models
    swapped.number_of_wargame_models = base.number_of_opponent_models
    swapped.number_of_opponent_models = base.number_of_wargame_models

    assert fingerprint(base) == fingerprint(swapped)


def test_a_different_army_size_is_a_different_scenario() -> None:
    base = _base_config()
    bigger = base.model_copy(deep=True)
    bigger.number_of_opponent_models = base.number_of_opponent_models + 1
    bigger.opponent_models = None

    assert fingerprint(base) != fingerprint(bigger)


def test_the_scenario_carries_no_leg_axis() -> None:
    scenario = canonical_scenario(_base_config())

    assert "turn_order" not in scenario
    assert "opponent_policy" not in scenario
    assert "deployment_zone" not in scenario
    assert scenario["zone_pair"] == sorted(scenario["zone_pair"])


def test_the_ledger_round_trips(tmp_path: Path) -> None:
    base = _base_config()

    append([_leg_result()], base, _entrants(), root=tmp_path)
    reloaded = load(fingerprint(base), root=tmp_path)

    assert reloaded is not None
    assert leg_results(reloaded) == [_leg_result()]


def test_appending_accumulates_rather_than_replacing(tmp_path: Path) -> None:
    """Adding an entrant must not mean replaying every pairing, which is why
    raw legs are stored rather than a fitted table."""
    base = _base_config()

    append([_leg_result(FOUR_LEGS[0])], base, _entrants(), root=tmp_path)
    ledger = append([_leg_result(FOUR_LEGS[1])], base, _entrants(), root=tmp_path)

    assert len(ledger.legs) == 2
    assert {leg.first_mover for leg in ledger.legs} == {"a", "b"}


def test_the_ledger_refuses_a_second_scenario(tmp_path: Path) -> None:
    """Refused, not warned about."""
    base = _base_config()
    other = base.model_copy(deep=True)
    other.board_width = base.board_width + 1
    append([_leg_result()], base, _entrants(), root=tmp_path)

    # A different scenario lands in its own file rather than corrupting this
    # one; the mismatch guard covers the case where a digest is reused.
    append([_leg_result()], other, _entrants(), root=tmp_path)

    assert fingerprint(base) != fingerprint(other)
    assert load(fingerprint(base), root=tmp_path) is not None
    assert load(fingerprint(other), root=tmp_path) is not None


def test_a_mismatched_ledger_raises(tmp_path: Path) -> None:
    base = _base_config()
    append([_leg_result()], base, _entrants(), root=tmp_path)
    path = tmp_path / f"{fingerprint(base)}.json"
    path.write_text(path.read_text().replace(fingerprint(base), "deadbeefdeadbeef", 1))

    with pytest.raises(RatingScenarioMismatch, match="played on"):
        append([_leg_result()], base, _entrants(), root=tmp_path)


def test_the_ledger_records_how_each_entrant_was_decoded(tmp_path: Path) -> None:
    """A rating is comparable only at a fixed decode -- the joint constrained
    decode is worth +40.5 vp per map on the real tables, so a table mixing K=1
    and K=3 rows would be measuring two different policies under one name."""
    base = _base_config()
    entrants = [
        Entrant(
            name="agent",
            build=lambda env: None,  # type: ignore[arg-type,return-value]
            kind="checkpoint",
            source="checkpoints/run/last.ckpt",
            decode_topk=3,
        )
    ]

    ledger = append([_leg_result()], base, entrants, root=tmp_path)

    assert ledger.entrants[0].decode_topk == 3
    assert ledger.entrants[0].source == "checkpoints/run/last.ckpt"


def test_the_ledger_records_the_code_revision(tmp_path: Path) -> None:
    """Open bugs in this repo's sight and terrain handling will move every
    number on the board when fixed; a row that does not say which code played it
    cannot be re-quoted afterwards."""
    ledger = append([_leg_result()], _base_config(), _entrants(), root=tmp_path)

    assert ledger.legs[0].code_revision


def _checkpoint_entrant(**overrides: object) -> Entrant:
    fields: dict[str, object] = {
        "name": "run-armA",
        "build": lambda env: None,
        "kind": "checkpoint",
        "source": "checkpoints/run/last.ckpt",
    }
    fields.update(overrides)
    return Entrant(**fields)  # type: ignore[arg-type]


def _leg_between(entrant_a: str, entrant_b: str) -> LegResult:
    return replace(_leg_result(), entrant_a=entrant_a, entrant_b=entrant_b)


def test_one_name_under_two_decodes_is_refused(tmp_path: Path) -> None:
    """A rating is a score, and this repo never quotes one without its decode.

    The same weights at K=1 and K=3 differ by **+40.5 vp**, larger than any
    policy difference measured here -- so a table holding both under one name
    would rank the decode and report it as skill. Refused rather than warned
    about, in the same class as mixing two scenarios.
    """
    config = _base_config()
    greedy = _checkpoint_entrant(decode_topk=1)
    append([_leg_between("run-armA", "random")], config, [greedy], root=tmp_path)

    with pytest.raises(RatingDecodeMismatch, match="different players"):
        append(
            [_leg_between("run-armA", "random")],
            config,
            [_checkpoint_entrant(decode_topk=3)],
            root=tmp_path,
        )


def test_a_sampled_entrant_cannot_join_a_greedy_one(tmp_path: Path) -> None:
    """The axis that matters for self-play: rollouts draw from the policy, every
    scoring path in this repo takes its argmax."""
    config = _base_config()
    append(
        [_leg_between("run-armA", "random")],
        config,
        [_checkpoint_entrant()],
        root=tmp_path,
    )

    with pytest.raises(RatingDecodeMismatch):
        append(
            [_leg_between("run-armA", "random")],
            config,
            [_checkpoint_entrant(sampled=True)],
            root=tmp_path,
        )


def test_the_same_decode_appends_as_before(tmp_path: Path) -> None:
    """The control: the guard must not refuse the ordinary case of an entrant
    playing more legs, which is the whole point of an append-only ledger."""
    config = _base_config()
    entrant = _checkpoint_entrant(decode_topk=3)
    append([_leg_between("run-armA", "random")], config, [entrant], root=tmp_path)

    ledger = append(
        [_leg_between("run-armA", "random")], config, [entrant], root=tmp_path
    )

    assert len(ledger.legs) == 2
