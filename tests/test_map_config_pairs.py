"""The map configs that are meant to differ only in the opponent, do.

Two published claims rest on this and neither is checkable by reading, because
each of these files is several hundred lines of which all but a handful are
comment:

**The four-opponent table is a single-variable comparison.** Every row is the
same agent on the same nine tables under the same referee, facing a different
opponent — so a row here and a row there differ by the opponent and nothing
else. If any other field drifts apart, the table silently becomes a comparison
of two things at once, and `CLAUDE.md`'s standing warning that "absolute score
measures the OPPONENT, not the agent" stops being the only caveat on it.

**`25v25_maps_coherency` is not a second scenario.** Stripped of comments it is
`25v25_maps_two_mode` with a weaker opponent, which is why nothing trains on it
and its agent column comes from scoring the `two_mode` lineage. The moment that
stops being true, that decision needs revisiting rather than inheriting.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic_yaml import parse_yaml_raw_as

from wargame_rl.wargame.envs.types.config import WargameEnvConfig

GOLDEN = Path("configs") / "golden"
EVALUATION = Path("configs") / "evaluation"

# The refereed evaluation family: one file per opponent, identical otherwise.
# `25v25_maps_take_centroid_refereed.yaml` and `..._repair.yaml` are deliberately
# absent — the first widens the observation and the second changes the referee,
# so neither belongs in a comparison that varies only the opponent.
OPPONENT_CONFIGS = [
    EVALUATION / "25v25_maps_take_opponent_refereed.yaml",
    EVALUATION / "25v25_maps_vs_squad_march_shoot.yaml",
    EVALUATION / "25v25_maps_vs_squad_march_deny.yaml",
    EVALUATION / "25v25_maps_vs_contest_and_spread.yaml",
    EVALUATION / "25v25_maps_vs_advance_and_shoot.yaml",
]


def _load(path: Path) -> dict[str, Any]:
    config = parse_yaml_raw_as(WargameEnvConfig, path.read_text())
    dumped: dict[str, Any] = config.model_dump()
    return dumped


def _differing_fields(left: Path, right: Path) -> set[str]:
    """Top-level config fields whose parsed values differ between two files.

    Parsed rather than textual: comments carry most of the bytes in these files
    and none of the meaning, and a default written out explicitly in one file is
    not a difference.
    """
    first, second = _load(left), _load(right)
    return {key for key in first if first[key] != second[key]}


@pytest.mark.parametrize("other", OPPONENT_CONFIGS[1:], ids=lambda p: p.stem)
def test_the_opponent_configs_differ_only_in_the_opponent(other: Path) -> None:
    assert _differing_fields(OPPONENT_CONFIGS[0], other) == {"opponent_policy"}


def test_the_two_golden_map_configs_differ_only_in_the_opponent() -> None:
    differing = _differing_fields(
        GOLDEN / "25v25_maps_two_mode.yaml", GOLDEN / "25v25_maps_coherency.yaml"
    )
    # `config_name` is the run label rather than a scenario field, so it is
    # expected to differ and says nothing about whether the game does.
    assert differing == {"config_name", "opponent_policy"}


def test_every_opponent_config_is_refereed() -> None:
    """The referee is what makes these scores comparable to the published table.

    Turning it off flatters the scripts by ~16 vp, because it taxes each policy
    by how often it breaks coherency — so an unrefereed row belongs to a
    different table, not a slightly noisier version of this one.
    """
    for path in OPPONENT_CONFIGS:
        config = parse_yaml_raw_as(WargameEnvConfig, path.read_text())
        assert config.coherency.enforce_move == "revert_unit", path
        assert config.coherency.attrition is True, path
