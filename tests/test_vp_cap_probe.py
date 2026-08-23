"""The cap probe must read the mission's numbers, not assume them.

`measure_vp_cap` reports percentages of VP discarded by `cap_per_turn`. If it
hardcoded 15 and 5 it would still print for a config with a different mission --
every figure wrong, nothing to show it. These pin the reading.
"""

from __future__ import annotations

import pytest

from scripts.measure_vp_cap import cap_and_rate, report
from scripts.scenario_overrides import load_env_config
from wargame_rl.wargame.envs.mission.vp_calculator import DefaultVPCalculator

GOLDEN = "configs/golden/25v25_maps_two_mode.yaml"


def test_cap_and_rate_defaults_match_the_calculator() -> None:
    """A config that names no params inherits the calculator's own defaults."""
    reference = DefaultVPCalculator()

    cap, per_objective = cap_and_rate(load_env_config(GOLDEN))

    assert (cap, per_objective) == (reference.cap_per_turn, reference.vp_per_objective)


def test_cap_and_rate_follows_the_config_not_the_default() -> None:
    """An explicit mission param wins, or every percentage printed is wrong."""
    config = load_env_config(GOLDEN)
    config.mission.params = {"cap_per_turn": 30, "vp_per_objective": 6}

    assert cap_and_rate(config) == (30, 6)


@pytest.mark.parametrize(
    ("counts", "cap", "per_objective", "expected_discarded"),
    [
        # Never above three: the cap takes nothing.
        ({2: 10}, 15, 5, "0.0%"),
        # Always at five, so two objectives in five are unpaid: 10 of 25.
        ({5: 10}, 15, 5, "40.0%"),
        # A raised cap pays what the default would have discarded.
        ({5: 10}, 30, 5, "0.0%"),
    ],
)
def test_report_prices_the_surplus_at_zero(
    counts: dict[int, int],
    cap: int,
    per_objective: int,
    expected_discarded: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Objectives beyond the cap must contribute nothing to earned VP."""
    from collections import Counter

    report("PLAYER", Counter(counts), cap, per_objective)

    assert f"discards {expected_discarded}" in capsys.readouterr().out
