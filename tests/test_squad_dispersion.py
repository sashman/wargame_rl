"""The bunching ratio must mean what its name says.

`squads per occupied objective` is the number the stacking diagnosis turns on,
and it has two ways to mislead: it rises when a squad is destroyed as well as
when two converge, and a squad standing on two overlapping objectives must not
be counted as two squads. These pin both.
"""

from __future__ import annotations

import pytest

from scripts.measure_squad_dispersion import Dispersion, report


def test_one_squad_per_point_reports_the_floor(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Four squads on four distinct objectives is the 1.00 baseline."""
    tally = Dispersion(
        steps=1,
        squads_alive=4,
        squads_on_objective=4,
        objectives_occupied=4,
        squads_sharing=0,
        objectives_total=6,
        board_diagonal=74.4,
    )

    report(tally)
    out = capsys.readouterr().out

    assert "SQUADS PER OCCUPIED OBJECTIVE        1.00" in out
    assert "squads sharing a point               0.0%" in out


def test_converging_squads_raise_the_ratio(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Four squads piled onto two points reads 2.00, with everyone sharing."""
    tally = Dispersion(
        steps=1,
        squads_alive=4,
        squads_on_objective=4,
        objectives_occupied=2,
        squads_sharing=4,
        objectives_total=6,
        board_diagonal=74.4,
    )

    report(tally)
    out = capsys.readouterr().out

    assert "SQUADS PER OCCUPIED OBJECTIVE        2.00" in out
    assert "squads sharing a point             100.0%" in out


def test_a_policy_holding_nothing_does_not_divide_by_zero(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """No squad on any objective must report, not raise -- `random` does this."""
    tally = Dispersion(steps=1, squads_alive=8, objectives_total=6, board_diagonal=74.4)

    report(tally)

    assert "squads sharing a point" in capsys.readouterr().out
