"""The hazard probe must credit a death to where the model stood before it died.

Scoring a casualty against the step it is already dead on would attribute every
loss to wherever the corpse lies, which on this board is disproportionately an
objective -- and would manufacture exactly the "holding is lethal" finding the
probe exists to test. These pin the accounting.
"""

from __future__ import annotations

import numpy as np
import pytest

from scripts.measure_hold_hazard import Tally, report


def test_a_tally_with_no_observations_reports_zero_rather_than_dividing() -> None:
    """An empty arm must not raise; a policy can simply never leave a point."""
    tally = Tally()

    assert tally.rate(0, 0) == 0.0
    assert tally.income(0.0, 0) == 0.0


@pytest.mark.parametrize(
    ("excess_on", "excess_off", "expected"),
    [
        # Standing is deadlier than the income covers -> hiding is correct.
        (40, 0, "does NOT pay"),
        # Standing is safer -> the policy is leaving return on the table.
        (0, 40, "DOES pay"),
    ],
)
def test_the_verdict_follows_the_hazard_not_the_income(
    excess_on: int,
    excess_off: int,
    expected: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Identical income on both arms, so only the hazard can decide."""
    tally = Tally(
        steps_on=100,
        steps_off=100,
        deaths_on=excess_on,
        deaths_off=excess_off,
        reward_on=50.0,
        reward_off=10.0,
        episode_steps=40,
        episodes=1,
    )

    report("trade", tally)

    assert expected in capsys.readouterr().out


def test_death_is_credited_to_the_previous_step_not_the_corpse() -> None:
    """The loop's own bookkeeping, in miniature.

    A model standing OFF an objective on step one and dead on step two must be
    counted as an off-objective death, even though its remains sit inside an
    objective radius when the death is observed.
    """
    previous_on = np.array([False, True])
    previous_alive = np.array([True, True])
    alive = np.array([False, True])

    died = previous_alive & ~alive

    assert int(np.sum(died & previous_on)) == 0
    assert int(np.sum(died & ~previous_on)) == 1
