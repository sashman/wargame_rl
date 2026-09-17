"""The bridge check's verdict: identical, diverging under a recorded rule, or a
design fault.

`just measure-bridge` used to exit 2 on any differing field. On a rung where
both sides field several shooting units the per-model facade always parts from
the phase facade under its own recorded rule (targets judged after an earlier
unit's casualties) -- the golden shooting config does -- so that verdict named
every real scenario a design fault. The exit code now follows the record.
"""

from __future__ import annotations

from collections import Counter

from scripts.measure_bridge import bridge_verdict


def test_identical_fields_pass() -> None:
    verdict, code = bridge_verdict([], Counter())
    assert code == 0
    assert "identical" in verdict


def test_a_divergence_under_a_recorded_rule_stands_and_names_the_rule() -> None:
    rules = Counter({"shooting.targets_judged_after_casualties": 8})
    verdict, code = bridge_verdict(["player_vp", "win_rate"], rules)
    assert code == 0
    assert "shooting.targets_judged_after_casualties" in verdict
    assert "player_vp" in verdict
    assert "own facade" in verdict


def test_a_divergence_with_no_recorded_rule_is_a_design_fault() -> None:
    verdict, code = bridge_verdict(["player_vp"], Counter())
    assert code == 2
    assert "design fault" in verdict
