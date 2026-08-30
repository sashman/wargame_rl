"""Baseline: the charging bar with the surplus-reallocation rule.

`squad_march_take_charge` plus `reallocate_surplus` — the fairness variant.
The play-time decode built on the same rule (`baseline/reallocation.py`) is
worth **+14.54 ± 3.81 vp to the agent** on 6 of 6 seeds, so any "beats the
bar" claim must be made against a bar allowed the same rule. This is that bar.
"""

from __future__ import annotations

from wargame_rl.wargame.envs.baseline.registry import register_baseline
from wargame_rl.wargame.envs.baseline.scripted_squad_march_charge import (
    ScriptedSquadMarchTakeChargePolicy,
)


class ScriptedSquadMarchTakeChargeReallocPolicy(ScriptedSquadMarchTakeChargePolicy):
    """The charging bar, redistributing one surplus squad per movement phase."""

    reallocate_surplus: bool = True


register_baseline(
    "squad_march_take_charge_realloc", ScriptedSquadMarchTakeChargeReallocPolicy
)
