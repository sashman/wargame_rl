"""Baseline: `squad_march_take`, but it runs when running is free.

Advance is a core rule, and until now no scripted policy could use it without
harming itself. The one heuristic that shipped -- "run while far, walk once
close" -- was measured at roughly **-78 vp to its own user** in the 2x2 on
`25v25_maps_advance_refereed`, because it never asked what the run cost. An
advance forfeits the unit's ENTIRE turn of shooting, and that price is paid
whether or not the unit had a shot to give.

This policy asks. It advances only when a normal move would have left no enemy
within weapon range of any member, so the fire it gives up is fire it never
had. Everything else -- the squad-centroid steering that keeps formation legal,
and `squad_march_take`'s allocation -- is inherited unchanged, so a paired
comparison against `squad_march_take` isolates the advance rule and nothing
else.

⚠ Its accept criterion is the paired difference against `squad_march_take`, not
its absolute score, and the record's own rule applies: quote a t AND a sign
count, because per-table differences on the map pool are heavy-tailed.
"""

from __future__ import annotations

from wargame_rl.wargame.envs.baseline.registry import register_baseline
from wargame_rl.wargame.envs.baseline.scripted_squad_march_take import (
    ScriptedSquadMarchTakePolicy,
)


class ScriptedSquadMarchTakeAdvancePolicy(ScriptedSquadMarchTakePolicy):
    """`squad_march_take` that advances only when it forfeits no shot."""

    advance_when_no_shot: bool = True


register_baseline("squad_march_take_advance", ScriptedSquadMarchTakeAdvancePolicy)
