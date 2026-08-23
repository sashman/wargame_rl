"""Baseline: advance only when the run lands the squad on the point this turn.

`squad_march_take_advance` prices the forfeited shooting and is **rejected** --
0 of 3 seed bases, -18.4 vp paired against plain `squad_march_take`. The
measurement that rejected it also refuted the obvious explanation: its advancing
moves end inside an enemy's weapon reach on **4.1%** of model-moves against
**22.4%** for its walking ones, so the run is not walking into danger. What it
buys is *time standing forward* -- episode exposure 0.2156 -> 0.2388, firepower
ratio 1.091 -> 1.004, `alive` 0.396 -> 0.349, `held` 2.57 -> 2.28.

So the gain has to be real, and "nearer" is not a gain. Control is a headcount
taken at the scoring moment, so a squad three inches closer scores exactly what
a squad nine inches closer scores. This policy spends the D6 only when it turns
a two-turn approach into a one-turn arrival, and keeps the no-shot clause on top
of it -- both halves of the trade, priced.

⚠ **NOT A BAR, and not to be developed further.** Per doctrine D-43, a move type
is a lever rather than an advantage, and a policy built to advance is built to be
wrong most of the time. This exists as a **measurement instrument** — it is what
priced the advance trade — and its number is a finding, not a target. Do not add
further advance-seeking policies; the right question is whether carrying the lever
costs the agent anything, which is answered by a `dark_action_slices` control.

⚠ Pre-registered before the number existed. **Accept** only on a paired
difference above zero with 3 of 3 seed bases positive and `held` no lower.
**Reject** on anything else -- including a positive difference with `held`
falling, which would be the same denial-not-offence trade the record is full of.
"""

from __future__ import annotations

from wargame_rl.wargame.envs.baseline.registry import register_baseline
from wargame_rl.wargame.envs.baseline.scripted_squad_march_take import (
    ScriptedSquadMarchTakePolicy,
)


class ScriptedSquadMarchTakeArrivePolicy(ScriptedSquadMarchTakePolicy):
    """`squad_march_take` that runs only to arrive, and only when it costs no shot."""

    advance_when_no_shot: bool = True
    advance_to_arrive: bool = True


register_baseline("squad_march_take_arrive", ScriptedSquadMarchTakeArrivePolicy)
