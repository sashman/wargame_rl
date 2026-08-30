"""Baseline: `squad_march_take` that charges when the charge actually lands.

⚠ **This exists because a bar that cannot use a core rule is not a bar.** Until
`BaselinePolicy.select_charge` existed, every scripted baseline and every
scripted opponent returned STAY in the charge phase, so an agent trained with
melee on would have been scored against a policy *physically incapable* of the
mechanic under test. That is verbatim the failure this project already paid for
on Advance, where the bar could not advance and the arm measured the bar.

## What it does

Rigid squad translation onto the nearest enemy unit, sized to put the closest
member a hair inside engagement range, declared only when a legal rung covers
that distance. The `charge_legality` mask remains the authority on eligibility
and on the 2D6 cap; this policy only chooses a direction and a length.

## ⚠ What its number is NOT

**It is not "the value of melee".** Six independently hand-rolled charging
scripts produced +6.5, +48.0, +52.0, +59.2, +82.9 and +88.8 vp for nominally the
same measurement — a **14x spread** — because each measured its own heuristic
rather than the mechanic. Any figure from this policy must be reported with

* the **ablation** — the same policy with `exclude_engaged_targets` off, which
  took a charging script from **+62.50 ± 14.74** to **−4.00 ± 17.39** and is the
  evidence that the charge's whole value is the shooting shield rather than the
  blade; and
* the **2x2** — both sides walking, both charging, and each alone. A symmetric
  change measured with both sides changed at once reads as zero: that error
  published "+15.5 to the bar" for Advance when the true figure was two
  self-inflicted wounds cancelling.

## Pre-registration

Written before any number exists, and the primary readouts are **mechanism
counts**, not vp — a lethality-negligible mechanic cannot move vp by more than
the estimator's own noise, so a vp gate is unpowered by construction.

**This policy is fit for use as a bar** iff, on the shipped melee config:

1. charges **declared** > 0 and charges **standing** > 0 per episode;
2. the standing fraction is above 0.5 — below that the rule is mostly proposing
   charges the referee reverts, and it is measuring its own geometry;
3. `coherent` is no lower than plain `squad_march_take` on the same layouts,
   since a rigid translation is supposed to preserve formation exactly;
4. with melee **off** it is byte-identical to `squad_march_take`.

Failing any of those it is an instrument that does not work, whatever its vp.
"""

from __future__ import annotations

from wargame_rl.wargame.envs.baseline.registry import register_baseline
from wargame_rl.wargame.envs.baseline.scripted_squad_march_take import (
    ScriptedSquadMarchTakePolicy,
)


class ScriptedSquadMarchTakeChargePolicy(ScriptedSquadMarchTakePolicy):
    """`squad_march_take` that charges whenever the charge would stand."""

    charge_when_it_lands: bool = True


register_baseline("squad_march_take_charge", ScriptedSquadMarchTakeChargePolicy)
