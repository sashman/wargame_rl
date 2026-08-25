"""Whether this scenario fights in melee at all."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class MeleeConfig(BaseModel):
    """The master switch for the charge and fight phases.

    The rules are `docs/rules/11-charge-phase.md` and `12-fight-phase.md`. Both
    are rated **absent** in `docs/rules/implementation-status.md`: the phases
    exist in `BattlePhase` and the clock has always ticked them, but only
    `"stay"` is legal and both sit in `skip_phases` by default.

    ⚠ **Turning this on creates a game state this environment has never
    entered.** Not a rare one -- an unreached one. `back_off_to_unengaged` walks
    every mover on both seats back out of contact, so engagement is measured at
    **0.0000%** of model-pairs over 60,520 observations. What that number does
    NOT mean is that contact is unreachable: the closest pair sits
    **1.000008740"** from an engagement range of 1.0, i.e. 8.7 micro-inches
    outside contact. What a charge needs first is the back-off **exemption**.

    ⚠ **RETRACTED: it needs the distance too.** That minimum was read as a
    typical value and it is not one. Only **0.081%** of living pairs are within
    1.001"; the median living pair is **27.25"** apart and the median
    charge-ELIGIBLE unit is **5.99"** from its nearest enemy. See
    `docs/melee.md`.

    So melee is opt-in per scenario and defaults **off**, like
    `n_advance_speed_bins`. Off, it registers no slice, draws no dice, adds no
    observation column and leaves `skip_phases` alone -- every golden config and
    every observation golden stays bit-identical. Turning it on voids every
    baseline and every agent score on that config, because it is a different
    game rather than a tuned one.
    """

    model_config = ConfigDict(extra="forbid")

    charge_range: float = Field(
        default=12.0,
        gt=0,
        description=(
            "How close, in inches, an enemy unit must be for a unit to declare "
            "a charge against it (docs/rules/11-charge-phase.md). Read only "
            "when `enabled`."
        ),
    )
    consolidate_distance: float = Field(
        default=3.0,
        gt=0,
        description=(
            "How far, in inches, a unit that fought may consolidate afterwards "
            "(docs/rules/12-fight-phase.md). Only the Objective mode is "
            "implemented, and it is the LAST of three ordered compulsory modes, "
            "so it fires only for a unit that ends the fight engaged with "
            "nobody and with no enemy within this distance. Read only when "
            "`enabled`."
        ),
    )
    enabled: bool = Field(
        default=False,
        description=(
            "Fight in melee: step the charge phase, resolve fights, and let a "
            "move end inside an enemy's engagement range. Default False is an "
            "exact no-op."
        ),
    )
    shield_engaged_targets: bool = Field(
        default=True,
        description=(
            "A unit locked in melee cannot be shot at, per "
            "`docs/rules/04-making-attacks.md`. ⚠ This exists ONLY so it can be "
            "ABLATED: a charging script measured +62.50 with it and -4.00 "
            "without, which is the evidence that the charge's value is the "
            "shooting shield and not the blade. Before this field the flag was "
            "hardwired to `enabled` at both mask call sites, so the ablation "
            "the melee pre-registration REQUIRES of every vp number was not "
            "expressible -- a requirement nothing could satisfy. Read only when "
            "`enabled`; turning it off is a deliberately incorrect game."
        ),
    )
