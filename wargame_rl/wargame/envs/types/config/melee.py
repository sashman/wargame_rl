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
    NOT mean is that contact is hard to reach: the minimum edge-to-edge gap is
    **1.000008740"** against an engagement range of 1.0, i.e. the army parks
    8.7 micro-inches outside contact and has done all along. A charge does not
    need to cross a gap; it needs the back-off exemption.

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
