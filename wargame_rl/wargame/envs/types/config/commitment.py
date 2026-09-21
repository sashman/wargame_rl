"""The commitment layer's switches (#384, Stage 0).

A commitment is a unit pointing at a thing on the board -- an objective to
take and hold, an enemy unit to attack -- that persists across steps, that
every soldier of the acting seat can see, and that the reward is keyed to.
Who WRITES commitments is the switch here; the state, the token fields and
the readouts exist whether or not anyone writes.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

CommitmentAssignment = Literal["none", "greedy"]


class CommitmentConfig(BaseModel):
    """Who writes a unit's commitments on the per-model facade.

    `none` (the default): nobody. The commitment state stays empty unless a
    scripted seat plans the phase (a script's own assignment is written for
    the readouts), the token columns the layer adds read zero with their
    presence flag off, and every calculator pays exactly as it did before
    the layer existed -- byte-identical observations up to the new zeroed
    columns, bit-identical reward.

    `greedy`: the ENVIRONMENT writes the ground commitment at deployment --
    the scripted bar's own rule, cheapest objective first, each taking the
    nearest unassigned squad, spares to their nearest -- and holds it until
    the unit dies or its objective is held by us WITHOUT it (another unit
    holds it, so this one is redundant and is re-assigned to the nearest
    objective that is neither ours nor claimed). With a writer on, the
    travel term pays distance closed toward the COMMITTED objective instead
    of its own per-step choice and the staying term pays for ending inside
    it. This is the lever of arm CM1 (#387): the plan given as an
    observation, to read whether the trainer executes a plan it is handed.

    The phase facade ignores this block; the whole-army trainer has its own
    declaration line (`declare_objectives`), a different mechanism.
    """

    model_config = ConfigDict(extra="forbid")

    assignment: CommitmentAssignment = Field(
        default="none",
        description="Who writes the ground commitment: 'none' or 'greedy' (the "
        "environment, at deployment, by the scripted bar's rule; sticky).",
    )
    combat_set_size: int = Field(
        default=2,
        ge=1,
        description="The cap K on a unit's combat set (ANY mode). Stage 0 has no "
        "writer for the combat slot; the cap sizes the state and the masks.",
    )

    @property
    def enabled(self) -> bool:
        """True when a writer is configured, i.e. the layer is ON for reward
        and observation."""
        return self.assignment != "none"
