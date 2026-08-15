"""How the unit coherency rule is enforced, if at all."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

# Mirrors `domain.rules_constants.COHERENCY_{NEAREST,FURTHEST}_IN`, duplicated
# rather than imported because config may not depend on the domain layer -- the
# same trade `INFANTRY_BASE_RADIUS_IN` in `env.py` already makes.
# `tests/test_coherent_deployment.py` pins the two together so neither drifts.
COHERENCY_NEAREST_IN = 2.0
COHERENCY_FURTHEST_IN = 9.0


class CoherencyConfig(BaseModel):
    """The coherency rule's distances, and which of its consequences bite.

    The rule is `docs/rules/03-moving.md` § Coherency. This environment has
    never enforced it -- `docs/rules/implementation-status.md` rates it
    **divergent**, approximated by one `group_max_distance` and priced by the
    `group_cohesion` reward. Measured with `just measure-coherency`, the whole
    ladder sits in breach: on `25v25_shooting_opponent.yaml` the scripted bar is
    fully coherent on **3.3%** of steps and the trained agent on **2.2%**, and
    deployment itself is coherent in **0** episodes out of 20.

    So every consequence here defaults to **off**. Turning one on is a scenario
    change that voids every baseline measured on that config, exactly as
    continuous space did -- the defaults exist so that adopting the rule is a
    deliberate, separately measured step rather than something a config picks up
    by upgrading.

    The two distances are always available, because reporting coherency costs
    nothing and a metric that only exists once enforcement is on cannot tell you
    what enforcement would cost.
    """

    model_config = ConfigDict(extra="forbid")

    nearest_distance: float = Field(
        gt=0,
        default=COHERENCY_NEAREST_IN,
        description=(
            "The chain distance, in inches: every model must be within this of "
            "at least one other model in its unit. The rules say 2. Measured "
            "base to base, like engagement range, so at the default 32mm base "
            "the centres may be up to 3.26in apart -- a band twice the width of "
            "the smallest move the action space can make."
        ),
    )
    furthest_distance: float = Field(
        gt=0,
        default=COHERENCY_FURTHEST_IN,
        description=(
            "The spread distance, in inches: every model must be within this of "
            "*every* other model in its unit. The rules say 9. This is the half "
            "with no existing approximation at all -- `group_cohesion` prices "
            "only the nearest neighbour, so a unit strung across the board pays "
            "nothing as long as each model has a partner."
        ),
    )
    enforce_at_deployment: bool = Field(
        default=False,
        description=(
            "Deploy each unit in coherency, as `03-moving.md` § Setting up "
            "requires. Off by default because it moves the baselines on its own: "
            "spawning squads together raised the shooting bar +58.9 -> +63.3 "
            "vp_margin and flipped the top two baselines "
            "(`configs/experiments/25v25_coherent_spawn.yaml`). This is the "
            "cheapest of the three and the prerequisite for the other two -- a "
            "force that starts in breach can only ever be caught up with."
        ),
    )
    enforce_move: str = Field(
        default="off",
        pattern="^(off|revert_unit|revert_model)$",
        description=(
            "Enforce coherency at the end of a move, the rules' *primary* "
            "consequence (`03-moving.md` § Making a move): a move that would end "
            "a unit out of coherency cannot be made, and its models return to "
            "where they started. `revert_unit` is the spec -- one model out of "
            "place cancels its whole unit's move. `revert_model` returns only "
            "the models outside their unit's coherent body, which diverges from "
            "the rule but replaces a 5-model cliff with a gradient.\n\n"
            "⚠ THIS IS A REFEREE, NOT A TEACHER — DO NOT TRAIN UNDER IT. "
            "Measured over ten runs on the real tables: enforcement guarantees a "
            "legal board, and a policy trained under it learns *less* about "
            "formation than one never exposed to it. With the referee switched "
            "off so the numbers describe the policy rather than the wrapper, "
            "training under enforcement lands at 0.569 units coherent against "
            "0.756-0.886 for `objective_hold.require_coherent` alone, and loses "
            "on unseen ground too -- 70.3 vp_margin on nine held-out tables "
            "against gate-only's 81.5. The cause is structural: every reverted "
            "action produces the identical outcome, so they share an advantage "
            "and the policy gradient inside that whole set is exactly zero. "
            "Train with the reward gate and no enforcement, then switch this on "
            "for play, where it costs nothing and makes the board legal.\n\n"
            "Off by default; every setting is a dynamics change that voids the "
            "baselines on that config."
        ),
    )
    attrition: bool = Field(
        default=False,
        description=(
            "Apply the rules' own enforcement: at the end of a player's turn, a "
            "unit out of coherency loses models one at a time until coherency is "
            "restored (`03-moving.md` § Regaining coherency). Destroyed, but "
            "triggering nothing that fires on a model being destroyed. This is "
            "the mechanism that makes the rule *binding* rather than priced, and "
            "unlike a reward term it leaves the action's stated consequence "
            "intact -- the move happens as asked, and the bill arrives after."
        ),
    )
