"""Board-wide reads: the data a plan is made of.

A **leaf** package. It may import `envs/domain`, `envs/types`, stdlib and numpy,
and nothing else -- `tests/test_board_layer_is_a_leaf.py` pins that with an AST
walk, and pins that only `renders/v2/control.py` may import it, which is v2's
single domain seam.

It lives inside the package rather than in `scripts/` for one reason: scripted
policies live in `envs/baseline/` and cannot import `scripts/`. The cheapest
form of any claim here is a scripted policy, so maths a policy cannot reach can
never be priced the way this project prices things.

Today it holds the sampling grid and the threat field. Other board-wide reads --
unit matchups, objective arrival -- are the same shape and belong here when they
land.

⚠ **Everything here is a NEXT-TURN quantity.** The opponent moves before it
shoots, so a read of what bears *now* answers a question nobody asks while
deciding where to move -- and answers it **false-safe**.

Every read states what it gets wrong with the direction of each bias. A tool
that hides which way it errs is worse than one that overstates.
"""

from wargame_rl.wargame.envs.board.grid import (
    DEFAULT_SPACING,
    BoardGrid,
    board_grid,
    board_grid_for,
)
from wargame_rl.wargame.envs.board.threat import (
    ReferenceModel,
    ThreatField,
    ThreatHorizon,
    VisibilityCache,
    attacker_stat_rows,
    move_reach,
    reachable_cells,
    reference_model,
    threat_field,
)

__all__ = [
    "DEFAULT_SPACING",
    "BoardGrid",
    "ReferenceModel",
    "ThreatField",
    "ThreatHorizon",
    "VisibilityCache",
    "attacker_stat_rows",
    "board_grid",
    "board_grid_for",
    "move_reach",
    "reachable_cells",
    "reference_model",
    "threat_field",
]
