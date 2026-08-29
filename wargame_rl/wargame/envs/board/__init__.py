"""Board-wide reads: the data a plan is made of.

A **leaf** package. It may import `envs/domain`, `envs/types`, stdlib and numpy,
and nothing else -- `tests/test_board_layer_is_a_leaf.py` pins that with an AST
walk, and pins that only `renders/v2/control.py` may import it, which is v2's
single domain seam.

It lives inside the package rather than in `scripts/` for one reason: scripted
policies live in `envs/baseline/` and cannot import `scripts/`. The cheapest
form of any claim here is a scripted policy, so maths a policy cannot reach can
never be priced the way this project prices things.

Today it holds the sampling grid, the threat field and the unit matchup table.

**The reads divide in two, and the division is the thing to keep straight.**

* Reads about **ground** -- the threat field, and objective arrival when it
  lands -- take a board and a turn. ⚠ **Every one of them is a NEXT-TURN
  quantity.** The opponent moves before it shoots, so a read of what bears
  *now* answers a question nobody asks while deciding where to move -- and
  answers it **false-safe**.
* Reads about **armies** -- the matchup table -- take two stat lines and have
  **no positions at all**. Range never enters them; where reach matters it is
  its own column. A matchup number that pretended to know range-to-target would
  be answering a ground question from a stat line, which cannot.

Every read states what it gets wrong with the direction of each bias. A tool
that hides which way it errs is worse than one that overstates.
"""

from wargame_rl.wargame.envs.board.grid import (
    DEFAULT_SPACING,
    BoardGrid,
    board_grid,
    board_grid_for,
)
from wargame_rl.wargame.envs.board.matchup import (
    Matchup,
    UnitProfile,
    exchange_ratio,
    matchup_matrix,
    matchup_table,
    unit_profiles,
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
    "Matchup",
    "ReferenceModel",
    "ThreatField",
    "ThreatHorizon",
    "UnitProfile",
    "VisibilityCache",
    "attacker_stat_rows",
    "board_grid",
    "board_grid_for",
    "exchange_ratio",
    "matchup_matrix",
    "matchup_table",
    "move_reach",
    "reachable_cells",
    "reference_model",
    "threat_field",
    "unit_profiles",
]
