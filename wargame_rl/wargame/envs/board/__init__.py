"""Board-wide reads: the data a plan is made of.

A **leaf** package. It may import `envs/domain`, `envs/types`, stdlib and numpy,
and nothing else -- `tests/test_board_layer_is_a_leaf.py` pins that with an AST
walk, and pins that only `renders/v2/control.py` may import it, which is v2's
single domain seam.

It lives inside the package rather than in `scripts/` for one reason: scripted
policies live in `envs/baseline/` and cannot import `scripts/`. The cheapest
form of any claim here is a scripted policy, so maths a policy cannot reach can
never be priced the way this project prices things.

`matchup` is the first tool in it, and it is the one that has **no positions**:
it reads the two armies' stat lines before a model has moved. Everything else
this package will hold answers a question about *ground*, and every one of those
is a NEXT-TURN quantity -- the opponent moves before it shoots, so a read of
what bears *now* answers a question nobody asks while deciding where to move.
Keeping the position-free tool separate from the position-bearing ones is the
whole reason the package is organised by question rather than by helper.
"""

from wargame_rl.wargame.envs.board.matchup import (
    Matchup,
    UnitProfile,
    exchange_ratio,
    matchup_matrix,
    matchup_table,
    unit_profiles,
)

__all__ = [
    "Matchup",
    "UnitProfile",
    "exchange_ratio",
    "matchup_matrix",
    "matchup_table",
    "unit_profiles",
]
