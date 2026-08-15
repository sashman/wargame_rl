"""Standing inside a ruin makes that whole ruin transparent to you.

This pins a **known divergence**, not a desired rule. `domain/sight.py` exempts
any footprint containing either endpoint, which is the spec's *obscuring*
see-out/see-into clause -- but the code has one terrain category, so every
feature gets it, including the ones the spec calls *solid*, where line of sight
"cannot be drawn across any enclosed gap". It is also unbounded: the exemption
covers the entire piece rather than the model's own position, so a model in a
20"-wide building shoots through all 20" of it.

It matters most on the real tables, where every objective marker sits inside a
ruin and the two centre markers share the largest central piece on 24 of the 45
maps -- so a unit holding the middle is *inside* the building it fires through.
That is what a recording shows as "shooting through terrain".

The test is here so that fixing this is a deliberate act with a visible diff.
`test_a_second_piece_on_the_line_still_blocks` is the one that bounds the
damage: were the exemption global rather than per-piece, a model in any ruin
would see the whole board and the divergence would be unusable rather than
merely wrong. See `docs/rules/implementation-status.md` § Terrain and
visibility.
"""

from __future__ import annotations

import pytest

from wargame_rl.wargame.envs.domain.sight import has_line_of_sight_between_points
from wargame_rl.wargame.envs.domain.terrain import Footprint, Terrain

# 20 wide and 8 deep, in cell coordinates. Comfortably wider than the 12" weapon
# range the 25v25 configs use, so "sees across it" is not a near-miss.
WIDE_RUIN = Footprint.from_cell_rect(20, 18, 39, 25)
MID_Y = 22.0


def sees(x0: float, x1: float, terrain: Terrain) -> bool:
    """Line of sight along y=MID_Y between two x positions."""
    return has_line_of_sight_between_points(
        x0, MID_Y, x1, MID_Y, terrain, None, sample_step=0.25
    )


def test_a_piece_blocks_when_neither_model_is_inside_it() -> None:
    # Arrange: the control. Without this the exemption tests below would pass
    # just as well against terrain that never blocked anything at all.
    terrain = Terrain([WIDE_RUIN])

    # Act / Assert
    assert not sees(19.0, 41.0, terrain)


@pytest.mark.parametrize(
    ("x0", "x1", "case"),
    [
        (21.0, 41.0, "shooter inside, target beyond the far wall"),
        (21.0, 39.0, "both inside, 18 inches apart"),
    ],
)
def test_standing_inside_makes_the_whole_piece_transparent(
    x0: float, x1: float, case: str
) -> None:
    # Arrange: the divergence. The spec's see-out clause is about seeing out of
    # the feature you occupy; here it clears the entire footprint for the ray.
    terrain = Terrain([WIDE_RUIN])

    # Act / Assert
    assert sees(x0, x1, terrain), case


def test_a_second_piece_on_the_line_still_blocks() -> None:
    # Arrange: the exemption is per-piece. A global one would let a model in any
    # ruin see the whole board, which is a different and far worse defect.
    near = Footprint.from_cell_rect(20, 18, 29, 25)
    far = Footprint.from_cell_rect(34, 18, 43, 25)
    terrain = Terrain([near, far])

    # Act / Assert: the shooter is inside `near`, but `far` is untouched by the
    # exemption and stops the ray.
    assert not sees(21.0, 50.0, terrain)
