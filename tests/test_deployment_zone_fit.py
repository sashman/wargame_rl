"""A deployed base must fit *wholly* inside its zone, on any polygon.

`_fits_in_zone` used to probe four cardinal points of the base, justified in its
own docstring by "the zones are convex or nearly so".

⚠ Both halves of that were wrong, and the first published correction of it was
wrong too. The zones are not convex (34 of the 45 real tables are triangles,
staircases and arcs) -- but convexity was never the property that mattered. **A
cardinal probe misses whenever the nearest boundary point is not in a cardinal
direction.** One edge at any angle to the axes causes it; no corner is required,
and it happens at convex and reflex vertices alike.
"""

from __future__ import annotations

import numpy as np

from wargame_rl.wargame.envs.domain.placement import _fits_in_zone
from wargame_rl.wargame.envs.types.geometry import Polygon

# A single angled edge -- the minimal failing case, no corner involved. The
# hypotenuse `x + y = 10` has normal (1,1)/sqrt(2), so a cardinal step of r
# closes only r/sqrt(2) of the gap to it.
TRIANGLE = Polygon.from_points([(0.0, 0.0), (10.0, 0.0), (0.0, 10.0)])

# An L, whose vertex at (5, 5) is REFLEX. Included because the first correction
# of this claimed reflex corners were safe ("a disc there has more room"). They
# are not: room in the interior angle is irrelevant when the nearest EDGE is
# still closer than the base radius and not on a cardinal axis.
L_SHAPE = Polygon.from_points(
    [(0.0, 0.0), (10.0, 0.0), (10.0, 5.0), (5.0, 5.0), (5.0, 10.0), (0.0, 10.0)]
)


def _four_point_probe(x: float, y: float, zone: Polygon, radius: float) -> bool:
    """The retired implementation, kept so the tests can prove they catch it."""
    return zone.contains(x, y) and all(
        zone.contains(x + dx, y + dy)
        for dx, dy in ((radius, 0.0), (-radius, 0.0), (0.0, radius), (0.0, -radius))
    )


def test_a_base_overhanging_an_angled_edge_is_rejected() -> None:
    """One angled edge is enough -- and the old probe accepted this."""
    # Arrange: on the bisector, 1.1 from the hypotenuse, with a 1.3 base.
    candidate, radius = (4.222, 4.222), 1.3

    # Act / Assert
    assert _fits_in_zone(candidate, TRIANGLE, radius) is False
    assert _four_point_probe(*candidate, TRIANGLE, radius) is True


def test_a_base_overhanging_a_REFLEX_corner_is_also_rejected() -> None:
    """The case the first correction wrongly declared safe."""
    # Arrange: near the reflex vertex of the L, nearest edge off-axis.
    candidate, radius = (4.99895813687647, 4.252286248490755), 0.8

    # Act / Assert
    assert _fits_in_zone(candidate, L_SHAPE, radius) is False
    assert _four_point_probe(*candidate, L_SHAPE, radius) is True


def test_a_base_with_room_on_every_side_is_accepted() -> None:
    """The guard must not reject legal ground, or deployment cannot place."""
    # Arrange / Act / Assert
    assert _fits_in_zone((3.232, 3.232), TRIANGLE, 1.3) is True
    assert _fits_in_zone((2.5, 2.5), L_SHAPE, 1.3) is True


def test_the_centre_still_has_to_be_inside() -> None:
    """Containment is checked before clearance; outside is outside."""
    # Arrange / Act / Assert
    assert _fits_in_zone((7.5, 7.5), L_SHAPE, 0.0) is False


def test_a_point_model_needs_only_containment() -> None:
    """`base_radius` 0 keeps every pre-base result reproducible."""
    # Arrange / Act / Assert
    assert _fits_in_zone((0.001, 0.001), L_SHAPE, 0.0) is True


def test_no_zone_accepts_everything() -> None:
    """A scenario without a polygon zone is unconstrained here."""
    # Arrange / Act / Assert
    assert _fits_in_zone((999.0, 999.0), None, 5.0) is True


def test_clearance_agrees_with_a_dense_rim_oracle() -> None:
    """Cross-check against sampling the base rim, on both shapes.

    ⚠ The oracle samples the RIM, not the disc, so it is only sound where no
    feature is thinner than the chord between samples. 512 samples at r=0.8
    gives a 0.0098 chord against fixture features of 5.0 -- three orders of
    margin. It would NOT be sound on a real staircase zone, which is why this
    is a cross-check on fixtures and not the primary assertion.
    """
    # Arrange
    rng = np.random.default_rng(0)
    angles = np.linspace(0.0, 2 * np.pi, 512, endpoint=False)
    radius = 0.8

    for zone in (TRIANGLE, L_SHAPE):
        for _ in range(400):
            x, y = rng.uniform(0.0, 10.0), rng.uniform(0.0, 10.0)
            if not zone.contains(x, y):
                continue

            # Act
            fits = _fits_in_zone((x, y), zone, radius)
            rim_inside = all(
                zone.contains(x + radius * np.cos(a), y + radius * np.sin(a))
                for a in angles
            )

            # Assert
            assert fits == rim_inside, f"disagreed at ({x:.6f}, {y:.6f})"
