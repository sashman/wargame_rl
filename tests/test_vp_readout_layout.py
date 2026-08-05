"""The VP readout must not move when the per-step delta appears.

`(+5)` shows up for a single step every few rounds. When the whole line was
rendered as one centred string, its arrival re-centred everything and the label
and number jumped sideways — legible when paused, unreadable in a recording.
"""

from __future__ import annotations

import numpy as np
import pygame
import pytest

from wargame_rl.wargame.envs.renders.human import HumanRender

WIDTH, HEIGHT = 600, 60
CENTER_X, CENTER_Y = WIDTH // 2, HEIGHT // 2
WHITE = (255, 255, 255)


@pytest.fixture(scope="module")
def font() -> pygame.font.Font:
    pygame.init()
    pygame.font.init()
    return pygame.font.Font(None, 24)


def _render(font: pygame.font.Font, value: int, delta: int) -> np.ndarray:
    """Draw one readout onto a blank surface and return its pixels."""
    renderer = HumanRender()
    renderer.window = pygame.Surface((WIDTH, HEIGHT))
    renderer.window.fill((0, 0, 0))
    renderer._draw_vp_readout(
        font, "Player VP:", value, delta, CENTER_X, CENTER_Y, WHITE
    )
    return pygame.surfarray.array3d(renderer.window)


def _ink_columns(pixels: np.ndarray) -> np.ndarray:
    """Which x columns contain any drawn pixel."""
    return np.flatnonzero(pixels.any(axis=(1, 2)))


def test_label_and_value_do_not_move_when_the_delta_appears(
    font: pygame.font.Font,
) -> None:
    """The regression: adding `(+5)` must not shift anything already drawn."""
    without = _render(font, 42, 0)
    with_delta = _render(font, 42, 5)

    # Everything up to where the delta starts must be pixel-identical.
    delta_start = _ink_columns(with_delta - np.minimum(without, with_delta)).min()
    assert np.array_equal(without[:delta_start], with_delta[:delta_start]), (
        "the label or value moved when the delta appeared"
    )


@pytest.mark.parametrize("value", [0, 7, 42, 285])
def test_value_keeps_a_fixed_right_edge_as_it_gains_digits(
    font: pygame.font.Font, value: int
) -> None:
    """The number is right-anchored, so more digits grow leftward.

    A left-anchored number would push the delta sideways every time the score
    crossed a power of ten — the same flicker in a slower form.
    """
    reference_right = _ink_columns(_render(font, 0, 0)).max()

    assert _ink_columns(_render(font, value, 0)).max() == reference_right


def test_delta_starts_at_the_same_place_regardless_of_value(
    font: pygame.font.Font,
) -> None:
    """A one-digit and a three-digit score put the delta in the same column."""
    small = _render(font, 7, 5)
    large = _render(font, 285, 5)

    small_delta_start = _ink_columns(
        small - np.minimum(_render(font, 7, 0), small)
    ).min()
    large_delta_start = _ink_columns(
        large - np.minimum(_render(font, 285, 0), large)
    ).min()

    assert small_delta_start == large_delta_start
