"""Resolve a monospace font path once, shared by every backend.

The HUD shows fast-changing numbers, so it wants tabular (monospace) digits that
stay aligned as values tick; the board labels keep the default proportional
face. The path is resolved through pygame (always a dependency here) and reused
by all backends, Pillow included, so the HUD looks the same whichever one draws.
``None`` when the system has no monospace font — a backend then falls back to its
default face, so the HUD still renders, just not tabular.
"""

from __future__ import annotations

import pygame

_paths: dict[bool, str | None] = {}


def mono_font_path(bold: bool = False) -> str | None:
    """System monospace font path (cached), or ``None`` if none is installed."""
    if bold not in _paths:
        if not pygame.font.get_init():
            pygame.font.init()
        _paths[bold] = pygame.font.match_font("monospace", bold=bold)
    return _paths[bold]
