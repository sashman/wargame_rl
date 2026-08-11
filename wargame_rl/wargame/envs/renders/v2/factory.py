"""One entry point to pick a renderer, so call sites stay declarative.

`legacy` returns the untouched `HumanRender`, which is the default everywhere, so
existing behaviour is unchanged until a caller opts into `v2`. Only pygame is
wired as an interactive backend in Phase 1; the AA and Pillow backends arrive in
the Phase 2 bake-off.
"""

from __future__ import annotations

from wargame_rl.wargame.envs.renders.human import HumanRender
from wargame_rl.wargame.envs.renders.renderer import Renderer
from wargame_rl.wargame.envs.renders.v2.backend import RenderBackend
from wargame_rl.wargame.envs.renders.v2.backends.pygame_backend import PygameBackend
from wargame_rl.wargame.envs.renders.v2.presenters.interactive import (
    InteractiveRenderer,
)
from wargame_rl.wargame.envs.renders.v2.presenters.recording import RecordingRenderer
from wargame_rl.wargame.envs.renders.v2.theme import DEFAULT_THEME, Theme


def _build_backend(name: str) -> RenderBackend:
    if name == "pygame":
        return PygameBackend()
    raise ValueError(f"unknown backend {name!r} (available: pygame)")


def build_renderer(
    name: str = "legacy",
    mode: str = "interactive",
    *,
    backend: str = "pygame",
    theme: Theme = DEFAULT_THEME,
) -> Renderer:
    """Construct a renderer.

    name: ``legacy`` (the current `HumanRender`) or ``v2``.
    mode: ``interactive`` (window) or ``recording`` (headless frames).
    backend: v2 drawing backend; only ``pygame`` in Phase 1.
    """
    if name == "legacy":
        return HumanRender()
    if name != "v2":
        raise ValueError(f"unknown renderer {name!r} (available: legacy, v2)")

    be = _build_backend(backend)
    if mode == "interactive":
        return InteractiveRenderer(be, theme)
    if mode == "recording":
        return RecordingRenderer(be, theme)
    raise ValueError(f"unknown mode {mode!r} (available: interactive, recording)")
