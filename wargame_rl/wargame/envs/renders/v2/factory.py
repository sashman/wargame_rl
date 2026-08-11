"""One entry point to pick a renderer, so call sites stay declarative.

`legacy` returns the untouched `HumanRender`, which is the default everywhere, so
existing behaviour is unchanged until a caller opts into `v2`. Three drawing
backends are wired for the v2 renderer: `pygame` (aliased, the legacy look),
`pygame_aa` (3× supersampled pygame) and `pillow`. The Phase 2 bake-off
(`scripts/render_bakeoff.py`) picked **pillow** as the default: it antialiases
shapes as well as the supersampled backend at ~pygame speed (1.04–1.09× vs
3.9–5.9×), so v2's default look is smooth without the supersample cost.
"""

from __future__ import annotations

from wargame_rl.wargame.envs.renders.human import HumanRender
from wargame_rl.wargame.envs.renders.renderer import Renderer
from wargame_rl.wargame.envs.renders.v2.backend import RenderBackend
from wargame_rl.wargame.envs.renders.v2.backends.pillow_backend import PillowBackend
from wargame_rl.wargame.envs.renders.v2.backends.pygame_aa_backend import (
    PygameAABackend,
)
from wargame_rl.wargame.envs.renders.v2.backends.pygame_backend import PygameBackend
from wargame_rl.wargame.envs.renders.v2.presenters.interactive import (
    InteractiveRenderer,
)
from wargame_rl.wargame.envs.renders.v2.presenters.recording import RecordingRenderer
from wargame_rl.wargame.envs.renders.v2.theme import DEFAULT_THEME, Theme

BACKENDS = ("pygame", "pygame_aa", "pillow")


def _build_backend(name: str) -> RenderBackend:
    if name == "pygame":
        return PygameBackend()
    if name == "pygame_aa":
        return PygameAABackend()
    if name == "pillow":
        return PillowBackend()
    raise ValueError(f"unknown backend {name!r} (available: {', '.join(BACKENDS)})")


def build_renderer(
    name: str = "legacy",
    mode: str = "interactive",
    *,
    backend: str = "pillow",
    theme: Theme = DEFAULT_THEME,
) -> Renderer:
    """Construct a renderer.

    name: ``legacy`` (the current `HumanRender`) or ``v2``.
    mode: ``interactive`` (window) or ``recording`` (headless frames).
    backend: v2 drawing backend — ``pygame``, ``pygame_aa`` or ``pillow``.
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
