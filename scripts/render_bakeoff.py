"""Compare the v2 drawing backends on quality and speed, so the default is picked
by evidence rather than taste.

Every backend rasterises the *same* `Scene` (`build_scene` is deterministic given
the same board state), so any visual difference is purely the backend's edge and
text quality, not different primitives. For each of a normal and a dense-terrain
scene this renders one frame per backend to a PNG, stitches a labelled contact
sheet for side-by-side eyeballing, and times steady-state renders to report
ms/frame. pygame is the aliased baseline, pygame_aa supersamples it, pillow draws
antialiased fills directly.

Usage: just render-bakeoff [out_dir] [n_timing]
"""

from __future__ import annotations

import os
import sys
import time

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from wargame_rl.wargame.envs.renders.renderer import FrameSource  # noqa: E402
from wargame_rl.wargame.envs.renders.v2 import build_renderer  # noqa: E402
from wargame_rl.wargame.envs.renders.v2.factory import BACKENDS  # noqa: E402
from wargame_rl.wargame.envs.types import WargameEnvAction  # noqa: E402
from wargame_rl.wargame.envs.types.config import WargameEnvConfig  # noqa: E402
from wargame_rl.wargame.envs.types.config import (  # noqa: E402
    OpponentPolicyConfig,
    RandomTerrainConfig,
    TerrainPieceConfig,
)
from wargame_rl.wargame.envs.wargame import WargameEnv  # noqa: E402

SEED = 7
WARMUP = 3


def _normal_config() -> WargameEnvConfig:
    """A representative mid-size board: two squads, objectives, a few ruins."""
    return WargameEnvConfig(
        board_width=30,
        board_height=22,
        number_of_wargame_models=12,
        number_of_opponent_models=12,
        number_of_objectives=3,
        number_of_battle_rounds=5,
        base_radius=1.0,
        opponent_policy=OpponentPolicyConfig(type="scripted_advance_to_objective"),
        render_mode=None,
        terrain=[
            TerrainPieceConfig(footprint=(8, 6, 12, 9)),
            TerrainPieceConfig(footprint=(18, 12, 23, 15)),
            TerrainPieceConfig(footprint=(13, 16, 16, 19)),
        ],
    )


def _dense_config() -> WargameEnvConfig:
    """Many small ruins — the case that stresses polygon edges and text."""
    return WargameEnvConfig(
        board_width=40,
        board_height=30,
        number_of_wargame_models=16,
        number_of_opponent_models=16,
        number_of_objectives=3,
        number_of_battle_rounds=5,
        base_radius=1.0,
        opponent_policy=OpponentPolicyConfig(type="scripted_advance_to_objective"),
        render_mode=None,
        random_terrain=RandomTerrainConfig(
            count=16, min_size=3, max_size=5, min_gap=1, mirror=True
        ),
    )


def _prepare_env(
    config: WargameEnvConfig, backend: str
) -> tuple[WargameEnv, FrameSource]:
    """Reset an env with a v2 recording renderer and take two moves for arrows."""
    renderer = build_renderer("v2", "recording", backend=backend)
    env = WargameEnv(config=config, renderer=renderer)
    env.reset(seed=SEED)
    env.action_space.seed(SEED)
    # A couple of steps so movement arrows appear — they exercise thick AA lines.
    for _ in range(2):
        action = WargameEnvAction(actions=[int(a) for a in env.action_space.sample()])
        _, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break
    assert isinstance(renderer, FrameSource)
    return env, renderer


def _time_render(env: WargameEnv, source: FrameSource, n: int) -> float:
    """Mean ms for render + frame capture over ``n`` calls after a warmup."""
    for _ in range(WARMUP):
        env.render()
        source.get_frame_array()
    start = time.perf_counter()
    for _ in range(n):
        env.render()
        source.get_frame_array()
    return (time.perf_counter() - start) / n * 1000.0


def _label_strip(width: int, text: str, height: int = 40) -> Image.Image:
    """A dark caption bar for the contact sheet."""
    strip = Image.new("RGB", (width, height), (30, 30, 33))
    draw = ImageDraw.Draw(strip)
    draw.text(
        (width // 2, height // 2),
        text,
        font=ImageFont.load_default(size=24),
        fill=(230, 230, 230),
        anchor="mm",
    )
    return strip


def _contact_sheet(
    frames: dict[str, np.ndarray], timings: dict[str, float]
) -> Image.Image:
    """Stack each backend's frame under its caption, side by side."""
    columns = []
    for backend, frame in frames.items():
        image = Image.fromarray(frame)
        caption = _label_strip(image.width, f"{backend}  ({timings[backend]:.1f} ms)")
        column = Image.new("RGB", (image.width, caption.height + image.height))
        column.paste(caption, (0, 0))
        column.paste(image, (0, caption.height))
        columns.append(column)
    total_w = sum(c.width for c in columns) + 8 * (len(columns) - 1)
    sheet = Image.new("RGB", (total_w, columns[0].height), (30, 30, 33))
    x = 0
    for column in columns:
        sheet.paste(column, (x, 0))
        x += column.width + 8
    return sheet


def _run_scene(
    name: str, config: WargameEnvConfig, out_dir: str, n_timing: int
) -> None:
    frames: dict[str, np.ndarray] = {}
    timings: dict[str, float] = {}
    for backend in BACKENDS:
        env, source = _prepare_env(config, backend)
        try:
            env.render()
            frame = source.get_frame_array()
            frames[backend] = frame
            timings[backend] = _time_render(env, source, n_timing)
            Image.fromarray(frame).save(os.path.join(out_dir, f"{name}_{backend}.png"))
        finally:
            env.close()

    sheet = _contact_sheet(frames, timings)
    sheet.save(os.path.join(out_dir, f"{name}_contact.png"))

    print(f"\n{name}  ({config.board_width}x{config.board_height} board)")
    baseline = timings[BACKENDS[0]]
    for backend in BACKENDS:
        ms = timings[backend]
        print(f"  {backend:10s} {ms:7.2f} ms/frame   {ms / baseline:5.2f}x baseline")


def main() -> None:
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "bakeoff_out"
    n_timing = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    os.makedirs(out_dir, exist_ok=True)

    _run_scene("normal", _normal_config(), out_dir, n_timing)
    _run_scene("dense", _dense_config(), out_dir, n_timing)

    print(
        f"\nPNGs + contact sheets written to {out_dir}/. "
        "Eyeball the *_contact.png sheets; ms/frame is steady-state render+capture.\n"
    )


if __name__ == "__main__":
    main()
