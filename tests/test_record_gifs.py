"""A GIF must keep the board's own colours, not an approximation of them.

The illustrations are regenerated often, so this is pinned rather than
eyeballed. The failure it guards against is silent: a GIF routed through a video
still *looks* like the board at a glance while every flat fill has drifted a
point or three — which is how the previous hand-made GIF shipped with the board
at (238, 229, 210) instead of the theme's (239, 231, 212).

`test_a_video_round_trip_is_what_drifts_them` is the reason the first test is
not vacuous. Quantising to a GIF palette turns out to preserve flat fills under
every setting tried, so a colour assertion alone would pass on the broken
pipeline too; the contrast test pins the thing that actually breaks, so nobody
"simplifies" the recorder by reusing an mp4 it already had.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import imageio.v3 as iio
import imageio_ffmpeg
import numpy as np
import pytest
from PIL import Image

from scripts.record_gifs import write_gif
from wargame_rl.wargame.envs.renders.v2.theme import THEMES

TABLETOP = THEMES["tabletop"].palette
FLAT_FILLS = (
    TABLETOP.board_bg,
    TABLETOP.deployment_zone,
    TABLETOP.opponent_zone,
)
# Big enough that scaling to 640 wide is a real downscale, as it is in practice.
FRAME_HEIGHT, FRAME_WIDTH = 400, 800


def _board_like_frame() -> np.ndarray:
    """Flat fills over most of the frame, noise over the rest.

    The noise stands in for antialiased edges and the HUD, which are what push a
    real frame past the 256 colours a GIF can hold.
    """
    frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
    band = FRAME_HEIGHT // (len(FLAT_FILLS) + 1)
    for index, colour in enumerate(FLAT_FILLS):
        frame[index * band : (index + 1) * band] = colour
    rng = np.random.default_rng(0)
    frame[len(FLAT_FILLS) * band :] = rng.integers(
        0, 256, (FRAME_HEIGHT - len(FLAT_FILLS) * band, FRAME_WIDTH, 3), dtype=np.uint8
    )
    return frame


def _first_frame_colours(path: Path) -> set[tuple[int, ...]]:
    frame = np.asarray(iio.imread(path, index=0))[:, :, :3]
    return {tuple(int(v) for v in pixel) for pixel in frame.reshape(-1, 3)}


def test_the_themes_own_colours_survive_exactly(tmp_path: Path) -> None:
    out = tmp_path / "board.gif"

    write_gif([_board_like_frame()] * 3, out)

    present = _first_frame_colours(out)
    drifted = [tuple(c) for c in FLAT_FILLS if tuple(c) not in present]
    assert not drifted, f"flat fills did not survive: {drifted}"


def test_a_video_round_trip_is_what_drifts_them(tmp_path: Path) -> None:
    """The trap, pinned: never build the GIF from a recorded video.

    H.264's RGB→YUV conversion is lossy on exactly the saturated flat areas a
    board is made of, so every fill comes back a point or three out.
    """
    frame = _board_like_frame()
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

    with tempfile.TemporaryDirectory() as work_dir:
        work = Path(work_dir)
        for index in range(4):
            Image.fromarray(frame).save(work / f"f{index:04d}.png")
        video = work / "clip.mp4"
        subprocess.run(
            [
                ffmpeg,
                "-y",
                "-v",
                "error",
                "-i",
                str(work / "f%04d.png"),
                "-pix_fmt",
                "yuv420p",
                "-vcodec",
                "libx264",
                str(video),
            ],
            check=True,
        )
        decoded = np.asarray(iio.imread(video, index=1))[:, :, :3]

    band = FRAME_HEIGHT // (len(FLAT_FILLS) + 1)
    for index, colour in enumerate(FLAT_FILLS):
        sampled = tuple(
            int(v) for v in decoded[index * band + band // 2, FRAME_WIDTH // 2]
        )
        assert sampled != tuple(colour), (
            f"{colour} survived the video round trip — if this now holds, the "
            f"contrast this test provides is gone and the colour test above is "
            f"no longer proving anything"
        )


def test_no_frames_is_an_error_not_an_empty_file(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no frames"):
        write_gif([], tmp_path / "empty.gif")
