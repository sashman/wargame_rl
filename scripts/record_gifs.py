"""Record one game per table and write a GIF of each, with exact colours.

The README's illustrations go stale every time the tables, the renderer or the
trained lineage change, so producing them is a recipe rather than a one-off:
`just record-gifs <policy|ckpt> <config> [tables]`.

**Frames go straight to the GIF, never through a video.** That is the whole
colour story, and it was measured rather than assumed: encoding to H.264 in
`yuv420p` and reading back shifts *every* flat fill by one to three points —
the tabletop board (239, 231, 212) comes back (238, 229, 210), the deployment
zone (223, 228, 207) comes back (221, 226, 204) — because the RGB→YUV
conversion is lossy on exactly the saturated flat areas a board is made of. The
previous hand-made GIF shipped with precisely that drift, and the `.mp4` files
beside it in `docs/images/` say why: it was converted from one. So this holds
the rendered arrays and writes the GIF from them.

The palette is still generated in two passes — one palette across the whole clip
(`stats_mode=full`) applied without dithering — because a frame carries ~1700
distinct colours and cannot fit losslessly in a GIF's 256. Measured, that choice
is *not* what saves the flat fills (a per-frame dithered palette keeps them
exact too, even at 85% noise); it is simply the better default, and keeps flat
areas flat instead of stippling two near colours together.

ffmpeg comes from `imageio_ffmpeg`'s bundled binary, so this needs nothing
installed on the machine.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path
from typing import cast

import imageio_ffmpeg
import numpy as np
from PIL import Image
from pydantic_yaml import parse_yaml_raw_as

from scripts.measure_maps import config_for_map, load_maps
from wargame_rl.wargame.envs.renders.renderer import FrameSource
from wargame_rl.wargame.envs.renders.v2 import build_renderer
from wargame_rl.wargame.envs.types import TerrainMapConfig, WargameEnvConfig
from wargame_rl.wargame.envs.wargame import WargameEnv
from wargame_rl.wargame.model.common.factory import create_environment

DEFAULT_MAPS_DIR = Path("configs/evaluation/maps_heldout")
DEFAULT_OUT_DIR = Path("docs/images")
DEFAULT_SEED = 700000
# 512 rather than 640: `check-added-large-files` caps a committed file at
# 500 KB, and a 41-frame board at 640 lands just over it -- which rejects the
# commit after the recording has already been paid for.
DEFAULT_WIDTH = 512
MAX_COMMITTABLE_KB = 500
DEFAULT_FPS = 8
# Videos are the v2 renderer on the tabletop theme, always — a GIF that does not
# look like the rest of the project's illustrations is worse than no GIF.
RENDERER, THEME = "v2", "tabletop"


class EpisodeResult:
    """One recorded game: its frames and the score they end on."""

    def __init__(
        self, frames: list[np.ndarray], player_vp: int, opponent_vp: int, alive: int
    ) -> None:
        self.frames = frames
        self.player_vp = player_vp
        self.opponent_vp = opponent_vp
        self.alive = alive

    @property
    def margin(self) -> int:
        return self.player_vp - self.opponent_vp


def record_episode(
    config: WargameEnvConfig,
    terrain_map: TerrainMapConfig,
    policy_or_checkpoint: str,
    seed: int,
    decode_topk: int,
) -> EpisodeResult:
    """Play one seeded game on one table, keeping every rendered frame."""
    # Imported here: resolving a checkpoint pulls in torch, and a caller that
    # only wants `write_gif` should not pay for that.
    from wargame_rl.wargame.selectors import build_action_selector

    # The selector is resolved against the env built for *this map*, because a
    # checkpoint's network is sized from the env it plays in and sizing it from
    # the scenario would give the wrong observation width.
    map_config = config_for_map(config, terrain_map)
    renderer = build_renderer(RENDERER, "recording", backend="pillow", theme=THEME)
    env = create_environment(env_config=map_config, renderer=renderer)
    renderer.setup(env)
    frame_source = cast(FrameSource, renderer)
    frame_source.epoch = 0

    unwrapped = cast(WargameEnv, env.unwrapped)
    select = build_action_selector(policy_or_checkpoint, unwrapped, decode_topk).select

    observation, _ = env.reset(seed=seed)
    frames: list[np.ndarray] = []
    env.render()
    frames.append(frame_source.get_frame_array())

    done = False
    while not done:
        action = select(observation, unwrapped)
        observation, _reward, terminated, truncated, _info = env.step(action)
        done = terminated or truncated
        env.render()
        frames.append(frame_source.get_frame_array())

    result = EpisodeResult(
        frames,
        int(unwrapped.player_vp),
        int(unwrapped.opponent_vp),
        int(sum(1 for m in unwrapped.wargame_models if m.is_alive)),
    )
    renderer.close()
    return result


def record_median_episode(
    config: WargameEnvConfig,
    terrain_map: TerrainMapConfig,
    policy_or_checkpoint: str,
    seed: int,
    decode_topk: int,
    episodes: int,
) -> EpisodeResult:
    """Play `episodes` consecutive seeds and keep the median game by margin.

    One fixed seed is not representative and misleads in both directions: on
    `table_35`, where the trained model averages +24 over thirty games, seed
    700000 alone comes out −100. Taking the median is a rule that can be stated
    in a caption and does not require knowing the answer first — unlike picking
    the best game, which is how the previous illustration was chosen.
    """
    results = [
        record_episode(
            config, terrain_map, policy_or_checkpoint, seed + offset, decode_topk
        )
        for offset in range(episodes)
    ]
    return sorted(results, key=lambda r: r.margin)[len(results) // 2]


def write_gif(
    frames: list[np.ndarray],
    path: Path,
    width: int = DEFAULT_WIDTH,
    fps: int = DEFAULT_FPS,
) -> None:
    """Write frames to a GIF whose flat fills keep their exact colours.

    Two ffmpeg passes: one palette generated across the whole clip, then applied
    without dithering. See the module docstring for why either alone is wrong.
    """
    if not frames:
        raise ValueError(f"no frames to write to {path}")

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    scale = f"scale={width}:-1:flags=lanczos"
    path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as work_dir:
        work = Path(work_dir)
        for index, frame in enumerate(frames):
            Image.fromarray(frame).save(work / f"f{index:04d}.png")
        pattern = str(work / "f%04d.png")
        palette = str(work / "palette.png")

        subprocess.run(
            [
                ffmpeg,
                "-y",
                "-v",
                "error",
                "-i",
                pattern,
                "-vf",
                f"{scale},palettegen=max_colors=256:stats_mode=full",
                palette,
            ],
            check=True,
        )
        subprocess.run(
            [
                ffmpeg,
                "-y",
                "-v",
                "error",
                "-framerate",
                str(fps),
                "-i",
                pattern,
                "-i",
                palette,
                "-lavfi",
                f"{scale}[x];[x][1:v]paletteuse=dither=none",
                "-loop",
                "0",
                str(path),
            ],
            check=True,
        )


def main() -> None:
    """CLI: `record_gifs <policy|ckpt> <config> [tables] [maps_dir] [out] [seed] [topk]`."""
    if len(sys.argv) < 3:
        raise SystemExit(
            "usage: python -m scripts.record_gifs <policy|ckpt> <env_config> "
            "[table_a,table_b] [maps_dir] [out_dir] [seed] [decode_topk] [width] "
            "[episodes]"
        )
    policy_or_checkpoint = sys.argv[1]
    config_path = Path(sys.argv[2])
    wanted = [t for t in (sys.argv[3] if len(sys.argv) > 3 else "").split(",") if t]
    maps_dir = (
        Path(sys.argv[4]) if len(sys.argv) > 4 and sys.argv[4] else DEFAULT_MAPS_DIR
    )
    out_dir = (
        Path(sys.argv[5]) if len(sys.argv) > 5 and sys.argv[5] else DEFAULT_OUT_DIR
    )
    seed = int(sys.argv[6]) if len(sys.argv) > 6 and sys.argv[6] else DEFAULT_SEED
    decode_topk = int(sys.argv[7]) if len(sys.argv) > 7 and sys.argv[7] else 3
    width = int(sys.argv[8]) if len(sys.argv) > 8 and sys.argv[8] else DEFAULT_WIDTH
    episodes = int(sys.argv[9]) if len(sys.argv) > 9 and sys.argv[9] else 1

    config = cast(
        WargameEnvConfig, parse_yaml_raw_as(WargameEnvConfig, config_path.read_text())
    )
    config.render_mode = None

    maps = {m.name: m for m in load_maps(maps_dir)}
    names = wanted or sorted(maps)
    missing = [n for n in names if n not in maps]
    if missing:
        raise SystemExit(f"not in {maps_dir}: {missing}. Available: {sorted(maps)}")

    drawn = f"median of {episodes} from seed {seed}" if episodes > 1 else f"seed {seed}"
    print(f"{policy_or_checkpoint} on {config_path.name}, {drawn}, K={decode_topk}\n")
    for name in names:
        terrain_map = maps[name]
        result = record_median_episode(
            config, terrain_map, policy_or_checkpoint, seed, decode_topk, episodes
        )
        path = out_dir / f"{name}.gif"
        write_gif(result.frames, path, width=width)
        shape = terrain_map.deployment.name if terrain_map.deployment else "rectangle"
        size_kb = path.stat().st_size / 1024
        # Flagged rather than raised: the recording is expensive and already
        # done, so the useful thing is to name the file that needs a smaller
        # width, not to throw the whole run away.
        oversize = (
            f"  <- over {MAX_COMMITTABLE_KB} KB, pre-commit will reject it"
            if size_kb > MAX_COMMITTABLE_KB
            else ""
        )
        print(
            f"  {name:<10} {shape:<18} {result.player_vp:>4}-{result.opponent_vp:<4} "
            f"({result.margin:+d})  {result.alive:>2} alive  "
            f"{len(result.frames):>3} frames  {size_kb:>5.0f} KB{oversize}"
        )


if __name__ == "__main__":
    main()
