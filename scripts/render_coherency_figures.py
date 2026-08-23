"""Render matched frames showing a unit out of coherency, and in it.

The report's numbers say coherency went from 0.55 to 1.000 and cost ~10 vp.
Neither number shows what the board actually looks like, which is the thing a
reader needs to judge whether the rule is being followed for real.

Two policies are run on the **same layout and seed** so the boards are directly
comparable, and every model the rules predicate calls out of coherency is ringed
in the frame. Detection uses `evaluate_coherency` itself -- the same predicate
the metric and the enforcement use -- so a ring is the engine's own answer and
not the figure's.
"""

from __future__ import annotations

import sys
from typing import cast

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pydantic_yaml import parse_yaml_raw_as

from wargame_rl.wargame.envs.domain.coherency import evaluate_coherency
from wargame_rl.wargame.envs.domain.entities import alive_mask_for
from wargame_rl.wargame.envs.renders.v2 import build_renderer
from wargame_rl.wargame.envs.renders.v2.fonts import mono_font_path
from wargame_rl.wargame.envs.types import WargameEnvConfig
from wargame_rl.wargame.envs.wargame import WargameEnv
from wargame_rl.wargame.selectors import build_action_selector

RING = (232, 62, 62)
RING_OK = (58, 176, 106)


def build(config_path: str) -> WargameEnv:
    """A rendering env on the tabletop theme, which is the house style."""
    config = cast(
        WargameEnvConfig, parse_yaml_raw_as(WargameEnvConfig, open(config_path).read())
    )
    config.render_mode = "rgb_array"
    renderer = build_renderer("v2", "recording", backend="pillow", theme="tabletop")
    return WargameEnv(config, renderer=renderer)


def breaches(env: WargameEnv) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
    """Models to ring, and the gaps to draw across a split squad.

    Ringing every model the predicate flags is technically right and visually
    wrong. The **spread condition is collective**: once one model exceeds the
    9" cap, no member is within the cap of every other, so `member_coherency`
    is False for the WHOLE unit -- and the frame then rings two tightly packed
    clusters, which reads as "packed models are illegal", the opposite of the
    truth.

    So this reports the *split* instead. Within a broken unit, the chain graph's
    largest component is the body and the rest are splinters; only splinters are
    ringed, and a line is drawn from each splinter to the body it left. That is
    the breach a reader can actually see.
    """
    models = env.wargame_models
    report = evaluate_coherency(
        positions=np.array([m.location for m in models], dtype=float),
        group_ids=np.array([m.group_id for m in models], dtype=np.intp),
        alive_mask=alive_mask_for(models),
        base_radii=np.array([m.base_radius for m in models], dtype=float),
        nearest_distance=env.rules_quantities.scale.to_units(
            env.config.coherency.nearest_distance
        ),
        furthest_distance=env.rules_quantities.scale.to_units(
            env.config.coherency.furthest_distance
        ),
    )
    ring = np.zeros(len(models), dtype=bool)
    gaps: list[tuple[np.ndarray, np.ndarray]] = []

    def centroid(indices: np.ndarray) -> np.ndarray:
        points = np.array([models[int(i)].location for i in indices], dtype=float)
        mean: np.ndarray = points.mean(axis=0)
        return mean

    for unit in report.units:
        if unit.coherent:
            continue
        counts = np.bincount(unit.component)
        biggest = int(counts.argmax())
        body = unit.member_indices[unit.component == biggest]
        for label in range(len(counts)):
            if label == biggest or counts[label] == 0:
                continue
            splinter = unit.member_indices[unit.component == label]
            ring[splinter] = True
            gaps.append((centroid(splinter), centroid(body)))
        if len(counts) == 1:
            # One connected clump whose overall span still breaks the 9" cap:
            # there is no splinter to ring, so mark the two ends of the span.
            points = np.array([models[int(i)].location for i in unit.member_indices])
            span = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)
            a, b = np.unravel_index(int(span.argmax()), span.shape)
            ring[unit.member_indices[[a, b]]] = True
            gaps.append((points[a].astype(float), points[b].astype(float)))
    return ring, gaps


def to_pixels(env: WargameEnv, frame: np.ndarray) -> tuple[float, float, float]:
    """Board-units -> pixels: scale and the board's top-left offset in the frame.

    The v2 frame carries HUD panels above and below the board, so the board does
    not start at y=0 and a naive scaling puts every ring in the wrong place.
    """
    height, width = frame.shape[0], frame.shape[1]
    board_w, board_h = float(env.config.board_width), float(env.config.board_height)
    scale = width / board_w
    board_px = board_h * scale
    return scale, 0.0, (height - board_px) / 2.0


def annotate(
    env: WargameEnv,
    frame: np.ndarray,
    flags: np.ndarray,
    gaps: list[tuple[np.ndarray, np.ndarray]],
    caption: str,
) -> Image.Image:
    """Ring every adrift model, tether it to its squad, and stamp a caption."""
    image = Image.fromarray(frame).convert("RGB")
    draw = ImageDraw.Draw(image)
    scale, off_x, off_y = to_pixels(env, frame)

    def px(point: np.ndarray) -> tuple[float, float]:
        return off_x + float(point[0]) * scale, off_y + float(point[1]) * scale

    for start, end in gaps:
        sx, sy = px(start)
        ex, ey = px(end)
        draw.line([sx, sy, ex, ey], fill=RING, width=3)
        draw.ellipse([ex - 6, ey - 6, ex + 6, ey + 6], fill=RING)

    for index, model in enumerate(env.wargame_models):
        if not model.is_alive or not flags[index]:
            continue
        x, y = px(model.location)
        r = max(11.0, float(model.base_radius) * scale * 2.6)
        for w in range(3):
            draw.ellipse([x - r - w, y - r - w, x + r + w, y + r + w], outline=RING)

    path = mono_font_path(bold=True)
    font = ImageFont.truetype(path, 26) if path else ImageFont.load_default()
    colour = RING if flags.any() else RING_OK
    draw.rectangle([0, 0, image.width, 40], fill=(28, 24, 20))
    draw.text((12, 8), caption, fill=colour, font=font)
    return image


def run(config_path: str, checkpoint: str, seed: int, step: int, out: str) -> int:
    """Play to `step` on `seed`, then write an annotated frame. Returns adrift."""
    env = build(config_path)
    select = build_action_selector(checkpoint, env).select
    observation, _ = env.reset(seed=seed)
    for _ in range(step):
        observation, _, terminated, truncated, _ = env.step(select(observation, env))
        if terminated or truncated:
            break
    env.render()
    frame = env.renderer.get_frame_array()  # type: ignore[union-attr]
    flags, gaps = breaches(env)
    label = "OUT OF COHERENCY" if flags.any() else "ALL UNITS IN COHERENCY"
    caption = (
        f"{label}  |  {len(gaps)} squad(s) split, {int(flags.sum())} models cut off"
    )
    annotate(env, frame, flags, gaps, caption).save(out)
    return int(flags.sum())


if __name__ == "__main__":
    config_path, checkpoint, seed, step, out = sys.argv[1:6]
    print(out, "adrift:", run(config_path, checkpoint, int(seed), int(step), out))
