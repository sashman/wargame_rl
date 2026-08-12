"""Step a match by hand, one action at a time, and rewind when it surprises you.

`play.py` watches a scripted policy at a fixed frame rate; `simulate.py` runs a
checkpoint through the shared agent loop. Neither lets you stop on a step and
interrogate it, and neither can go back. This drives the loop itself so the
match advances only when asked, and keeps deep copies of the env so any step can
be taken back and re-run.

    just debug                                        # golden 25v25, squad_march_shoot
    just debug configs/dev/tiny.yaml random           # a small board, random actions
    just debug configs/golden/25v25_curriculum.yaml checkpoints/<run>/last.ckpt

Both a scripted baseline name and a checkpoint path are accepted — they reduce
to the same `ActionSelector`, which is what `measure-checkpoint` and
`measure-baselines` already score through.

A config with `skip_phases: []` steps one *sub-phase* at a time (command,
movement, shooting, charge, fight) rather than one round, which is the finest
granularity the env exposes without new code.

Keys are listed in the window itself — press [Tab].
"""

from __future__ import annotations

from pathlib import Path

import typer
from loguru import logger
from pydantic_yaml import parse_yaml_file_as

from wargame_rl.wargame.envs.baseline.evaluate import ActionSelector, selector_for
from wargame_rl.wargame.envs.baseline.registry import (
    build_baseline_policy,
    get_registry,
)
from wargame_rl.wargame.envs.debug import run_session
from wargame_rl.wargame.envs.renders.v2.factory import build_backend, resolve_theme
from wargame_rl.wargame.envs.renders.v2.presenters.debug import (
    DebugControls,
    DebugPresenter,
)
from wargame_rl.wargame.envs.types import WargameEnvConfig
from wargame_rl.wargame.envs.wargame import WargameEnv

app = typer.Typer(add_completion=False)


def build_selector_for(driver: str, env: WargameEnv) -> tuple[ActionSelector, str]:
    """Resolve a baseline name or a checkpoint path to a selector and a label.

    Torch is imported only on the checkpoint branch — a scripted session should
    not pay for it.
    """
    path = Path(driver)
    if path.suffix == ".ckpt" or path.exists():
        if not path.exists():
            raise typer.BadParameter(f"no checkpoint at {driver!r}")
        from scripts.measure_checkpoint import build_selector

        select, _net = build_selector(str(path), env)
        return select, path.parent.name

    available = sorted(get_registry())
    if driver not in available:
        raise typer.BadParameter(
            f"unknown policy {driver!r}; try one of {available}, or a .ckpt path"
        )
    return selector_for(build_baseline_policy(driver)), driver


@app.command()
def debug(
    env_config_path: str = typer.Argument(
        "configs/golden/25v25_shooting_opponent.yaml",
        help="Environment config to step through.",
    ),
    driver: str = typer.Argument(
        "squad_march_shoot",
        help="Scripted baseline name, or a path to a .ckpt, driving the player.",
    ),
    theme: str = typer.Argument("default", help="Renderer theme: default | tabletop."),
    seed: int = typer.Option(700000, help="Episode seed."),
    fps: int = typer.Option(4, help="Frames per second while playing."),
    backend: str = typer.Option("pillow", help="Drawing backend."),
    undo_depth: int = typer.Option(100, help="How many steps can be taken back."),
) -> None:
    """Open a window on one episode and step it under your control."""
    config = parse_yaml_file_as(WargameEnvConfig, env_config_path)
    controls = DebugControls()
    renderer = DebugPresenter(build_backend(backend), controls, resolve_theme(theme))
    # `build_info` is left at its default: the info dict is part of what there is
    # to inspect, and training envs are the only callers that turn it off.
    env = WargameEnv(config=config, renderer=renderer)
    # The interactive presenter reads the frame rate off the view's metadata in
    # `setup`, so it has to be set before `reset` triggers that call.
    env.metadata = {**WargameEnv.metadata, "render_fps": fps}

    select, label = build_selector_for(driver, env)
    renderer.run_label = f"{label} · {env_config_path.split('/')[-1]}"

    logger.info(
        f"Debugging {env_config_path} with {label}. Opens paused — "
        "[.] steps, [,] steps back, [Space] plays, [Tab] lists the keys."
    )
    try:
        env = run_session(
            env, renderer, controls, select, seed=seed, undo_depth=undo_depth
        )
    except KeyboardInterrupt:
        logger.info("Interrupted.")
    finally:
        renderer.close()
        env.close()


if __name__ == "__main__":
    app()
