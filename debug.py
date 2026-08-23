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

import typer
from loguru import logger
from pydantic_yaml import parse_yaml_file_as

from wargame_rl.wargame.envs.baseline.evaluate import ActionSelector
from wargame_rl.wargame.envs.debug import run_session
from wargame_rl.wargame.envs.debug.reproduce import (
    build_env,
    provenance_of,
    read_log,
    recorded_actions,
)
from wargame_rl.wargame.envs.debug.reproduce import (
    reset_options as reproduce_reset_options,
)
from wargame_rl.wargame.envs.renders.v2.control import THREAT_SMOOTHING, THREAT_SPACING
from wargame_rl.wargame.envs.renders.v2.factory import (
    build_backend,
    resolve_theme,
    threat_options,
)
from wargame_rl.wargame.envs.renders.v2.presenters.debug import (
    DebugControls,
    DebugPresenter,
)
from wargame_rl.wargame.envs.types import WargameEnvConfig
from wargame_rl.wargame.envs.wargame import WargameEnv
from wargame_rl.wargame.selectors import build_action_selector

app = typer.Typer(add_completion=False)

# Sentinel: a recording names its own driver, but only if the user did not.
_DEFAULT_DRIVER = "squad_march_shoot"


def build_selector_for(driver: str, env: WargameEnv) -> tuple[ActionSelector, str]:
    """Resolve a baseline name or a checkpoint path to a selector and a label.

    Torch is imported only on the checkpoint branch — a scripted session should
    not pay for it, which is why `wargame_rl.wargame.selectors` defers it.
    """
    try:
        resolved = build_action_selector(driver, env)
    except ValueError as error:
        raise typer.BadParameter(str(error)) from error
    return resolved.select, resolved.label


@app.command()
def debug(
    env_config_path: str = typer.Argument(
        "configs/golden/25v25_shooting_opponent.yaml",
        help="Environment config to step through.",
    ),
    driver: str = typer.Argument(
        _DEFAULT_DRIVER,
        help="Scripted baseline name, or a path to a .ckpt, driving the player.",
    ),
    theme: str = typer.Argument("default", help="Renderer theme: default | tabletop."),
    show_threat_range: bool = typer.Option(
        False,
        "--threat-range",
        help="Outline the ground each side can shoot: weapon range intersected with line of sight.",
    ),
    show_engagement_range: bool = typer.Option(
        False,
        "--engagement-range",
        help="Shade each side's engagement range — inside it, a model may not shoot at all.",
    ),
    threat_grid: float = typer.Option(
        THREAT_SPACING,
        help="Sampling grid for the threat sweep, in inches. Coarser is cheaper and blurs small cover pockets.",
    ),
    threat_smoothing: int = typer.Option(
        THREAT_SMOOTHING,
        help="Chaikin smoothing passes on the threat outline. 0 draws the sampled shape exactly.",
    ),
    seed: int = typer.Option(700000, help="Episode seed."),
    from_recording: str | None = typer.Option(
        None,
        help="Recreate the episode a recording came from, exactly. Overrides "
        "the config, the seed and (unless one is given) the driver.",
    ),
    follow: bool = typer.Option(
        True,
        help="With --from-recording, replay the recording's own actions until "
        "you change something. --no-follow drives with the policy instead, "
        "which reproduces the scenario but not necessarily the match.",
    ),
    fps: int = typer.Option(4, help="Frames per second while playing."),
    backend: str = typer.Option("pillow", help="Drawing backend."),
    undo_depth: int = typer.Option(100, help="How many steps can be taken back."),
) -> None:
    """Open a window on one episode and step it under your control."""
    controls = DebugControls()
    renderer = DebugPresenter(
        build_backend(backend),
        controls,
        resolve_theme(theme),
        threat_options(
            show_threat_range, show_engagement_range, threat_grid, threat_smoothing
        ),
    )
    source = env_config_path.split("/")[-1]
    reset_options: dict[str, int] | None = None
    recorded: list[list[int] | None] | None = None

    if from_recording is not None:
        # Everything the episode needs comes off the recording -- the config
        # included, so a scenario file edited since it was made cannot quietly
        # reproduce something that merely looks like it.
        log = read_log(from_recording)
        provenance = provenance_of(log, from_recording)
        env = build_env(provenance, renderer=renderer)
        if follow:
            # The actions matter more than the driver for a *training*
            # recording: the network that played it was mid-training and was
            # never saved, so no driver can reproduce its decisions.
            recorded = recorded_actions(log)
        reset_options = reproduce_reset_options(provenance)
        # `run_session` owns the reset; seeding there would discard the
        # generator state `build_env` just installed.
        seed = None  # type: ignore[assignment]
        source = from_recording.split("/")[-1]
        if driver == _DEFAULT_DRIVER and provenance.driver:
            driver = provenance.driver
    else:
        config = parse_yaml_file_as(WargameEnvConfig, env_config_path)
        # `build_info` is left at its default: the info dict is part of what
        # there is to inspect, and training envs are the only callers that turn
        # it off.
        env = WargameEnv(config=config, renderer=renderer)

    # The interactive presenter reads the frame rate off the view's metadata in
    # `setup`, so it has to be set before `reset` triggers that call.
    env.metadata = {**WargameEnv.metadata, "render_fps": fps}

    select, label = build_selector_for(driver, env)
    renderer.run_label = f"{label} · {source}"

    driving = (
        f"replaying {len(recorded) - 1} recorded steps, then {label}"
        if recorded
        else label
    )
    logger.info(
        f"Debugging {source} with {driving}. Opens paused — "
        "[.] steps, [,] steps back, [Space] plays, [Tab] lists the keys."
    )
    try:
        env = run_session(
            env,
            renderer,
            controls,
            select,
            seed=seed,
            undo_depth=undo_depth,
            reset_options=reset_options,
            recorded=recorded,
        )
    except KeyboardInterrupt:
        logger.info("Interrupted.")
    finally:
        renderer.close()
        env.close()


if __name__ == "__main__":
    app()
