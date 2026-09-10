"""Watch the per-model facade play, one decision per frame, with no checkpoint.

The tests prove a script plays the same game through both facades and that a
random legal seat cannot break the env. This is the rung they cannot replace:
does one-model-at-a-time play LOOK like the rules? The board is drawn between
decisions with the pending one highlighted -- which models may be named,
which unit is open, which model must act next -- and the HUD's context slot
says what is being asked.

    just play-per-model                                   # golden map pool, squad_march_take
    just play-per-model configs/dev/tiny.yaml random      # random legal decisions
    just play-per-model configs/dev/tiny.yaml set_network # the set network, fresh weights
    just play-per-model <config> <policy> tabletop "" phase   # one frame per phase, today's look
    just record-per-model <config> <policy> out.mp4       # headless, to a file

Opens PAUSED: [.] steps one decision, [Space] plays, [Tab] lists the keys.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

import typer
from loguru import logger
from pydantic_yaml import parse_yaml_file_as

from wargame_rl.wargame.envs.baseline.registry import get_registry
from wargame_rl.wargame.envs.per_model import (
    PerModelAction,
    PerModelEnv,
    PerModelObservation,
    StepKind,
)
from wargame_rl.wargame.envs.renders.human import QuitRequested
from wargame_rl.wargame.envs.renders.v2.control import THREAT_SMOOTHING, THREAT_SPACING
from wargame_rl.wargame.envs.renders.v2.decision import highlight_from
from wargame_rl.wargame.envs.renders.v2.factory import (
    build_backend,
    resolve_theme,
    threat_options,
)
from wargame_rl.wargame.envs.renders.v2.presenters.per_model import (
    PerModelPresenter,
    PerModelRecorder,
)
from wargame_rl.wargame.envs.state import EventLogExporter, JsonMatchCodec
from wargame_rl.wargame.envs.types import WargameEnvConfig
from wargame_rl.wargame.selectors import (
    RANDOM_POLICY,
    SET_NETWORK_POLICY,
    build_per_model_chooser,
)

app = typer.Typer(add_completion=False)

Chooser = Callable[[PerModelObservation], PerModelAction]


def build_chooser(env: PerModelEnv, policy: str, seed: int) -> Chooser:
    """A scripted baseline through the adapter, the random legal seat, the set
    network at fresh weights, or a `.pt` played greedily -- resolved by
    `wargame_rl.wargame.selectors`, the one resolver for both facades."""
    choose = build_per_model_chooser(policy, [env], seed=seed).choose
    return lambda observation: choose([env], [observation])[0]


def describe(action: PerModelAction, observation: PerModelObservation) -> str:
    """The last decision in a few characters, for the caption."""
    if action.kind is StepKind.close_turn:
        return ""
    text = f"m{action.model} "
    if action.kind is StepKind.open:
        text += f"decl {action.value}"
        roll = max(
            float(observation.revealed_advance_roll[action.model]),
            float(observation.revealed_charge_roll[action.model]),
        )
        if roll > 0:
            text += f" r{roll:g}"
    elif action.kind is StepKind.target:
        text += "declined" if action.value < 0 else f"-> u{action.value}"
    else:
        text += f"act {action.value}"
    return text


def play_episode(
    env: PerModelEnv,
    presenter: PerModelPresenter | PerModelRecorder,
    chooser: Chooser,
    *,
    seed: int,
    cadence: str,
) -> None:
    """One episode, rendered at the chosen cadence; pause and step honoured."""
    observation, info = env.reset(seed=seed)
    presenter.setup(env)
    controls = presenter.controls
    detail = ""
    done = False
    interactive = isinstance(presenter, PerModelPresenter)
    while not done:
        point = observation.decision
        if cadence == "decision":
            controls.decision = highlight_from(
                point, sub_step=env.sub_step, detail=detail
            )
            presenter.render(env)
            # Paused: keep presenting the same frame so the keys are read,
            # until play resumes or one step is asked for.
            while interactive and controls.paused and not controls.take_step():
                presenter.render(env)
                if presenter.wants_quit:  # type: ignore[union-attr]
                    raise QuitRequested
        action = chooser(observation)
        observation, _reward, done, _truncated, info = env.step(action)
        detail = describe(action, observation)
        if cadence == "phase" and (info.get("reward_settled") or done):
            controls.decision = highlight_from(
                observation.decision, sub_step=env.sub_step, detail=detail
            )
            presenter.render(env)
            while interactive and controls.paused and not controls.take_step():
                presenter.render(env)
                if presenter.wants_quit:  # type: ignore[union-attr]
                    raise QuitRequested
        if interactive and presenter.wants_quit:  # type: ignore[union-attr]
            raise QuitRequested


@app.command()
def play(
    env_config_path: str = typer.Argument(
        "configs/golden/25v25_maps_two_mode.yaml", help="Environment config to play."
    ),
    policy: str = typer.Argument(
        "squad_march_take",
        help="Scripted baseline driving the player, `random` for random legal decisions, `set_network` for the set network at fresh weights, or a per-model checkpoint (`*.pt`) played greedily.",
    ),
    theme: str = typer.Argument("default", help="Renderer theme: default | tabletop."),
    cadence: str = typer.Option(
        "decision",
        help="`decision` draws a frame before every decision; `phase` at every settled reward window, today's look.",
    ),
    out: str = typer.Option(
        "", help="Write an MP4 here instead of opening a window (headless)."
    ),
    show_threat_range: bool = typer.Option(False, "--threat-range"),
    show_engagement_range: bool = typer.Option(False, "--engagement-range"),
    show_threat_field: bool = typer.Option(False, "--threat-field"),
    threat_grid: float = typer.Option(THREAT_SPACING),
    threat_smoothing: int = typer.Option(THREAT_SMOOTHING),
    episodes: int = typer.Option(0, help="Episodes to play; 0 plays until you quit."),
    seed: int = typer.Option(700000, help="Seed of the first episode."),
    events: str = typer.Option(
        "",
        help="Also write the LAST episode's event log here, at the `--cadence` given.",
    ),
    fps: int = typer.Option(4, help="Frames per second."),
    backend: str = typer.Option("pillow", help="Drawing backend."),
) -> None:
    """Play episodes in a window until Esc, or record them to a file."""
    if (
        policy not in (RANDOM_POLICY, SET_NETWORK_POLICY)
        and not policy.endswith(".pt")
        and policy not in get_registry()
    ):
        raise typer.BadParameter(
            f"unknown policy {policy!r}; try `random`, `{SET_NETWORK_POLICY}` or one "
            f"of {sorted(get_registry())}"
        )
    if cadence not in ("decision", "phase"):
        raise typer.BadParameter("cadence must be `decision` or `phase`")
    config = parse_yaml_file_as(WargameEnvConfig, env_config_path)
    options = threat_options(
        show_threat_range,
        show_engagement_range,
        threat_grid,
        threat_smoothing,
        show_threat_field=show_threat_field,
    )
    presenter: PerModelPresenter | PerModelRecorder
    if out:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        presenter = PerModelRecorder(
            build_backend(backend), resolve_theme(theme), options
        )
        episodes = episodes or 1
    else:
        presenter = PerModelPresenter(
            build_backend(backend), resolve_theme(theme), options
        )
    exporter = EventLogExporter() if events else None
    env = PerModelEnv(
        config,
        state_exporters=[exporter] if exporter is not None else None,
        record_cadence="decision" if cadence == "decision" else "phase",
    )
    env.driver_label = policy
    env.metadata = {**env.metadata, "render_fps": fps}
    presenter.run_label = f"{policy} · {env_config_path.split('/')[-1]} · per-model"
    chooser = build_chooser(env, policy, seed)

    logger.info(f"Playing {env_config_path} with {policy}, one frame per {cadence}.")
    episode = 0
    try:
        while episodes == 0 or episode < episodes:
            play_episode(env, presenter, chooser, seed=seed + episode, cadence=cadence)
            logger.info(
                f"episode {episode}: VP {env.player_vp}-{env.opponent_vp}, "
                f"reward {env.episode_reward:.2f}, {env.episode_step} decisions, "
                f"{len(env.divergences)} divergences from the phase facade"
            )
            episode += 1
    except (QuitRequested, KeyboardInterrupt):
        logger.info("Closing.")
    finally:
        if isinstance(presenter, PerModelRecorder) and presenter.frames:
            presenter.export_mp4(out, fps=fps)
            logger.info(f"Wrote {len(presenter.frames)} frames to {out}.")
        if exporter is not None and exporter.log.events:
            Path(events).parent.mkdir(parents=True, exist_ok=True)
            Path(events).write_bytes(JsonMatchCodec().encode(exporter.log))
            logger.info(f"Wrote the last episode's event log to {events}.")
        presenter.close()


if __name__ == "__main__":
    app()
