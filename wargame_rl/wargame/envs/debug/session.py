"""The driving loop for a hand-stepped match.

Separate from `debug.py` so it can be tested without a window: the loop only
needs something that renders and a `DebugControls` to read, both of which a test
can supply.

The loop presents on *every* pass, including passes where nothing advances —
that is what keeps the window responsive while paused, and what lets a key
pressed during a pause be seen at all.
"""

from __future__ import annotations

from collections import deque

from loguru import logger

from wargame_rl.wargame.envs.baseline.evaluate import ActionSelector
from wargame_rl.wargame.envs.debug.undo import DEFAULT_DEPTH, UndoStack
from wargame_rl.wargame.envs.renders.human import QuitRequested
from wargame_rl.wargame.envs.renders.renderer import Renderer
from wargame_rl.wargame.envs.renders.v2.presenters.debug import (
    DebugControls,
    MoveRecord,
)
from wargame_rl.wargame.envs.state.snapshot import describe_action
from wargame_rl.wargame.envs.types import WargameEnvAction
from wargame_rl.wargame.envs.types.game_timing import BattlePhase
from wargame_rl.wargame.envs.wargame import WargameEnv


def record_moves(
    env: WargameEnv, action: WargameEnvAction, before: list[tuple[float, float]]
) -> dict[int, MoveRecord]:
    """Where each player model meant to land, against where it ended up.

    Built here rather than in the presenter for two reasons: the env keeps only
    the *most recent* action, so by the time a shooting step is on screen the
    movement that produced these positions is gone; and decoding an action needs
    the action handler, which the renderer layer may not import.

    The gap between intended and actual is board clamping and `resolve_move`
    collision displacement — the one thing about a move that no other view in
    the renderer shows.
    """
    handler = env.player_action_handler
    shooting = handler.shooting_slice
    # With no shoot targets registered there is no shooting slice; an empty range
    # past the end of the action space means no action is ever read as a shot.
    shoot_start = shooting.start if shooting is not None else env.n_actions
    shoot_end = shooting.end if shooting is not None else env.n_actions
    moves: dict[int, MoveRecord] = {}
    for index, model in enumerate(env.player_models):
        if index >= len(action.actions):
            break
        chosen = int(action.actions[index])
        displacement = handler.decode_action(chosen)
        start = before[index]
        moves[index] = MoveRecord(
            text=describe_action(
                chosen,
                env.config.n_movement_angles,
                env.config.n_speed_bins,
                shoot_start,
                shoot_end,
            ),
            intended=(
                start[0] + float(displacement[0]),
                start[1] + float(displacement[1]),
            ),
            actual=(float(model.location[0]), float(model.location[1])),
        )
    return moves


def run_session(
    env: WargameEnv,
    renderer: Renderer,
    controls: DebugControls,
    select: ActionSelector,
    *,
    seed: int,
    undo_depth: int = DEFAULT_DEPTH,
) -> WargameEnv:
    """Step one episode under the user's control until they quit.

    Returns the env the session ended on, which is not necessarily the one
    passed in — stepping back replaces it with a restored copy, and the caller
    needs the current one to close and to report on.

    Quitting is the normal way out of a debug session, so the `QuitRequested`
    the renderer raises is caught here rather than propagated.
    """
    undo = UndoStack(undo_depth)
    # The move records live on the controls, not in the env, so they need their
    # own history — otherwise a rewind leaves the panel describing a step that
    # has just been undone. Pushed and popped in lockstep with the env states.
    move_history: deque[dict[int, MoveRecord]] = deque(maxlen=undo_depth)
    observation, _info = env.reset(seed=seed)
    done = False

    try:
        while True:
            renderer.render(env)

            if controls.step_back:
                controls.step_back = False
                previous = undo.pop()
                if previous is None:
                    logger.info("Nothing to step back to.")
                else:
                    # The restored state predates the step that produced `done`,
                    # so a finished episode becomes live again.
                    env, done = previous, False
                    observation = env.observation
                    controls.moves = move_history.pop() if move_history else {}
                continue

            if done:
                # Nothing to step forward into; drop the request rather than let
                # it fire later against a rewound state.
                controls.step_once = False
                continue
            if controls.paused and not controls.step_once:
                continue
            controls.step_once = False

            undo.push(env)
            move_history.append(dict(controls.moves))
            action = select(observation, env)
            # Captured before the step, because the step is what moves them.
            moving = env.game_clock_state.phase is BattlePhase.movement
            before = [
                (float(m.location[0]), float(m.location[1])) for m in env.player_models
            ]
            observation, _reward, terminated, truncated, _info = env.step(action)
            if moving:
                controls.moves = record_moves(env, action, before)
            done = terminated or truncated
            if done:
                logger.info(
                    f"Episode over: VP {env.player_vp}-{env.opponent_vp}, "
                    f"reward {env.episode_reward:.2f}. [,] steps back into it."
                )
    except QuitRequested:
        logger.info("Closing.")
    return env
