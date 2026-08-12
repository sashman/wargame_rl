"""The driving loop for a hand-stepped match.

Separate from `debug.py` so it can be tested without a window: the loop only
needs something that renders and a `DebugControls` to read, both of which a test
can supply.

The loop presents on *every* pass, including passes where nothing advances —
that is what keeps the window responsive while paused, and what lets a key
pressed during a pause be seen at all.
"""

from __future__ import annotations

from loguru import logger

from wargame_rl.wargame.envs.baseline.evaluate import ActionSelector
from wargame_rl.wargame.envs.debug.undo import DEFAULT_DEPTH, UndoStack
from wargame_rl.wargame.envs.renders.human import QuitRequested
from wargame_rl.wargame.envs.renders.renderer import Renderer
from wargame_rl.wargame.envs.renders.v2.presenters.debug import DebugControls
from wargame_rl.wargame.envs.wargame import WargameEnv


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
            action = select(observation, env)
            observation, _reward, terminated, truncated, _info = env.step(action)
            done = terminated or truncated
            if done:
                logger.info(
                    f"Episode over: VP {env.player_vp}-{env.opponent_vp}, "
                    f"reward {env.episode_reward:.2f}. [,] steps back into it."
                )
    except QuitRequested:
        logger.info("Closing.")
    return env
