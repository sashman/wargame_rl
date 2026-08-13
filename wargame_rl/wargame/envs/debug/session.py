"""The driving loop for a hand-stepped match.

Separate from `debug.py` so it can be tested without a window: the loop only
needs something that renders and a `DebugControls` to read, both of which a test
can supply.

The loop presents on *every* pass, including passes where nothing advances —
that is what keeps the window responsive while paused, and what lets a key
pressed during a pause be seen at all.
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np
from loguru import logger

from wargame_rl.wargame.envs.baseline.evaluate import ActionSelector
from wargame_rl.wargame.envs.debug.overrides import OverridableOpponentPolicy
from wargame_rl.wargame.envs.debug.undo import DEFAULT_DEPTH, UndoStack
from wargame_rl.wargame.envs.renders.human import QuitRequested
from wargame_rl.wargame.envs.renders.renderer import Renderer
from wargame_rl.wargame.envs.renders.v2.presenters.debug import (
    OPPONENT,
    PLAYER,
    DebugControls,
    MoveRecord,
    Order,
)
from wargame_rl.wargame.envs.state.snapshot import describe_action
from wargame_rl.wargame.envs.types import WargameEnvAction, WargameEnvObservation
from wargame_rl.wargame.envs.types.game_timing import BattlePhase
from wargame_rl.wargame.envs.wargame import WargameEnv
from wargame_rl.wargame.envs.wargame_model import WargameModel

# A click asking for a move: (side, index, board_x, board_y).
OrderRequest = tuple[str, int, float, float]


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


def resolve_order(
    env: WargameEnv, observation: WargameEnvObservation, request: OrderRequest
) -> tuple[tuple[str, int], Order]:
    """Turn a clicked point into the action that gets closest to it.

    The bins are coarse — `n_movement_angles` x `n_speed_bins` — so the model
    will not land where the click was, and `landing` is the honest answer rather
    than the request. That gap is worth seeing: it is the same discretisation the
    policy is choosing under.

    Legality is checked against the action mask the network is given, so an
    order refused here is refused for exactly the reason the agent's would be —
    most often that this is not the movement phase. Only the player's mask is
    published, so an opponent order goes through unchecked.
    """
    side, index, x, y = request
    models = env.player_models if side == PLAYER else env.opponent_models
    model = models[index]
    handler = (
        env.player_action_handler if side == PLAYER else env.opponent_action_handler
    )
    start = (float(model.location[0]), float(model.location[1]))
    delta = (x - start[0], y - start[1])
    action = handler.best_action_toward(delta[0], delta[1], math.hypot(*delta))
    displacement = handler.decode_action(action)
    shooting = handler.shooting_slice
    shoot_start = shooting.start if shooting is not None else env.n_actions
    shoot_end = shooting.end if shooting is not None else env.n_actions

    legal, reason = True, None
    mask = observation.action_mask
    if side == PLAYER and mask is not None and index < len(mask):
        legal = bool(mask[index][action])
        if not legal:
            reason = _refusal(env, model)
    return (side, index), Order(
        action=action,
        text=describe_action(
            action,
            env.config.n_movement_angles,
            env.config.n_speed_bins,
            shoot_start,
            shoot_end,
        ),
        landing=(start[0] + float(displacement[0]), start[1] + float(displacement[1])),
        legal=legal,
        reason=reason,
    )


def _refusal(env: WargameEnv, model: WargameModel) -> str:
    """Why the mask said no, in the terms the user is thinking in.

    "The action mask refuses this move" is true of every refusal and explains
    none of them. The two that actually happen are a casualty — restricted to
    STAY, and the panel would otherwise report a dead model as merely illegal —
    and clicking during a phase that is not movement.
    """
    if not model.is_alive:
        return "Killed — a casualty has only STAY available."
    phase = env.game_clock_state.phase
    if phase is not BattlePhase.movement:
        return (
            f"Not legal now — the phase is {phase.value if phase else 'over'}, "
            "not movement."
        )
    return "The action mask refuses this move."


def apply_orders(
    env: WargameEnv, action: WargameEnvAction, orders: dict[tuple[str, int], Order]
) -> None:
    """Overwrite the policy's choices with the human's, on both sides.

    The player's actions are edited in place before `step`. The opponent's
    cannot be: `run_after_player_action` runs its whole turn *inside* that same
    `step`, so the only seam is the policy, which is wrapped. The wrapper is
    installed fresh each step because a rewind hands back a *different* env
    object — one carrying a deep copy of whatever was installed when it was
    captured.
    """
    for (side, index), order in orders.items():
        if side == PLAYER and order.legal and index < len(action.actions):
            action.actions[index] = order.action
    opponent = {
        index: order.action
        for (side, index), order in orders.items()
        if side == OPPONENT and order.legal
    }
    inner = env.opponent_policy
    if inner is None:
        return
    base = inner.inner if isinstance(inner, OverridableOpponentPolicy) else inner
    env.set_opponent_policy(
        OverridableOpponentPolicy(base, opponent) if opponent else base
    )


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
    controls.undo_depth = 0
    done = False

    # Dice for a reroll come off the session's own generator rather than the
    # clock, so a whole session — orders, redos and all — replays from its seed.
    redo_rng = np.random.default_rng(seed)

    try:
        while True:
            renderer.render(env)

            if controls.order_at is not None:
                request, controls.order_at = controls.order_at, None
                key, order = resolve_order(env, observation, request)
                controls.orders[key] = order
                if not order.legal:
                    logger.info(f"Refused: {order.reason}")
                continue

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
                controls.undo_depth = len(undo)
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
            controls.undo_depth = len(undo)
            action = select(observation, env)
            apply_orders(env, action, controls.orders)
            controls.orders.clear()
            if controls.reroll_dice:
                env.reseed_combat(int(redo_rng.integers(0, 2**31)))
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
