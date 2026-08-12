"""Rewind a live env by keeping deep copies of it.

`WargameEnv.load_state(snapshot)` already exists and is the obvious thing to
reach for, but it is lossy in three ways that matter here:

* the combat RNG is never restored (only reseeded in `reset`), so a rewound step
  rolls different dice;
* terrain is not restored — `load_state` never calls `set_terrain`;
* the reward-shaping memory (`previous_closest_objective_distance`,
  `best_closest_objective_distance`) is cleared *deliberately* by
  `state/restore.py`.

The point of rewinding here is to re-run a step and compare, so a restore that
changes the reward for an identical action defeats the feature. That is not
hypothetical: on the fixture in `tests/test_debug_session.py`, replaying one step
through `load_state` scores **0.0 against the live env's -0.3** for the very same
action, because the cleared shaping memory zeroes `closest_objective`'s
`base_penalty`. A deep copy is lossless, carries the RNG with it, and makes
holding the dice free rather than a plumbing exercise.

The renderer is detached across the copy because a pygame window is not
copyable; everything else the env holds is plain data.
"""

from __future__ import annotations

import copy
from collections import deque

from wargame_rl.wargame.envs.wargame import WargameEnv

DEFAULT_DEPTH = 100


def capture_state(env: WargameEnv) -> WargameEnv:
    """Deep-copy `env` so it can be restored later.

    The copy is a whole env rather than a slice of one: the opponent policy
    holds a reference back to the env it drives, so copying the object graph in
    a single call is what keeps the two consistent.
    """
    renderer = env.renderer
    env.renderer = None
    try:
        return copy.deepcopy(env)
    finally:
        env.renderer = renderer


class UndoStack:
    """A bounded history of env states, newest last."""

    def __init__(self, depth: int = DEFAULT_DEPTH) -> None:
        self._states: deque[WargameEnv] = deque(maxlen=depth)

    def __len__(self) -> int:
        return len(self._states)

    def push(self, env: WargameEnv) -> None:
        """Record the state to come back to, before stepping past it."""
        self._states.append(capture_state(env))

    def pop(self) -> WargameEnv | None:
        """The most recently recorded state, or None when there is no history."""
        if not self._states:
            return None
        return self._states.pop()

    def clear(self) -> None:
        """Drop the history — used when a new episode starts."""
        self._states.clear()
