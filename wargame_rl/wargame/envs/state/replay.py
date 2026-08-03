"""Deterministic replay controller for reconstructing historical game states.

Given an EventLog, the ReplayController can seek to any recorded step by
finding the nearest anchor snapshot and applying forward deltas.
"""

from __future__ import annotations

from wargame_rl.wargame.envs.state.event_log import EventLog
from wargame_rl.wargame.envs.state.snapshot import GameStateSnapshot


class ReplayController:
    """Reconstructs game states from an EventLog.

    Supports random-access seek to any step recorded in the log,
    sequential iteration, and range queries.

    Args:
        event_log: A populated EventLog containing at least a reset event.
    """

    def __init__(self, event_log: EventLog) -> None:
        if len(event_log) == 0:
            raise ValueError("Cannot replay an empty EventLog")
        self._log = event_log

    @property
    def event_log(self) -> EventLog:
        return self._log

    @property
    def first_step(self) -> int:
        """First recorded step number (from the reset event)."""
        from wargame_rl.wargame.envs.state.events import ResetEvent

        first = self._log.events[0]
        assert isinstance(first, ResetEvent)
        return first.snapshot.step

    @property
    def last_step(self) -> int:
        """Last recorded step number."""
        from wargame_rl.wargame.envs.state.events import ResetEvent, StepEvent

        last = self._log.events[-1]
        if isinstance(last, ResetEvent):
            return last.snapshot.step
        assert isinstance(last, StepEvent)
        return last.delta.step

    @property
    def total_steps(self) -> int:
        """Number of step events (excluding the reset)."""
        return len(self._log) - 1

    def seek(self, step: int) -> GameStateSnapshot:
        """Reconstruct the game state at the given step.

        Raises:
            ValueError: If step is outside the recorded range.
        """
        return self._log.snapshot_at(step)

    def iter_snapshots(self) -> list[GameStateSnapshot]:
        """Reconstruct all snapshots in order (reset + every step).

        Returns a list of GameStateSnapshot for the entire episode.
        Useful for full replay or analysis pipelines.
        """
        from wargame_rl.wargame.envs.state.events import (
            ResetEvent,
            StepEvent,
            apply_delta,
        )

        if not self._log.events:
            return []

        first = self._log.events[0]
        assert isinstance(first, ResetEvent)
        snapshots = [first.snapshot]
        current = first.snapshot

        for event in self._log.events[1:]:
            assert isinstance(event, StepEvent)
            # Prefer the anchor when present: it is authoritative and resyncs the
            # walk, so a field missing from a delta cannot persist for the rest
            # of the episode.
            current = (
                event.anchor
                if event.anchor is not None
                else apply_delta(current, event.delta)
            )
            snapshots.append(current)

        return snapshots

    def snapshot_range(self, start_step: int, end_step: int) -> list[GameStateSnapshot]:
        """Reconstruct snapshots for a range of steps [start_step, end_step].

        Raises:
            ValueError: If start_step > end_step or steps are out of range.
        """
        if start_step > end_step:
            raise ValueError(
                f"start_step ({start_step}) must be <= end_step ({end_step})"
            )

        snapshots: list[GameStateSnapshot] = []
        for step in range(start_step, end_step + 1):
            try:
                snapshots.append(self.seek(step))
            except ValueError:
                break
        return snapshots
