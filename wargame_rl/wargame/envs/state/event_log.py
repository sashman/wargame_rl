"""Append-only event log for recording complete match histories.

The EventLog accumulates MatchEvents in order, inserting full snapshot
anchors at configurable intervals to allow efficient random-access seek.
"""

from __future__ import annotations

from wargame_rl.wargame.envs.state.events import (
    MatchEvent,
    ResetEvent,
    StepEvent,
    apply_delta,
    compute_delta,
)
from wargame_rl.wargame.envs.state.snapshot import EpisodeProvenance, GameStateSnapshot


class EventLog:
    """Append-only ordered event log for a single episode.

    Stores a sequence of MatchEvents and provides random-access state
    reconstruction via anchored snapshots + forward delta application.

    Args:
        anchor_interval: Insert a full snapshot anchor every N steps.
            Lower values trade storage for faster seek. Default 10.
    """

    def __init__(
        self,
        anchor_interval: int = 10,
        provenance: EpisodeProvenance | None = None,
    ) -> None:
        self._anchor_interval = anchor_interval
        self._events: list[MatchEvent] = []
        self._last_snapshot: GameStateSnapshot | None = None
        self._provenance = provenance

    @property
    def events(self) -> list[MatchEvent]:
        """All recorded events in order."""
        return self._events

    @property
    def anchor_interval(self) -> int:
        return self._anchor_interval

    @property
    def provenance(self) -> EpisodeProvenance | None:
        """How to boot this episode again; None on recordings made before it
        was written down, which therefore cannot be reproduced."""
        return self._provenance

    def set_provenance(self, provenance: EpisodeProvenance) -> None:
        """Record the episode's inputs. Set at reset, before any step."""
        self._provenance = provenance

    def __len__(self) -> int:
        return len(self._events)

    def record_reset(self, snapshot: GameStateSnapshot) -> None:
        """Record an episode reset with its initial full snapshot."""
        self._events = [ResetEvent(snapshot=snapshot)]
        self._last_snapshot = snapshot

    def record_step(self, snapshot: GameStateSnapshot) -> None:
        """Record a step by computing delta from previous state.

        Automatically inserts a full anchor snapshot at the configured interval.
        """
        if self._last_snapshot is None:
            raise RuntimeError(
                "record_step called before record_reset — "
                "no previous snapshot available"
            )

        delta = compute_delta(self._last_snapshot, snapshot)
        is_anchor = (snapshot.step % self._anchor_interval) == 0
        event = StepEvent(
            delta=delta,
            anchor=snapshot if is_anchor else None,
        )
        self._events.append(event)
        self._last_snapshot = snapshot

    def snapshot_at(self, step: int) -> GameStateSnapshot:
        """Reconstruct the game state at the given step number.

        Finds the nearest anchor at or before the requested step, then
        applies forward deltas to reach the target.

        Raises:
            ValueError: If step is not within the recorded range.
        """
        if not self._events:
            raise ValueError("EventLog is empty")

        reset_event = self._events[0]
        assert isinstance(reset_event, ResetEvent)

        if step == reset_event.snapshot.step:
            return reset_event.snapshot

        anchor_snapshot = reset_event.snapshot
        anchor_event_idx = 0

        for i in range(1, len(self._events)):
            event = self._events[i]
            assert isinstance(event, StepEvent)
            if event.delta.step > step:
                break
            if event.anchor is not None and event.delta.step <= step:
                anchor_snapshot = event.anchor
                anchor_event_idx = i

        state = anchor_snapshot
        start = anchor_event_idx + 1 if anchor_event_idx > 0 else 1
        for i in range(start, len(self._events)):
            event = self._events[i]
            assert isinstance(event, StepEvent)
            if event.delta.step > step:
                break
            state = apply_delta(state, event.delta)

        if state.step != step:
            raise ValueError(
                f"Step {step} not found in event log "
                f"(range: {reset_event.snapshot.step}–{self._last_step})"
            )
        return state

    @property
    def _last_step(self) -> int:
        """Step number of the most recently recorded event."""
        if not self._events:
            return -1
        last = self._events[-1]
        if isinstance(last, ResetEvent):
            return last.snapshot.step
        assert isinstance(last, StepEvent)
        return last.delta.step
