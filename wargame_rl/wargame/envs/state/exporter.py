"""StateExporter protocol and EventLogExporter implementation.

The StateExporter protocol defines lifecycle hooks that WargameEnv calls
after reset() and step(). EventLogExporter is the concrete implementation
that records events into an EventLog.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from wargame_rl.wargame.envs.state.event_log import EventLog
from wargame_rl.wargame.envs.state.snapshot import GameStateSnapshot


@runtime_checkable
class StateExporter(Protocol):
    """Protocol for receiving game state at env lifecycle points."""

    def on_reset(self, snapshot: GameStateSnapshot) -> None:
        """Called after env.reset() with the initial state snapshot."""
        ...

    def on_step(self, snapshot: GameStateSnapshot) -> None:
        """Called after env.step() with the post-step state snapshot."""
        ...


class EventLogExporter:
    """Concrete StateExporter that records events into an EventLog.

    Args:
        anchor_interval: Full snapshot anchor frequency. See EventLog.
    """

    def __init__(self, anchor_interval: int = 10) -> None:
        self._log = EventLog(anchor_interval=anchor_interval)

    @property
    def log(self) -> EventLog:
        """Access the underlying EventLog for replay or serialisation."""
        return self._log

    def on_reset(self, snapshot: GameStateSnapshot) -> None:
        """Record episode reset."""
        self._log.record_reset(snapshot)

    def on_step(self, snapshot: GameStateSnapshot) -> None:
        """Record a step event."""
        self._log.record_step(snapshot)
