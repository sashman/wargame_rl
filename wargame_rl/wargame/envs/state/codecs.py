"""Pluggable codec registry for serialising and deserialising EventLogs.

Extends the SnapshotEncoder pattern (SGS-04) with full match-level codecs
that handle the entire event stream, not just individual snapshots.
"""

from __future__ import annotations

import json
from typing import Any, Protocol, runtime_checkable

from pydantic import TypeAdapter

from wargame_rl.wargame.envs.state.event_log import EventLog
from wargame_rl.wargame.envs.state.events import MatchEvent, ResetEvent, StepEvent

_event_adapter: TypeAdapter[MatchEvent] = TypeAdapter(MatchEvent)


@runtime_checkable
class MatchCodec(Protocol):
    """Protocol for encoding/decoding complete match event logs."""

    def encode(self, event_log: EventLog) -> bytes:
        """Serialise an EventLog to bytes."""
        ...

    def decode(self, data: bytes) -> EventLog:
        """Deserialise bytes back into an EventLog."""
        ...

    def content_type(self) -> str:
        """MIME type for the encoded format."""
        ...


class JsonMatchCodec:
    """JSON codec for match event logs.

    Serialises the full event list as a JSON array with metadata.
    """

    def encode(self, event_log: EventLog) -> bytes:
        """Serialise an EventLog to JSON bytes."""
        payload: dict[str, Any] = {
            "version": "1.0",
            "anchor_interval": event_log.anchor_interval,
            "events": [
                _event_adapter.dump_python(e, mode="python") for e in event_log.events
            ],
        }
        return json.dumps(payload, default=_json_default).encode("utf-8")

    def decode(self, data: bytes) -> EventLog:
        """Deserialise JSON bytes into an EventLog."""
        payload = json.loads(data)
        anchor_interval = payload.get("anchor_interval", 10)
        log = EventLog(anchor_interval=anchor_interval)

        raw_events = payload["events"]
        for raw in raw_events:
            event = _event_adapter.validate_python(raw)
            if isinstance(event, ResetEvent):
                log.record_reset(event.snapshot)
            elif isinstance(event, StepEvent):
                if event.anchor is not None:
                    log.record_step(event.anchor)
                else:
                    from wargame_rl.wargame.envs.state.events import apply_delta

                    assert log._last_snapshot is not None
                    reconstructed = apply_delta(log._last_snapshot, event.delta)
                    log.record_step(reconstructed)
        return log

    def content_type(self) -> str:
        return "application/json"


def _json_default(obj: object) -> Any:
    """Handle non-standard types during JSON serialisation."""
    if hasattr(obj, "model_dump"):
        return obj.model_dump()  # type: ignore[union-attr]
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

CODEC_REGISTRY: dict[str, type[MatchCodec]] = {
    "json": JsonMatchCodec,  # type: ignore[dict-item]
}


def build_codec(codec_type: str) -> MatchCodec:
    """Build a codec instance by registry key.

    Raises:
        ValueError: If codec_type is not registered.
    """
    cls = CODEC_REGISTRY.get(codec_type)
    if cls is None:
        available = ", ".join(sorted(CODEC_REGISTRY.keys()))
        raise ValueError(f"Unknown codec type '{codec_type}'. Available: {available}")
    return cls()
