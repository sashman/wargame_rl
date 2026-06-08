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
    """JSONL codec for match event logs.

    Serialises each event as a separate JSON line (newline-delimited JSON).
    First line is a header with version and anchor_interval metadata.
    """

    def encode(self, event_log: EventLog) -> bytes:
        """Serialise an EventLog to JSONL bytes (one JSON object per line)."""
        lines: list[str] = []
        header: dict[str, Any] = {
            "type": "header",
            "version": "1.0",
            "anchor_interval": event_log.anchor_interval,
        }
        lines.append(json.dumps(header))
        for event in event_log.events:
            raw = _event_adapter.dump_python(event, mode="python")
            lines.append(json.dumps(raw, default=_json_default))
        return ("\n".join(lines) + "\n").encode("utf-8")

    def decode(self, data: bytes) -> EventLog:
        """Deserialise JSONL bytes into an EventLog."""
        text = data.decode("utf-8")
        lines = [line for line in text.splitlines() if line.strip()]

        header = json.loads(lines[0])
        anchor_interval = header.get("anchor_interval", 10)
        log = EventLog(anchor_interval=anchor_interval)

        for line in lines[1:]:
            raw = json.loads(line)
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
        return "application/x-ndjson"


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
