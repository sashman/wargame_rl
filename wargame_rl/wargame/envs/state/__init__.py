"""Canonical game-state snapshot: serialisable representation of env state."""

from __future__ import annotations

from wargame_rl.wargame.envs.state.analysis import MatchAnalysis, analyze_match
from wargame_rl.wargame.envs.state.codecs import (
    CODEC_REGISTRY,
    JsonMatchCodec,
    MatchCodec,
    build_codec,
)
from wargame_rl.wargame.envs.state.event_log import EventLog
from wargame_rl.wargame.envs.state.events import (
    MatchEvent,
    ModelDelta,
    ResetEvent,
    StateDelta,
    StepEvent,
    apply_delta,
    compute_delta,
)
from wargame_rl.wargame.envs.state.exporter import EventLogExporter, StateExporter
from wargame_rl.wargame.envs.state.narrator import StepNarrator
from wargame_rl.wargame.envs.state.replay import ReplayController
from wargame_rl.wargame.envs.state.snapshot import (
    GameStateSnapshot,
    JsonEncoder,
    SnapshotEncoder,
    build_snapshot,
    describe_action,
    validate_snapshot,
)

__all__ = [
    "CODEC_REGISTRY",
    "MatchAnalysis",
    "analyze_match",
    "EventLog",
    "EventLogExporter",
    "GameStateSnapshot",
    "JsonEncoder",
    "JsonMatchCodec",
    "MatchCodec",
    "MatchEvent",
    "ModelDelta",
    "ReplayController",
    "ResetEvent",
    "SnapshotEncoder",
    "StateDelta",
    "StateExporter",
    "StepEvent",
    "StepNarrator",
    "apply_delta",
    "build_codec",
    "build_snapshot",
    "compute_delta",
    "describe_action",
    "validate_snapshot",
]
