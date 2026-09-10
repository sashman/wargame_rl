"""Which facade made a recording, and at what cadence.

Two application contexts play the same game (`docs/ddd-envs.md` § Two
application contexts over one domain) and both write the same event log.
The header's provenance says which one wrote it, because the tools that
read a log differ in what they can do with each: `analyze_match`'s per-step
rates assume one snapshot per phase step, and `debug.py` can only step the
phase facade.

This lives in `state/`, not in `per_model/`, because the codec has to
construct the per-model provenance when it decodes a header and `state/`
may not import a facade. `per_model/types.py` re-exports these names.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from wargame_rl.wargame.envs.state.snapshot import EpisodeProvenance

PHASE_FACADE_TAG = "phase"
PER_MODEL_FACADE_TAG = "per_model"

# What one recorded snapshot spans. `phase`: one settled reward window -- the
# phase facade's step -- so the log is schema-identical to the phase facade's
# and every reader works. `decision`: one `step()` call of the per-model
# facade, with the decision on the snapshot; replay, narrate and render
# accept it, `analyze_match` refuses.
Cadence = Literal["phase", "decision"]

CADENCES: tuple[Cadence, ...] = ("phase", "decision")


class PerModelProvenance(EpisodeProvenance):
    """How to boot this episode again, stamped with the facade that played it.

    A subclass rather than a sibling so every `EpisodeProvenance` annotation
    on the exporter and the event log admits it unchanged. `extra="forbid"`
    where the parent is lenient: the parent's leniency is what silently
    swallowed a `facade` key before `decode_provenance` existed.
    """

    model_config = ConfigDict(extra="forbid")

    facade: Literal["per_model"] = "per_model"
    cadence: Cadence = "phase"


def facade_of(provenance: dict[str, Any] | BaseModel) -> str:
    """Which facade an artefact belongs to; an untagged one is the phase facade's."""
    data = provenance.model_dump() if isinstance(provenance, BaseModel) else provenance
    tag = data.get("facade")
    return PHASE_FACADE_TAG if tag is None else str(tag)


def cadence_of(provenance: dict[str, Any] | BaseModel) -> Cadence:
    """What one snapshot of the recording spans; the phase facade's is `phase`."""
    data = provenance.model_dump() if isinstance(provenance, BaseModel) else provenance
    cadence = data.get("cadence", "phase")
    if cadence not in CADENCES:
        raise ValueError(f"unknown recording cadence {cadence!r}")
    return cadence  # type: ignore[no-any-return]


def decode_provenance(raw: dict[str, Any]) -> EpisodeProvenance:
    """Rebuild a header's provenance as the model the facade tag names.

    Refuses an unknown tag by name rather than falling through to the
    lenient parent, which would drop the tag and read the recording as the
    phase facade's -- the silent misread this function exists to prevent.
    """
    tag = raw.get("facade")
    if tag is None:
        return EpisodeProvenance(**raw)
    if tag == PER_MODEL_FACADE_TAG:
        return PerModelProvenance(**raw)
    raise ValueError(f"recording made by an unknown facade {tag!r}")


__all__ = [
    "CADENCES",
    "Cadence",
    "PER_MODEL_FACADE_TAG",
    "PHASE_FACADE_TAG",
    "PerModelProvenance",
    "cadence_of",
    "decode_provenance",
    "facade_of",
]
