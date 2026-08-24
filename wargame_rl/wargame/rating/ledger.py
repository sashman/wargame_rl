"""The rating ledger: raw per-layout legs, keyed by the scenario they were played on.

**A rating is meaningful only within one scenario.** A number measured on
`configs/golden/25v25_shooting_opponent.yaml` says nothing about the real map
pool, and the whole top of `CLAUDE.md` is a monument to numbers quoted from a
different environment. So the ledger *enforces* that rather than documenting it:
two fingerprints in one table is **refused**, not warned about. A warning in a
log is what the TF32 and `last.ckpt` episodes show gets ignored.

**The ledger stores raw legs, not fitted ratings.** Three reasons, each on its
own sufficient: the bootstrap resamples layouts and needs the rows; adding one
entrant would otherwise mean replaying every pairing; and recalibrating the
margin scale would mean replaying everything. `just elo-table` fits on read.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from wargame_rl.wargame.envs.types import WargameEnvConfig
from wargame_rl.wargame.rating.arena import LegResult
from wargame_rl.wargame.rating.entrant import Entrant
from wargame_rl.wargame.rating.schedule import Leg, Seat, Zone

LEDGER_ROOT = Path("ratings")

# Not part of the scenario, and each for its own reason.
_DROPPED_FIELDS = frozenset(
    {
        # Presentation.
        "render_mode",
        "config_name",
        # This is entrant B, not the scenario. Dropping it is what lets a table
        # hold every pairing.
        "opponent_policy",
        # This is the leg axis. Dropping it is what lets the four legs of one
        # pairing share a fingerprint -- without it the feature does not work
        # at all.
        "turn_order",
        # Measurement instrumentation; it changes no outcome.
        "track_exposure",
        # Shapes training, not an argmax playout.
        "reward_phases",
        "terminal_success_bonus",
        "terminal_vp_bonus",
        # Canonicalised into `zone_pair` and `army_pair` below.
        "deployment_zone",
        "opponent_deployment_zone",
        "number_of_wargame_models",
        "number_of_opponent_models",
        "models",
        "opponent_models",
    }
)


class RatingScenarioMismatch(ValueError):
    """A leg was played on a different scenario from the ledger's."""


class RatingDecodeMismatch(ValueError):
    """One name, two decodes -- which is two players wearing one label."""


class LedgerEntrant(BaseModel):
    """Who played, and how their actions were decoded."""

    model_config = ConfigDict(extra="forbid")

    name: str
    kind: str
    source: str | None = None
    parent: str | None = None
    # How the actions were chosen. `decode_topk` > 1 is joint constrained
    # decoding, worth +40.5 vp on this project's own measurement -- so the same
    # weights at K=1 and K=3 are not the same player and must not share a
    # rating. `sampled` is the other axis: self-play rollouts draw from the
    # policy rather than taking its argmax, which is a third player again.
    decode_topk: int = 1
    sampled: bool = False


class LegRecord(BaseModel):
    """One leg's per-layout rows, as played."""

    model_config = ConfigDict(extra="forbid")

    entrant_a: str
    entrant_b: str
    a_zone: str
    first_mover: str
    layout_seeds: list[int]
    combat_seeds: list[int]
    margins: list[float]
    wins: list[float]
    objectives_held: list[float]
    coherency_rate: float | None = None
    opponent_coherency_rate: float | None = None
    code_revision: str | None = None
    recorded_at: str


class Ledger(BaseModel):
    """Every leg ever played on one scenario."""

    model_config = ConfigDict(extra="forbid")

    fingerprint: str
    scenario: dict[str, Any]
    entrants: list[LedgerEntrant]
    legs: list[LegRecord]


def canonical_scenario(config: WargameEnvConfig) -> dict[str, Any]:
    """What defines the game, with the leg axes and the entrants taken out.

    The zone pair and the army pair are **sorted** rather than dropped, so a
    zone swap leaves the fingerprint unchanged while a genuinely different board
    or a genuinely different army does not.

    Rule of thumb for anything added later: *if it changes what happens on the
    board, it belongs in the fingerprint.*
    """
    dumped = config.model_dump(mode="json")
    scenario = {
        key: value for key, value in dumped.items() if key not in _DROPPED_FIELDS
    }
    scenario["zone_pair"] = sorted(
        json.dumps(zone, sort_keys=True)
        for zone in (dumped["deployment_zone"], dumped["opponent_deployment_zone"])
    )
    scenario["army_pair"] = sorted(
        json.dumps({"count": count, "models": models}, sort_keys=True, default=str)
        for count, models in (
            (dumped["number_of_wargame_models"], dumped["models"]),
            (dumped["number_of_opponent_models"], dumped["opponent_models"]),
        )
    )
    return scenario


def fingerprint(config: WargameEnvConfig) -> str:
    """A stable digest of the scenario. Sixteen hex characters is plenty here."""
    payload = json.dumps(
        canonical_scenario(config), sort_keys=True, separators=(",", ":"), default=str
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def path_for(digest: str, root: Path = LEDGER_ROOT) -> Path:
    """Where a scenario's ledger lives."""
    return root / f"{digest}.json"


def load(digest: str, root: Path = LEDGER_ROOT) -> Ledger | None:
    """Read a ledger, or `None` if this scenario has never been rated."""
    path = path_for(digest, root)
    if not path.exists():
        return None
    ledger: Ledger = Ledger.model_validate_json(path.read_text())
    return ledger


def append(
    legs: Sequence[LegResult],
    config: WargameEnvConfig,
    entrants: Sequence[Entrant],
    root: Path = LEDGER_ROOT,
) -> Ledger:
    """Add legs to this scenario's ledger, creating it if needed.

    Refuses outright when `config` is not the scenario the ledger already holds.
    """
    digest = fingerprint(config)
    existing = load(digest, root)
    if existing is not None and existing.fingerprint != digest:
        raise RatingScenarioMismatch(
            f"ledger at {path_for(digest, root)} holds scenario "
            f"{existing.fingerprint} but these legs were played on {digest}"
        )

    revision = _code_revision()
    stamp = datetime.now(UTC).isoformat(timespec="seconds")
    records = [_record(leg, revision, stamp) for leg in legs]

    known = {
        entrant.name: entrant for entrant in (existing.entrants if existing else [])
    }  # type: ignore[misc]
    for entrant in entrants:
        record = LedgerEntrant(
            name=entrant.name,
            kind=entrant.kind,
            source=entrant.source,
            parent=entrant.parent,
            decode_topk=entrant.decode_topk,
            sampled=entrant.sampled,
        )
        _require_same_decode(known.get(entrant.name), record)  # type: ignore[arg-type]
        known[entrant.name] = record  # type: ignore[assignment]

    ledger = Ledger(
        fingerprint=digest,
        scenario=canonical_scenario(config),
        entrants=sorted(known.values(), key=lambda entry: entry.name),  # type: ignore[arg-type]
        legs=[*(existing.legs if existing else []), *records],
    )
    path = path_for(digest, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(ledger.model_dump_json(indent=2) + "\n")
    return ledger


def leg_results(ledger: Ledger) -> list[LegResult]:
    """Rebuild the played legs, so a table can be fitted without replaying."""
    return [
        LegResult(
            entrant_a=record.entrant_a,
            entrant_b=record.entrant_b,
            leg=Leg(Zone(record.a_zone), Seat(record.first_mover)),
            layout_seeds=tuple(record.layout_seeds),
            combat_seeds=tuple(record.combat_seeds),
            margins=tuple(record.margins),
            wins=tuple(record.wins),
            objectives_held=tuple(record.objectives_held),
            coherency_rate=record.coherency_rate,
            opponent_coherency_rate=record.opponent_coherency_rate,
        )
        for record in ledger.legs
    ]


def _record(leg: LegResult, revision: str | None, stamp: str) -> LegRecord:
    return LegRecord(
        entrant_a=leg.entrant_a,
        entrant_b=leg.entrant_b,
        a_zone=leg.leg.a_zone.value,
        first_mover=leg.leg.first_mover.value,
        layout_seeds=list(leg.layout_seeds),
        combat_seeds=list(leg.combat_seeds),
        margins=list(leg.margins),
        wins=list(leg.wins),
        objectives_held=list(leg.objectives_held),
        coherency_rate=leg.coherency_rate,
        opponent_coherency_rate=leg.opponent_coherency_rate,
        code_revision=revision,
        recorded_at=stamp,
    )


def _require_same_decode(
    existing: LedgerEntrant | None, arriving: LedgerEntrant
) -> None:
    """Refuse one name recorded under two decodes.

    CLAUDE.md's standing rule is never to quote a score without saying how it
    was decoded, and a rating is a score. The same weights at K=1 and K=3 differ
    by **+40.5 vp** here, which is larger than any policy difference this repo
    has ever measured -- so a table holding both under one name would rank the
    decode and call it skill. Refused rather than warned about, in the same
    class as mixing two scenarios.
    """
    if existing is None:
        return
    if (existing.decode_topk, existing.sampled) == (
        arriving.decode_topk,
        arriving.sampled,
    ):
        return
    raise RatingDecodeMismatch(
        f"entrant {arriving.name!r} is already in this ledger decoded at "
        f"topk={existing.decode_topk} sampled={existing.sampled}, but these "
        f"legs were played at topk={arriving.decode_topk} "
        f"sampled={arriving.sampled}. Those are different players; give the "
        f"second one its own name."
    )


def _code_revision() -> str | None:
    """The commit these legs were played at.

    Recorded because a rating is only reproducible against the code that played
    it -- and there are open bugs in this repo's line-of-sight and terrain
    handling whose fixes will move every number on the board.
    """
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
