"""Booting the episode a recording came from, so it can be stepped by hand.

A recording carries the full state of every step but replaying it is a *video*:
there is no env, so nothing can be asked "what if". This turns the provenance
block in the log header back into a live environment reproducing that episode —
same layout, same dice, same everything — which is what makes `just debug`
usable on a game you have already watched go wrong.

**How exact, measured.** Two rebuilds from one provenance are bit-identical to
each other, and a rebuild replaying a `simulate` recording's own actions matched
its snapshots exactly over 20 steps, actions included. Against a *training*
recording the agreement is exact at 38 of 40 steps with a transient ~9e-13 on a
single model at two of them, which self-corrects. Both rebuilds agree with each
other there, so the difference is between the training process and any
reproduction, not between reproductions — unexplained, and small enough that it
changes no decision the debugger is used to make. Do not quote this path as
bit-identical against a training recording until that is understood.

Lives here rather than in `state/` because it constructs a `WargameEnv`, and
`state/` is imported *by* the env.
"""

from __future__ import annotations

from pathlib import Path

from wargame_rl.wargame.envs.state.codecs import JsonMatchCodec
from wargame_rl.wargame.envs.state.event_log import EventLog
from wargame_rl.wargame.envs.state.snapshot import EpisodeProvenance
from wargame_rl.wargame.envs.types import WargameEnvConfig
from wargame_rl.wargame.envs.wargame import WargameEnv


def read_log(path: str | Path) -> EventLog:
    """Decode a recording. Split out so a caller wanting both the provenance and
    the actions does not parse the file twice."""
    return JsonMatchCodec().decode(Path(path).read_bytes())


def recorded_actions(log: EventLog) -> list[list[int] | None]:
    """The player's action at each step, indexed by step number.

    Index 0 is unused — a reset executes nothing — so entry `n` is the action
    that produced the snapshot at step `n`. That indexing is what lets a session
    follow the recording through a rewind for free: `env.current_turn` is
    restored with the env, so the next action is always looked up by where the
    match actually is rather than by how many steps have been taken.

    Reconstructed once at startup rather than seeked per step: a seek walks from
    the nearest anchor, and the whole point is to do that walk a single time.
    """
    actions: list[list[int] | None] = [None]
    for step in range(1, len(log.events)):
        snapshot = log.snapshot_at(step)
        recorded = snapshot.player_actions
        actions.append(list(recorded) if recorded is not None else None)
    return actions


def read_provenance(path: str | Path) -> EpisodeProvenance:
    """The provenance block from a recording's reset event.

    Raises with the schema version when the recording predates 2.4, because
    "reproduce this" and "this file cannot say how" are worth telling apart —
    the second is not fixable by trying again.
    """
    return provenance_of(read_log(path), path)


def provenance_of(log: EventLog, path: str | Path) -> EpisodeProvenance:
    """The provenance from an already-decoded log; `path` only names it in the
    error, so a caller that wants the actions too parses the file once."""
    if log.provenance is None:
        raise ValueError(
            f"{path} carries no provenance. It was recorded before the inputs "
            "were written down, so the episode cannot be recreated from it — "
            "re-record it, or supply the config and seed by hand."
        )
    return log.provenance


def build_env(
    provenance: EpisodeProvenance, renderer: object | None = None
) -> WargameEnv:
    """A fresh env with the episode's generator state installed, **not yet reset**.

    Split from the reset because the caller owns it: `run_session` resets the
    env itself, and resetting here as well would draw from the stream twice and
    land on a different layout. Whoever resets must pass `reset_options`.
    """
    env = WargameEnv(config=WargameEnvConfig(**provenance.config), renderer=renderer)  # type: ignore[arg-type]
    env.driver_label = provenance.driver
    # Touching `np_random` first materialises the generator, so this replaces a
    # real state rather than one created lazily part-way through `reset`.
    env.np_random.bit_generator.state = provenance.rng_state
    return env


def reset_options(provenance: EpisodeProvenance) -> dict[str, int]:
    """What `reset` needs beyond the generator state.

    The combat seed is passed explicitly rather than left to be re-derived, so
    the dice survive a change to how the derivation works.
    """
    return {"combat_seed": provenance.combat_seed}


def rebuild(provenance: EpisodeProvenance) -> WargameEnv:
    """A fresh env, reset into the exact episode the provenance describes."""
    env = build_env(provenance)
    env.reset(options=reset_options(provenance))
    return env


def rebuild_from_recording(path: str | Path) -> WargameEnv:
    """`read_provenance` then `rebuild`, which is what every caller wants."""
    return rebuild(read_provenance(path))
