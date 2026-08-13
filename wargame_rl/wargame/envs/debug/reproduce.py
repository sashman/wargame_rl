"""Booting the episode a recording came from, so it can be stepped by hand.

A recording carries the full state of every step but replaying it is a *video*:
there is no env, so nothing can be asked "what if". This turns the provenance
block on the reset snapshot back into a live environment that will produce the
identical match — same layout, same dice, same everything — which is what makes
`just debug` usable on a game you have already watched go wrong.

Lives here rather than in `state/` because it constructs a `WargameEnv`, and
`state/` is imported *by* the env.
"""

from __future__ import annotations

from pathlib import Path

from wargame_rl.wargame.envs.state.codecs import JsonMatchCodec
from wargame_rl.wargame.envs.state.snapshot import EpisodeProvenance
from wargame_rl.wargame.envs.types import WargameEnvConfig
from wargame_rl.wargame.envs.wargame import WargameEnv


def read_provenance(path: str | Path) -> EpisodeProvenance:
    """The provenance block from a recording's reset event.

    Raises with the schema version when the recording predates 2.4, because
    "reproduce this" and "this file cannot say how" are worth telling apart —
    the second is not fixable by trying again.
    """
    source = Path(path)
    log = JsonMatchCodec().decode(source.read_bytes())
    if log.provenance is None:
        raise ValueError(
            f"{source} carries no provenance. It was recorded before the inputs "
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
