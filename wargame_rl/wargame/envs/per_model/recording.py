"""Record one episode of the per-model facade to a match event log.

The per-model counterpart of `envs/baseline/evaluate.py::record_episode`. At
`phase` cadence the log is schema-identical to the phase facade's, so `just
replay`, `replay-render` and `analyze` read it unchanged; at `decision`
cadence there is one snapshot per decision, which replay and render accept
and `analyze_match` refuses by name.

Takes a chooser FACTORY rather than a chooser: the env is built here, and a
scripted seat has to be installed on it before `reset` plans the command
phase. `selectors.build_per_model_chooser` has exactly that shape.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

from wargame_rl.wargame.envs.per_model.env import PerModelEnv
from wargame_rl.wargame.envs.per_model.types import BatchChooser
from wargame_rl.wargame.envs.state import EventLogExporter, JsonMatchCodec
from wargame_rl.wargame.envs.state.provenance import Cadence
from wargame_rl.wargame.envs.types import WargameEnvConfig

ChooserFactory = Callable[[Sequence[PerModelEnv]], BatchChooser]


def record_episode(
    chooser_for: ChooserFactory,
    config: WargameEnvConfig,
    seed: int,
    output_path: Path,
    *,
    cadence: Cadence = "phase",
    combat_seed: int | None = None,
    driver: str | None = None,
    anchor_interval: int = 10,
) -> Path:
    """Play one seeded episode with event recording on and write the log.

    Returns the path written. One episode per file, because `EventLog`
    holds only the most recent episode.
    """
    exporter = EventLogExporter(anchor_interval=anchor_interval)
    env = PerModelEnv(config, state_exporters=[exporter], record_cadence=cadence)
    env.driver_label = driver
    choose = chooser_for([env])
    options = None if combat_seed is None else {"combat_seed": combat_seed}
    observation, _ = env.reset(seed=seed, options=options)
    terminated = False
    while not terminated:
        action = choose([env], [observation])[0]
        observation, _reward, terminated, _truncated, _info = env.step(action)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(JsonMatchCodec().encode(exporter.log))
    return output_path


__all__ = ["ChooserFactory", "record_episode"]
