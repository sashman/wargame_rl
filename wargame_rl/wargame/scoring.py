"""Score a policy spec on a config, through whichever facade owns it.

The one place that decides which facade a spec plays: a `.pt` is the
per-model facade's and runs through `envs/per_model/evaluate.py` in waves;
everything else (a baseline name or a `.ckpt`) is the phase facade's and runs
through `envs/baseline/evaluate.py` exactly as it always has. Both produce
one `EvalResult`, so `measure-checkpoint`, `measure-maps` and `measure-paired`
print one table and pair one per-episode tuple.

An application service beside `selectors.py`, not part of it: resolving a
spec into something that plays is one job, building the envs and running
the episodes is another. Torch is imported only on the checkpoint branches,
inside the resolver.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from wargame_rl.wargame.envs.baseline.evaluate import evaluate_selector, record_episode
from wargame_rl.wargame.envs.evaluation import EVAL_WAVE_SIZE, EvalResult
from wargame_rl.wargame.envs.per_model.env import PerModelEnv
from wargame_rl.wargame.envs.per_model.evaluate import evaluate_per_model_chooser
from wargame_rl.wargame.envs.per_model.recording import (
    record_episode as record_per_model_episode,
)
from wargame_rl.wargame.envs.state.provenance import Cadence
from wargame_rl.wargame.envs.types import WargameEnvConfig
from wargame_rl.wargame.selectors import (
    build_action_selector,
    build_per_model_chooser,
    is_per_model_checkpoint,
)

if TYPE_CHECKING:
    from wargame_rl.wargame.envs.wargame import WargameEnv

# The opponent policy that seats a whole-phase network on the other side.
# The per-model facade drives its opponent through a scripted seat, and a
# network there is the self-play question (#274), not this build.
_NETWORK_OPPONENT = "model"


def evaluate_spec(
    spec: str,
    env_config: WargameEnvConfig,
    seeds: Sequence[int],
    name: str,
    *,
    decode_topk: int = 1,
    decode_stay: bool = False,
    combat_seeds: Sequence[int] | None = None,
    wave_size: int = EVAL_WAVE_SIZE,
) -> EvalResult:
    """Score `spec` on `env_config` over `seeds`, on the facade that owns it.

    `decode_topk` and `decode_stay` are the phase facade's play-time decodes;
    the per-model facade runs no decode, so they are refused with a `.pt`
    rather than silently ignored -- a row that says "K=3" must have run one.
    """
    if is_per_model_checkpoint(spec):
        if decode_topk != 1 or decode_stay:
            raise ValueError(
                f"{spec!r} plays the per-model facade, which runs no joint "
                "decode; drop decode_topk / decode_stay"
            )
        opponent = env_config.opponent_policy
        if opponent is not None and opponent.type == _NETWORK_OPPONENT:
            raise ValueError(
                "the per-model facade cannot seat a whole-phase network as "
                "its opponent (self-play under the per-model step is #274)"
            )
        envs = [
            PerModelEnv(env_config) for _ in range(max(1, min(wave_size, len(seeds))))
        ]
        chooser = build_per_model_chooser(spec, envs)
        return evaluate_per_model_chooser(
            chooser.choose, envs, seeds, name, combat_seeds=combat_seeds
        )
    env = _phase_env(env_config)
    try:
        selector = build_action_selector(spec, env, decode_topk, decode_stay)
        return evaluate_selector(
            selector.select,
            env,
            list(seeds),
            name,
            combat_seeds=None if combat_seeds is None else list(combat_seeds),
        )
    finally:
        env.close()


def record_spec(
    spec: str,
    env_config: WargameEnvConfig,
    seed: int,
    output_path: Path,
    *,
    decode_topk: int = 1,
    decode_stay: bool = False,
    cadence: Cadence = "phase",
) -> Path:
    """Record one episode of `spec` on `env_config` to an event log."""
    if is_per_model_checkpoint(spec):
        if decode_topk != 1 or decode_stay:
            raise ValueError(
                f"{spec!r} plays the per-model facade, which runs no decode"
            )
        return record_per_model_episode(
            lambda envs: build_per_model_chooser(spec, envs).choose,
            env_config,
            seed,
            output_path,
            cadence=cadence,
            driver=spec,
        )
    if cadence != "phase":
        raise ValueError(
            f"{spec!r} plays the phase facade, which records at phase cadence only"
        )
    env = _phase_env(env_config)
    try:
        select = build_action_selector(spec, env, decode_topk, decode_stay).select
    finally:
        env.close()
    return record_episode(select, env_config, seed, output_path)


def record_per_model(
    spec: str,
    env_config: WargameEnvConfig,
    seed: int,
    output_path: Path,
    *,
    cadence: Cadence = "phase",
) -> Path:
    """Record one episode of `spec` played on the PER-MODEL facade, whatever
    `spec` is -- a `.pt`, a baseline name, `random` or `set_network`.

    `record_spec` lets the spec choose the facade, which sends a baseline
    name to the phase facade (the bar's own game). This is the other
    question: how does the per-model facade play this policy.
    """
    return record_per_model_episode(
        lambda envs: build_per_model_chooser(spec, envs, seed=seed).choose,
        env_config,
        seed,
        output_path,
        cadence=cadence,
        driver=spec,
    )


def _phase_env(env_config: WargameEnvConfig) -> WargameEnv:
    """The phase facade's env, through the factory that registers the `model`
    opponent key -- imported here, not at module scope, because that factory
    pulls in torch and a scripted score should not pay for it."""
    from wargame_rl.wargame.model.common.factory import create_environment

    return create_environment(env_config=env_config)


__all__ = ["evaluate_spec", "record_per_model", "record_spec"]
