"""Turn a policy name or a checkpoint path into something that plays.

Two facades, two contracts, one resolver: `build_action_selector` gives the
phase facade an `ActionSelector` (one action per model per phase) for a
baseline name or a `.ckpt`; `build_per_model_chooser` gives the per-model
facade a batch chooser (one decision per env per call) for a baseline name,
`random`, `set_network` or a `.pt`. Each refuses the other's checkpoint by
name -- a `.pt` used to fall into the Lightning loader and die there.

Every tool here names a policy the same way: a baseline registry key, or a path
to a `.ckpt`. Four near-duplicate resolvers had grown to do that -- in
`scripts/measure_maps.py`, `debug.py`, `scripts/measure_paired_policies.py` and
`scripts/measure_income_share.py` -- **with two different precedences**, two
trying the filesystem first and two trying the registry first. They disagree
only for a baseline name that is also an existing path, which is why nobody
noticed; the cost was four places to change and a fifth about to be written for
the rating arena.

This module sits *above* `envs/` and beside `model/`: it reaches for the
baseline registry and, on the checkpoint branch, for a network. Nothing in
`envs/` may import it -- `tests/test_import_direction.py` pins that.

**Torch is imported inside the checkpoint branch, not at module scope.** A
scripted `just debug` session should not pay for it, and
`envs/baseline/evaluate.py` imports torch-free today. That property is easy to
undo by accident and nothing else in the suite would notice, so
`tests/test_action_selector_resolution.py` asserts it in a subprocess.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from wargame_rl.wargame.envs.baseline.evaluate import ActionSelector, selector_for
from wargame_rl.wargame.envs.baseline.registry import (
    build_baseline_policy,
    get_registry,
)
from wargame_rl.wargame.envs.per_model.evaluate import random_chooser, scripted_chooser
from wargame_rl.wargame.envs.per_model.types import BatchChooser

if TYPE_CHECKING:
    from wargame_rl.wargame.envs.per_model.env import PerModelEnv
    from wargame_rl.wargame.envs.wargame import WargameEnv
    from wargame_rl.wargame.model.net import TransformerNetwork

CHECKPOINT_SUFFIX = ".ckpt"
PER_MODEL_CHECKPOINT_SUFFIX = ".pt"
# The two policies the per-model facade offers that are not scripts: the
# random legal seat, and the set network at fresh weights (the whole pipeline
# seen playing legally, not a skill).
RANDOM_POLICY = "random"
SET_NETWORK_POLICY = "set_network"

# `<scenario>-<YYYY-MM-DD-HH-MM-SS>[-<suffix>]`. The scenario part is identical
# across the arms of a screen, so the run suffix is the only part that says
# which arm a row belongs to.
_RUN_SUFFIX = re.compile(r"\d{4}(?:-\d{2}){5}-(.+)$")


@dataclass(frozen=True, slots=True)
class ResolvedSelector:
    """A playable policy, plus enough provenance to label a row of a table."""

    select: ActionSelector
    label: str
    kind: Literal["baseline", "checkpoint"]
    source: str | None
    network: TransformerNetwork | None


def is_checkpoint(spec: str) -> bool:
    """True when `spec` names a checkpoint rather than a registered baseline.

    Reads the suffix rather than the filesystem, so a mistyped path is reported
    as a missing checkpoint instead of an unknown baseline -- the two mistakes
    need different fixes.
    """
    return Path(spec).suffix == CHECKPOINT_SUFFIX


def is_per_model_checkpoint(spec: str) -> bool:
    """True when `spec` names a per-model checkpoint (`model/per_model/checkpoint.py`)."""
    return Path(spec).suffix == PER_MODEL_CHECKPOINT_SUFFIX


def label_for(checkpoint_path: str) -> str:
    """Name a run by its `--run-suffix`, falling back to the directory name."""
    directory = Path(checkpoint_path).parent.name
    match = _RUN_SUFFIX.search(directory)
    return match.group(1) if match else directory


def build_action_selector(
    spec: str,
    env: WargameEnv,
    decode_topk: int = 1,
    decode_stay: bool = False,
    reallocate: bool = False,
    charge_decode: bool = False,
    reallocate_max: int = 1,
) -> ResolvedSelector:
    """Resolve `spec` against the filesystem first, then the baseline registry.

    Path-first is the `measure_maps` precedence. The registry-first variants it
    replaces differ only for a baseline name that is also an existing file.

    A checkpoint's network is **sized from `env`**, so the caller must pass the
    env the selector will actually play in -- `measure_maps` rebuilds one per
    map for exactly this reason, since each map carries its own objective and
    terrain counts.

    `decode_topk` > 1 replaces the independent per-model argmax with joint
    constrained decoding (`model/common/decoding.py`); `decode_stay` stands a
    unit still when the top-K set yields no legal combination at all. Both are
    ignored on the baseline branch.

    `reallocate` applies the surplus-reallocation decode AFTER the joint one
    (`model/common/reallocation_decode.py`), worth **+8.3 ± 4.25 vp** on frozen
    weights. It is ignored on the baseline branch too — the scripted bar
    already allocates globally, and the redirect on it measured **exactly
    zero** (docs/melee-teaching-goal.md §29).
    """
    if is_per_model_checkpoint(spec):
        raise ValueError(
            f"{spec!r} is a per-model checkpoint; it plays the per-model facade, "
            "and only `measure-checkpoint`, `measure-maps` and `measure-paired` "
            "score one (through `wargame_rl.wargame.scoring`)"
        )
    if is_checkpoint(spec) or Path(spec).exists():
        return _resolve_checkpoint(
            spec,
            env,
            decode_topk,
            decode_stay,
            reallocate,
            charge_decode,
            reallocate_max,
        )
    return _resolve_baseline(spec)


def _resolve_baseline(spec: str) -> ResolvedSelector:
    registry = get_registry()
    if spec not in registry:
        raise ValueError(
            f"'{spec}' is neither a checkpoint path nor a baseline. "
            f"Known baselines: {', '.join(sorted(registry))}"
        )
    return ResolvedSelector(
        select=selector_for(build_baseline_policy(spec)),
        label=spec,
        kind="baseline",
        source=None,
        network=None,
    )


def _resolve_checkpoint(
    spec: str,
    env: WargameEnv,
    decode_topk: int,
    decode_stay: bool,
    reallocate: bool = False,
    charge_decode: bool = False,
    reallocate_max: int = 1,
) -> ResolvedSelector:
    """Load a policy network and wrap it as an `ActionSelector`.

    Greedy (argmax) rather than sampled: this measures the policy the agent
    would play, not the exploration distribution around it. The network applies
    the action mask internally, so illegal actions cannot be selected.
    """
    if not Path(spec).exists():
        raise ValueError(f"no checkpoint at {spec!r}")

    # Deferred deliberately -- see the module docstring.
    import torch

    from wargame_rl.wargame.envs.types import WargameEnvAction, WargameEnvObservation
    from wargame_rl.wargame.model.common.charge_decode import apply_charge_decode
    from wargame_rl.wargame.model.common.decoding import decode_joint_coherent
    from wargame_rl.wargame.model.common.observation import observation_to_tensor
    from wargame_rl.wargame.model.common.reallocation_decode import apply_reallocation
    from wargame_rl.wargame.model.net import TransformerNetwork

    policy_net = TransformerNetwork.from_checkpoint(env, spec)
    policy_net.eval()

    def select(
        observation: WargameEnvObservation, env_: WargameEnv
    ) -> WargameEnvAction:
        with torch.no_grad():
            state = observation_to_tensor(observation, policy_net.device)
            logits = policy_net(state)
            actions = [int(a) for a in logits.argmax(dim=-1).flatten().tolist()]
            if decode_topk > 1:
                log_probs = torch.log_softmax(logits, dim=-1).squeeze(0).cpu().numpy()
                actions = decode_joint_coherent(
                    log_probs,
                    actions,
                    env_,
                    decode_topk,
                    include_stay=decode_stay,
                )
            if charge_decode:
                # The JOINT move a factored policy cannot express, for units
                # the policy itself declared. Execution only, never the choice.
                actions = apply_charge_decode(actions, env_)
            if reallocate:
                # AFTER the joint decode: the redirect is rigid, so a squad the
                # joint decode certified coherent stays coherent, and the env's
                # referee judges the result either way.
                actions = apply_reallocation(
                    actions, env_, max_redirects=reallocate_max
                )
        return WargameEnvAction(actions=actions)

    return ResolvedSelector(
        select=select,
        label=label_for(spec),
        kind="checkpoint",
        source=spec,
        network=policy_net,
    )


# --------------------------------------------------------------- per-model


@dataclass(frozen=True, slots=True)
class ResolvedChooser:
    """A playable per-model policy, plus enough provenance to label a row."""

    choose: BatchChooser
    label: str
    kind: Literal["baseline", "random", "set_network", "checkpoint"]
    source: str | None


def build_per_model_chooser(
    spec: str, envs: Sequence[PerModelEnv], *, seed: int = 0
) -> ResolvedChooser:
    """Resolve `spec` into a batch chooser seated on every env in `envs`.

    Takes the envs rather than one env because a scripted seat is installed
    per env, before any reset (a script plans its command phase inside
    `reset`), and a checkpoint's head is sized from the env's action encoding.
    `seed` drives the random seat and the fresh-weight network only.
    """
    if not envs:
        raise ValueError("build_per_model_chooser needs at least one env")
    if is_checkpoint(spec):
        raise ValueError(
            f"{spec!r} is a whole-phase checkpoint; the per-model facade cannot "
            "play it (use `build_action_selector`)"
        )
    if spec == RANDOM_POLICY:
        return ResolvedChooser(random_chooser(seed), spec, "random", None)
    if spec == SET_NETWORK_POLICY:
        return ResolvedChooser(
            _fresh_network_chooser(envs, seed), spec, "set_network", None
        )
    if is_per_model_checkpoint(spec) or Path(spec).exists():
        if not Path(spec).exists():
            raise ValueError(f"no per-model checkpoint at {spec!r}")
        return ResolvedChooser(
            _per_model_checkpoint_chooser(spec, envs),
            label_for(spec),
            "checkpoint",
            spec,
        )
    registry = get_registry()
    if spec not in registry:
        raise ValueError(
            f"'{spec}' is neither a per-model checkpoint path nor a baseline. "
            f"Known baselines: {', '.join(sorted(registry))}, plus "
            f"`{RANDOM_POLICY}` and `{SET_NETWORK_POLICY}`"
        )
    return ResolvedChooser(
        scripted_chooser(build_baseline_policy(spec), envs), spec, "baseline", None
    )


def _expected_displacements(env: PerModelEnv) -> int:
    """The displacement head's width this env's action encoding needs."""
    handler = env.player_action_handler
    advance = handler.advance_slice
    return 1 + handler.movement_slice.size + (advance.size if advance else 0)


def _per_model_checkpoint_chooser(
    spec: str, envs: Sequence[PerModelEnv]
) -> BatchChooser:
    """A trained set network, greedy: the policy the checkpoint would play."""
    # Deferred deliberately -- see the module docstring.
    import torch

    from wargame_rl.wargame.envs.per_model.types import (
        PerModelAction,
        PerModelObservation,
    )
    from wargame_rl.wargame.model.per_model import SetAgent, load_checkpoint

    loaded = load_checkpoint(
        Path(spec), expected_n_displacements=_expected_displacements(envs[0])
    )
    loaded.network.eval()
    agent = SetAgent(loaded.network, greedy=True)

    def choose(
        envs_: Sequence[PerModelEnv], observations: Sequence[PerModelObservation]
    ) -> list[PerModelAction]:
        with torch.no_grad():
            return [d.action for d in agent.act_batch(envs_, observations)]

    return choose


def _fresh_network_chooser(envs: Sequence[PerModelEnv], seed: int) -> BatchChooser:
    """The set network at fresh weights, sampled: the pipeline, not a skill."""
    import torch

    from wargame_rl.wargame.envs.per_model.types import (
        PerModelAction,
        PerModelObservation,
    )
    from wargame_rl.wargame.model.per_model import SetAgent, SetNetwork

    torch.manual_seed(seed)
    agent = SetAgent(SetNetwork.from_env(envs[0]))
    generator = torch.Generator().manual_seed(seed)

    def choose(
        envs_: Sequence[PerModelEnv], observations: Sequence[PerModelObservation]
    ) -> list[PerModelAction]:
        with torch.no_grad():
            return [
                d.action
                for d in agent.act_batch(envs_, observations, generator=generator)
            ]

    return choose


__all__ = [
    "CHECKPOINT_SUFFIX",
    "PER_MODEL_CHECKPOINT_SUFFIX",
    "RANDOM_POLICY",
    "SET_NETWORK_POLICY",
    "ResolvedChooser",
    "ResolvedSelector",
    "build_action_selector",
    "build_per_model_chooser",
    "is_checkpoint",
    "is_per_model_checkpoint",
    "label_for",
]
