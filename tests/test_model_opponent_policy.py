"""A checkpoint can play the opponent seat.

The guarantees worth pinning are not "it produces an action": they are that
seating a network changes **nothing else** about the episode, and that the two
ways this could be silently wrong both raise instead.
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
import torch
from pydantic_yaml import parse_yaml_raw_as

from wargame_rl.wargame.envs.types import OpponentPolicyConfig, WargameEnvConfig
from wargame_rl.wargame.envs.wargame import WargameEnv
from wargame_rl.wargame.model.common.factory import create_environment
from wargame_rl.wargame.model.net import TransformerNetwork
from wargame_rl.wargame.model.opponent.network_policy import (
    AsymmetricArmiesError,
    NetworkOpponentPolicy,
)
from wargame_rl.wargame.selectors import build_action_selector

CONFIG = "configs/dev/4v4_two_phases.yaml"
POLICY_PREFIX = "ppo_model.policy_network."
SEED = 900_000


def _config() -> WargameEnvConfig:
    with open(CONFIG) as handle:
        config: WargameEnvConfig = parse_yaml_raw_as(WargameEnvConfig, handle.read())
    config.render_mode = None
    return config


def _checkpoint(tmp_path: Path, config: WargameEnvConfig) -> str:
    """A real checkpoint, sized from the *opponent* seat of `config`.

    Written from an env whose armies are swapped, so the weights are the shape
    the opponent seat needs — which on a symmetric config is the same shape
    either way, and is exactly what makes a seat bug here invisible.
    """
    env = WargameEnv(config=config.model_copy(deep=True), renderer=None)
    torch.manual_seed(0)
    policy = TransformerNetwork.policy_from_env(env=env)
    env.close()
    path = tmp_path / "run-2026-08-19-00-00-00-opp" / "last.ckpt"
    path.parent.mkdir(parents=True)
    torch.save(
        {"state_dict": {POLICY_PREFIX + k: v for k, v in policy.state_dict().items()}},
        path,
    )
    return str(path)


def _with_model_opponent(
    config: WargameEnvConfig, checkpoint: str, **params: object
) -> WargameEnvConfig:
    updated: WargameEnvConfig = config.model_copy(deep=True)
    updated.opponent_policy = OpponentPolicyConfig(
        type="model", params={"checkpoint": checkpoint, **params}
    )
    return updated


def test_a_model_opponent_plays_a_whole_episode(tmp_path: Path) -> None:
    base = _config()
    env = create_environment(
        env_config=_with_model_opponent(base, _checkpoint(tmp_path, base))
    )
    select = build_action_selector("squad_march", env).select

    observation, _ = env.reset(seed=SEED)
    terminated = truncated = False
    steps = 0
    while not (terminated or truncated):
        observation, _r, terminated, truncated, _i = env.step(select(observation, env))
        steps += 1

    assert steps > 1
    env.close()


def test_a_model_opponent_draws_nothing_from_the_layout_rng(tmp_path: Path) -> None:
    """Seating a network must not move the layout.

    This is what pins the lazy sizing. `TransformerNetwork.from_env` calls
    `env.reset()`, and the opponent policy is constructed *inside*
    `WargameEnv.__init__` — so building the network there would both re-enter a
    half-built env and consume draws from the layout RNG, shifting every seeded
    episode. The failure would look like "the scenario changed", not like a
    crash.
    """
    base = _config()
    checkpoint = _checkpoint(tmp_path, base)

    scripted = create_environment(env_config=base.model_copy(deep=True))
    scripted.reset(seed=SEED)
    expected = [tuple(m.location) for m in scripted.wargame_models]
    objectives = [tuple(o.location) for o in scripted.objectives]
    scripted.close()

    networked = create_environment(env_config=_with_model_opponent(base, checkpoint))
    networked.reset(seed=SEED)
    got = [tuple(m.location) for m in networked.wargame_models]
    got_objectives = [tuple(o.location) for o in networked.objectives]
    networked.close()

    assert got == expected
    assert got_objectives == objectives


def test_a_model_opponent_survives_a_deep_copy(tmp_path: Path) -> None:
    """Lightning deep-copies the env in `save_hyperparameters`.

    The mirror's `__getattr__` recursed forever under `copy.deepcopy` once
    before (PR #204), killing every training run on such a config at startup.
    A torch module now hangs off that mirror, so the hazard is larger, not
    smaller. Steps the copy rather than only building it, so playability is
    pinned and not merely constructibility.
    """
    base = _config()
    env = create_environment(
        env_config=_with_model_opponent(base, _checkpoint(tmp_path, base))
    )
    observation, _ = env.reset(seed=SEED)

    duplicate = copy.deepcopy(env)
    select = build_action_selector("squad_march", duplicate).select
    duplicate.step(select(observation, duplicate))

    env.close()
    duplicate.close()


def test_an_asymmetric_army_is_refused(tmp_path: Path) -> None:
    """`_alive_feature_index` counts back from the trailing expected-damage
    block and `_alive_from_features` falls back to **all alive** when the index
    lands out of range — so unequal armies do not raise on their own, they just
    make the seat read corpses as live models."""
    base = _config()
    checkpoint = _checkpoint(tmp_path, base)
    asymmetric = _with_model_opponent(base, checkpoint)
    asymmetric.number_of_opponent_models = base.number_of_wargame_models + 1
    asymmetric.opponent_models = None

    with pytest.raises(AsymmetricArmiesError, match="equal armies"):
        create_environment(env_config=asymmetric)


def test_a_checkpoint_with_an_unknown_prefix_raises(tmp_path: Path) -> None:
    """Never silently load nothing.

    `_apply_warm_start_weights` in `train.py` uses `strict=False` with no prefix
    rewriting, so a wrong prefix there loads zero tensors and trains a random
    network while reporting a warm start. This path must not copy that.
    """
    base = _config()
    env = WargameEnv(config=base.model_copy(deep=True), renderer=None)
    torch.manual_seed(0)
    policy = TransformerNetwork.policy_from_env(env=env)
    env.close()
    path = tmp_path / "bad.ckpt"
    torch.save(
        {"state_dict": {"layer." + k: v for k, v in policy.state_dict().items()}}, path
    )

    with pytest.raises(ValueError, match="No policy network"):
        create_environment(env_config=_with_model_opponent(base, str(path)))


def test_a_missing_checkpoint_fails_at_construction(tmp_path: Path) -> None:
    """Loudly, and before a scoring run has spent anything."""
    with pytest.raises((FileNotFoundError, OSError)):
        create_environment(
            env_config=_with_model_opponent(_config(), str(tmp_path / "absent.ckpt"))
        )


def test_the_opponent_shoots_when_its_action_space_can(tmp_path: Path) -> None:
    """`shoots` gates whether the env refines this policy's mask with range,
    line of sight and engagement validity. Left False for a policy that can emit
    a shot, every shot would be resolved unchecked."""
    base = _config()
    env = create_environment(
        env_config=_with_model_opponent(base, _checkpoint(tmp_path, base))
    )
    policy = env.opponent_policy

    assert isinstance(policy, NetworkOpponentPolicy)
    assert policy.shoots is (env.opponent_action_handler.shooting_slice is not None)
    env.close()


def test_the_network_is_sized_from_the_opponent_seat(tmp_path: Path) -> None:
    """Reading `env._action_handler` through the mirror falls through to the
    *player's* handler. On a symmetric config that is the same width, so the
    only way to see the difference is to check which handler was consulted."""
    base = _config()
    env = create_environment(
        env_config=_with_model_opponent(base, _checkpoint(tmp_path, base))
    )
    observation, _ = env.reset(seed=SEED)
    policy = env.opponent_policy
    assert isinstance(policy, NetworkOpponentPolicy)

    assert policy.mirror.player_action_handler is env.opponent_action_handler
    env.close()


def test_an_unregistered_model_key_names_the_import_that_fixes_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`model` is registered a layer above `envs`, so it is absent whenever an
    env is built without going through the factory. The message has to name the
    import, or keeping the layering honest costs a debugging session."""
    from wargame_rl.wargame.envs.opponent import registry

    without_model = {k: v for k, v in registry.get_registry().items() if k != "model"}
    monkeypatch.setattr(registry, "_REGISTRY", without_model)
    monkeypatch.setattr(registry, "_auto_register", lambda: None)

    with pytest.raises(ValueError, match=r"wargame_rl\.wargame\.model\.opponent"):
        registry.build_opponent_policy(
            OpponentPolicyConfig(type="model", params={"checkpoint": "x.ckpt"}),
            None,  # type: ignore[arg-type]
        )


def test_the_mask_is_not_recomputed_for_the_opponent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The env has already built a rules-legal mask and passes it in.

    Letting `build_observation` build its own would run
    `compute_unit_shooting_masks`, and with it the line-of-sight pass, a second
    time in every shooting phase — so the mask is spliced rather than recomputed.
    """
    from wargame_rl.wargame.envs.env_components import observation_builder

    calls = {"n": 0}
    original = observation_builder.compute_unit_shooting_masks

    def counted(*args: object, **kwargs: object) -> np.ndarray:
        calls["n"] += 1
        return original(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(observation_builder, "compute_unit_shooting_masks", counted)

    base = _config()
    env = create_environment(
        env_config=_with_model_opponent(base, _checkpoint(tmp_path, base))
    )
    select = build_action_selector("squad_march", env).select
    observation, _ = env.reset(seed=SEED)
    before = calls["n"]
    env.step(select(observation, env))

    # The player's observation may build one; the opponent's must not add
    # another on top of the mask the env already handed it.
    assert calls["n"] - before <= 1
    env.close()
