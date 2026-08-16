"""The clone's checkpoint keys must land on the real policy network.

`train.py::_apply_warm_start_weights` loads a warm start with
``load_state_dict(strict=False)``. That is deliberate — a warm start is allowed
to supply the policy without the critic — but it means a **wrong key prefix
loads nothing at all, silently**, and the run trains a random network while
reporting that it warm-started. The repo has been bitten by this family before:
`torch.compile` is not wired precisely because it prefixes every key with
``_orig_mod.``.

So the prefix is pinned here rather than trusted. If `PPOLightning`'s attribute
names ever change, this fails loudly instead of producing a run whose "warm
start" was a no-op.
"""

from __future__ import annotations

import torch

from scripts.behaviour_clone import POLICY_PREFIX
from wargame_rl.wargame.envs.types import WargameEnvConfig
from wargame_rl.wargame.envs.wargame import WargameEnv
from wargame_rl.wargame.model.net import TransformerNetwork
from wargame_rl.wargame.model.ppo.lightning import PPOLightning
from wargame_rl.wargame.model.ppo.ppo import PPO_Transformer


def _make_module() -> tuple[PPOLightning, WargameEnv]:
    env = WargameEnv(
        config=WargameEnvConfig(
            render_mode=None,
            number_of_wargame_models=3,
            number_of_objectives=2,
            number_of_battle_rounds=4,
        )
    )
    module = PPOLightning(
        env=env,
        ppo_model=PPO_Transformer.from_env(env),
        log=False,
        n_steps=8,
        batch_size=4,
        n_epochs=1,
        num_rollout_envs=2,
    )
    return module, env


def test_clone_prefix_matches_every_policy_tensor() -> None:
    """Every cloned tensor must have a home in the Lightning module."""
    module, env = _make_module()
    policy = TransformerNetwork.policy_from_env(env)

    cloned = {POLICY_PREFIX + key for key in policy.state_dict()}
    target = set(module.state_dict())

    missing = cloned - target
    assert not missing, (
        f"{len(missing)} cloned keys have no home, e.g. {sorted(missing)[:3]}"
    )
    assert cloned, "no policy tensors produced"


def test_loading_a_clone_actually_changes_the_weights() -> None:
    """The end the prefix serves: `strict=False` must not silently no-op.

    Asserting the key names alone would still pass if `load_state_dict` ignored
    them, so this loads a clone whose weights are deliberately distinct and
    checks the module's policy tensors actually moved.
    """
    module, env = _make_module()
    policy = TransformerNetwork.policy_from_env(env)

    with torch.no_grad():
        for tensor in policy.state_dict().values():
            if tensor.is_floating_point():
                tensor.fill_(0.5)
    state_dict = {POLICY_PREFIX + k: v for k, v in policy.state_dict().items()}

    before = module.state_dict()[POLICY_PREFIX + "game_embedding.weight"].clone()
    module.load_state_dict(state_dict, strict=False)
    after = module.state_dict()[POLICY_PREFIX + "game_embedding.weight"]

    assert not torch.equal(before, after), "warm start was a silent no-op"
    assert torch.allclose(after, torch.full_like(after, 0.5))
