"""Darkening a slice restores PAIRING to an action-space experiment.

Adding actions is the least measurable class of change in this project: it
widens the policy head, which changes how much RNG `seed_everything` consumes,
so an arm and its control no longer start from the same weights and the paired
estimator -- worth roughly an order of magnitude here -- is lost. The advance
move was called REJECTED on an unpaired reading at t = -3.20, p = 0.085.

`dark_action_slices` registers a slice at full width and valid in NO phase. The
arm and the control then share a parameter shape, so their initial weights are
bit-identical and the difference between them is paired again.

⚠ The third test is the one that costs GPU-hours: a control trained WITHOUT the
slice is **not** reusable, because the narrower head changes the whole draw.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from pytorch_lightning import seed_everything

from wargame_rl.wargame.envs.env_components.actions import ActionHandler
from wargame_rl.wargame.envs.types import WargameEnvConfig
from wargame_rl.wargame.envs.types.game_timing import BattlePhase
from wargame_rl.wargame.model.common.factory import create_environment
from wargame_rl.wargame.model.net import TransformerNetwork


@pytest.fixture
def base_env_config() -> WargameEnvConfig:
    """A small scenario -- these tests are about action-space shape, not play."""
    return WargameEnvConfig(render_mode=None, number_of_battle_rounds=4)


def _config(
    env_config: WargameEnvConfig, bins: int, dark: list[str]
) -> WargameEnvConfig:
    config: WargameEnvConfig = env_config.model_copy(deep=True)
    config.n_advance_speed_bins = bins
    config.dark_action_slices = dark
    return config


def test_a_dark_slice_keeps_its_width_and_loses_every_phase(
    base_env_config: WargameEnvConfig,
) -> None:
    """The action space is the same size; none of the slice's actions is valid."""
    lit = ActionHandler(_config(base_env_config, 3, []))
    dark = ActionHandler(_config(base_env_config, 3, ["advance"]))

    assert dark.registry.n_actions == lit.registry.n_actions

    advance = dark.registry.slice_for("advance")
    for phase in BattlePhase:
        mask = dark.registry.get_action_mask(phase)
        assert not mask[advance.start : advance.end].any(), (
            f"a darkened action was valid in {phase}"
        )
    lit_mask = lit.registry.get_action_mask(BattlePhase.movement)
    assert lit_mask[advance.start : advance.end].any(), "the lit control is vacuous"


def test_arm_and_dark_control_start_from_bit_identical_weights(
    base_env_config: WargameEnvConfig,
) -> None:
    """The whole point: same shape, same RNG draw, same weights, paired arms."""
    arm_config = _config(base_env_config, 3, [])
    control_config = _config(base_env_config, 3, ["advance"])

    seed_everything(1, workers=True)
    arm = TransformerNetwork.from_env(create_environment(env_config=arm_config), True)
    seed_everything(1, workers=True)
    control = TransformerNetwork.from_env(
        create_environment(env_config=control_config), True
    )

    arm_params = dict(arm.named_parameters())
    control_params = dict(control.named_parameters())
    assert set(arm_params) == set(control_params)
    for name, tensor in arm_params.items():
        assert torch.equal(tensor, control_params[name]), f"{name} differs at init"


def test_a_control_trained_without_the_slice_is_NOT_reusable(
    base_env_config: WargameEnvConfig,
) -> None:
    """The expensive half, pinned so nobody assumes the old control still pairs.

    A narrower policy head consumes less RNG, so *every* shared tensor drawn
    after it differs. This is why darkening restores pairing but does not make
    an existing control free: it has to be retrained with the slice darkened.
    """
    seed_everything(1, workers=True)
    narrow = TransformerNetwork.from_env(
        create_environment(env_config=_config(base_env_config, 0, [])), True
    )
    seed_everything(1, workers=True)
    wide = TransformerNetwork.from_env(
        create_environment(env_config=_config(base_env_config, 3, ["advance"])), True
    )

    assert wide.n_actions > narrow.n_actions
    narrow_params = dict(narrow.named_parameters())
    wide_params = dict(wide.named_parameters())
    shared = [
        name
        for name, tensor in narrow_params.items()
        if name in wide_params and tensor.shape == wide_params[name].shape
    ]
    differing = [
        name
        for name in shared
        if not torch.equal(narrow_params[name], wide_params[name])
    ]
    assert differing, (
        "a narrower head drew the same weights -- if this ever passes, the old "
        "control IS reusable and this test should be deleted, not weakened"
    )


def test_a_darkened_action_receives_exactly_zero_gradient() -> None:
    """Masked columns must not move, and must not perturb the gradient norm.

    Load-bearing because `grad_clipped_fraction` is 1.0 for whole runs: the clip
    threshold, not the learning rate, sets the effective step size, so a dark
    row contributing *any* gradient would shrink every other parameter's step.
    """
    torch.manual_seed(0)
    n_live, n_dark = 12, 5
    head = torch.nn.Linear(8, n_live + n_dark)
    features = torch.randn(4, 8)

    logits = head(features)
    mask = torch.ones(4, n_live + n_dark, dtype=torch.bool)
    mask[:, n_live:] = False
    distribution = torch.distributions.Categorical(
        logits=logits.masked_fill(~mask, float("-inf"))
    )
    loss = -distribution.log_prob(torch.zeros(4, dtype=torch.long)).mean()
    loss.backward()

    gradient = head.weight.grad
    assert gradient is not None
    dark_gradient = gradient[n_live:]
    assert not torch.isnan(gradient).any(), "masking produced NaN gradients"
    assert torch.all(dark_gradient == 0.0), "a darkened action moved its weights"


def test_an_unknown_slice_name_raises(base_env_config: WargameEnvConfig) -> None:
    """A typo would silently darken nothing and quietly unpair the experiment."""
    with pytest.raises(ValueError, match="unknown slices"):
        ActionHandler(_config(base_env_config, 3, ["advnace"]))


def test_the_darkened_game_is_the_narrow_game_for_a_non_advancing_policy(
    base_env_config: WargameEnvConfig,
) -> None:
    """The cross-config bridge: the board a policy plays on must be the same.

    Without this the dark control could be a different scenario, and the paired
    difference would measure the scenario rather than the feature.
    """
    narrow = create_environment(env_config=_config(base_env_config, 0, []))
    dark = create_environment(env_config=_config(base_env_config, 3, ["advance"]))

    narrow.reset(seed=7)
    dark.reset(seed=7)

    assert len(narrow.player_models) == len(dark.player_models)
    assert len(narrow.objectives) == len(dark.objectives)
    for left, right in zip(narrow.player_models, dark.player_models):
        assert np.allclose(left.location, right.location)
    for left_obj, right_obj in zip(narrow.objectives, dark.objectives):
        assert np.allclose(left_obj.location, right_obj.location)
