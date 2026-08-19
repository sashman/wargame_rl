"""The constrained joint distribution PPO would train against.

The single most important property here is the LAST test: the argmax of this
distribution must be the combination `decode_joint_coherent` plays. If the
sampler and the decoder disagreed, training would optimise one constraint while
play enforced another -- which is the exact defect this work exists to remove,
reintroduced one layer down.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from wargame_rl.wargame.envs.types import WargameEnvConfig
from wargame_rl.wargame.envs.types.config import ModelConfig
from wargame_rl.wargame.envs.wargame import WargameEnv
from wargame_rl.wargame.model.common.decoding import (
    decode_joint_coherent,
    legal_unit_candidates,
)
from wargame_rl.wargame.model.common.joint_policy import (
    joint_entropy,
    joint_log_probs,
    joint_scores,
)


def test_scores_are_the_sum_of_member_log_probs() -> None:
    """The joint score is what the independent policy already says."""
    model_log_probs = torch.log(
        torch.tensor([[0.5, 0.5], [0.25, 0.75]])
    )  # (k=2, n_actions=2)
    combos = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]])

    scores = joint_scores(model_log_probs, combos)

    expected = torch.tensor(
        [
            np.log(0.5) + np.log(0.25),
            np.log(0.5) + np.log(0.75),
            np.log(0.5) + np.log(0.25),
            np.log(0.5) + np.log(0.75),
        ],
        dtype=scores.dtype,
    )
    assert torch.allclose(scores, expected, atol=1e-6)


def test_probability_mass_sums_to_one_over_the_legal_set() -> None:
    """The whole point: mass is redistributed onto what can execute."""
    model_log_probs = torch.log(torch.tensor([[0.5, 0.5], [0.25, 0.75]]))
    combos = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]])
    legal = torch.tensor([True, False, True, False])

    probs = joint_log_probs(model_log_probs, combos, legal).exp()

    assert probs[legal].sum().item() == pytest.approx(1.0, abs=1e-5)
    # Illegal combinations are not merely unlikely, they are unreachable.
    assert probs[~legal].sum().item() < 1e-9


def test_an_all_illegal_unit_does_not_produce_nan() -> None:
    """The degenerate row must stay differentiable, not poison the gradient.

    A unit whose entire candidate set is illegal scored every entry at -inf,
    and `log_softmax` of an all--inf row is NaN -- which would propagate into
    every parameter on the backward pass.
    """
    model_log_probs = torch.log(torch.tensor([[0.5, 0.5], [0.5, 0.5]]))
    combos = torch.tensor([[0, 0], [1, 1]])
    legal = torch.tensor([False, False])

    out = joint_log_probs(model_log_probs, combos, legal)

    assert torch.isfinite(out).all(), "an all-illegal unit produced non-finite logits"


def test_gradient_flows_to_the_member_log_probs() -> None:
    """PPO needs this differentiable, and only through the legal set."""
    model_log_probs = torch.log(
        torch.tensor([[0.5, 0.5], [0.25, 0.75]], requires_grad=True)
    ).detach()
    model_log_probs.requires_grad_(True)
    combos = torch.tensor([[0, 0], [1, 1]])
    legal = torch.tensor([True, True])

    joint_log_probs(model_log_probs, combos, legal)[0].backward()

    assert model_log_probs.grad is not None
    assert torch.isfinite(model_log_probs.grad).all()
    assert model_log_probs.grad.abs().sum() > 0


def test_entropy_is_zero_when_one_combination_is_legal() -> None:
    """A unit with no choice has no entropy -- and must not report noise."""
    model_log_probs = torch.log(torch.tensor([[0.5, 0.5], [0.25, 0.75]]))
    combos = torch.tensor([[0, 0], [0, 1], [1, 0]])
    legal = torch.tensor([True, False, False])

    entropy = joint_entropy(joint_log_probs(model_log_probs, combos, legal), legal)

    assert entropy.item() == pytest.approx(0.0, abs=1e-6)


def test_batched_matches_looping_one_at_a_time() -> None:
    """The update runs this over a minibatch; batching must not change it."""
    torch.manual_seed(0)
    batch, k, n_actions, n_combos = 4, 3, 5, 6
    model_log_probs = torch.log_softmax(torch.randn(batch, k, n_actions), dim=-1)
    combos = torch.randint(0, n_actions, (batch, n_combos, k))
    legal = torch.rand(batch, n_combos) > 0.3

    batched = joint_log_probs(model_log_probs, combos, legal)
    one_by_one = torch.stack(
        [joint_log_probs(model_log_probs[i], combos[i], legal[i]) for i in range(batch)]
    )

    assert torch.allclose(batched, one_by_one, atol=1e-6)


def test_the_argmax_is_what_the_decoder_actually_plays() -> None:
    """Sampler and decoder must agree, or training optimises the wrong game.

    `decode_joint_coherent` picks the most probable legal combination; this
    distribution puts its mode on that same combination. Checked against the
    real env so the legality both sides see is the real geometry, not a
    reimplementation of it.
    """
    # One unit of four, not four units of one: `group_id` defaults to the
    # model's own index, and a one-model unit is coherent by definition, so a
    # default env has nothing joint to decode and this test would be vacuous.
    env = WargameEnv(
        config=WargameEnvConfig(
            render_mode=None,
            number_of_wargame_models=4,
            number_of_objectives=2,
            number_of_battle_rounds=4,
            models=[ModelConfig(group_id=0) for _ in range(4)],
        )
    )
    observation, _ = env.reset(seed=7)
    assert len({m.group_id for m in env.player_models}) == 1, "expected one unit"
    rng = np.random.default_rng(11)
    assert observation.action_mask is not None
    mask = np.asarray(observation.action_mask)
    logits = np.where(mask, rng.normal(size=mask.shape), -np.inf)
    log_probs = logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))

    baseline = [int(row.argmax()) for row in log_probs]
    decoded = decode_joint_coherent(log_probs, baseline, env, top_k=3)
    units = legal_unit_candidates(log_probs, env, top_k=3)
    assert units, "no unit produced a candidate set; the test proves nothing"

    for unit in units:
        if not unit.legal.any():
            continue
        mode = int(
            joint_log_probs(
                torch.tensor(log_probs[unit.member_indices], dtype=torch.float64),
                torch.tensor(unit.combos, dtype=torch.long),
                torch.tensor(unit.legal),
            ).argmax()
        )
        chosen = [decoded[i] for i in unit.member_indices]
        assert list(unit.combos[mode]) == chosen, (
            "the joint distribution's mode is not the combination the decoder plays"
        )
    env.close()
