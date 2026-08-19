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

from wargame_rl.wargame.envs.types import WargameEnvConfig, WargameEnvObservation
from wargame_rl.wargame.envs.types.config import ModelConfig
from wargame_rl.wargame.envs.wargame import WargameEnv
from wargame_rl.wargame.model.common.decoding import (
    decode_joint_coherent,
    legal_unit_candidates,
)
from wargame_rl.wargame.model.common.joint_policy import (
    combo_pattern,
    combos_from_topk,
    joint_entropy,
    joint_log_probs,
    joint_scores,
    sample_joint_actions,
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


def _one_unit_env(n_models: int = 4) -> WargameEnv:
    """One unit of `n_models`, deployed IN formation.

    Placed explicitly rather than randomly: models scattered across a
    deployment zone start incoherent, no combination of moves is legal, and the
    joint sampler correctly declines to act -- which makes every test here pass
    vacuously while proving nothing. The chain distance is 2", so 1.5" apart
    leaves the unit coherent with room to move.
    """
    return WargameEnv(
        config=WargameEnvConfig(
            render_mode=None,
            number_of_wargame_models=n_models,
            number_of_objectives=2,
            number_of_battle_rounds=4,
            models=[
                ModelConfig(group_id=0, x=10.0 + 1.5 * i, y=10.0)
                for i in range(n_models)
            ],
        )
    )


def _masked_log_probs(observation: WargameEnvObservation, seed: int) -> np.ndarray:
    """Random but mask-respecting per-model log-probs."""
    rng = np.random.default_rng(seed)
    mask = np.asarray(observation.action_mask)
    logits = np.where(mask, rng.normal(size=mask.shape), -np.inf)
    normalised: np.ndarray = logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))
    return normalised


def test_the_stored_log_prob_is_the_density_of_what_executed() -> None:
    """The property a broken PPO ratio would violate, and nothing else would.

    The update recomputes the joint log-probability from the *stored* candidate
    set. At unchanged parameters that must reproduce the number recorded during
    the rollout exactly -- if it does not, `exp(new - old)` is not 1 at the
    start of an epoch, the importance ratio is measured against the wrong
    distribution, and PPO optimises a quantity nobody chose. It would train
    happily and look completely normal.
    """
    env = _one_unit_env()
    observation, _ = env.reset(seed=7)
    log_probs = _masked_log_probs(observation, 11)
    baseline = [int(row.argmax()) for row in log_probs]

    actions, draws = sample_joint_actions(
        log_probs, baseline, env, top_k=3, rng=np.random.default_rng(3)
    )
    assert draws, "no unit was decoded jointly; the test proves nothing"

    for draw in draws:
        pattern = combo_pattern(len(draw.member_indices), 3)
        combos = combos_from_topk(draw.topk_actions, pattern)
        recomputed = joint_log_probs(
            torch.tensor(log_probs[draw.member_indices], dtype=torch.float64),
            torch.tensor(combos, dtype=torch.long),
            torch.tensor(draw.legal),
        )[draw.chosen]

        assert float(recomputed) == pytest.approx(draw.log_prob, abs=1e-9)
        # And the executed action really is the sampled combination.
        for slot, index in enumerate(draw.member_indices):
            assert actions[index] == int(combos[draw.chosen, slot])
    env.close()


def test_a_sampled_move_is_always_coherency_legal() -> None:
    """Sampling must draw only from the legal set, on every draw."""
    env = _one_unit_env()
    observation, _ = env.reset(seed=5)
    log_probs = _masked_log_probs(observation, 2)
    baseline = [int(row.argmax()) for row in log_probs]
    rng = np.random.default_rng(99)

    for _ in range(50):
        _actions, draws = sample_joint_actions(log_probs, baseline, env, 3, rng)
        for draw in draws:
            assert draw.legal[draw.chosen], "sampled an illegal combination"
    env.close()


def test_sampling_explores_rather_than_collapsing_to_the_decode() -> None:
    """Greedy would give PPO one action per state and nothing to improve."""
    env = _one_unit_env()
    observation, _ = env.reset(seed=5)
    log_probs = _masked_log_probs(observation, 2)
    baseline = [int(row.argmax()) for row in log_probs]
    rng = np.random.default_rng(4)

    seen = {
        tuple(sample_joint_actions(log_probs, baseline, env, 3, rng)[0])
        for _ in range(40)
    }

    assert len(seen) > 1, "the joint sampler is behaving greedily"
    env.close()


def test_a_padded_slot_never_doubles_a_combinations_mass() -> None:
    """A member with fewer than K legal actions must not distort the joint.

    Padding each member's candidate list to a rectangular shape repeats an
    action, so the repeated combination appears twice. Left legal, the outcome
    would carry twice the probability it should -- a silent bias toward
    whatever the mask happened to restrict.
    """
    env = _one_unit_env()
    observation, _ = env.reset(seed=13)
    mask = np.asarray(observation.action_mask)
    # Leave the first model exactly two legal actions against a top_k of 3.
    restricted = np.zeros_like(mask)
    restricted[0, :2] = True
    restricted[1:] = mask[1:]
    rng = np.random.default_rng(6)
    logits = np.where(restricted, rng.normal(size=mask.shape), -np.inf)
    log_probs = logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))

    units = legal_unit_candidates(log_probs, env, top_k=3)
    assert units, "no candidate set built"
    unit = units[0]

    padded = ~unit.slot_valid[0]
    assert padded.any(), "the restriction did not produce a padded slot"
    reaches_padding = np.isin(
        unit.combos[:, 0], unit.topk_actions[0][padded]
    ) & ~np.isin(unit.combos[:, 0], unit.topk_actions[0][unit.slot_valid[0]])
    assert not unit.legal[reaches_padding].any(), (
        "a combination using a padded slot was left legal, doubling its mass"
    )
    env.close()
