"""Autoregressive shooting decode: no two models may claim the same target.

The decode exists because the default policy is factorized -- one Categorical
over the whole (n_models, n_actions) tensor, every row sampled independently --
over a permutation-equivariant backbone. Identically-placed models therefore
emit identical greedy actions, and `_resolve_shooting_action` silently discards
a shot whose target died earlier in the same phase. Measured on the batch-3
control, that discarded 36-40% of ordered shots.

The property that matters for PPO correctness is that `evaluate_actions`
reconstructs exactly the conditionals the rollout sampled under; a mismatch
there corrupts the importance ratio without raising anything.
"""

from __future__ import annotations

import torch

from wargame_rl.wargame.model.ppo.networks import (
    decode_distinct_targets,
    greedy_actions,
    shooting_decode_applies,
    targets_taken_earlier,
)

# A small stand-in for the real layout: 2 non-shooting actions (stay + one
# movement) then 4 target columns, so `start` is nonzero as it is in the env.
N_ACTIONS = 6
SHOOTING_SLICE = (2, 6)


def _movement_phase_logits(n_models: int = 3) -> torch.Tensor:
    """Logits with the whole shooting slice masked out, as outside shooting."""
    logits = torch.zeros(1, n_models, N_ACTIONS)
    logits[..., SHOOTING_SLICE[0] : SHOOTING_SLICE[1]] = float("-inf")
    return logits


def test_greedy_without_a_slice_is_a_plain_argmax() -> None:
    """Flag off must leave behaviour bit-identical, or every checkpoint shifts."""
    logits = torch.randn(2, 5, N_ACTIONS)

    assert torch.equal(greedy_actions(logits, None), logits.argmax(dim=-1))


def test_all_models_preferring_one_target_get_distinct_ones() -> None:
    """The defect, directly: three models whose best target is the same one."""
    logits = torch.full((1, 3, N_ACTIONS), -1.0)
    # Target column 2 is every model's favourite, then 3, then 4.
    logits[0, :, 2] = 5.0
    logits[0, :, 3] = 4.0
    logits[0, :, 4] = 3.0

    plain = logits.argmax(dim=-1)
    decoded = greedy_actions(logits, SHOOTING_SLICE)

    assert plain.tolist() == [[2, 2, 2]], "precondition: the factorized collapse"
    assert decoded.tolist() == [[2, 3, 4]]


def test_a_model_with_no_free_target_holds_fire_rather_than_breaking() -> None:
    """Stay is outside the slice, so it is always available as a fallback.

    Without this the last model would have every legal action at -inf and the
    Categorical would produce NaN rather than an action.
    """
    logits = torch.full((1, 2, N_ACTIONS), float("-inf"))
    logits[0, :, 0] = 0.0  # stay
    logits[0, :, 2] = 5.0  # the single shared legal target

    decoded = greedy_actions(logits, SHOOTING_SLICE)

    assert decoded.tolist() == [[2, 0]]


def test_movement_phase_skips_the_decode_entirely() -> None:
    """Nothing to de-duplicate when no target is legal, so take the cheap path."""
    logits = _movement_phase_logits()
    logits[0, :, 1] = 2.0

    assert not shooting_decode_applies(logits, SHOOTING_SLICE)
    assert torch.equal(greedy_actions(logits, SHOOTING_SLICE), logits.argmax(dim=-1))


def test_taken_earlier_is_exclusive_so_a_model_never_blocks_itself() -> None:
    """The stored action of model i must stay legal when i is re-evaluated."""
    actions = torch.tensor([[2, 3, 2]])

    forbidden = targets_taken_earlier(actions, N_ACTIONS, SHOOTING_SLICE)

    assert not forbidden[0, 0, 2], "model 0 blocked its own choice"
    assert not forbidden[0, 1, 3], "model 1 blocked its own choice"
    assert forbidden[0, 1, 2], "model 1 should see model 0's target as taken"
    assert forbidden[0, 2, 2] and forbidden[0, 2, 3]


def test_non_shooting_actions_are_never_forbidden() -> None:
    """Two models may both stay; only target columns are exclusive."""
    actions = torch.tensor([[0, 0, 1]])

    forbidden = targets_taken_earlier(actions, N_ACTIONS, SHOOTING_SLICE)

    assert not forbidden.any()


def test_evaluate_side_masking_reproduces_the_rollout_conditionals() -> None:
    """The PPO correctness property, stated as an equality.

    `decode_distinct_targets` returns the log-probs the rollout sampled under;
    `targets_taken_earlier` is what `evaluate_actions` rebuilds them from. If
    these disagree the importance ratio is silently wrong.
    """
    torch.manual_seed(0)
    logits = torch.randn(4, 5, N_ACTIONS)

    actions, rollout_log_probs = decode_distinct_targets(
        logits, SHOOTING_SLICE, deterministic=False
    )

    forbidden = targets_taken_earlier(actions, N_ACTIONS, SHOOTING_SLICE)
    replayed = torch.distributions.Categorical(
        logits=logits.masked_fill(forbidden, float("-inf"))
    ).log_prob(actions)

    assert torch.allclose(rollout_log_probs, replayed, atol=1e-6)


def test_sampled_targets_are_distinct_within_a_step() -> None:
    """The invariant itself, over a batch, under sampling rather than argmax."""
    torch.manual_seed(1)
    logits = torch.randn(8, 4, N_ACTIONS)

    actions, _log_probs = decode_distinct_targets(
        logits, SHOOTING_SLICE, deterministic=False
    )

    start, end = SHOOTING_SLICE
    for row in actions:
        shots = [int(a) for a in row if start <= int(a) < end]
        assert len(shots) == len(set(shots))
