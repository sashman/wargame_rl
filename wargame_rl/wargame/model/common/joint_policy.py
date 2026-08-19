"""The unit's joint action distribution, constrained before sampling.

The policy emits one categorical per model, but unit coherency is a property of
the *combination* those models land in. Play time already fixes this by decoding
(`decoding.py`) — but the policy is then **trained under one rule and played
under another**, so it spends probability mass on combinations that never
execute and never receives a gradient saying so.

This module makes the constraint part of the distribution:

    pi_unit(a | s) = softmax over LEGAL combos of  sum_i log pi_i(a_i | s)

At K=3 on a five-model unit that is <=243 terms, so the joint log-probability is
**exactly computable** — no approximation, no sampling estimate.

**Why this is PPO-correct, and the trap it avoids.** The constraint is applied
to the distribution *before* sampling, which is the same argument that makes
action masking valid: the action that was sampled is the action that executes,
so the stored log-probability is the density of what actually happened. Applying
the decoder as a *filter* after sampling would break exactly that — the executed
action would not be the sampled one, and the importance ratio would be computed
against the wrong distribution. It would also look like it was working.

**Why the unit and not the army.** Summing log-probs over all 25 models made one
importance ratio for the whole joint action, and `eps_clip=0.2` was breached at
0.0073 nats of change per model (`exp(25 * 0.0073) = 1.200`). At the unit that
same per-model change gives `exp(5 * 0.0073) = 1.037` — inside the clip with
roughly 5x margin. The unit is the level the *constraint* lives at anyway.

**The candidate set is frozen from the behaviour policy.** Top-K truncation is
what makes the enumeration tractable, but it means the support depends on the
logits, so a new policy would rank a different top-K and could assign the stored
action zero probability. The candidate set and its legality mask are therefore
recorded during the rollout and reused unchanged in the update — exactly how the
action mask is already treated. The residual off-policy bias is of the same kind
and size as the mask's.
"""

from __future__ import annotations

import torch
from torch import Tensor

# Illegal combinations are scored at this rather than `-inf`: a unit whose whole
# candidate set is illegal would otherwise produce an all-`-inf` row, and
# `log_softmax` would return NaN and poison the gradient. Large enough that a
# legal combination always dominates, finite enough to stay differentiable.
ILLEGAL_SCORE = -1e9


def joint_scores(model_log_probs: Tensor, combos: Tensor) -> Tensor:
    """Unnormalised joint score of each candidate combination.

    Args:
        model_log_probs: ``(..., k, n_actions)`` per-model log-probabilities for
            one unit's members, already masked by the network.
        combos: ``(..., n_combos, k)`` action ids per candidate.

    Returns:
        ``(..., n_combos)`` — the sum of member log-probabilities, which is the
        log-probability the *independent* policy assigns that combination.
    """
    gathered = torch.gather(
        model_log_probs.unsqueeze(-3).expand(
            *combos.shape[:-1], model_log_probs.shape[-2], model_log_probs.shape[-1]
        ),
        -1,
        combos.unsqueeze(-1),
    ).squeeze(-1)
    return gathered.sum(dim=-1)


def joint_log_probs(model_log_probs: Tensor, combos: Tensor, legal: Tensor) -> Tensor:
    """``(..., n_combos)`` log-probabilities over the legal combinations.

    Renormalised across the legal set only, so the distribution sums to one over
    combinations that can actually execute.
    """
    scores = joint_scores(model_log_probs, combos)
    scores = torch.where(legal, scores, torch.full_like(scores, ILLEGAL_SCORE))
    return torch.log_softmax(scores, dim=-1)


def joint_entropy(log_probs: Tensor, legal: Tensor) -> Tensor:
    """``(...)`` entropy of the constrained joint, in nats.

    Computed over the legal set only. A unit with one legal combination has zero
    entropy, which is correct — it has no choice to make.
    """
    probs = log_probs.exp()
    terms = torch.where(legal, probs * log_probs, torch.zeros_like(probs))
    return -terms.sum(dim=-1)
