"""Turn a victory-point margin into the score a rating is fitted on.

Standard Elo takes an outcome `S in {0, 1/2, 1}`. This repo has already measured
that **win rate cannot resolve differences under ~7pp** on these configs while
`vp_margin` separates cleanly: TF32 cost 8.5 vp_margin on both seeds and moved
win rate 0.705 -> 0.65, inside the noise and invisible. A rating built on
win/draw/loss inherits that blindness exactly.

So the margin enters the rating -- and it enters through the **score**, not
through the update rule:

    s = 1 / (1 + exp(-m / s_m))        m = own VP - enemy VP, from A's side

**This is not a redefinition of Elo.** With `s_m` fitted so that
`s ~ P(win | margin m)`, the noisy 0/1 outcome has been replaced by its own
conditional expectation -- same expected rating, strictly lower variance. A
rating point keeps meaning what it means everywhere else: a win probability.

Nothing here imports from `wargame_rl`.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

# Illustrative until re-fitted on a recorded corpus for the scenario in hand.
# Pinned rather than fitted per run so a rating reproduces; `fit_margin_scale`
# is what re-derives it, and the fit belongs in the ledger beside the ratings.
DEFAULT_MARGIN_SCALE: float = 50.0


def margin_score(
    margin: NDArray[np.float64] | np.ndarray,
    margin_scale: float = DEFAULT_MARGIN_SCALE,
) -> NDArray[np.float64]:
    """Map victory-point margins to scores in `[0, 1]`.

    Three properties earn their keep here:

    - `s(0) = 0.5` **exactly**, so a VP tie is a draw with no special case.
    - It saturates. Per-episode `vp_margin` sd is 45-50 on 25v25, so blowouts
      are common and an unbounded score would let one layout dominate a rating.
    - `margin_scale <= 0` degrades to the win indicator, which is the check that
      this *generalises* plain Elo rather than replacing it.

    Computed through `expit`'s stable branch rather than `1/(1+exp(-x))`, which
    overflows on the blowouts this scenario actually produces.
    """
    margins = np.asarray(margin, dtype=np.float64)
    if margin_scale <= 0.0:
        return np.where(margins > 0.0, 1.0, np.where(margins < 0.0, 0.0, 0.5))
    return _sigmoid(margins / margin_scale)


def fit_margin_scale(
    margins: NDArray[np.float64] | np.ndarray,
    wins: NDArray[np.float64] | np.ndarray,
    max_iter: int = 100,
    tol: float = 1e-10,
) -> float:
    """Fit `s_m` by logistic regression of the win indicator on the margin.

    Fit **once** over a recorded corpus for a scenario, then pin the result and
    re-fit only when the scenario changes. It must be calibrated per scenario
    rather than assumed: where VP is capped -- and the cap is what makes the
    current scenario a denial game -- the margin distribution is bounded, and a
    scale fitted on an uncapped scenario would be wrong in a direction nobody
    would notice.

    Maximised over `beta = 1 / s_m`, in which the log-likelihood is concave, so
    Newton converges from any start. Returns `s_m`.
    """
    margin_values = np.asarray(margins, dtype=np.float64)
    win_values = np.asarray(wins, dtype=np.float64)
    if margin_values.shape != win_values.shape:
        raise ValueError(
            f"margins and wins must align: {margin_values.shape} != {win_values.shape}"
        )
    if margin_values.size == 0:
        raise ValueError("cannot fit a margin scale from an empty corpus")
    if not (np.any(win_values > 0.5) and np.any(win_values < 0.5)):
        raise ValueError(
            "the corpus must contain both wins and losses; an all-one-way "
            "corpus carries no information about the scale"
        )

    beta = 1.0 / DEFAULT_MARGIN_SCALE
    for _ in range(max_iter):
        expected = _sigmoid(margin_values * beta)
        gradient = float(np.sum(margin_values * (win_values - expected)))
        curvature = float(np.sum(margin_values**2 * expected * (1.0 - expected)))
        if curvature <= 0.0:
            break
        step = gradient / curvature
        beta += step
        if abs(step) < tol:
            break

    if beta <= 0.0:
        raise ValueError(
            "the fitted scale is not positive, which means margin and win "
            "anti-correlate in this corpus -- check the sign convention"
        )
    return 1.0 / beta


def _sigmoid(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """`1 / (1 + exp(-x))`, without overflowing on large negative `x`.

    The naive form evaluates `exp(+large)` and returns `inf`, then `nan`. A
    margin of several hundred is ordinary here, so this is reached in practice
    rather than only in principle.
    """
    positive = values >= 0.0
    result = np.empty_like(values)
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponential = np.exp(values[~positive])
    result[~positive] = exponential / (1.0 + exponential)
    return result
