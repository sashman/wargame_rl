"""Which frozen opponent to train against next.

Prioritised fictitious self-play: sample a snapshot with probability
proportional to a function of `p`, the learner's chance of beating it. Uniform
sampling over a pool spends most of its games on opponents already beaten
comfortably; weighting by `p` spends them where the gradient is.

`p` comes from the **rating table**, through `elo.win_probability`, rather than
from a win-rate counter maintained alongside it. One estimator, fitted from
every game either policy has played, instead of two that can disagree.

Nothing here imports from `wargame_rl`, so the sampler is testable with no env,
no torch and no I/O -- the same split `score.py` and `elo.py` keep.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray

Mode = Literal["hard", "even", "uniform"]

# Enough mass that no snapshot is starved. A pool entry that is never sampled
# is a pool entry that cannot catch the learner forgetting how to beat it, and
# forgetting is the failure a pool exists to prevent -- so the floor is not a
# tuning knob so much as the thing that makes the pool a pool.
DEFAULT_UNIFORM_FLOOR = 0.1


def pfsp_weights(
    win_probability: NDArray[np.float64] | np.ndarray,
    mode: Mode = "hard",
    uniform_floor: float = DEFAULT_UNIFORM_FLOOR,
) -> NDArray[np.float64]:
    """Sampling weights over a pool, one per snapshot, summing to one.

    Args:
        win_probability: `p` per snapshot -- the learner's expected score
            against it. Must be in `[0, 1]`.
        mode: which opponents to prefer.

            - `hard` weights `(1 - p)^2`: the ones the learner loses to. The
              square rather than `1 - p` because a linear weight still spends a
              third of its games on opponents it beats 2:1.
            - `even` weights `p(1 - p)`: the ones it is level with, which is
              where a game is most informative about *which* of two policies is
              better and where the ratings themselves sharpen fastest.
            - `uniform` weights everything alike. **The control**, and the thing
              to run an arm against before believing a PFSP number: a pool on
              its own changes training, and the schedule is a separate claim.
        uniform_floor: share of the mass spread evenly over the pool regardless
            of `p`, so nothing is starved.

    Returns:
        Weights in pool order, summing to 1.

    An entirely-beaten pool -- every `p` at 1.0 under `hard`, or every `p` at 0
    or 1 under `even` -- has no signal to prioritise on, so the weighting is
    degenerate and the result falls back to uniform rather than dividing by
    zero. That case is not hypothetical: it is exactly the state a pool reaches
    when the learner has outgrown everything in it, and the right response is to
    keep playing while a new snapshot is taken, not to crash.
    """
    probability = np.asarray(win_probability, dtype=np.float64)
    if probability.ndim != 1 or probability.size == 0:
        raise ValueError(
            f"win_probability must be a non-empty 1-D array, got shape "
            f"{probability.shape}"
        )
    if np.any(probability < 0.0) or np.any(probability > 1.0):
        raise ValueError("win_probability must lie in [0, 1]")
    if not 0.0 <= uniform_floor <= 1.0:
        raise ValueError(f"uniform_floor must lie in [0, 1], got {uniform_floor}")

    if mode == "uniform":
        return np.full(probability.size, 1.0 / probability.size)
    if mode == "hard":
        raw = np.square(1.0 - probability)
    elif mode == "even":
        raw = probability * (1.0 - probability)
    else:
        raise ValueError(f"unknown PFSP mode {mode!r}; expected hard, even or uniform")

    uniform = np.full(probability.size, 1.0 / probability.size)
    total = float(raw.sum())
    if total <= 0.0:
        return uniform

    prioritised = raw / total
    return (1.0 - uniform_floor) * prioritised + uniform_floor * uniform


def sample_opponent(
    weights: NDArray[np.float64] | np.ndarray,
    rng: np.random.Generator,
) -> int:
    """Draw one pool index.

    Takes a `Generator` rather than a seed because the caller owns the stream:
    training is deterministic given seed, config and code, and an opponent
    schedule drawn from a shared or wall-clock-seeded source would end that.
    The stream is also **separate** from the layout and dice streams, so
    switching self-play on does not shift the board a control was trained on.
    """
    probabilities = np.asarray(weights, dtype=np.float64)
    return int(rng.choice(probabilities.size, p=probabilities))
