from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class WargameEnvAction:
    """Action structure for the Wargame environment.

    ``actions`` is a list of ints — one per wargame model.
    """

    actions: list[int]

    @classmethod
    def random(cls, mask: np.ndarray) -> WargameEnvAction:
        """Sample a random action per model, respecting a boolean mask.

        Parameters
        ----------
        mask:
            ``(n_models, n_actions)`` boolean array where True = valid.
        """
        actions = [int(np.random.choice(np.where(m)[0])) for m in mask]
        return cls(actions)
