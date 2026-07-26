from dataclasses import dataclass

import numpy as np


@dataclass
class WargameTerrainObservation:
    """Observation structure for a single terrain footprint.

    Stores the normalized bounding-box corners of a ruin footprint.
    Each coordinate is normalized to [-1, 1] using the board half-dimensions.
    """

    footprint: np.ndarray  # [x0_norm, y0_norm, x1_norm, y1_norm]

    @property
    def size(self) -> int:
        return int(self.footprint.size)
