from dataclasses import dataclass

import numpy as np

# Vertex budget per terrain piece in the observation. A batch stacks terrain across
# episodes, so the token width cannot vary. Outlines with more vertices than this are
# truncated; the shipped generators stay well under it.
MAX_TERRAIN_VERTICES = 8


@dataclass
class WargameTerrainObservation:
    """Observation structure for a single terrain footprint.

    Stores the outline as a fixed number of vertices, each normalized to [-1, 1] using
    the board half-dimensions, followed by the true vertex count as a fraction of the
    budget. Outlines shorter than the budget repeat their last vertex, which adds
    zero-length edges and so changes neither the shape nor any geometric test.
    """

    footprint: np.ndarray  # [x0, y0, ... x7, y7, n_vertices / MAX_TERRAIN_VERTICES]

    @property
    def size(self) -> int:
        return int(self.footprint.size)
