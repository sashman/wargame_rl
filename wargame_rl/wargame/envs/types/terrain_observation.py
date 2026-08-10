from dataclasses import dataclass

import numpy as np

# Vertices an outline may carry in the observation. A terrain token is
# `2 * TERRAIN_VERTEX_BUDGET + 1` wide: the padded vertices plus the real vertex
# count, which is what tells a padding repeat from a genuine repeated vertex.
#
# Declared here, in `types/`, because both the observation builder (envs) and the
# tensor pipeline (model) need it and envs must not depend on model.
TERRAIN_VERTEX_BUDGET = 8


@dataclass
class WargameTerrainObservation:
    """Observation structure for a single terrain piece.

    Carries the piece's **outline**, as vertices normalised to [-1, 1] by the
    board half-dimensions, padded to a common budget by repeating the last
    vertex, plus the real vertex count.

    This used to be the bounding box, and that was the whole problem: an
    L-shaped ruin and a solid block produced identical four-number tokens, so
    the policy could not have told them apart even in principle. Every cover
    experiment in this repo was run against that input.

    The vertex count rides along because padding is indistinguishable from a
    genuinely repeated vertex otherwise. It is normalised to [0, 1] so it sits
    on the same scale as the coordinates.
    """

    outline: np.ndarray  # [x0, y0, x1, y1, ..., vertex_count] all in [-1, 1]

    @property
    def size(self) -> int:
        return int(self.outline.size)
