"""Processing steps specific to this model."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace

import numpy as np


@dataclass(frozen=True)
class Threshold:
    """Bound a field to what its transform can actually represent.

    The reflectivity scale saturates at both ends, so the inverse cannot emit
    anything below `floor`: without this, every dry pixel in the domain carries
    a uniform 0.0365 mm/h of drizzle that no forecast intended.

    The ceiling guards the other end. The published inference path omits the
    clamp used during training, and the inverse grows exponentially, so a single
    member drifting out of range turns into hundreds of mm/h.
    """

    floor: float
    ceiling: float

    def __post_init__(self):
        if not 0 <= self.floor < self.ceiling:
            raise ValueError("Threshold needs 0 <= floor < ceiling")

    def apply(self, prediction):
        values = np.asarray(prediction.values, dtype=np.float32).copy()
        np.minimum(values, self.ceiling, out=values)
        # Below the representable floor there is no information, only the
        # saturation artefact, so it becomes an honest zero. The comparison is
        # tolerant because the floor is the image of a float32 round trip and
        # lands a fraction above or below itself depending on the path taken.
        values[values <= self.floor * (1.0 + 1e-5)] = 0.0
        return replace(prediction, values=values)

    def describe(self) -> Mapping[str, object]:
        return {"kind": "threshold", "floor": self.floor, "ceiling": self.ceiling}
