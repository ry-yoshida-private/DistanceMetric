import numpy as np
from dataclasses import dataclass

from enum import Enum

from .array_types import FloatArray

class DistanceResultType(Enum):
    DISTANCE = "distance"
    SIMILARITY = "similarity"

@dataclass
class DistanceResult:
    value: FloatArray
    type: DistanceResultType

    @property
    def min(self) -> float:
        return float(np.min(self.value))

    @property
    def max(self) -> float:
        return float(np.max(self.value))

    @property
    def best_score(self) -> float:
        match self.type:
            case DistanceResultType.DISTANCE:
                return self.min
            case DistanceResultType.SIMILARITY:
                return self.max

    @property
    def worst_score(self) -> float:
        match self.type:
            case DistanceResultType.DISTANCE:
                return self.max
            case DistanceResultType.SIMILARITY:
                return self.min

    @property
    def shape(self) -> tuple[int, ...]:
        return self.value.shape
