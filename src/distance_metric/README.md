# Distance Metric

## Overview

NumPy-based pairwise, cross, and elementwise distance and similarity calculators. Array parameters use typed `NDArray` aliases (`NumericArray`, `BinaryArray`, `FloatArray`) from [`array_types.py`](array_types.py) for static analysis compatibility.

## Components

| Component | Description |
| --- | --- |
| [`array_types.py`](array_types.py) | Shared `NDArray` aliases for numeric, binary, and float arrays |
| [`calculator.py`](calculator.py) | Abstract `DistanceCalculator` base with pairwise, cross, and elementwise APIs |
| [`result.py`](result.py) | `DistanceResult` wrapper and `DistanceResultType` enum |
| [`metric.py`](metric.py) | Root distance metric registry enum |
| [`calculators/`](calculators/) | Metric family packages and concrete calculator implementations |

## Examples

```python
import numpy as np
from distance_metric.calculators import MinkowskiDistanceMetric
from distance_metric.calculators.minkowski import EuclideanDistanceCalculator

x = np.array([[0.0, 0.0], [3.0, 4.0]], dtype=np.float64)
calc = EuclideanDistanceCalculator()
result = calc.pairwise(x)
print(result.value.shape)  # (2, 2)
```
