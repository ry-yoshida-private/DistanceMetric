# calculators

## Overview

Concrete Minkowski-family calculators: fixed-Lp variants (Manhattan, Euclidean, Chebyshev), configurable Lp, and squared Euclidean. Each module defines a `DistanceCalculator` subclass wired from [`metric.py`](../metric.py).

## Components

| Path | Description |
|------|-------------|
| [`manhattan.py`](manhattan.py) | L1 (Manhattan) distance. |
| [`euclidean.py`](euclidean.py) | L2 (Euclidean) distance. |
| [`minkowski.py`](minkowski.py) | General Lp distance with a user-set exponent (`norm_order`) on the instance. |
| [`chebyshev.py`](chebyshev.py) | L∞ (Chebyshev) distance. |
| [`squared_euclidean.py`](squared_euclidean.py) | Squared Euclidean distance (no final square root). |

## Examples

Full runnable snippets for every metric live in [`../README.md`](../README.md). Minimal sanity checks:

```python
import numpy as np
from distance_metric.calculators.minkowski import EuclideanDistanceCalculator

calc = EuclideanDistanceCalculator()
result = calc.pairwise(array=np.array([[0.0, 0.0], [3.0, 4.0]], dtype=np.float64))
assert result.value.shape == (2, 2)
```

```python
import numpy as np
from distance_metric.calculators.minkowski import MinkowskiDistanceCalculator

calc = MinkowskiDistanceCalculator(norm_order=1.5)
result = calc.cross(
    query_array=np.array([[1.0, 2.0]], dtype=np.float64),
    gallery_array=np.array([[4.0, 8.0]], dtype=np.float64),
)
assert result.value.shape == (1, 1)
```
