# calculators

## Overview

Implementations of **scale- and covariance-aware** Euclidean variants.

## Components

| Path | Description |
|------|-------------|
| [`mahalanobis.py`](mahalanobis.py) | Mahalanobis distance using inverse covariance (or pseudoinverse) supplied at call time. |
| [`seuclidean.py`](seuclidean.py) | Standardized Euclidean: Euclidean length after per-dimension scaling by variances. |

## Examples

### [`mahalanobis.py`](mahalanobis.py)

Requires `vi` or `cov` on every call.

```python
import numpy as np
from distance_metric.calculators.covariance_normalized import MahalanobisDistanceCalculator

calc = MahalanobisDistanceCalculator()
result = calc.pairwise(
    array=np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float64),
    cov=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64),
)
assert result.value.shape == (2, 2)
```

### [`seuclidean.py`](seuclidean.py)

Requires `variance` shaped like one sample.

```python
import numpy as np
from distance_metric.calculators.covariance_normalized import SEuclideanDistanceCalculator

calc = SEuclideanDistanceCalculator()
result = calc.cross(
    query_array=np.array([[1.0, 2.0]], dtype=np.float64),
    gallery_array=np.array([[4.0, 6.0]], dtype=np.float64),
    variance=np.array([1.0, 4.0], dtype=np.float64),
)
assert result.value.shape == (1, 1)
```
