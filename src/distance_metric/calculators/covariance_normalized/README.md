# covariance_normalized

## Overview

**Covariance-normalized** distances: Mahalanobis (quadratic form with inverse covariance or pseudoinverse) and standardized Euclidean (per-feature variance scaling). Pass **`vi`** or **`cov`** (Mahalanobis) and **`variance`** (SEuclidean) into `pairwise`, `cross`, or `elementwise`.

## Components

| Path | Description |
|------|-------------|
| [`metric.py`](metric.py) | `CovarianceNormalizedDistanceMetric` enum. |
| [`calculators/`](calculators/) | `MahalanobisDistanceCalculator` and `SEuclideanDistanceCalculator`. |

## Examples

### Mahalanobis with `cov`

```python
import numpy as np
from distance_metric import CovarianceNormalizedDistanceMetric, DistanceResult

calc = CovarianceNormalizedDistanceMetric.MAHALANOBIS.calculator
x: np.ndarray = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]], dtype=np.float64)
cov: np.ndarray = np.array([[1.0, 0.1], [0.1, 1.0]], dtype=np.float64)
result: DistanceResult = calc.pairwise(array=x, cov=cov)
assert result.value.shape == (3, 3)
```

### Mahalanobis with `vi`

```python
import numpy as np
from distance_metric import CovarianceNormalizedDistanceMetric

calc = CovarianceNormalizedDistanceMetric.MAHALANOBIS.calculator
vi: np.ndarray = np.linalg.inv(np.array([[1.0, 0.1], [0.1, 1.0]], dtype=np.float64))
result = calc.cross(
    query_array=np.array([[0.0, 0.0]], dtype=np.float64),
    gallery_array=np.array([[1.0, 2.0]], dtype=np.float64),
    vi=vi,
)
assert result.value.shape == (1, 1)
```

### Standardized Euclidean (`variance`)

`variance` must match the trailing shape of each sample (here a length-2 vector).

```python
import numpy as np
from distance_metric import CovarianceNormalizedDistanceMetric

calc = CovarianceNormalizedDistanceMetric.SEUCLIDEAN.calculator
x: np.ndarray = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]], dtype=np.float64)
variance: np.ndarray = np.array([1.0, 4.0], dtype=np.float64)
result = calc.pairwise(array=x, variance=variance)
assert result.value.shape == (3, 3)
```
