# calculators

## Overview

Concrete **distance and similarity calculators** grouped by mathematical family. Each subdirectory exposes an enum subclass of `DistanceMetric` (lazy-loaded calculators) and one module per metric implementation. The package [`__init__.py`](__init__.py) re-exports calculators and enums for convenient imports.

## Components

| Path | Description |
|------|-------------|
| [`cross_elementwise.py`](cross_elementwise.py) | `CrossElementwiseCalculatorBase`: broadcast query vs gallery, per-coordinate terms, then reduce to an `(n, m)` matrix. |
| [`binary/`](binary/) | Binary / set-style metrics (Hamming, Jaccard). |
| [`covariance_normalized/`](covariance_normalized/) | Mahalanobis and standardized Euclidean (covariance / variance kwargs). |
| [`geodesic_sequence/`](geodesic_sequence/) | Haversine (geo) and Dynamic Time Warping (sequences). |
| [`information_theoretic/`](information_theoretic/) | KL, Jensen–Shannon, Bhattacharyya, Hellinger on nonnegative mass vectors. |
| [`minkowski/`](minkowski/) | Lp family: Manhattan, Euclidean, Minkowski, Chebyshev, squared Euclidean. |
| [`ratio_based/`](ratio_based/) | Canberra and Bray–Curtis for nonnegative features. |
| [`similarity_correlation/`](similarity_correlation/) | Cosine, Pearson correlation distance, Spearman, Kendall. |
| [`transport/`](transport/) | Huber loss sum and 1-D Wasserstein-style distance. |

## Examples

### Import from the package root

```python
from distance_metric import BinaryDistanceMetric, MinkowskiDistanceMetric

hamming = BinaryDistanceMetric.HAMMING.calculator
euclidean = MinkowskiDistanceMetric.EUCLIDEAN.calculator
```

### Import concrete calculator classes

```python
from distance_metric.calculators.binary import HammingDistanceCalculator
from distance_metric.calculators.minkowski import EuclideanDistanceCalculator

HammingDistanceCalculator()
EuclideanDistanceCalculator()
```

### Shared kwargs pattern

Family README files document required kwargs. Typical calls look like:

```python
import numpy as np
from distance_metric import CovarianceNormalizedDistanceMetric

calc = CovarianceNormalizedDistanceMetric.MAHALANOBIS.calculator
x = np.array([[0.0, 0.0], [1.0, 2.0]], dtype=np.float64)
cov = np.eye(2, dtype=np.float64)
out = calc.pairwise(array=x, cov=cov)
```

See [`cross_elementwise.py`](cross_elementwise.py) for the broadcasting template many metrics inherit.
