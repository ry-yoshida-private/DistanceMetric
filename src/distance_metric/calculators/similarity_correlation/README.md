# similarity_correlation

## Overview

**Similarity and correlation** metrics expressed as distance-style scores along rows: **cosine**, **Pearson correlation distance** (centering per row in cross mode), **Spearman**, and **Kendall**. Results use `DistanceResult` like other calculators.

## Components

| Path | Description |
|------|-------------|
| [`metric.py`](metric.py) | `SimilarityCorrelationDistanceMetric` enum. |
| [`calculators/`](calculators/) | Cosine, correlation, Spearman, and Kendall calculators. |

## Examples

### Cosine distance

Distance is one minus cosine similarity on the last axis.

```python
import numpy as np
from distance_metric import SimilarityCorrelationDistanceMetric

calc = SimilarityCorrelationDistanceMetric.COSINE.calculator
query_array = np.array([[1.0, 0.0, 1.0]], dtype=np.float64)
gallery_array = np.array([[1.0, 1.0, 0.0]], dtype=np.float64)
result = calc.cross(query_array=query_array, gallery_array=gallery_array)
assert result.value.shape == (1, 1)
```

### Correlation, Spearman, Kendall

```python
import numpy as np
from distance_metric import SimilarityCorrelationDistanceMetric

x = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float64)
y = np.array([[4.0, 3.0, 2.0, 1.0]], dtype=np.float64)

corr = SimilarityCorrelationDistanceMetric.CORRELATION.calculator.cross(x, y)
spearman = SimilarityCorrelationDistanceMetric.SPEARMAN.calculator.cross(x, y)
kendall = SimilarityCorrelationDistanceMetric.KENDALL.calculator.cross(x, y)
assert corr.value.shape == spearman.value.shape == kendall.value.shape == (1, 1)
```
