# transport

## Overview

**Transport-flavored** metrics: summed **Huber** losses per coordinate (threshold **`delta`** must be positive) and **1-D Wasserstein-style** distance from sorting aligned coordinates along the last axis.

## Components

| Path | Description |
|------|-------------|
| [`metric.py`](metric.py) | `TransportDistanceMetric` enum. |
| [`calculators/`](calculators/) | `HuberDistanceCalculator` and `WassersteinDistanceCalculator`. |

## Examples

### Huber distance

```python
import numpy as np
from distance_metric import TransportDistanceMetric

calc = TransportDistanceMetric.HUBER_DISTANCE.calculator
query_array = np.array([[0.0, 3.0]], dtype=np.float64)
gallery_array = np.array([[0.0, 10.0]], dtype=np.float64)
result = calc.cross(query_array=query_array, gallery_array=gallery_array, delta=1.0)
assert result.value.shape == (1, 1)
```

### Wasserstein (sorted marginals)

```python
import numpy as np
from distance_metric import TransportDistanceMetric

calc = TransportDistanceMetric.WASSERSTEIN.calculator
query_array = np.array([[3.0, 1.0, 2.0]], dtype=np.float64)
gallery_array = np.array([[1.0, 2.0, 3.0]], dtype=np.float64)
result = calc.cross(query_array=query_array, gallery_array=gallery_array)
assert result.value.shape == (1, 1)
```
