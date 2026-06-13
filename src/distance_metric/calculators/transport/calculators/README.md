# calculators

## Overview

Robust **loss** and **sorted-marginal** distance calculators.

## Components

| Path | Description |
|------|-------------|
| [`huber_distance.py`](huber_distance.py) | Sum of Huber losses between aligned coordinates; uses `delta` kwarg. |
| [`wasserstein.py`](wasserstein.py) | Mean absolute difference between sorted coordinates (equal weights per dimension). |

## Examples

### [`huber_distance.py`](huber_distance.py)

```python
import numpy as np
from distance_metric.calculators.transport import HuberDistanceCalculator

calc = HuberDistanceCalculator()
result = calc.pairwise(array=np.array([[0.0, 1.0], [2.0, -3.0]], dtype=np.float64), delta=1.0)
assert result.value.shape == (2, 2)
```

### [`wasserstein.py`](wasserstein.py)

```python
import numpy as np
from distance_metric.calculators.transport import WassersteinDistanceCalculator

calc = WassersteinDistanceCalculator()
result = calc.cross(
    query_array=np.array([[0.0, 10.0, 5.0]], dtype=np.float64),
    gallery_array=np.array([[10.0, 0.0, 5.0]], dtype=np.float64),
)
assert result.value.shape == (1, 1)
```
