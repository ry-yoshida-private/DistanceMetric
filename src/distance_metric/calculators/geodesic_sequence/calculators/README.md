# calculators

## Overview

Calculators for **geo coordinates** and **time series / sequences**.

## Components

| Path | Description |
|------|-------------|
| [`haversine.py`](haversine.py) | Haversine formula for shortest distance on a sphere between (lat, lon) samples. |
| [`dynamic_time_warping.py`](dynamic_time_warping.py) | DTW path cost between sequences along the row axis; Numba fast paths with NumPy fallback. |
| [`utils/`](utils/) | DTW Numba kernels (`dtw_numba_kernels.py`) and pair-cost dispatch / NumPy fallback (`dtw_pair_cost.py`). |

## Examples

### [`haversine.py`](haversine.py)

```python
import numpy as np
from distance_metric.calculators.geodesic_sequence import HaversineDistanceCalculator

calc = HaversineDistanceCalculator()
result = calc.cross(
    query_array=np.array([[0.0, 0.0]], dtype=np.float64),
    gallery_array=np.array([[0.0, np.radians(1.0)]], dtype=np.float64),
    radius=6371.0,
)
assert result.value.shape == (1, 1)
```

### [`dynamic_time_warping.py`](dynamic_time_warping.py)

```python
import numpy as np
from distance_metric.calculators.geodesic_sequence import DynamicTimeWarpingDistanceCalculator

calc = DynamicTimeWarpingDistanceCalculator()
result = calc.pairwise(array=np.array([[1.0, 3.0, 2.0], [1.0, 2.0, 4.0]], dtype=np.float64))
assert result.value.shape == (2, 2)
```
