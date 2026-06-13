# geodesic_sequence

## Overview

**Spatial and sequential** metrics: great-circle (**Haversine**) distance on latitude–longitude pairs (last axis: latitude, longitude in radians), and **Dynamic Time Warping** alignment cost between rows treated as sequences. DTW `cross`/`pairwise` require **full-array broadcast compatibility** on `query_array` and `gallery_array` (for example one query row vs many gallery rows). Pass **`radius`** for Haversine.

## Components

| Path | Description |
|------|-------------|
| [`metric.py`](metric.py) | `GeodesicSequenceDistanceMetric` enum. |
| [`calculators/`](calculators/) | `HaversineDistanceCalculator` and `DynamicTimeWarpingDistanceCalculator`. |

## Examples

### Haversine (pairwise)

Distances use the same unit as `radius` (default Earth radius in kilometers).

```python
import numpy as np
from distance_metric import GeodesicSequenceDistanceMetric, DistanceResult

calc = GeodesicSequenceDistanceMetric.HAVERSINE.calculator
coords: np.ndarray = np.array(
    [
        [np.radians(35.68), np.radians(139.76)],
        [np.radians(34.69), np.radians(135.50)],
    ],
    dtype=np.float64,
)
result: DistanceResult = calc.pairwise(array=coords, radius=6371.0)
assert result.value.shape == (2, 2)
```

### Dynamic Time Warping (cross)

Use batches that broadcast together; here one query sequence versus two gallery sequences of equal length.

```python
import numpy as np
from distance_metric import GeodesicSequenceDistanceMetric

calc = GeodesicSequenceDistanceMetric.DYNAMIC_TIME_WARPING.calculator
query_array = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]], dtype=np.float64)
gallery_array = np.array(
    [[1.0, 1.0, 1.0, 1.0, 1.0], [5.0, 4.0, 3.0, 2.0, 1.0]],
    dtype=np.float64,
)
result = calc.cross(query_array=query_array, gallery_array=gallery_array)
assert result.value.shape == (1, 2)
```
