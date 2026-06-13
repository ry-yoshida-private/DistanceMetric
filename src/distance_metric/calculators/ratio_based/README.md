# ratio_based

## Overview

**Ratio-based** distances suited to nonnegative abundance or count-like features: **Canberra** (normalized absolute gaps per coordinate) and **Bray–Curtis** (gap sum relative to combined row mass with correct row coupling in cross mode).

## Components

| Path | Description |
|------|-------------|
| [`metric.py`](metric.py) | `RatioBasedDistanceMetric` enum. |
| [`calculators/`](calculators/) | `CanberraDistanceCalculator` and `BrayCurtisDistanceCalculator`. |

## Examples

### Canberra

```python
import numpy as np
from distance_metric import RatioBasedDistanceMetric

calc = RatioBasedDistanceMetric.CANBERRA.calculator
query_array = np.array([[1.0, 2.0, 0.5]], dtype=np.float64)
gallery_array = np.array([[2.0, 0.0, 1.0]], dtype=np.float64)
result = calc.cross(query_array=query_array, gallery_array=gallery_array)
assert result.value.shape == (1, 1)
```

### Bray–Curtis

```python
import numpy as np
from distance_metric import RatioBasedDistanceMetric

calc = RatioBasedDistanceMetric.BRAY_CURTIS.calculator
query_array = np.array([[1.0, 2.0, 3.0]], dtype=np.float64)
gallery_array = np.array([[3.0, 2.0, 1.0]], dtype=np.float64)
result = calc.cross(query_array=query_array, gallery_array=gallery_array)
assert result.value.shape == (1, 1)
```
