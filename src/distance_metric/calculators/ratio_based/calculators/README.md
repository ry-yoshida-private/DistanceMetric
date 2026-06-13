# calculators

## Overview

Implementations of **relative-difference** metrics on nonnegative vectors.

## Components

| Path | Description |
|------|-------------|
| [`canberra.py`](canberra.py) | Canberra distance: sum of \|x_i - y_i\| / (\|x_i\| + \|y_i\|) style terms per coordinate. |
| [`bray_curtis.py`](bray_curtis.py) | Bray–Curtis dissimilarity with batch coupling for denominators in cross mode. |

## Examples

### [`canberra.py`](canberra.py)

```python
import numpy as np
from distance_metric.calculators.ratio_based import CanberraDistanceCalculator

calc = CanberraDistanceCalculator()
result = calc.pairwise(array=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64))
assert result.value.shape == (2, 2)
```

### [`bray_curtis.py`](bray_curtis.py)

```python
import numpy as np
from distance_metric.calculators.ratio_based import BrayCurtisDistanceCalculator

calc = BrayCurtisDistanceCalculator()
result = calc.cross(
    query_array=np.array([[10.0, 0.0]], dtype=np.float64),
    gallery_array=np.array([[5.0, 5.0]], dtype=np.float64),
)
assert result.value.shape == (1, 1)
```
