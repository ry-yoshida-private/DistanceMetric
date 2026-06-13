# calculators

## Overview

Concrete calculators for **binary / set** comparisons on aligned feature vectors.

## Components

| Path | Description |
|------|-------------|
| [`hamming.py`](hamming.py) | Hamming distance: fraction (or count semantics per implementation) of disagreeing coordinates. |
| [`jaccard.py`](jaccard.py) | Jaccard-derived distance from boolean or set-like masks along features. |

## Examples

### [`hamming.py`](hamming.py)

```python
import numpy as np
from distance_metric.calculators.binary import HammingDistanceCalculator

calc = HammingDistanceCalculator()
result = calc.elementwise(
    query_array=np.array([0.0, 1.0, 1.0], dtype=np.float64),
    gallery_array=np.array([1.0, 1.0, 0.0], dtype=np.float64),
)
assert result.value.shape == (1,)
```

### [`jaccard.py`](jaccard.py)

```python
import numpy as np
from distance_metric.calculators.binary import JaccardDistanceCalculator

calc = JaccardDistanceCalculator()
result = calc.cross(
    query_array=np.array([[1, 0, 1]], dtype=np.float64),
    gallery_array=np.array([[1, 1, 0]], dtype=np.float64),
)
assert result.value.shape == (1, 1)
```
