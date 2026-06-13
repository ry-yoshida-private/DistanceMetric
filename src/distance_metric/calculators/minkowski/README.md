# minkowski

## Overview

This submodule implements **Minkowski-family** distances on NumPy arrays: Lp norms (Manhattan, Euclidean, configurable Minkowski), **Chebyshev** (L∞), and **squared Euclidean**. Each metric name is exposed as `MinkowskiDistanceMetric`, which resolves to the matching calculator instance under `calculators/`.

## Components

| Path | Description |
|------|-------------|
| [`calculator.py`](calculator.py) | Abstract base `MinkowskiDistanceCalculatorBase`: shared Lp logic (powered differences, reduction) for most metrics in this family. |
| [`metric.py`](metric.py) | `MinkowskiDistanceMetric` enum: string codes and `.calculator` mapping to each concrete calculator. |
| [`calculators/`](calculators/) | Concrete distance calculators (one module per metric variant). Per-file summary: [`calculators/README.md`](calculators/README.md). |

## Example

After installing the project (for example `pip install -e .` from the repo root), import `DistanceCalculator`, `DistanceResult`, and `MinkowskiDistanceMetric` from `distance_metric`, and concrete calculator classes from `distance_metric.calculators.minkowski` when you need them.

Use the `.calculator` property for a ready-to-use instance. For `MINKOWSKI`, the enum uses `norm_order=2.0` (Lp with *p* = 2); for a different exponent, construct `MinkowskiDistanceCalculator(norm_order=...)` (see the Minkowski subsection).

### Euclidean (L2)

```python
import numpy as np
from distance_metric import (
    DistanceCalculator,
    DistanceResult,
    MinkowskiDistanceMetric,
)
from distance_metric.calculators.minkowski import EuclideanDistanceCalculator

calc: DistanceCalculator = MinkowskiDistanceMetric.EUCLIDEAN.calculator
# or: calc = EuclideanDistanceCalculator()

x: np.ndarray = np.array([[0.0, 0.0], [3.0, 4.0]], dtype=np.float64)
result: DistanceResult = calc.pairwise(array=x)
print(result.value)  # np.ndarray, shape (n, n) == (2, 2)
```

### Manhattan (L1)

```python
import numpy as np
from distance_metric import (
    DistanceCalculator,
    DistanceResult,
    MinkowskiDistanceMetric,
)
from distance_metric.calculators.minkowski import ManhattanDistanceCalculator

calc: DistanceCalculator = MinkowskiDistanceMetric.MANHATTAN.calculator
# or: calc = ManhattanDistanceCalculator()

q: np.ndarray = np.array([[1.0, 2.0]], dtype=np.float64)
g: np.ndarray = np.array([[4.0, 6.0]], dtype=np.float64)
result: DistanceResult = calc.cross(query_array=q, gallery_array=g)
print(result.value)  # np.ndarray, shape (n, m) == (1, 1)
```

### Minkowski (Lp)

```python
import numpy as np
from distance_metric import (
    DistanceCalculator,
    DistanceResult,
    MinkowskiDistanceMetric,
)
from distance_metric.calculators.minkowski import MinkowskiDistanceCalculator

calc: DistanceCalculator = MinkowskiDistanceMetric.MINKOWSKI.calculator
# default is p=2; for a custom Lp exponent use e.g.:
# calc = MinkowskiDistanceCalculator(norm_order=1.5)

q: np.ndarray = np.array([1.0, 2.0], dtype=np.float64)
g: np.ndarray = np.array([4.0, 6.0], dtype=np.float64)
result: DistanceResult = calc.elementwise(query_array=q, gallery_array=g)
print(result.value)  # np.ndarray, scalar (0-d)
```

### Chebyshev (L∞)

```python
import numpy as np
from distance_metric import (
    DistanceCalculator,
    DistanceResult,
    MinkowskiDistanceMetric,
)
from distance_metric.calculators.minkowski import ChebyshevDistanceCalculator

calc: DistanceCalculator = MinkowskiDistanceMetric.CHEBYSHEV.calculator
# or: calc = ChebyshevDistanceCalculator()

x: np.ndarray = np.array([[0.0, 0.0], [1.0, 2.0]], dtype=np.float64)
result: DistanceResult = calc.pairwise(array=x)
print(result.value)  # np.ndarray, shape (2, 2)
```

### Squared Euclidean

```python
import numpy as np
from distance_metric import (
    DistanceCalculator,
    DistanceResult,
    MinkowskiDistanceMetric,
)
from distance_metric.calculators.minkowski import SquaredEuclideanDistanceCalculator

calc: DistanceCalculator = MinkowskiDistanceMetric.SQUARED_EUCLIDEAN.calculator
# or: calc = SquaredEuclideanDistanceCalculator()

x: np.ndarray = np.array([[0.0, 0.0], [3.0, 4.0]], dtype=np.float64)
result: DistanceResult = calc.pairwise(array=x)
print(result.value)  # np.ndarray, shape (2, 2)
```
