# Binary Distance Calculators

## Overview

Boolean- and integer-coded vector distances (Hamming, Jaccard). Inputs use
`BinaryArray` (`bool` or integer dtypes); floating-point feature vectors are
rejected at runtime.

## Components

| Component | Description |
| --- | --- |
| [`base.py`](base.py) | `BinaryCrossElementwiseCalculatorBase` with `BinaryArray` public API |
| [`metric.py`](metric.py) | `BinaryDistanceMetric` enum and calculator factory |
| [`calculators/hamming.py`](calculators/hamming.py) | Normalized coordinate mismatch fraction |
| [`calculators/jaccard.py`](calculators/jaccard.py) | Set-based Jaccard distance on membership vectors |

## Examples

```python
import numpy as np
from distance_metric.calculators.binary import HammingDistanceCalculator

# Boolean vectors
bits = np.array([[True, False, True], [False, True, False]])
calc = HammingDistanceCalculator()
print(calc.pairwise(bits).value)

# Integer-coded 0/1 vectors
codes = np.array([[1, 0, 1], [0, 1, 0]], dtype=np.int8)
print(calc.pairwise(codes).value)
```
