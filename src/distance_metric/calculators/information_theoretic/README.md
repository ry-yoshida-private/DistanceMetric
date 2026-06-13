# information_theoretic

## Overview

**Information-theoretic** divergences and distances treating vectors as nonnegative mass over aligned discrete bins. **KL divergence is asymmetric** (swapping query and gallery changes the value). Several calculators accept **`eps`** smoothing in kwargs.

## Components

| Path | Description |
|------|-------------|
| [`metric.py`](metric.py) | `InformationTheoreticDistanceMetric` enum. |
| [`calculators/`](calculators/) | KL, Jensen–Shannon, Bhattacharyya, and Hellinger calculators. |

## Examples

### KL divergence (directed)

```python
import numpy as np
from distance_metric import InformationTheoreticDistanceMetric

calc = InformationTheoreticDistanceMetric.KL_DIVERGENCE.calculator
p = np.array([[0.2, 0.3, 0.5]], dtype=np.float64)
q = np.array([[0.1, 0.4, 0.5]], dtype=np.float64)
forward = calc.cross(query_array=p, gallery_array=q)
backward = calc.cross(query_array=q, gallery_array=p)
assert forward.value.shape == (1, 1)
assert not np.allclose(forward.value, backward.value)
```

### Jensen–Shannon (symmetric)

```python
import numpy as np
from distance_metric import InformationTheoreticDistanceMetric

calc = InformationTheoreticDistanceMetric.JENSEN_SHANNON_DIVERGENCE.calculator
p = np.array([[0.25, 0.25, 0.25, 0.25]], dtype=np.float64)
q = np.array([[0.5, 0.5, 0.0, 0.0]], dtype=np.float64)
result = calc.cross(query_array=p, gallery_array=q, eps=1e-12)
assert result.value.shape == (1, 1)
```

### Bhattacharyya and Hellinger

```python
import numpy as np
from distance_metric import InformationTheoreticDistanceMetric

p = np.array([[0.25, 0.25, 0.25, 0.25]], dtype=np.float64)
q = np.array([[0.5, 0.5, 0.0, 0.0]], dtype=np.float64)
bh = InformationTheoreticDistanceMetric.BHATTACHARYYA.calculator.cross(p, q)
hell = InformationTheoreticDistanceMetric.HELLINGER.calculator.cross(p, q)
assert bh.value.shape == hell.value.shape == (1, 1)
```
