# calculators

## Overview

Concrete **divergence** calculators for nonnegative mass vectors.

## Components

| Path | Description |
|------|-------------|
| [`kl_divergence.py`](kl_divergence.py) | Kullback–Leibler divergence (query vs gallery asymmetric). |
| [`jensen_shannon_divergence.py`](jensen_shannon_divergence.py) | Jensen–Shannon divergence (symmetric blend of KL terms). |
| [`bhattacharyya.py`](bhattacharyya.py) | Bhattacharyya-related distance on distributions. |
| [`hellinger.py`](hellinger.py) | Hellinger distance derived from Hellinger affinity. |

## Examples

Mass vectors should be nonnegative; optional `eps` floors smoothing for logarithms where supported.

### [`kl_divergence.py`](kl_divergence.py)

```python
import numpy as np
from distance_metric.calculators.information_theoretic import KLDivergenceDistanceCalculator

calc = KLDivergenceDistanceCalculator()
result = calc.cross(
    query_array=np.array([[0.6, 0.4]], dtype=np.float64),
    gallery_array=np.array([[0.5, 0.5]], dtype=np.float64),
    eps=1e-12,
)
assert result.value.shape == (1, 1)
```

### [`jensen_shannon_divergence.py`](jensen_shannon_divergence.py)

```python
import numpy as np
from distance_metric.calculators.information_theoretic import JensenShannonDivergenceDistanceCalculator

calc = JensenShannonDivergenceDistanceCalculator()
result = calc.pairwise(
    array=np.array([[0.5, 0.5, 0.0], [0.25, 0.25, 0.5]], dtype=np.float64),
    eps=1e-12,
)
assert result.value.shape == (2, 2)
```

### [`bhattacharyya.py`](bhattacharyya.py)

```python
import numpy as np
from distance_metric.calculators.information_theoretic import BhattacharyyaDistanceCalculator

calc = BhattacharyyaDistanceCalculator()
result = calc.cross(
    query_array=np.array([[1.0, 2.0, 3.0]], dtype=np.float64),
    gallery_array=np.array([[3.0, 2.0, 1.0]], dtype=np.float64),
)
assert result.value.shape == (1, 1)
```

### [`hellinger.py`](hellinger.py)

```python
import numpy as np
from distance_metric.calculators.information_theoretic import HellingerDistanceCalculator

calc = HellingerDistanceCalculator()
result = calc.cross(
    query_array=np.array([[1.0, 0.0]], dtype=np.float64),
    gallery_array=np.array([[0.25, 0.75]], dtype=np.float64),
)
assert result.value.shape == (1, 1)
```
