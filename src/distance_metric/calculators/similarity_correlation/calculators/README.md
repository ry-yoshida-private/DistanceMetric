# calculators

## Overview

Concrete calculators derived from **vector similarity** and **rank / correlation** statistics.

## Components

| Path | Description |
|------|-------------|
| [`cosine.py`](cosine.py) | Cosine-based distance along the last axis. |
| [`correlation.py`](correlation.py) | Pearson correlation transformed into a distance after centering. |
| [`spearman.py`](spearman.py) | Distance from Spearman rank correlation per row/pair. |
| [`kendall.py`](kendall.py) | Kendall tau-based distance from pairwise ordering concordance. |

## Examples

### [`cosine.py`](cosine.py)

```python
import numpy as np
from distance_metric.calculators.similarity_correlation import CosineDistanceCalculator

calc = CosineDistanceCalculator()
result = calc.cross(
    query_array=np.array([[3.0, 4.0]], dtype=np.float64),
    gallery_array=np.array([[5.0, 12.0]], dtype=np.float64),
)
assert result.value.shape == (1, 1)
```

### [`correlation.py`](correlation.py)

```python
import numpy as np
from distance_metric.calculators.similarity_correlation import CorrelationDistanceCalculator

calc = CorrelationDistanceCalculator()
result = calc.pairwise(array=np.array([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]], dtype=np.float64))
assert result.value.shape == (2, 2)
```

### [`spearman.py`](spearman.py)

```python
import numpy as np
from distance_metric.calculators.similarity_correlation import SpearmanDistanceCalculator

calc = SpearmanDistanceCalculator()
result = calc.cross(
    query_array=np.array([[10.0, 20.0, 30.0]], dtype=np.float64),
    gallery_array=np.array([[30.0, 20.0, 10.0]], dtype=np.float64),
)
assert result.value.shape == (1, 1)
```

### [`kendall.py`](kendall.py)

```python
import numpy as np
from distance_metric.calculators.similarity_correlation import KendallDistanceCalculator

calc = KendallDistanceCalculator()
result = calc.pairwise(array=np.array([[1.0, 3.0, 2.0], [1.0, 2.0, 4.0]], dtype=np.float64))
assert result.value.shape == (2, 2)
```
