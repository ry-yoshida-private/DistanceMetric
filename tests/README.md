# Tests

## Overview

Pytest suite covering the full `distance_metric` package under `src/`: all 24
concrete calculators, `DistanceResult`, metric enum factories, and cross-cutting
API consistency.

## Components

| Component | Description |
| --- | --- |
| [`conftest.py`](conftest.py) | Shared numeric, binary, and probability fixtures |
| [`test_result.py`](test_result.py) | `DistanceResult` property helpers |
| [`test_api_consistency.py`](test_api_consistency.py) | `pairwise` / `cross` / `elementwise` consistency |
| [`test_metric_registry.py`](test_metric_registry.py) | Parametrized smoke tests for all metric enums |
| [`calculators/binary/`](calculators/binary/) | Hamming and Jaccard |
| [`calculators/minkowski/`](calculators/minkowski/) | Lp norm family |
| [`calculators/covariance_normalized/`](calculators/covariance_normalized/) | Mahalanobis and SEuclidean |
| [`calculators/geodesic_sequence/`](calculators/geodesic_sequence/) | Haversine and DTW |
| [`calculators/information_theoretic/`](calculators/information_theoretic/) | KL, JS, Bhattacharyya, Hellinger |
| [`calculators/ratio_based/`](calculators/ratio_based/) | Canberra and Bray-Curtis |
| [`calculators/similarity_correlation/`](calculators/similarity_correlation/) | Cosine, correlation, Spearman, Kendall |
| [`calculators/transport/`](calculators/transport/) | Huber and Wasserstein |

## Examples

```bash
pip install -e ".[dev]"
pytest
pytest tests/calculators -v
```
