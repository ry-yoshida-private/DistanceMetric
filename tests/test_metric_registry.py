"""Smoke tests for every metric enum calculator factory."""

from __future__ import annotations

from enum import Enum
from typing import Any

import numpy as np
import pytest

from distance_metric.calculator import DistanceCalculator
from distance_metric.calculators import (
    BinaryDistanceMetric,
    CovarianceNormalizedDistanceMetric,
    GeodesicSequenceDistanceMetric,
    InformationTheoreticDistanceMetric,
    MinkowskiDistanceMetric,
    RatioBasedDistanceMetric,
    SimilarityCorrelationDistanceMetric,
    TransportDistanceMetric,
)
from distance_metric.result import DistanceResultType


def _cross_kwargs(metric: Enum) -> dict[str, Any]:
    """Return metric-specific keyword arguments for cross()."""
    match metric:
        case CovarianceNormalizedDistanceMetric.MAHALANOBIS:
            return {"cov": np.eye(2, dtype=np.float64)}
        case CovarianceNormalizedDistanceMetric.SEUCLIDEAN:
            return {"variance": np.array([1.0, 1.0], dtype=np.float64)}
        case GeodesicSequenceDistanceMetric.HAVERSINE:
            return {"radius": 1.0}
        case TransportDistanceMetric.HUBER_DISTANCE:
            return {"delta": 1.0}
        case _:
            return {}


def _query_gallery(metric: Enum) -> tuple[np.ndarray, np.ndarray]:
    """Return minimal query and gallery arrays for a metric family."""
    match metric:
        case BinaryDistanceMetric.HAMMING | BinaryDistanceMetric.JACCARD:
            query = np.array([[True, False, True]])
            gallery = np.array([[False, True, False], [True, False, True]])
        case GeodesicSequenceDistanceMetric.HAVERSINE:
            query = np.array([[0.0, 0.0]])
            gallery = np.array([[0.1, 0.1], [0.0, 0.2]])
        case GeodesicSequenceDistanceMetric.DYNAMIC_TIME_WARPING:
            query = np.array([[1.0, 2.0, 3.0]])
            gallery = np.array([[1.0, 2.5, 3.0], [0.0, 1.0, 2.0]])
        case (
            InformationTheoreticDistanceMetric.KL_DIVERGENCE
            | InformationTheoreticDistanceMetric.JENSEN_SHANNON_DIVERGENCE
            | InformationTheoreticDistanceMetric.BHATTACHARYYA
            | InformationTheoreticDistanceMetric.HELLINGER
        ):
            query = np.array([[0.5, 0.5]])
            gallery = np.array([[0.25, 0.75], [0.5, 0.5]])
        case SimilarityCorrelationDistanceMetric.KENDALL:
            query = np.array([[1.0, 2.0, 3.0]])
            gallery = np.array([[1.0, 3.0, 2.0], [3.0, 2.0, 1.0]])
        case _:
            query = np.array([[1.0, 2.0]])
            gallery = np.array([[4.0, 6.0], [1.0, 2.0]])
    return query, gallery


ALL_METRICS: list[Enum] = [
    *BinaryDistanceMetric,
    *MinkowskiDistanceMetric,
    *CovarianceNormalizedDistanceMetric,
    *GeodesicSequenceDistanceMetric,
    *InformationTheoreticDistanceMetric,
    *RatioBasedDistanceMetric,
    *SimilarityCorrelationDistanceMetric,
    *TransportDistanceMetric,
]


@pytest.mark.parametrize("metric", ALL_METRICS, ids=lambda metric: metric.name)
class TestMetricRegistry:
    """Every enum member resolves to a working calculator."""

    def test_calculator_returns_distance_result(self, metric: Enum) -> None:
        calculator: DistanceCalculator = metric.calculator
        query, gallery = _query_gallery(metric)
        kwargs = _cross_kwargs(metric)
        result = calculator.cross(query, gallery, **kwargs)
        assert result.type is DistanceResultType.DISTANCE
        assert result.value.shape == (query.shape[0], gallery.shape[0])
        assert np.all(np.isfinite(result.value))

    def test_calculator_metric_round_trip(self, metric: Enum) -> None:
        calculator = metric.calculator
        assert calculator.metric is metric
