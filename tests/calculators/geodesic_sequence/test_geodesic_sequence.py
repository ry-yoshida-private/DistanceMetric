"""Tests for geodesic and sequence distance calculators."""

from __future__ import annotations

import numpy as np
import pytest

from distance_metric.calculators.geodesic_sequence import (
    DynamicTimeWarpingDistanceCalculator,
    GeodesicSequenceDistanceMetric,
    HaversineDistanceCalculator,
)


class TestHaversineDistance:
    """Great-circle distance on a sphere."""

    @pytest.fixture
    def calculator(self) -> HaversineDistanceCalculator:
        return HaversineDistanceCalculator()

    def test_identical_points_zero_distance(
        self,
        calculator: HaversineDistanceCalculator,
    ) -> None:
        points = np.array([[0.5, 1.0], [0.5, 1.0]])
        result = calculator.pairwise(points, radius=1.0)
        np.testing.assert_allclose(np.diag(result.value), 0.0)

    def test_invalid_last_axis_raises(
        self,
        calculator: HaversineDistanceCalculator,
    ) -> None:
        query = np.array([[0.0, 0.0, 0.0]])
        gallery = np.array([[0.1, 0.1, 0.1]])
        with pytest.raises(ValueError, match="latitude, longitude"):
            calculator.cross(query, gallery)

    def test_metric_enum(self, calculator: HaversineDistanceCalculator) -> None:
        assert calculator.metric is GeodesicSequenceDistanceMetric.HAVERSINE


class TestDynamicTimeWarpingDistance:
    """Dynamic time warping distance."""

    @pytest.fixture
    def calculator(self) -> DynamicTimeWarpingDistanceCalculator:
        return DynamicTimeWarpingDistanceCalculator()

    def test_identical_series_zero_distance(
        self,
        calculator: DynamicTimeWarpingDistanceCalculator,
    ) -> None:
        series = np.array([[1.0, 2.0, 3.0]])
        result = calculator.pairwise(series)
        np.testing.assert_allclose(result.value, np.array([[0.0]]))

    def test_cross_returns_finite_matrix(
        self,
        calculator: DynamicTimeWarpingDistanceCalculator,
    ) -> None:
        query = np.array([[1.0, 2.0, 3.0]])
        gallery = np.array([[1.0, 2.5, 3.0], [0.0, 1.0, 2.0]])
        result = calculator.cross(query, gallery)
        assert result.value.shape == (1, 2)
        assert np.all(result.value >= 0.0)

    def test_elementwise_shape(
        self,
        calculator: DynamicTimeWarpingDistanceCalculator,
    ) -> None:
        result = calculator.elementwise(
            np.array([1.0, 2.0]),
            np.array([1.0, 3.0]),
        )
        assert result.value.shape == (1,)

    def test_metric_enum(self, calculator: DynamicTimeWarpingDistanceCalculator) -> None:
        assert calculator.metric is GeodesicSequenceDistanceMetric.DYNAMIC_TIME_WARPING
