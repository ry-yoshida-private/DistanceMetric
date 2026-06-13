"""Tests for Minkowski-family distance calculators."""

from __future__ import annotations

import numpy as np
import pytest

from distance_metric.calculators.minkowski import (
    ChebyshevDistanceCalculator,
    EuclideanDistanceCalculator,
    ManhattanDistanceCalculator,
    MinkowskiDistanceCalculator,
    MinkowskiDistanceMetric,
    SquaredEuclideanDistanceCalculator,
)
from distance_metric.result import DistanceResultType


class TestManhattanDistance:
    """L1 distance."""

    @pytest.fixture
    def calculator(self) -> ManhattanDistanceCalculator:
        return ManhattanDistanceCalculator()

    def test_cross_known_value(self, calculator: ManhattanDistanceCalculator) -> None:
        query = np.array([[1.0, 2.0]])
        gallery = np.array([[4.0, 6.0]])
        result = calculator.cross(query, gallery)
        np.testing.assert_allclose(result.value, np.array([[7.0]]))

    def test_metric_enum(self, calculator: ManhattanDistanceCalculator) -> None:
        assert calculator.metric is MinkowskiDistanceMetric.MANHATTAN


class TestEuclideanDistance:
    """L2 distance."""

    @pytest.fixture
    def calculator(self) -> EuclideanDistanceCalculator:
        return EuclideanDistanceCalculator()

    def test_pairwise_right_triangle(
        self,
        calculator: EuclideanDistanceCalculator,
        numeric_batch: np.ndarray,
    ) -> None:
        result = calculator.pairwise(numeric_batch)
        np.testing.assert_allclose(result.value[0, 1], 5.0)
        assert result.type is DistanceResultType.DISTANCE

    def test_elementwise(
        self,
        calculator: EuclideanDistanceCalculator,
    ) -> None:
        result = calculator.elementwise(np.array([1.0, 2.0]), np.array([4.0, 6.0]))
        np.testing.assert_allclose(result.value, np.array([5.0]))


class TestSquaredEuclideanDistance:
    """Squared L2 distance."""

    @pytest.fixture
    def calculator(self) -> SquaredEuclideanDistanceCalculator:
        return SquaredEuclideanDistanceCalculator()

    def test_cross_is_square_of_euclidean(
        self,
        calculator: SquaredEuclideanDistanceCalculator,
    ) -> None:
        query = np.array([[1.0, 2.0]])
        gallery = np.array([[4.0, 6.0]])
        squared = calculator.cross(query, gallery).value
        euclidean = EuclideanDistanceCalculator().cross(query, gallery).value
        np.testing.assert_allclose(squared, euclidean**2)


class TestChebyshevDistance:
    """L-infinity distance."""

    @pytest.fixture
    def calculator(self) -> ChebyshevDistanceCalculator:
        return ChebyshevDistanceCalculator()

    def test_cross_uses_max_coordinate_gap(
        self,
        calculator: ChebyshevDistanceCalculator,
    ) -> None:
        query = np.array([[1.0, 10.0]])
        gallery = np.array([[4.0, 3.0]])
        result = calculator.cross(query, gallery)
        np.testing.assert_allclose(result.value, np.array([[7.0]]))


class TestMinkowskiDistance:
    """Configurable Lp distance."""

    def test_norm_order_three(self) -> None:
        calculator = MinkowskiDistanceCalculator(norm_order=3.0)
        query = np.array([[0.0, 0.0]])
        gallery = np.array([[3.0, 4.0]])
        result = calculator.cross(query, gallery)
        np.testing.assert_allclose(result.value, np.array([[91.0 ** (1.0 / 3.0)]]))

    def test_metric_enum(self) -> None:
        calculator = MinkowskiDistanceCalculator()
        assert calculator.metric is MinkowskiDistanceMetric.MINKOWSKI
