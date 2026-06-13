"""Tests for transport distance calculators."""

from __future__ import annotations

import numpy as np
import pytest

from distance_metric.calculators.transport import (
    HuberDistanceCalculator,
    TransportDistanceMetric,
    WassersteinDistanceCalculator,
)


class TestHuberDistance:
    """Huber loss distance."""

    @pytest.fixture
    def calculator(self) -> HuberDistanceCalculator:
        return HuberDistanceCalculator()

    def test_identical_rows_zero(
        self,
        calculator: HuberDistanceCalculator,
    ) -> None:
        batch = np.array([[0.0, 1.0, 2.0]])
        result = calculator.pairwise(batch, delta=1.0)
        np.testing.assert_allclose(result.value, np.array([[0.0]]))

    def test_quadratic_region(self, calculator: HuberDistanceCalculator) -> None:
        query = np.array([[0.0, 0.5]])
        gallery = np.array([[0.0, 0.0]])
        result = calculator.cross(query, gallery, delta=1.0)
        np.testing.assert_allclose(result.value, np.array([[0.125]]))

    def test_metric_enum(self, calculator: HuberDistanceCalculator) -> None:
        assert calculator.metric is TransportDistanceMetric.HUBER_DISTANCE


class TestWassersteinDistance:
    """Sorted L1 (1D Wasserstein) distance."""

    @pytest.fixture
    def calculator(self) -> WassersteinDistanceCalculator:
        return WassersteinDistanceCalculator()

    def test_identical_rows_zero(
        self,
        calculator: WassersteinDistanceCalculator,
    ) -> None:
        batch = np.array([[1.0, 2.0, 3.0]])
        result = calculator.pairwise(batch)
        np.testing.assert_allclose(result.value, np.array([[0.0]]))

    def test_reversed_order_zero_when_sorted_marginals_match(
        self,
        calculator: WassersteinDistanceCalculator,
    ) -> None:
        query = np.array([[1.0, 2.0, 3.0]])
        gallery = np.array([[3.0, 2.0, 1.0]])
        result = calculator.cross(query, gallery)
        np.testing.assert_allclose(result.value, np.array([[0.0]]))

    def test_different_marginals_positive_distance(
        self,
        calculator: WassersteinDistanceCalculator,
    ) -> None:
        query = np.array([[1.0, 2.0, 3.0]])
        gallery = np.array([[1.0, 1.0, 4.0]])
        result = calculator.cross(query, gallery)
        assert result.value[0, 0] > 0.0

    def test_metric_enum(self, calculator: WassersteinDistanceCalculator) -> None:
        assert calculator.metric is TransportDistanceMetric.WASSERSTEIN
