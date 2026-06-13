"""Tests for ratio-based distance calculators."""

from __future__ import annotations

import numpy as np
import pytest

from distance_metric.calculators.ratio_based import (
    BrayCurtisDistanceCalculator,
    CanberraDistanceCalculator,
    RatioBasedDistanceMetric,
)


class TestCanberraDistance:
    """Canberra distance."""

    @pytest.fixture
    def calculator(self) -> CanberraDistanceCalculator:
        return CanberraDistanceCalculator()

    def test_identical_rows_zero(
        self,
        calculator: CanberraDistanceCalculator,
    ) -> None:
        batch = np.array([[1.0, 2.0, 0.0]])
        result = calculator.pairwise(batch)
        np.testing.assert_allclose(result.value, np.array([[0.0]]))

    def test_cross_known_value(self, calculator: CanberraDistanceCalculator) -> None:
        query = np.array([[1.0, 0.0]])
        gallery = np.array([[0.0, 1.0]])
        result = calculator.cross(query, gallery)
        np.testing.assert_allclose(result.value, np.array([[2.0]]))

    def test_metric_enum(self, calculator: CanberraDistanceCalculator) -> None:
        assert calculator.metric is RatioBasedDistanceMetric.CANBERRA


class TestBrayCurtisDistance:
    """Bray-Curtis distance."""

    @pytest.fixture
    def calculator(self) -> BrayCurtisDistanceCalculator:
        return BrayCurtisDistanceCalculator()

    def test_identical_rows_zero(
        self,
        calculator: BrayCurtisDistanceCalculator,
    ) -> None:
        batch = np.array([[1.0, 1.0], [2.0, 0.0]])
        result = calculator.pairwise(batch)
        np.testing.assert_allclose(np.diag(result.value), 0.0)

    def test_cross_returns_values_in_unit_interval(
        self,
        calculator: BrayCurtisDistanceCalculator,
    ) -> None:
        query = np.array([[1.0, 2.0, 0.0]])
        gallery = np.array([[2.0, 0.0, 1.0], [1.0, 2.0, 0.0]])
        result = calculator.cross(query, gallery)
        assert np.all(result.value >= 0.0)
        assert np.all(result.value <= 1.0)

    def test_metric_enum(self, calculator: BrayCurtisDistanceCalculator) -> None:
        assert calculator.metric is RatioBasedDistanceMetric.BRAY_CURTIS
