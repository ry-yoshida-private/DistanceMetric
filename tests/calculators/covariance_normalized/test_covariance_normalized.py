"""Tests for covariance-normalized distance calculators."""

from __future__ import annotations

import numpy as np
import pytest

from distance_metric.calculators.covariance_normalized import (
    CovarianceNormalizedDistanceMetric,
    MahalanobisDistanceCalculator,
    SEuclideanDistanceCalculator,
)


class TestMahalanobisDistance:
    """Mahalanobis distance with vi or cov."""

    @pytest.fixture
    def calculator(self) -> MahalanobisDistanceCalculator:
        return MahalanobisDistanceCalculator()

    def test_cross_with_identity_covariance(
        self,
        calculator: MahalanobisDistanceCalculator,
        identity_covariance: np.ndarray,
    ) -> None:
        query = np.array([[0.0, 0.0]])
        gallery = np.array([[1.0, 0.0], [0.0, 1.0]])
        result = calculator.cross(query, gallery, cov=identity_covariance)
        np.testing.assert_allclose(result.value, np.array([[1.0, 1.0]]))

    def test_missing_vi_and_cov_raises(
        self,
        calculator: MahalanobisDistanceCalculator,
    ) -> None:
        query = np.array([[0.0, 0.0]])
        gallery = np.array([[1.0, 0.0]])
        with pytest.raises(ValueError, match="requires vi or cov"):
            calculator.cross(query, gallery)

    def test_metric_enum(self, calculator: MahalanobisDistanceCalculator) -> None:
        assert calculator.metric is CovarianceNormalizedDistanceMetric.MAHALANOBIS


class TestSEuclideanDistance:
    """Standardized Euclidean distance."""

    @pytest.fixture
    def calculator(self) -> SEuclideanDistanceCalculator:
        return SEuclideanDistanceCalculator()

    def test_cross_with_unit_variance(
        self,
        calculator: SEuclideanDistanceCalculator,
        unit_variance: np.ndarray,
    ) -> None:
        query = np.array([[0.0, 0.0]])
        gallery = np.array([[3.0, 4.0]])
        result = calculator.cross(query, gallery, variance=unit_variance)
        np.testing.assert_allclose(result.value, np.array([[5.0]]))

    def test_missing_variance_raises(
        self,
        calculator: SEuclideanDistanceCalculator,
    ) -> None:
        query = np.array([[0.0, 0.0]])
        gallery = np.array([[1.0, 0.0]])
        with pytest.raises(ValueError, match="requires variance"):
            calculator.cross(query, gallery)

    def test_metric_enum(self, calculator: SEuclideanDistanceCalculator) -> None:
        assert calculator.metric is CovarianceNormalizedDistanceMetric.SEUCLIDEAN
