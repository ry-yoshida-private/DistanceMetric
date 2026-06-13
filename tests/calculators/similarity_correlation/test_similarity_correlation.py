"""Tests for similarity and correlation distance calculators."""

from __future__ import annotations

import numpy as np
import pytest

from distance_metric.calculators.similarity_correlation import (
    CorrelationDistanceCalculator,
    CosineDistanceCalculator,
    KendallDistanceCalculator,
    SimilarityCorrelationDistanceMetric,
    SpearmanDistanceCalculator,
)


class TestCosineDistance:
    """Cosine distance."""

    @pytest.fixture
    def calculator(self) -> CosineDistanceCalculator:
        return CosineDistanceCalculator()

    def test_identical_rows_zero(
        self,
        calculator: CosineDistanceCalculator,
    ) -> None:
        batch = np.array([[1.0, 2.0, 3.0]])
        result = calculator.pairwise(batch)
        np.testing.assert_allclose(result.value, np.array([[0.0]]), atol=1e-12)

    def test_opposite_direction_maximum_distance(
        self,
        calculator: CosineDistanceCalculator,
    ) -> None:
        query = np.array([[1.0, 0.0]])
        gallery = np.array([[-1.0, 0.0]])
        result = calculator.cross(query, gallery)
        np.testing.assert_allclose(result.value, np.array([[2.0]]))

    def test_metric_enum(self, calculator: CosineDistanceCalculator) -> None:
        assert calculator.metric is SimilarityCorrelationDistanceMetric.COSINE


class TestCorrelationDistance:
    """Pearson correlation distance."""

    @pytest.fixture
    def calculator(self) -> CorrelationDistanceCalculator:
        return CorrelationDistanceCalculator()

    def test_perfect_positive_correlation_zero(
        self,
        calculator: CorrelationDistanceCalculator,
    ) -> None:
        query = np.array([[1.0, 2.0, 3.0]])
        gallery = np.array([[2.0, 4.0, 6.0]])
        result = calculator.cross(query, gallery)
        np.testing.assert_allclose(result.value, np.array([[0.0]]), atol=1e-12)

    def test_metric_enum(self, calculator: CorrelationDistanceCalculator) -> None:
        assert calculator.metric is SimilarityCorrelationDistanceMetric.CORRELATION


class TestSpearmanDistance:
    """Spearman correlation distance."""

    @pytest.fixture
    def calculator(self) -> SpearmanDistanceCalculator:
        return SpearmanDistanceCalculator()

    def test_identical_rows_zero(
        self,
        calculator: SpearmanDistanceCalculator,
    ) -> None:
        batch = np.array([[3.0, 1.0, 2.0]])
        result = calculator.pairwise(batch)
        np.testing.assert_allclose(result.value, np.array([[0.0]]), atol=1e-12)

    def test_metric_enum(self, calculator: SpearmanDistanceCalculator) -> None:
        assert calculator.metric is SimilarityCorrelationDistanceMetric.SPEARMAN


class TestKendallDistance:
    """Kendall concordance distance."""

    @pytest.fixture
    def calculator(self) -> KendallDistanceCalculator:
        return KendallDistanceCalculator()

    def test_identical_ordering_zero(
        self,
        calculator: KendallDistanceCalculator,
    ) -> None:
        query = np.array([[1.0, 2.0, 3.0]])
        gallery = np.array([[1.0, 2.0, 3.0]])
        result = calculator.cross(query, gallery)
        np.testing.assert_allclose(result.value, np.array([[0.0]]), atol=1e-12)

    def test_elementwise_shape(self, calculator: KendallDistanceCalculator) -> None:
        result = calculator.elementwise(
            np.array([1.0, 2.0, 3.0]),
            np.array([1.0, 3.0, 2.0]),
        )
        assert result.value.shape == (1,)

    def test_metric_enum(self, calculator: KendallDistanceCalculator) -> None:
        assert calculator.metric is SimilarityCorrelationDistanceMetric.KENDALL
