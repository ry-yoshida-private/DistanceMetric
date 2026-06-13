"""Tests for information-theoretic distance calculators."""

from __future__ import annotations

import numpy as np
import pytest

from distance_metric.calculators.information_theoretic import (
    BhattacharyyaDistanceCalculator,
    HellingerDistanceCalculator,
    InformationTheoreticDistanceMetric,
    JensenShannonDivergenceDistanceCalculator,
    KLDivergenceDistanceCalculator,
)


class TestKLDivergenceDistance:
    """Directed KL divergence."""

    @pytest.fixture
    def calculator(self) -> KLDivergenceDistanceCalculator:
        return KLDivergenceDistanceCalculator()

    def test_identical_distributions_zero(
        self,
        calculator: KLDivergenceDistanceCalculator,
    ) -> None:
        mass = np.array([[0.5, 0.5]])
        result = calculator.pairwise(mass)
        np.testing.assert_allclose(result.value, np.array([[0.0]]))

    def test_not_symmetric_in_query_and_gallery(
        self,
        calculator: KLDivergenceDistanceCalculator,
        probability_query: np.ndarray,
        probability_gallery: np.ndarray,
    ) -> None:
        forward = calculator.cross(probability_query, probability_gallery).value[0, 0]
        reverse = calculator.cross(probability_gallery[:1], probability_query).value[0, 0]
        assert forward != reverse

    def test_metric_enum(self, calculator: KLDivergenceDistanceCalculator) -> None:
        assert calculator.metric is InformationTheoreticDistanceMetric.KL_DIVERGENCE


class TestJensenShannonDivergenceDistance:
    """Symmetric Jensen-Shannon divergence."""

    @pytest.fixture
    def calculator(self) -> JensenShannonDivergenceDistanceCalculator:
        return JensenShannonDivergenceDistanceCalculator()

    def test_identical_distributions_zero(
        self,
        calculator: JensenShannonDivergenceDistanceCalculator,
    ) -> None:
        mass = np.array([[0.25, 0.75]])
        result = calculator.pairwise(mass)
        np.testing.assert_allclose(result.value, np.array([[0.0]]))

    def test_metric_enum(
        self,
        calculator: JensenShannonDivergenceDistanceCalculator,
    ) -> None:
        assert calculator.metric is InformationTheoreticDistanceMetric.JENSEN_SHANNON_DIVERGENCE


class TestBhattacharyyaDistance:
    """Bhattacharyya distance."""

    @pytest.fixture
    def calculator(self) -> BhattacharyyaDistanceCalculator:
        return BhattacharyyaDistanceCalculator()

    def test_cross_returns_finite_values(
        self,
        calculator: BhattacharyyaDistanceCalculator,
        probability_query: np.ndarray,
        probability_gallery: np.ndarray,
    ) -> None:
        result = calculator.cross(probability_query, probability_gallery)
        assert np.all(np.isfinite(result.value))

    def test_metric_enum(self, calculator: BhattacharyyaDistanceCalculator) -> None:
        assert calculator.metric is InformationTheoreticDistanceMetric.BHATTACHARYYA


class TestHellingerDistance:
    """Hellinger distance."""

    @pytest.fixture
    def calculator(self) -> HellingerDistanceCalculator:
        return HellingerDistanceCalculator()

    def test_identical_distributions_zero(
        self,
        calculator: HellingerDistanceCalculator,
    ) -> None:
        mass = np.array([[0.6, 0.4]])
        result = calculator.pairwise(mass)
        np.testing.assert_allclose(result.value, np.array([[0.0]]))

    def test_metric_enum(self, calculator: HellingerDistanceCalculator) -> None:
        assert calculator.metric is InformationTheoreticDistanceMetric.HELLINGER
