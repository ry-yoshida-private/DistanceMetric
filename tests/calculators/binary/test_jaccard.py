"""Tests for JaccardDistanceCalculator."""

from __future__ import annotations

import numpy as np
import pytest

from distance_metric.calculators.binary import JaccardDistanceCalculator
from distance_metric.calculators.binary.calculators.jaccard import (
    JaccardDistanceCalculator as JaccardClass,
)
from distance_metric.calculators.binary.metric import BinaryDistanceMetric
from distance_metric.result import DistanceResultType


@pytest.fixture
def calculator() -> JaccardDistanceCalculator:
    """Return a fresh Jaccard calculator instance."""
    return JaccardDistanceCalculator()


class TestJaccardPairwise:
    """Pairwise Jaccard distance on batched rows."""

    def test_boolean_vectors(self, calculator: JaccardDistanceCalculator) -> None:
        array = np.array([[True, False, True], [False, True, False]])
        result = calculator.pairwise(array)
        expected = np.array([[0.0, 1.0], [1.0, 0.0]])
        np.testing.assert_allclose(result.value, expected)
        assert result.type is DistanceResultType.DISTANCE

    def test_integer_coded_vectors(self, calculator: JaccardDistanceCalculator) -> None:
        array = np.array([[1, 0, 1], [0, 1, 0]], dtype=np.int8)
        result = calculator.pairwise(array)
        expected = np.array([[0.0, 1.0], [1.0, 0.0]])
        np.testing.assert_allclose(result.value, expected)

    def test_partial_overlap(self, calculator: JaccardDistanceCalculator) -> None:
        query = np.array([[1, 1, 0, 0]], dtype=np.int8)
        gallery = np.array([[1, 0, 1, 0]], dtype=np.int8)
        result = calculator.cross(query, gallery)
        # intersection=1 (first position), union=3 -> distance = 1 - 1/3
        np.testing.assert_allclose(result.value, np.array([[2.0 / 3.0]]))

    def test_identical_rows_zero_distance(self, calculator: JaccardDistanceCalculator) -> None:
        array = np.array([[True, False], [False, True]])
        result = calculator.pairwise(array)
        np.testing.assert_allclose(np.diag(result.value), 0.0)


class TestJaccardCross:
    """Cross Jaccard distance between query and gallery batches."""

    def test_cross_two_batches(self, calculator: JaccardDistanceCalculator) -> None:
        query = np.array([[True, True, False]])
        gallery = np.array([[True, False, False], [False, False, False]])
        result = calculator.cross(query, gallery)
        # q vs g0: inter=1, union=2 -> 0.5
        # q vs g1: inter=0, union=2 -> 1.0
        np.testing.assert_allclose(result.value, np.array([[0.5, 1.0]]))


class TestJaccardElementwise:
    """Elementwise Jaccard distance for a single pair."""

    def test_disjoint_sets(self, calculator: JaccardDistanceCalculator) -> None:
        query = np.array([True, True, False])
        gallery = np.array([False, False, True])
        result = calculator.elementwise(query, gallery)
        np.testing.assert_allclose(result.value, np.array([1.0]))

    def test_identical_sets(self, calculator: JaccardDistanceCalculator) -> None:
        query = np.array([True, False, True])
        gallery = np.array([True, False, True])
        result = calculator.elementwise(query, gallery)
        np.testing.assert_allclose(result.value, np.array([0.0]))


class TestJaccardDtypeValidation:
    """Runtime rejection of unsupported floating-point inputs."""

    def test_pairwise_rejects_float64(self, calculator: JaccardDistanceCalculator) -> None:
        array = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
        with pytest.raises(ValueError, match="boolean or integer dtype"):
            calculator.pairwise(array)


class TestJaccardMetricRegistry:
    """Enum factory resolves to JaccardDistanceCalculator."""

    def test_metric_calculator_property(self) -> None:
        calc = BinaryDistanceMetric.JACCARD.calculator
        assert isinstance(calc, JaccardClass)
