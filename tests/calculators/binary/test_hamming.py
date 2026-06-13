"""Tests for HammingDistanceCalculator."""

from __future__ import annotations

import numpy as np
import pytest

from distance_metric.calculators.binary import HammingDistanceCalculator
from distance_metric.calculators.binary.calculators.hamming import (
    HammingDistanceCalculator as HammingClass,
)
from distance_metric.calculators.binary.metric import BinaryDistanceMetric
from distance_metric.result import DistanceResultType


@pytest.fixture
def calculator() -> HammingDistanceCalculator:
    """Return a fresh Hamming calculator instance."""
    return HammingDistanceCalculator()


class TestHammingPairwise:
    """Pairwise Hamming distance on batched rows."""

    def test_boolean_vectors(self, calculator: HammingDistanceCalculator) -> None:
        array = np.array([[True, False, True], [False, True, False]])
        result = calculator.pairwise(array)
        expected = np.array([[0.0, 1.0], [1.0, 0.0]])
        np.testing.assert_allclose(result.value, expected)
        assert result.type is DistanceResultType.DISTANCE
        assert result.shape == (2, 2)

    def test_integer_coded_vectors(self, calculator: HammingDistanceCalculator) -> None:
        array = np.array([[1, 0, 1], [0, 1, 0]], dtype=np.int8)
        result = calculator.pairwise(array)
        expected = np.array([[0.0, 1.0], [1.0, 0.0]])
        np.testing.assert_allclose(result.value, expected)

    def test_integer_symbol_equality(self, calculator: HammingDistanceCalculator) -> None:
        query = np.array([[1, 2, 3]], dtype=np.int32)
        gallery = np.array([[1, 2, 4]], dtype=np.int32)
        result = calculator.cross(query, gallery)
        np.testing.assert_allclose(result.value, np.array([[1.0 / 3.0]]))

    def test_identical_rows_zero_distance(self, calculator: HammingDistanceCalculator) -> None:
        array = np.array([[True, True], [False, True]])
        result = calculator.pairwise(array)
        assert result.min == 0.0
        np.testing.assert_allclose(np.diag(result.value), 0.0)


class TestHammingCross:
    """Cross Hamming distance between query and gallery batches."""

    def test_cross_two_batches(self, calculator: HammingDistanceCalculator) -> None:
        query = np.array([[True, False]])
        gallery = np.array([[False, True], [True, False]])
        result = calculator.cross(query, gallery)
        np.testing.assert_allclose(result.value, np.array([[1.0, 0.0]]))


class TestHammingElementwise:
    """Elementwise Hamming distance for a single pair."""

    def test_partial_mismatch(self, calculator: HammingDistanceCalculator) -> None:
        query = np.array([True, False, True])
        gallery = np.array([True, True, False])
        result = calculator.elementwise(query, gallery)
        np.testing.assert_allclose(result.value, np.array([2.0 / 3.0]))
        assert result.value.shape == (1,)

    def test_broadcast_compatible_shapes(self, calculator: HammingDistanceCalculator) -> None:
        query = np.array([[True, False]])
        gallery = np.array([[True], [False]])
        result = calculator.elementwise(query, gallery)
        assert result.value.shape == (1,)


class TestHammingDtypeValidation:
    """Runtime rejection of unsupported floating-point inputs."""

    def test_pairwise_rejects_float64(self, calculator: HammingDistanceCalculator) -> None:
        array = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
        with pytest.raises(ValueError, match="boolean or integer dtype"):
            calculator.pairwise(array)

    def test_cross_rejects_float32(self, calculator: HammingDistanceCalculator) -> None:
        query = np.array([[1.0, 0.0]], dtype=np.float32)
        gallery = np.array([[0.0, 1.0]], dtype=np.float32)
        with pytest.raises(ValueError, match="query_array must have boolean or integer dtype"):
            calculator.cross(query, gallery)


class TestHammingMetricRegistry:
    """Enum factory resolves to HammingDistanceCalculator."""

    def test_metric_calculator_property(self) -> None:
        calc = BinaryDistanceMetric.HAMMING.calculator
        assert isinstance(calc, HammingClass)
