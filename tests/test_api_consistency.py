"""Cross-cutting API consistency tests."""

from __future__ import annotations

import numpy as np
import pytest

from distance_metric.calculators.minkowski import EuclideanDistanceCalculator


class TestApiConsistency:
    """Public calculator APIs stay mutually consistent."""

    @pytest.fixture
    def calculator(self) -> EuclideanDistanceCalculator:
        return EuclideanDistanceCalculator()

    def test_pairwise_matches_cross_with_same_batch(
        self,
        calculator: EuclideanDistanceCalculator,
        numeric_batch: np.ndarray,
    ) -> None:
        pairwise = calculator.pairwise(numeric_batch)
        cross = calculator.cross(numeric_batch, numeric_batch)
        np.testing.assert_allclose(pairwise.value, cross.value)

    def test_elementwise_matches_cross_single_cell(
        self,
        calculator: EuclideanDistanceCalculator,
    ) -> None:
        query = np.array([1.0, 2.0])
        gallery = np.array([4.0, 6.0])
        elementwise = calculator.elementwise(query, gallery)
        cross = calculator.cross(query[None, :], gallery[None, :])
        np.testing.assert_allclose(elementwise.value[0], cross.value[0, 0])

    def test_pairwise_diagonal_is_zero(
        self,
        calculator: EuclideanDistanceCalculator,
        numeric_batch: np.ndarray,
    ) -> None:
        result = calculator.pairwise(numeric_batch)
        np.testing.assert_allclose(np.diag(result.value), 0.0)

    def test_incompatible_broadcast_raises(
        self,
        calculator: EuclideanDistanceCalculator,
    ) -> None:
        query = np.array([[1.0, 2.0, 3.0]])
        gallery = np.array([[1.0, 2.0]])
        with pytest.raises(ValueError):
            calculator.elementwise(query, gallery)
