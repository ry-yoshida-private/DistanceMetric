"""Tests for DistanceResult."""

from __future__ import annotations

import numpy as np

from distance_metric.result import DistanceResult, DistanceResultType


class TestDistanceResult:
    """DistanceResult score helpers."""

    def test_min_and_max(self) -> None:
        result = DistanceResult(
            value=np.array([[1.0, 3.0], [2.0, 0.5]]),
            type=DistanceResultType.DISTANCE,
        )
        assert result.min == 0.5
        assert result.max == 3.0
        assert result.shape == (2, 2)

    def test_best_and_worst_score_for_distance(self) -> None:
        result = DistanceResult(
            value=np.array([[1.0, 3.0]]),
            type=DistanceResultType.DISTANCE,
        )
        assert result.best_score == 1.0
        assert result.worst_score == 3.0

    def test_best_and_worst_score_for_similarity(self) -> None:
        result = DistanceResult(
            value=np.array([[0.2, 0.9]]),
            type=DistanceResultType.SIMILARITY,
        )
        assert result.best_score == 0.9
        assert result.worst_score == 0.2
