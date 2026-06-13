"""
Spearman rank-based correlation distance.

Ranks each row then applies correlation distance on ranks.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from .correlation import CorrelationDistanceCalculator

from ....array_types import FloatArray, NumericArray

if TYPE_CHECKING:
    from ..metric import SimilarityCorrelationDistanceMetric


class SpearmanDistanceCalculator(CorrelationDistanceCalculator):
    """
    Spearman correlation distance via rank transforms along each row.

    Notes:
    -----
    Ranks use argsort twice (average ranks for ties are not implemented).
    """

    @staticmethod
    def _rank_values(values: NumericArray) -> FloatArray:
        """
        Ordinal ranks for a one-dimensional vector.

        Parameters:
        ----------
        values: NumericArray
            One-dimensional sample along a row.

        Returns:
        --------
        FloatArray
            Float ranks via argsort(argsort).
        """
        return np.argsort(np.argsort(values)).astype(float)

    def _cross_array(
        self,
        query_array: NumericArray,
        gallery_array: NumericArray,
        **kwargs: Any,
    ) -> FloatArray:
        """
        Rank each row, then correlation distance on ranks.

        Parameters:
        ----------
        query_array: NumericArray
            Two-dimensional or higher; axis 1 is rank axis.
        gallery_array: NumericArray
            Same row layout as query_array.
        **kwargs: Any
            Forwarded.

        Returns:
        --------
        FloatArray
            Spearman-based distance matrix (n, m).
        """
        query_rank = np.apply_along_axis(self._rank_values, 1, query_array)
        gallery_rank = np.apply_along_axis(self._rank_values, 1, gallery_array)
        return super()._cross_array(query_rank, gallery_rank, **kwargs)

    @property
    def metric(self) -> SimilarityCorrelationDistanceMetric:
        """
        Enum tag for this calculator.

        Returns:
        --------
        SimilarityCorrelationDistanceMetric
            Always SPEARMAN.
        """
        from ..metric import SimilarityCorrelationDistanceMetric
        return SimilarityCorrelationDistanceMetric.SPEARMAN
