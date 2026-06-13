"""
Dynamic Time Warping (DTW) distance between sequences.

Aligns two temporal or ordered sequences with a minimum-cost path through a local
cost matrix. Cross mode evaluates DTW independently for every query row against
every gallery row.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ...cross_elementwise import CrossElementwiseCalculatorBase

from ....array_types import FloatArray, NumericArray

if TYPE_CHECKING:
    from ..metric import GeodesicSequenceDistanceMetric


class DynamicTimeWarpingDistanceCalculator(CrossElementwiseCalculatorBase):
    """
    Classic DTW with unit step transitions (diagonal, up, left).

    Notes:
    -----
    Scalars use absolute difference cost; vector steps use Euclidean norm along the
    last axis. _cross_array overrides the cross-elementwise default implementation.
    """

    def _elementwise_values(
        self,
        query_array: NumericArray,
        gallery_array: NumericArray,
        **kwargs: Any,
    ) -> FloatArray:
        """
        Optimal alignment cost for one query sequence vs one gallery sequence.

        Parameters:
        ----------
        query_array: NumericArray
            One-dimensional sequence or shape (T, D) time series.
        gallery_array: NumericArray
            Same dimensional convention as query_array.
        **kwargs: Any
            Unused.

        Returns:
        --------
        FloatArray
            One-element float vector holding the total DTW cost for this pair.
        """
        q = np.asarray(query_array)
        g = np.asarray(gallery_array)
        n, m = q.shape[0], g.shape[0]
        if q.ndim == 1 and g.ndim == 1:
            cost = np.abs(q[:, None] - g[None, :])
        else:
            cost = np.linalg.norm(q[:, None, ...] - g[None, ...], axis=-1)
        dp = np.full((n + 1, m + 1), np.inf, dtype=float)
        dp[0, 0] = 0.0
        for k in range(2, n + m + 1):
            i = np.arange(max(1, k - m), min(n, k - 1) + 1)
            j = k - i
            dp[i, j] = cost[i - 1, j - 1] + np.minimum(
                np.minimum(dp[i - 1, j], dp[i, j - 1]),
                dp[i - 1, j - 1],
            )
        return np.asarray([dp[n, m]], dtype=float)

    def _reduce_elementwise_values(
        self,
        values: NumericArray,
        query_array: NumericArray,
        gallery_array: NumericArray,
        **kwargs: Any,
    ) -> float:
        """
        Extract scalar DTW cost from the one-element container.

        Parameters:
        ----------
        values: NumericArray
            Length-one array from _elementwise_values.
        query_array: NumericArray
            Unused.
        gallery_array: NumericArray
            Unused.
        **kwargs: Any
            Unused.

        Returns:
        --------
        float
            Total alignment cost.
        """
        return float(values[0])

    def _cross_array(
        self,
        query_array: NumericArray,
        gallery_array: NumericArray,
        **kwargs: Any,
    ) -> FloatArray:
        """
        Fill an (n, m) matrix by running DTW for each row pair.

        Parameters:
        ----------
        query_array: NumericArray
            Batch of sequences with shape (n, *) where leading axis indexes queries.
        gallery_array: NumericArray
            Batch of sequences with shape (m, *) sharing trailing layout with queries.
        **kwargs: Any
            Forwarded to _elementwise_values and _reduce_elementwise_values.

        Returns:
        --------
        FloatArray
            Shape (n, m); entry (i, j) is DTW cost between query_array[i] and gallery_array[j].

        Notes:
        -----
        Validates broadcast compatibility between rows via _validate_broadcast_compatible on full batches.
        """
        self._validate_broadcast_compatible(
            query_array=query_array,
            gallery_array=gallery_array,
        )
        n, m = query_array.shape[0], gallery_array.shape[0]
        out = np.empty((n, m), dtype=float)
        for flat_idx in range(n * m):
            i, j = divmod(flat_idx, m)
            out[i, j] = self._reduce_elementwise_values(
                values=self._elementwise_values(query_array[i], gallery_array[j], **kwargs),
                query_array=query_array[i],
                gallery_array=gallery_array[j],
                **kwargs,
            )
        return out

    @property
    def metric(self) -> GeodesicSequenceDistanceMetric:
        """
        Enum tag for this calculator.

        Returns:
        --------
        GeodesicSequenceDistanceMetric
            Always DYNAMIC_TIME_WARPING.
        """
        from ..metric import GeodesicSequenceDistanceMetric

        return GeodesicSequenceDistanceMetric.DYNAMIC_TIME_WARPING
