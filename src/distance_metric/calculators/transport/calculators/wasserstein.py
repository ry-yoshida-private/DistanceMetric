"""
One-dimensional Wasserstein-style distance via sorted samples.

Sorts each marginal along the last axis and averages absolute gaps between sorted entries.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ...cross_elementwise import CrossElementwiseCalculatorBase

from ....array_types import FloatArray, NumericArray

if TYPE_CHECKING:
    from ..metric import TransportDistanceMetric


class WassersteinDistanceCalculator(CrossElementwiseCalculatorBase):
    """
    Mean absolute difference between aligned sorted coordinates.

    Notes:
    -----
    Sort-then-compare pattern for empirical vectors or histograms on matched bins.
    """

    def _elementwise_values(
        self,
        query_array: NumericArray,
        gallery_array: NumericArray,
        **kwargs: Any,
    ) -> FloatArray:
        """
        Absolute gaps between sorted coordinates along the last axis.

        Parameters:
        ----------
        query_array: NumericArray
            Batch tensor; last axis indexes bins or samples to sort.
        gallery_array: NumericArray
            Same layout as query_array for broadcasting.
        **kwargs: Any
            Unused.

        Returns:
        --------
        FloatArray
            |sort(q) - sort(g)| element-wise after sorting along axis -1.
        """
        self._validate_broadcast_compatible(
            query_array=query_array,
            gallery_array=gallery_array,
        )
        q = np.sort(query_array, axis=-1)
        g = np.sort(gallery_array, axis=-1)
        return np.abs(q - g)

    def _reduce_elementwise_values(
        self,
        values: NumericArray,
        query_array: NumericArray,
        gallery_array: NumericArray,
        **kwargs: Any,
    ) -> FloatArray:
        """
        Average coordinate gaps over remaining feature axes.

        Parameters:
        ----------
        values: NumericArray
            Sorted gaps in cross layout.
        query_array: NumericArray
            Unused.
        gallery_array: NumericArray
            Unused.
        **kwargs: Any
            Unused.

        Returns:
        --------
        FloatArray
            Shape (n, m) mean absolute gaps between sorted vectors.
        """
        return np.mean(values, axis=self._sample_value_axes(values))

    @property
    def metric(self) -> TransportDistanceMetric:
        """
        Enum tag for this calculator.

        Returns:
        --------
        TransportDistanceMetric
            Always WASSERSTEIN.
        """
        from ..metric import TransportDistanceMetric

        return TransportDistanceMetric.WASSERSTEIN
