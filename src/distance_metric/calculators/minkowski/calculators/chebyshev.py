"""
Chebyshev (L-infinity) distance calculator.

Uses maximum absolute coordinate difference; separate from the Minkowski base class.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ...cross_elementwise import CrossElementwiseCalculatorBase

from ....array_types import FloatArray, NumericArray

if TYPE_CHECKING:
    from ..metric import MinkowskiDistanceMetric


class ChebyshevDistanceCalculator(CrossElementwiseCalculatorBase):
    """
    Maximum absolute coordinate gap per sample pair (L-infinity).

    Notes:
    -----
    Implemented as max over absolute differences, not via the Minkowski sum hook.
    """

    def _elementwise_values(
        self,
        query_array: NumericArray,
        gallery_array: NumericArray,
        **kwargs: Any,
    ) -> FloatArray:
        """
        Absolute coordinate-wise distances before taking the max.

        Parameters:
        ----------
        query_array: NumericArray
            Query batch broadcast-compatible with gallery_array.
        gallery_array: NumericArray
            Gallery batch with matching broadcast rules.
        **kwargs: Any
            Unused.

        Returns:
        --------
        FloatArray
            Absolute differences with the same broadcast shape as inputs.
        """
        self._validate_broadcast_compatible(
            query_array=query_array,
            gallery_array=gallery_array,
        )
        return np.abs(query_array - gallery_array)

    def _reduce_elementwise_values(
        self,
        values: NumericArray,
        query_array: NumericArray,
        gallery_array: NumericArray,
        **kwargs: Any,
    ) -> FloatArray:
        """
        Maximum over all feature axes for each query-gallery pair.

        Parameters:
        ----------
        values: NumericArray
            Absolute differences in cross layout starting with (n, m, ...).
        query_array: NumericArray
            Unused.
        gallery_array: NumericArray
            Unused.
        **kwargs: Any
            Unused.

        Returns:
        --------
        FloatArray
            Shape (n, m); entry (i, j) is the Chebyshev distance between row i and row j.
        """
        return np.max(values, axis=self._sample_value_axes(values))

    @property
    def metric(self) -> MinkowskiDistanceMetric:
        """
        Enum tag for this calculator.

        Returns:
        --------
        MinkowskiDistanceMetric
            Always MinkowskiDistanceMetric.CHEBYSHEV.
        """
        from ..metric import MinkowskiDistanceMetric

        return MinkowskiDistanceMetric.CHEBYSHEV
