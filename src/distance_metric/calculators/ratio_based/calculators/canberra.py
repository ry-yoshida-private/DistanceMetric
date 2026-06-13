"""
Canberra distance for nonnegative or general numeric vectors.

Each coordinate contributes |q-g| / (|q|+|g|) with a small denominator floor.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ...cross_elementwise import CrossElementwiseCalculatorBase

from ....array_types import FloatArray, NumericArray

if TYPE_CHECKING:
    from ..metric import RatioBasedDistanceMetric


class CanberraDistanceCalculator(CrossElementwiseCalculatorBase):
    """
    Sum of relative absolute differences; emphasizes small magnitudes.

    Notes:
    -----
    Denominator uses max(..., 1e-12) for numerical stability.
    """

    def _elementwise_values(
        self,
        query_array: NumericArray,
        gallery_array: NumericArray,
        **kwargs: Any,
    ) -> FloatArray:
        """
        Normalized absolute gaps per coordinate.

        Parameters:
        ----------
        query_array: NumericArray
            Query batch broadcast-compatible with gallery_array.
        gallery_array: NumericArray
            Gallery batch matching query_array rules.
        **kwargs: Any
            Unused.

        Returns:
        --------
        FloatArray
            |q-g| / max(|q|+|g|, tiny) with broadcast shape.
        """
        self._validate_broadcast_compatible(
            query_array=query_array,
            gallery_array=gallery_array,
        )
        num = np.abs(query_array - gallery_array)
        den = np.abs(query_array) + np.abs(gallery_array)
        return num / np.maximum(den, 1e-12)

    def _reduce_elementwise_values(
        self,
        values: NumericArray,
        query_array: NumericArray,
        gallery_array: NumericArray,
        **kwargs: Any,
    ) -> FloatArray:
        """
        Sum Canberra terms across features.

        Parameters:
        ----------
        values: NumericArray
            Per-coordinate Canberra contributions in cross layout.
        query_array: NumericArray
            Unused.
        gallery_array: NumericArray
            Unused.
        **kwargs: Any
            Unused.

        Returns:
        --------
        FloatArray
            Shape (n, m) Canberra distances.
        """
        return np.sum(values, axis=self._sample_value_axes(values))

    @property
    def metric(self) -> RatioBasedDistanceMetric:
        """
        Enum tag for this calculator.

        Returns:
        --------
        RatioBasedDistanceMetric
            Always CANBERRA.
        """
        from ..metric import RatioBasedDistanceMetric

        return RatioBasedDistanceMetric.CANBERRA
