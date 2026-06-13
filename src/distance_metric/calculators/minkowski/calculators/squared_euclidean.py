"""
Squared Euclidean distance calculator.

Sums squared differences without taking the square root.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ...cross_elementwise import CrossElementwiseCalculatorBase

from ....array_types import FloatArray, NumericArray

if TYPE_CHECKING:
    from ..metric import MinkowskiDistanceMetric


class SquaredEuclideanDistanceCalculator(CrossElementwiseCalculatorBase):
    """
    Sum of squared coordinate differences (no square root).

    Notes:
    -----
    Often used where ranking by distance is enough without taking sqrt.
    """

    def _elementwise_values(
        self,
        query_array: NumericArray,
        gallery_array: NumericArray,
        **kwargs: Any,
    ) -> FloatArray:
        """
        Squared coordinate-wise gaps.

        Parameters:
        ----------
        query_array: NumericArray
            Query batch broadcast-compatible with gallery_array.
        gallery_array: NumericArray
            Gallery batch matching query_array broadcast rules.
        **kwargs: Any
            Unused.

        Returns:
        --------
        FloatArray
            Element-wise (q - g)^2 with broadcast shape.
        """
        self._validate_broadcast_compatible(
            query_array=query_array,
            gallery_array=gallery_array,
        )
        d = query_array - gallery_array
        return d * d

    def _reduce_elementwise_values(
        self,
        values: NumericArray,
        query_array: NumericArray,
        gallery_array: NumericArray,
        **kwargs: Any,
    ) -> FloatArray:
        """
        Sum squared gaps over feature axes.

        Parameters:
        ----------
        values: NumericArray
            Squared differences in cross layout (n, m, *features).
        query_array: NumericArray
            Unused.
        gallery_array: NumericArray
            Unused.
        **kwargs: Any
            Unused.

        Returns:
        --------
        FloatArray
            Shape (n, m) containing summed squared Euclidean distances.
        """
        return np.sum(values, axis=self._sample_value_axes(values))

    @property
    def metric(self) -> MinkowskiDistanceMetric:
        """
        Enum tag for this calculator.

        Returns:
        --------
        MinkowskiDistanceMetric
            Always MinkowskiDistanceMetric.SQUARED_EUCLIDEAN.
        """
        from ..metric import MinkowskiDistanceMetric
        return MinkowskiDistanceMetric.SQUARED_EUCLIDEAN
