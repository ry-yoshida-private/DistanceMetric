"""
Hellinger distance between nonnegative distributions on shared bins.

Clamps mass at zero before the square-root and squaring steps.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ...cross_elementwise import CrossElementwiseCalculatorBase

from ....array_types import FloatArray, NumericArray

if TYPE_CHECKING:
    from ..metric import InformationTheoreticDistanceMetric


class HellingerDistanceCalculator(CrossElementwiseCalculatorBase):
    """
    Hellinger-style distance from nonnegative bin masses.

    Notes:
    -----
    Inputs are floored at zero before root and square operations.
    """

    def _elementwise_values(
        self,
        query_array: NumericArray,
        gallery_array: NumericArray,
        **kwargs: Any,
    ) -> FloatArray:
        """
        Per-bin terms before summing and final aggregation.

        Parameters:
        ----------
        query_array: NumericArray
            Nonnegative bin masses.
        gallery_array: NumericArray
            Nonnegative bin masses aligned with query_array.
        **kwargs: Any
            Unused.

        Returns:
        --------
        FloatArray
            Intermediate nonnegative terms aligned with broadcast bins.
        """
        self._validate_broadcast_compatible(
            query_array=query_array,
            gallery_array=gallery_array,
        )
        return (np.sqrt(np.maximum(query_array, 0.0)) - np.sqrt(np.maximum(gallery_array, 0.0))) ** 2

    def _reduce_elementwise_values(
        self,
        values: NumericArray,
        query_array: NumericArray,
        gallery_array: NumericArray,
        **kwargs: Any,
    ) -> FloatArray:
        """
        Aggregate per-bin terms into Hellinger distances for each batch pair.

        Parameters:
        ----------
        values: NumericArray
            Per-bin intermediate terms in cross layout.
        query_array: NumericArray
            Unused.
        gallery_array: NumericArray
            Unused.
        **kwargs: Any
            Unused.

        Returns:
        --------
        FloatArray
            Shape (n, m) Hellinger distances.
        """
        return np.sqrt(0.5 * np.sum(values, axis=self._sample_value_axes(values)))

    @property
    def metric(self) -> InformationTheoreticDistanceMetric:
        """
        Enum tag for this calculator.

        Returns:
        --------
        InformationTheoreticDistanceMetric
            Always HELLINGER.
        """
        from ..metric import InformationTheoreticDistanceMetric
        return InformationTheoreticDistanceMetric.HELLINGER
