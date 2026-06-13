"""
Huber loss summed across coordinates.

Quadratic for small residuals and linear beyond threshold delta.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ...cross_elementwise import CrossElementwiseCalculatorBase

from ....array_types import FloatArray, NumericArray

if TYPE_CHECKING:
    from ..metric import TransportDistanceMetric


class HuberDistanceCalculator(CrossElementwiseCalculatorBase):
    """
    Sum of per-coordinate Huber losses with threshold delta.

    Notes:
    -----
    Pass delta through cross, pairwise, or elementwise kwargs.
    """

    def _elementwise_values(
        self,
        query_array: NumericArray,
        gallery_array: NumericArray,
        *,
        delta: float = 1.0,
        **kwargs: Any,
    ) -> FloatArray:
        """
        Huber kernel values per coordinate.

        Parameters:
        ----------
        query_array: NumericArray
            Query batch matching gallery_array.
        gallery_array: NumericArray
            Gallery batch matching query_array.
        delta: float, optional
            Threshold between quadratic and linear regions; must be positive.
        **kwargs: Any
            Unused.

        Returns:
        --------
        FloatArray
            Per-coordinate Huber loss contributions.

        Raises:
        -------
        ValueError
            If delta is not positive.
        """
        self._validate_broadcast_compatible(
            query_array=query_array,
            gallery_array=gallery_array,
        )
        delta = float(delta)
        if delta <= 0:
            raise ValueError("delta must be greater than 0.")
        a = np.abs(query_array - gallery_array)
        return np.where(a <= delta, 0.5 * a * a, delta * (a - 0.5 * delta))

    def _reduce_elementwise_values(
        self,
        values: NumericArray,
        query_array: NumericArray,
        gallery_array: NumericArray,
        **kwargs: Any,
    ) -> FloatArray:
        """
        Sum Huber losses over feature axes.

        Parameters:
        ----------
        values: NumericArray
            Per-coordinate Huber terms in cross layout.
        query_array: NumericArray
            Unused.
        gallery_array: NumericArray
            Unused.
        **kwargs: Any
            Unused.

        Returns:
        --------
        FloatArray
            Shape (n, m) total Huber distances.
        """
        return np.sum(values, axis=self._sample_value_axes(values))

    @property
    def metric(self) -> TransportDistanceMetric:
        """
        Enum tag for this calculator.

        Returns:
        --------
        TransportDistanceMetric
            Always HUBER_DISTANCE.
        """
        from ..metric import TransportDistanceMetric

        return TransportDistanceMetric.HUBER_DISTANCE
