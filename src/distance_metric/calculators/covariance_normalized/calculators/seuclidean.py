"""
Standardized Euclidean (SEuclidean) distance.

Divides squared differences by per-coordinate variance, then takes the square root
of the sum.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ...cross_elementwise import CrossElementwiseCalculatorBase

from ....array_types import FloatArray, NumericArray

if TYPE_CHECKING:
    from ..metric import CovarianceNormalizedDistanceMetric


class SEuclideanDistanceCalculator(CrossElementwiseCalculatorBase):
    """
    Euclidean distance with per-coordinate variance scaling.

    Notes:
    -----
    Supply variance on every call; its shape must match a single sample (trailing
    dimensions of the batch rows).
    """

    def _elementwise_values(
        self,
        query_array: NumericArray,
        gallery_array: NumericArray,
        *,
        variance: FloatArray | None = None,
        **kwargs: Any,
    ) -> FloatArray:
        """
        Per-coordinate scaled squared differences (d^2 / v).

        Parameters:
        ----------
        query_array: NumericArray
            Query batch; must match gallery_array for broadcasting.
        gallery_array: NumericArray
            Gallery batch; same shape rules.
        variance: FloatArray, optional
            Per-coordinate variances. Required; must match each sample shape.
        **kwargs: Any
            Unused.

        Returns:
        --------
        FloatArray
            (q - g)^2 / max(variance, tiny) with broadcast shape.

        Raises:
        -------
        ValueError
            If variance is None or its shape does not match each sample.
        """
        self._validate_broadcast_compatible(
            query_array=query_array,
            gallery_array=gallery_array,
        )
        if variance is None:
            raise ValueError("SEuclidean distance requires variance.")
        variance = np.asarray(variance, dtype=float)
        expected_shape = query_array.shape[2:] if query_array.ndim >= 3 else query_array.shape
        if variance.shape != expected_shape:
            raise ValueError("variance must have the same shape as each sample.")
        d = query_array - gallery_array
        return (d * d) / np.maximum(variance, 1e-12)

    def _reduce_elementwise_values(
        self,
        values: NumericArray,
        query_array: NumericArray,
        gallery_array: NumericArray,
        *,
        variance: FloatArray | None = None,
        **kwargs: Any,
    ) -> FloatArray:
        """
        Square root of the sum of scaled squared differences.

        Parameters:
        ----------
        values: NumericArray
            Scaled squares from _elementwise_values; cross shape (n, m, *features).
        query_array: NumericArray
            Unused; variance was applied in the elementwise step.
        gallery_array: NumericArray
            Unused.
        variance: FloatArray, optional
            Echoed for signature compatibility; not read here.
        **kwargs: Any
            Unused.

        Returns:
        --------
        FloatArray
            Shape (n, m) standardized Euclidean distances.
        """
        return np.sqrt(np.sum(values, axis=self._sample_value_axes(values)))

    @property
    def metric(self) -> CovarianceNormalizedDistanceMetric:
        """
        Enum tag for this calculator.

        Returns:
        --------
        CovarianceNormalizedDistanceMetric
            Always SEUCLIDEAN.
        """
        from ..metric import CovarianceNormalizedDistanceMetric
        return CovarianceNormalizedDistanceMetric.SEUCLIDEAN
