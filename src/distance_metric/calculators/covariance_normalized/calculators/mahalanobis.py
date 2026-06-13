"""
Mahalanobis distance with inverse covariance or covariance matrix.

Combines each difference vector with vi (or pinv of cov) per batch pair.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ...cross_elementwise import CrossElementwiseCalculatorBase

from ....array_types import FloatArray, NumericArray

if TYPE_CHECKING:
    from ..metric import CovarianceNormalizedDistanceMetric


class MahalanobisDistanceCalculator(CrossElementwiseCalculatorBase):
    """
    Elliptical distance induced by a positive semidefinite Gram factor.

    Notes:
    -----
    Pass vi (inverse covariance) or cov (covariance, then pinv) through cross,
    pairwise, or elementwise. At least one of vi or cov is required in reduce.
    """

    def _elementwise_values(
        self,
        query_array: NumericArray,
        gallery_array: NumericArray,
        *,
        vi: FloatArray | None = None,
        cov: FloatArray | None = None,
        **kwargs: Any,
    ) -> FloatArray:
        """
        Coordinate-wise differences before combining with vi.

        Parameters:
        ----------
        query_array: NumericArray
            Query batch; must broadcast with gallery_array.
        gallery_array: NumericArray
            Gallery batch; same shape rules as query_array.
        vi: FloatArray, optional
            Passed through to reduce; not read here.
        cov: FloatArray, optional
            Passed through to reduce; not read here.
        **kwargs: Any
            Additional options for subclasses; unused.

        Returns:
        --------
        FloatArray
            query_array minus gallery_array, same broadcast shape.
        """
        self._validate_broadcast_compatible(
            query_array=query_array,
            gallery_array=gallery_array,
        )
        return query_array - gallery_array

    def _reduce_elementwise_values(
        self,
        values: NumericArray,
        query_array: NumericArray,
        gallery_array: NumericArray,
        *,
        vi: FloatArray | None = None,
        cov: FloatArray | None = None,
        **kwargs: Any,
    ) -> FloatArray:
        """
        Combine differences with vi and return nonnegative scalar distances per pair.

        Parameters:
        ----------
        values: NumericArray
            Differences in cross layout (n, m, *d); last axis is feature index.
        query_array: NumericArray
            Unused.
        gallery_array: NumericArray
            Unused.
        vi: FloatArray, optional
            Inverse covariance matrix, shape (d, d). Preferred if known.
        cov: FloatArray, optional
            Covariance; vi is set to pinv(cov) when vi is None.
        **kwargs: Any
            Unused.

        Returns:
        --------
        FloatArray
            Shape (n, m) non-negative Mahalanobis distances.

        Raises:
        -------
        ValueError
            If both vi and cov are None.
        """
        if vi is None:
            if cov is None:
                raise ValueError("Mahalanobis distance requires vi or cov.")
            vi = np.linalg.pinv(np.asarray(cov))
        vi = np.asarray(vi, dtype=float)
        quad = np.einsum("...i,ij,...j->...", values, vi, values)
        return np.sqrt(np.maximum(quad, 0.0))

    @property
    def metric(self) -> CovarianceNormalizedDistanceMetric:
        """
        Enum tag for this calculator.

        Returns:
        --------
        CovarianceNormalizedDistanceMetric
            Always MAHALANOBIS.
        """
        from ..metric import CovarianceNormalizedDistanceMetric

        return CovarianceNormalizedDistanceMetric.MAHALANOBIS
