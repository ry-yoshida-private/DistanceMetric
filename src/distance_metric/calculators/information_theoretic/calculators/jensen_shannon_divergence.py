"""
Jensen–Shannon divergence for two nonnegative mass vectors.

Builds a mixture of the two masses and compares each side to that mixture.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ...cross_elementwise import CrossElementwiseCalculatorBase

from ....array_types import FloatArray, NumericArray

if TYPE_CHECKING:
    from ..metric import InformationTheoreticDistanceMetric


class JensenShannonDivergenceDistanceCalculator(CrossElementwiseCalculatorBase):
    """
    Symmetric Jensen–Shannon style divergence on aligned bins.

    Notes:
    -----
    Pass eps in cross or pairwise to control the floor before logarithms.
    """

    def _elementwise_values(
        self,
        query_array: NumericArray,
        gallery_array: NumericArray,
        *,
        eps: float = 1e-12,
        **kwargs: Any,
    ) -> FloatArray:
        """
        Per-bin Jensen–Shannon terms before summing.

        Parameters:
        ----------
        query_array: NumericArray
            Mass p; floored with eps.
        gallery_array: NumericArray
            Mass q; floored with eps.
        eps: float, optional
            Small positive floor to avoid log(0).
        **kwargs: Any
            Unused.

        Returns:
        --------
        FloatArray
            Per-bin divergence contributions before summing over features.
        """
        self._validate_broadcast_compatible(
            query_array=query_array,
            gallery_array=gallery_array,
        )
        eps = float(eps)
        p = np.maximum(query_array, eps)
        q = np.maximum(gallery_array, eps)
        m = 0.5 * (p + q)
        return 0.5 * (p * np.log(p / m) + q * np.log(q / m))

    def _reduce_elementwise_values(
        self,
        values: NumericArray,
        query_array: NumericArray,
        gallery_array: NumericArray,
        **kwargs: Any,
    ) -> FloatArray:
        """
        Sum divergence contributions across bins.

        Parameters:
        ----------
        values: NumericArray
            Per-bin JS terms in cross layout.
        query_array: NumericArray
            Unused.
        gallery_array: NumericArray
            Unused.
        **kwargs: Any
            Unused.

        Returns:
        --------
        FloatArray
            Shape (n, m) Jensen–Shannon divergences.
        """
        return np.sum(values, axis=self._sample_value_axes(values))

    @property
    def metric(self) -> InformationTheoreticDistanceMetric:
        """
        Enum tag for this calculator.

        Returns:
        --------
        InformationTheoreticDistanceMetric
            Always JENSEN_SHANNON_DIVERGENCE.
        """
        from ..metric import InformationTheoreticDistanceMetric

        return InformationTheoreticDistanceMetric.JENSEN_SHANNON_DIVERGENCE
