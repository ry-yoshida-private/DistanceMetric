"""
Kendall-style distance from pairwise concordance of coordinate ordering.

Compares relative order of feature pairs between query and gallery samples.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ...cross_elementwise import CrossElementwiseCalculatorBase

from ....array_types import FloatArray, NumericArray

if TYPE_CHECKING:
    from ..metric import SimilarityCorrelationDistanceMetric


class KendallDistanceCalculator(CrossElementwiseCalculatorBase):
    """
    Concordance-based rank association converted to a distance.

    Notes:
    -----
    Uses upper-triangular index pairs along the last axis. Ties yield zero contribution.
    """

    def _elementwise_values(
        self,
        query_array: NumericArray,
        gallery_array: NumericArray,
        **kwargs: Any,
    ) -> FloatArray:
        """
        Concordance sign per unordered feature pair from upper-triangular indices.

        Parameters:
        ----------
        query_array: NumericArray
            Samples with feature dimension last.
        gallery_array: NumericArray
            Matching gallery samples.
        **kwargs: Any
            Unused.

        Returns:
        --------
        FloatArray
            Values in {-1, 0, 1} indicating discordant, tie, or concordant pairs per bundle.
        """
        self._validate_broadcast_compatible(
            query_array=query_array,
            gallery_array=gallery_array,
        )
        n = query_array.shape[-1]
        i_idx, j_idx = np.triu_indices(n, k=1)
        dq = query_array[..., i_idx] - query_array[..., j_idx]
        dg = gallery_array[..., i_idx] - gallery_array[..., j_idx]
        return np.sign(dq * dg).astype(float)

    def _reduce_elementwise_values(
        self,
        values: NumericArray,
        query_array: NumericArray,
        gallery_array: NumericArray,
        **kwargs: Any,
    ) -> FloatArray:
        """
        Map concordance and discordance counts to a distance in (0, 1) range.

        Parameters:
        ----------
        values: NumericArray
            Pairwise sign indicators in cross layout.
        query_array: NumericArray
            Unused.
        gallery_array: NumericArray
            Unused.
        **kwargs: Any
            Unused.

        Returns:
        --------
        FloatArray
            Shape (n, m) Kendall distances.

        Notes:
        -----
        Denominator uses concordant plus discordant counts with epsilon stability.
        """
        concordant = np.sum(values > 0, axis=self._sample_value_axes(values)).astype(float)
        discordant = np.sum(values < 0, axis=self._sample_value_axes(values)).astype(float)
        denom = concordant + discordant
        tau = (concordant - discordant) / np.maximum(denom, 1e-12)
        return 1.0 - tau

    @property
    def metric(self) -> SimilarityCorrelationDistanceMetric:
        """
        Enum tag for this calculator.

        Returns:
        --------
        SimilarityCorrelationDistanceMetric
            Always KENDALL.
        """
        from ..metric import SimilarityCorrelationDistanceMetric
        return SimilarityCorrelationDistanceMetric.KENDALL
