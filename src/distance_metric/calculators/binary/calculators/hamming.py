"""
Hamming distance calculator for boolean and integer-coded vectors.

Counts mismatched coordinates between two samples and averages over feature axes.
Equality is evaluated per coordinate; floating-point dtypes are not supported.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ..base import BinaryCrossElementwiseCalculatorBase

from ....array_types import BinaryArray, ElementwiseValuesArray, FloatArray

if TYPE_CHECKING:
    from ..metric import BinaryDistanceMetric


class HammingDistanceCalculator(BinaryCrossElementwiseCalculatorBase):
    """
    Normalized Hamming distance between aligned boolean or integer-coded samples.

    Each coordinate contributes 1 if query and gallery differ and 0 otherwise.
    Boolean arrays compare membership directly; integer arrays compare symbol
    equality (typical use: 0/1 bit vectors). The cross-elementwise base averages
    mismatch indicators over feature axes, yielding the fraction of disagreeing
    positions for flat vectors.
    """

    def _elementwise_values(
        self,
        query_array: BinaryArray,
        gallery_array: BinaryArray,
        **kwargs: Any,
    ) -> FloatArray:
        """
        Per-coordinate mismatch indicators before averaging.

        Parameters:
        ----------
        query_array: BinaryArray
            Query tensor; must broadcast with gallery_array.
        gallery_array: BinaryArray
            Gallery tensor with the same shape as query_array for this call.
        **kwargs: Any
            Reserved; unused.

        Returns:
        --------
        FloatArray
            Float array of zeros and ones with the broadcast shape of the inputs,
            where 1 marks positions where query_array != gallery_array.
        """
        self._validate_broadcast_compatible(
            query_array=query_array,
            gallery_array=gallery_array,
        )
        return np.asarray(query_array != gallery_array, dtype=np.float64)

    def _reduce_elementwise_values(
        self,
        values: ElementwiseValuesArray,
        query_array: BinaryArray,
        gallery_array: BinaryArray,
        **kwargs: Any,
    ) -> FloatArray:
        """
        Mean mismatch over all feature axes for each query-gallery pair.

        Parameters:
        ----------
        values: ElementwiseValuesArray
            Output of _elementwise_values. In cross mode, shape begins with
            (n, m, ...) where trailing axes are feature dimensions.
        query_array: BinaryArray
            Original query batch; not read in this implementation.
        gallery_array: BinaryArray
            Original gallery batch; not read in this implementation.
        **kwargs: Any
            Reserved; unused.

        Returns:
        --------
        FloatArray
            Array of shape (n, m) containing the mean of values along every axis
            after the first two batch dimensions.
        """
        return np.mean(values, axis=self._sample_value_axes(values))

    @property
    def metric(self) -> BinaryDistanceMetric:
        """
        Enum member identifying this implementation.

        Returns:
        --------
        BinaryDistanceMetric
            Always BinaryDistanceMetric.HAMMING.
        """
        from ..metric import BinaryDistanceMetric
        return BinaryDistanceMetric.HAMMING
