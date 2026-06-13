"""
Jaccard distance calculator for binary feature vectors.

Nonzero entries are treated as set membership after boolean conversion. Batch
cross mode compares every query row to every gallery row via the binary
cross-elementwise machinery.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ..base import BinaryCrossElementwiseCalculatorBase

from ....array_types import BinaryArray, ElementwiseValuesArray, FloatArray

if TYPE_CHECKING:
    from ..metric import BinaryDistanceMetric


class JaccardDistanceCalculator(BinaryCrossElementwiseCalculatorBase):
    """
    Jaccard distance on boolean or integer-coded binary vectors.

    Intersection and union are accumulated per coordinate channel, then combined
    into a scalar per query-gallery pair. Integer values are coerced with
    astype(bool) before logical operations.
    """

    def _elementwise_values(
        self,
        query_array: BinaryArray,
        gallery_array: BinaryArray,
        **kwargs: Any,
    ) -> FloatArray:
        """
        Build per-coordinate intersection and union indicators for reduction.

        Parameters:
        ----------
        query_array: BinaryArray
            Query batch, broadcast-compatible with gallery_array. Integer values
            are interpreted as false (0) or true (nonzero) per element.
        gallery_array: BinaryArray
            Gallery batch with the same shape as query_array for this call.
        **kwargs: Any
            Reserved; unused.

        Returns:
        --------
        FloatArray
            Stacked array with an axis at index 2: channel 0 is logical AND,
            channel 1 is logical OR (float-cast), per coordinate.
        """
        self._validate_broadcast_compatible(
            query_array=query_array,
            gallery_array=gallery_array,
        )
        q = query_array.astype(bool)
        g = gallery_array.astype(bool)
        return np.stack([np.logical_and(q, g), np.logical_or(q, g)], axis=2).astype(float)

    def _reduce_elementwise_values(
        self,
        values: ElementwiseValuesArray,
        query_array: BinaryArray,
        gallery_array: BinaryArray,
        **kwargs: Any,
    ) -> FloatArray:
        """
        Aggregate channels into Jaccard distance per batch pair.

        Parameters:
        ----------
        values: ElementwiseValuesArray
            Output of _elementwise_values in cross layout: batch n, batch m,
            channel axis (AND/OR), then feature dimensions.
        query_array: BinaryArray
            Original query batch; not read in this implementation.
        gallery_array: BinaryArray
            Original gallery batch; not read in this implementation.
        **kwargs: Any
            Reserved; unused.

        Returns:
        --------
        FloatArray
            Shape (n, m). Denominator uses a small floor to avoid division by zero.

        Notes:
        -----
        Axes 0–2 are query index, gallery index, and channel (AND/OR). Remaining
        axes are feature dimensions summed into intersection and union counts.
        """
        feature_axes = tuple(range(2, values.ndim - 1))
        inter = np.sum(values[:, :, 0, ...], axis=feature_axes)
        union = np.sum(values[:, :, 1, ...], axis=feature_axes)
        return 1.0 - (inter / np.maximum(union, 1e-12))

    @property
    def metric(self) -> BinaryDistanceMetric:
        """
        Enum member identifying this implementation.

        Returns:
        --------
        BinaryDistanceMetric
            Always BinaryDistanceMetric.JACCARD.
        """
        from ..metric import BinaryDistanceMetric
        return BinaryDistanceMetric.JACCARD
