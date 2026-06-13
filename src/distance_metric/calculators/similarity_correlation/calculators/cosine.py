"""
Cosine distance calculator.

Cosine distance is one minus cosine similarity along the last axis.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ...cross_elementwise import CrossElementwiseCalculatorBase

from ....array_types import FloatArray, NumericArray

if TYPE_CHECKING:
    from ..metric import SimilarityCorrelationDistanceMetric


class CosineDistanceCalculator(CrossElementwiseCalculatorBase):
    """
    Cosine distance 1 minus cosine similarity along the last axis.

    Notes:
    -----
    _cross_array is overridden to insert query and gallery batch axes explicitly.
    """

    def _elementwise_values(
        self,
        query_array: NumericArray,
        gallery_array: NumericArray,
        **kwargs: Any,
    ) -> FloatArray:
        """
        Element-wise products before summing into a dot product.

        Parameters:
        ----------
        query_array: NumericArray
            Query tensor broadcast-compatible with gallery_array.
        gallery_array: NumericArray
            Gallery tensor matching query_array rules.
        **kwargs: Any
            Unused.

        Returns:
        --------
        FloatArray
            query_array * gallery_array with broadcast shape.
        """
        self._validate_broadcast_compatible(
            query_array=query_array,
            gallery_array=gallery_array,
        )
        return query_array * gallery_array

    def _reduce_elementwise_values(
        self,
        values: NumericArray,
        query_array: NumericArray,
        gallery_array: NumericArray,
        **kwargs: Any,
    ) -> FloatArray:
        """
        Turn dot products and L2 norms into cosine distance per batch pair.

        Parameters:
        ----------
        values: NumericArray
            Element-wise products in cross layout (n, m, *features).
        query_array: NumericArray
            Original query rows (n, *features) for norms.
        gallery_array: NumericArray
            Original gallery rows (m, *features) for norms.
        **kwargs: Any
            Unused.

        Returns:
        --------
        FloatArray
            Shape (n, m) cosine distances.
        """
        dot = np.sum(values, axis=self._sample_value_axes(values))
        q_norm = np.linalg.norm(query_array, axis=-1)
        g_norm = np.linalg.norm(gallery_array, axis=-1)
        denom = q_norm[:, None] * g_norm[None, :]
        return 1.0 - (dot / np.maximum(denom, 1e-12))

    def _cross_array(
        self,
        query_array: NumericArray,
        gallery_array: NumericArray,
        **kwargs: Any,
    ) -> FloatArray:
        """
        Broadcast cross layout then reduce to cosine distances.

        Parameters:
        ----------
        query_array: NumericArray
            Shape (n, *sample_shape).
        gallery_array: NumericArray
            Shape (m, *sample_shape).
        **kwargs: Any
            Forwarded to _elementwise_values and _reduce_elementwise_values.

        Returns:
        --------
        FloatArray
            Cosine distance matrix (n, m) as float.
        """
        np.broadcast_shapes(query_array.shape[1:], gallery_array.shape[1:])
        values = self._elementwise_values(
            query_array[:, None, ...], gallery_array[None, ...], **kwargs
        )
        return np.asarray(
            self._reduce_elementwise_values(
                values=values,
                query_array=query_array,
                gallery_array=gallery_array,
                **kwargs,
            ),
            dtype=float,
        )

    @property
    def metric(self) -> SimilarityCorrelationDistanceMetric:
        """
        Enum tag for this calculator.

        Returns:
        --------
        SimilarityCorrelationDistanceMetric
            Always COSINE.
        """
        from ..metric import SimilarityCorrelationDistanceMetric

        return SimilarityCorrelationDistanceMetric.COSINE
