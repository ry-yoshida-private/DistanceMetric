"""
Great-circle (Haversine) distance on a sphere.

Last dimension must be latitude then longitude in radians (or consistent angular
units). Output distance uses the same unit as radius (default Earth radius in km).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ...cross_elementwise import CrossElementwiseCalculatorBase

from ....array_types import FloatArray, NumericArray

if TYPE_CHECKING:
    from ..metric import GeodesicSequenceDistanceMetric


class HaversineDistanceCalculator(CrossElementwiseCalculatorBase):
    """
    Great-circle distance from latitude-longitude samples on a sphere.

    Notes:
    -----
    Pass radius as keyword to cross, pairwise, or elementwise to scale arc length.
    """

    def _elementwise_values(
        self,
        query_array: NumericArray,
        gallery_array: NumericArray,
        **kwargs: Any,
    ) -> FloatArray:
        """
        Haversine a term per coordinate pair (before central angle).

        Parameters:
        ----------
        query_array: NumericArray
            Shape (..., 2) with latitude and longitude in last axis.
        gallery_array: NumericArray
            Same trailing layout as query_array.
        **kwargs: Any
            Unused.

        Returns:
        --------
        FloatArray
            Haversine of half central angle, expanded with a trailing length-1 axis.

        Raises:
        -------
        ValueError
            If the last dimension is not of size two.
        """
        self._validate_broadcast_compatible(
            query_array=query_array,
            gallery_array=gallery_array,
        )
        if query_array.shape[-1] != 2:
            raise ValueError("Haversine distance expects samples as [latitude, longitude].")
        lat1, lon1 = query_array[..., 0], query_array[..., 1]
        lat2, lon2 = gallery_array[..., 0], gallery_array[..., 1]
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
        return np.asarray(a, dtype=float)[..., None]

    def _reduce_elementwise_values(
        self,
        values: NumericArray,
        query_array: NumericArray,
        gallery_array: NumericArray,
        *,
        radius: float = 6371.0,
        **kwargs: Any,
    ) -> FloatArray:
        """
        Convert haversine a to arc length on the sphere.

        Parameters:
        ----------
        values: NumericArray
            Output of _elementwise_values; last axis holds the a channel.
        query_array: NumericArray
            Unused.
        gallery_array: NumericArray
            Unused.
        radius: float, optional
            Sphere radius; default is Earth mean radius in kilometers.
        **kwargs: Any
            Unused.

        Returns:
        --------
        FloatArray
            Distances with the same length unit as radius.
        """
        radius = float(radius)
        a = values[..., 0]
        c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(np.maximum(1.0 - a, 0.0)))
        return radius * c

    @property
    def metric(self) -> GeodesicSequenceDistanceMetric:
        """
        Enum tag for this calculator.

        Returns:
        --------
        GeodesicSequenceDistanceMetric
            Always HAVERSINE.
        """
        from ..metric import GeodesicSequenceDistanceMetric
        return GeodesicSequenceDistanceMetric.HAVERSINE
