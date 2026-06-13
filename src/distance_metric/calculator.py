from __future__ import annotations

import numpy as np

from abc import ABC, abstractmethod
from typing import Any, ClassVar

from .result import DistanceResult, DistanceResultType

from .array_types import BroadcastArray, FloatArray, NumericArray


class DistanceCalculator(ABC):
    """
    Base calculator. Numeric computation lives in _cross_array; public methods
    return DistanceResult with value and semantic type (distance vs similarity).
    """

    result_type: ClassVar[DistanceResultType] = DistanceResultType.DISTANCE

    def pairwise(self, array: NumericArray, **kwargs: Any) -> DistanceResult:
        """
        Calculate the pairwise distance between all elements in the array.

        Parameters:
        ----------
        array: NumericArray
            The array to calculate the pairwise distance between with shape (n, *).

        Returns:
        --------
        DistanceResult
            Pairwise matrix in the value field with shape (n, n).
        """
        return self.cross(array, array, **kwargs)

    def cross(
        self,
        query_array: NumericArray,
        gallery_array: NumericArray,
        **kwargs: Any,
    ) -> DistanceResult:
        """
        Calculate the cross distance between the query and gallery arrays.

        Parameters:
        ----------
        query_array: NumericArray
            The query array to calculate the cross distance with shape (n, *).
        gallery_array: NumericArray
            The gallery array to calculate the cross distance with shape (m, *).

        Returns:
        --------
        DistanceResult
            Cross matrix in the value field with shape (n, m).
        """
        return self._wrap(self._cross_array(query_array, gallery_array, **kwargs))

    @abstractmethod
    def _cross_array(
        self,
        query_array: NumericArray,
        gallery_array: NumericArray,
        **kwargs: Any,
    ) -> FloatArray:
        """
        Return the raw numeric matrix (distance or similarity values).

        Subclasses implement computation here; the public cross method wraps
        the result in DistanceResult using the result_type class attribute.
        """

    def _wrap(self, value: FloatArray) -> DistanceResult:
        return DistanceResult(value=value, type=self.result_type)

    def elementwise(
        self,
        query_array: NumericArray,
        gallery_array: NumericArray,
        **kwargs: Any,
    ) -> DistanceResult:
        """
        Compare two samples (typically one pair) and return their metric value.

        Parameters:
        ----------
        query_array: NumericArray
            Query tensor for one logical pair before batching.
        gallery_array: NumericArray
            Gallery tensor for the same pair. Must be broadcast-compatible with
            query_array per NumPy rules; identical shapes are not required.

        Returns:
        --------
        DistanceResult
            value is always a rank-1 numpy.ndarray (shape (1,) for a single scalar
            score), never a Python float.

        Notes:
        -----
        Broadcast compatibility is checked via _validate_broadcast_compatible.
        Inputs are then promoted with a leading batch axis so _cross_array runs
        in the same batched mode as cross.
        """
        self._validate_broadcast_compatible(
            query_array=query_array,
            gallery_array=gallery_array,
        )
        inner = self._cross_array(
            query_array=query_array[None, ...],
            gallery_array=gallery_array[None, ...],
            **kwargs,
        )
        inner_arr = np.asarray(inner)
        cell = inner_arr.reshape(inner_arr.shape[0], inner_arr.shape[1])[0, 0]
        return self._wrap(np.atleast_1d(cell))

    def _validate_broadcast_compatible(
        self,
        query_array: BroadcastArray,
        gallery_array: BroadcastArray,
    ) -> None:
        """
        Ensure query_array and gallery_array are NumPy broadcast-compatible.

        Uses numpy.broadcast_shapes; raises ValueError if the shapes cannot
        be broadcast together. Matching shapes are not required.
        """
        np.broadcast_shapes(query_array.shape, gallery_array.shape)
