"""
Binary cross-elementwise distance pattern.

Distances in this family compare discrete coordinates (boolean membership or
integer category codes). Floating-point feature vectors are rejected at runtime.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any

import numpy as np

from ...array_types import BinaryArray, ElementwiseValuesArray, FloatArray
from ...result import DistanceResult
from ..cross_elementwise import CrossElementwiseCalculatorBase


class BinaryCrossElementwiseCalculatorBase(CrossElementwiseCalculatorBase):
    """
    Base for metrics on boolean or integer-coded binary vectors.

    Public entry points accept BinaryArray only. Each concrete metric compares
    coordinates with equality or logical set operations, then reduces to a float
    distance matrix.
    """

    def pairwise(self, array: BinaryArray, **kwargs: Any) -> DistanceResult:
        """
        Calculate pairwise distances for boolean or integer-coded rows.

        Parameters:
        ----------
        array: BinaryArray
            Batch with shape (n, *features). Dtype must be boolean or integer.

        Returns:
        --------
        DistanceResult
            Pairwise matrix in the value field with shape (n, n).
        """
        return self.cross(array, array, **kwargs)

    def cross(
        self,
        query_array: BinaryArray,
        gallery_array: BinaryArray,
        **kwargs: Any,
    ) -> DistanceResult:
        """
        Calculate cross distances between query and gallery batches.

        Parameters:
        ----------
        query_array: BinaryArray
            Query batch with shape (n, *features). Dtype must be boolean or integer.
        gallery_array: BinaryArray
            Gallery batch with shape (m, *features). Dtype must be boolean or integer.

        Returns:
        --------
        DistanceResult
            Cross matrix in the value field with shape (n, m).
        """
        return self._wrap(self._cross_array(query_array, gallery_array, **kwargs))

    def elementwise(
        self,
        query_array: BinaryArray,
        gallery_array: BinaryArray,
        **kwargs: Any,
    ) -> DistanceResult:
        """
        Compare one query-gallery pair and return a rank-1 distance array.

        Parameters:
        ----------
        query_array: BinaryArray
            Query tensor for one logical pair before batching.
        gallery_array: BinaryArray
            Gallery tensor broadcast-compatible with query_array.

        Returns:
        --------
        DistanceResult
            value is always a rank-1 FloatArray (shape (1,) for a single score).
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

    @staticmethod
    def _validate_binary_dtype(array: BinaryArray, parameter_name: str) -> None:
        """
        Reject floating-point arrays for binary-family metrics.

        Parameters:
        ----------
        array: BinaryArray
            Candidate input array.
        parameter_name: str
            Name used in the raised ValueError.

        Raises:
        ------
        ValueError
            If dtype is neither boolean nor integer.
        """
        dtype = np.dtype(array.dtype)
        is_boolean = np.issubdtype(dtype, np.bool_)
        is_integer = np.issubdtype(dtype, np.integer)
        if not (is_boolean or is_integer):
            raise ValueError(
                f"{parameter_name} must have boolean or integer dtype, got {dtype}"
            )

    def _cross_array(
        self,
        query_array: BinaryArray,
        gallery_array: BinaryArray,
        **kwargs: Any,
    ) -> FloatArray:
        """
        Compute cross distances after validating binary dtypes.

        Parameters:
        ----------
        query_array: BinaryArray
            Query batch with shape (n, *sample_shape).
        gallery_array: BinaryArray
            Gallery batch with shape (m, *sample_shape).

        Returns:
        --------
        FloatArray
            Cross distance matrix of shape (n, m).
        """
        self._validate_binary_dtype(query_array, "query_array")
        self._validate_binary_dtype(gallery_array, "gallery_array")
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

    @abstractmethod
    def _elementwise_values(
        self,
        query_array: BinaryArray,
        gallery_array: BinaryArray,
        **kwargs: Any,
    ) -> FloatArray:
        """
        Per-coordinate terms before aggregation on binary inputs.

        Parameters:
        ----------
        query_array: BinaryArray
            Query tensor broadcast-compatible with gallery_array for this call.
        gallery_array: BinaryArray
            Gallery tensor with the same broadcasting rules as query_array.

        Returns:
        --------
        FloatArray
            Terms that _reduce_elementwise_values will aggregate.
        """

    @abstractmethod
    def _reduce_elementwise_values(
        self,
        values: ElementwiseValuesArray,
        query_array: BinaryArray,
        gallery_array: BinaryArray,
        **kwargs: Any,
    ) -> FloatArray | float:
        """
        Reduce element-wise binary terms into final distance scores.

        Parameters:
        ----------
        values: ElementwiseValuesArray
            Element-wise values from _elementwise_values with shape (n, m, *features).
        query_array: BinaryArray
            Original query batch with shape (n, *features).
        gallery_array: BinaryArray
            Original gallery batch with shape (m, *features).

        Returns:
        --------
        FloatArray | float
            Reduced distance values, usually shape (n, m).
        """
