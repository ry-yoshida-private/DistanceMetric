"""Shared NumPy array type aliases for static type checking.

NumericArray
    Integer or floating element dtypes. Use for continuous numeric calculator
    inputs and for hook method parameters on CrossElementwiseCalculatorBase
    subclasses.

BinaryArray
    Boolean or integer element dtypes. Use for set- and symbol-comparison metrics
    (Hamming, Jaccard) where coordinates are discrete membership or category codes.

FloatArray
    Floating element dtypes only. Use for distance-matrix results, auxiliary
    matrices (covariance, inverse covariance, variance), element-wise terms
    after reduction hooks, and standalone helpers that require real-valued data.

BroadcastArray
    Union of BinaryArray and NumericArray. Use only for shape-only helpers
    (for example broadcast compatibility checks) that do not depend on dtype.

ElementwiseValuesArray
    Union of NumericArray and FloatArray. Use for tensors emitted by
    _elementwise_values before reduction in cross-elementwise calculators.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

NumericArray = NDArray[np.integer[Any] | np.floating[Any]]
BinaryArray = NDArray[np.bool_ | np.integer[Any]]
FloatArray = NDArray[np.floating[Any]]
BroadcastArray = BinaryArray | NumericArray
ElementwiseValuesArray = NumericArray | FloatArray
