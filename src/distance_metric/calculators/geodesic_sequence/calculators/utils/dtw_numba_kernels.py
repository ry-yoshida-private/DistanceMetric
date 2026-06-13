"""
Numba-accelerated DTW dynamic programming.

Computes the classic DTW alignment cost with unit-step transitions (horizontal,
vertical, diagonal) for contiguous float64 inputs: univariate shape (time,)
or multivariate shape (time, features).

Private ``@njit`` kernels implement the recurrence; ``dtw_cost_*`` functions are
the stable entry points used by ``dynamic_time_warping``.
"""

from __future__ import annotations

from typing import TypeAlias

import numpy as np
import numpy.typing as npt
from numba import njit

Float64Array: TypeAlias = npt.NDArray[np.float64]
"""Any-dimensional ndarray holding IEEE binary64 samples (caller fixes rank)."""


@njit(cache=True)
def _minimum_previous_accumulated_cost(
    accumulated: Float64Array,
    row_index: int,
    column_index: int,
) -> np.float64:
    """
    Minimum accumulated cost among the three admissible predecessor cells.

    Parameters
    ----------
    accumulated : Float64Array
        DP table of shape (n + 1, m + 1); unused entries must be positive
        infinity so they never minimize the recurrence.
    row_index : int
        Target row in ``accumulated``, in ``1 .. n`` inclusive.
    column_index : int
        Target column in ``accumulated``, in ``1 .. m`` inclusive.

    Returns
    -------
    np.float64
        Minimum of the up, left, and diagonal neighbors at the given indices.
    """
    from_up: np.float64 = accumulated[row_index - 1, column_index]
    from_left: np.float64 = accumulated[row_index, column_index - 1]
    from_diagonal: np.float64 = accumulated[row_index - 1, column_index - 1]
    best: np.float64 = from_up if from_up < from_left else from_left
    if from_diagonal < best:
        best = from_diagonal
    return best


@njit(cache=True)
def _dtw_cost_univariate_kernel(query: Float64Array, gallery: Float64Array):
    n_time: int = int(query.shape[0])
    m_time: int = int(gallery.shape[0])
    accumulated = np.full((n_time + 1, m_time + 1), np.inf, dtype=np.float64)
    accumulated[0, 0] = 0.0
    for row_i in range(1, n_time + 1):
        for col_j in range(1, m_time + 1):
            local_cost: np.float64 = np.abs(query[row_i - 1] - gallery[col_j - 1])
            accumulated[row_i, col_j] = local_cost + _minimum_previous_accumulated_cost(
                accumulated,
                row_i,
                col_j,
            )
    return accumulated[n_time, m_time]


@njit(cache=True)
def _dtw_cost_multivariate_kernel(query: Float64Array, gallery: Float64Array):
    n_time: int = int(query.shape[0])
    m_time: int = int(gallery.shape[0])
    n_features: int = int(query.shape[1])
    accumulated = np.full((n_time + 1, m_time + 1), np.inf, dtype=np.float64)
    accumulated[0, 0] = 0.0
    for row_i in range(1, n_time + 1):
        for col_j in range(1, m_time + 1):
            squared_sum: np.float64 = np.float64(0.0)
            for feat_k in range(n_features):
                delta: np.float64 = query[row_i - 1, feat_k] - gallery[col_j - 1, feat_k]
                squared_sum += delta * delta
            local_cost: np.float64 = np.sqrt(squared_sum)
            accumulated[row_i, col_j] = local_cost + _minimum_previous_accumulated_cost(
                accumulated,
                row_i,
                col_j,
            )
    return accumulated[n_time, m_time]


def dtw_cost_univariate(query: Float64Array, gallery: Float64Array) -> np.float64:
    """
    DTW total cost for two one-dimensional sequences.

    Parameters
    ----------
    query : Float64Array
        Shape (n,), dtype float64. Query series samples along axis 0.
    gallery : Float64Array
        Shape (m,), dtype float64. Gallery series samples along axis 0.

    Returns
    -------
    np.float64
        Minimum cumulative alignment cost under standard DTW recurrence.

    Notes
    -----
    Local cost is the absolute difference between paired samples.
    """
    raw = _dtw_cost_univariate_kernel(query, gallery)
    return np.float64(raw)


def dtw_cost_multivariate(query: Float64Array, gallery: Float64Array) -> np.float64:
    """
    DTW total cost for two multivariate sequences.

    Parameters
    ----------
    query : Float64Array
        Shape (n, d), dtype float64. Rows index time; columns index features.
    gallery : Float64Array
        Shape (m, d), dtype float64. Same layout as query.

    Returns
    -------
    np.float64
        Minimum cumulative alignment cost under standard DTW recurrence.

    Notes
    -----
    Local cost is the Euclidean distance between feature vectors at paired time
    indices (square root of the sum of squared coordinate differences).
    """
    raw = _dtw_cost_multivariate_kernel(query, gallery)
    return np.float64(raw)
