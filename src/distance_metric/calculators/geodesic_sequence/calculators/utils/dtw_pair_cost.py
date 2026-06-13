"""
DTW pair-cost helpers: Numba-backed fast paths and NumPy fallback.

Dispatches (1,) / (T, D) contiguous float64 pairs to compiled kernels; other
broadcast layouts build a local cost matrix and run anti-diagonal DP.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from distance_metric.calculators.geodesic_sequence.calculators.utils.dtw_numba_kernels import (
    dtw_cost_multivariate,
    dtw_cost_univariate,
)


def _dtw_antidiagonal_numpy(cost: np.ndarray) -> float:
    """
    Classic DTW DP using anti-diagonal NumPy vectorization.

    Parameters
    ----------
    cost : np.ndarray
        Shape (n, m). Pairwise local costs between series positions.

    Returns
    -------
    float
        Total DTW alignment cost.
    """
    n, m = cost.shape
    dp = np.full((n + 1, m + 1), np.inf, dtype=float)
    dp[0, 0] = 0.0
    for k in range(2, n + m + 1):
        i_idx: npt.NDArray[np.intp] = np.arange(
            max(1, k - m), min(n, k - 1) + 1, dtype=np.intp
        )
        j_idx: npt.NDArray[np.intp] = np.asarray(k - i_idx, dtype=np.intp)
        dp[i_idx, j_idx] = cost[i_idx - 1, j_idx - 1] + np.minimum(
            np.minimum(dp[i_idx - 1, j_idx], dp[i_idx, j_idx - 1]),
            dp[i_idx - 1, j_idx - 1],
        )
    return float(dp[n, m])


def _numpy_local_cost_matrix(query_array: np.ndarray, gallery_array: np.ndarray) -> np.ndarray:
    """
    Build the DTW local cost matrix from broadcastable query and gallery tensors.

    Parameters
    ----------
    query_array : np.ndarray
        Samples along axis 0; trailing axes follow NumPy broadcasting rules with
        gallery_array.
    gallery_array : np.ndarray
        Samples along axis 0; trailing axes follow NumPy broadcasting rules with
        query_array.

    Returns
    -------
    np.ndarray
        Shape (n, m) float array of pairwise local costs.
    """
    q = np.asarray(query_array)
    g = np.asarray(gallery_array)
    if q.ndim == 1 and g.ndim == 1:
        return np.abs(q[:, None] - g[None, :])
    return np.linalg.norm(q[:, None, ...] - g[None, ...], axis=-1)


def compute_pair_dtw_cost(query_array: np.ndarray, gallery_array: np.ndarray) -> float:
    """
    Total DTW cost for one query sequence versus one gallery sequence.

    Parameters
    ----------
    query_array : np.ndarray
        One-dimensional sequence or shape (T, D) time series, or other layouts
        handled by the NumPy fallback path.
    gallery_array : np.ndarray
        Same dimensional conventions as query_array where Numba applies.

    Returns
    -------
    float
        Minimum cumulative alignment cost.

    Raises
    ------
    ValueError
        If either sequence has zero length along axis 0, or if multivariate
        Numba inputs disagree on the feature dimension.
    """
    q = np.asarray(query_array)
    g = np.asarray(gallery_array)
    if q.shape[0] == 0 or g.shape[0] == 0:
        raise ValueError("DTW requires non-empty sequences along the time axis.")
    match (q.ndim, g.ndim):
        case (1, 1):
            return float(
                dtw_cost_univariate(
                    np.ascontiguousarray(q, dtype=np.float64),
                    np.ascontiguousarray(g, dtype=np.float64),
                )
            )
        case (2, 2):
            if q.shape[1] != g.shape[1]:
                raise ValueError(
                    "Multivariate DTW requires matching feature dimensions between sequences."
                )
            return float(
                dtw_cost_multivariate(
                    np.ascontiguousarray(q, dtype=np.float64),
                    np.ascontiguousarray(g, dtype=np.float64),
                )
            )
        case _:
            cost = _numpy_local_cost_matrix(q, g)
            return _dtw_antidiagonal_numpy(cost)
