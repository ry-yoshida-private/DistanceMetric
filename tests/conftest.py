"""Shared pytest fixtures for distance_metric tests."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def numeric_batch() -> np.ndarray:
    """Two-row float batch for pairwise tests."""
    return np.array([[0.0, 0.0], [3.0, 4.0]], dtype=np.float64)


@pytest.fixture
def numeric_query() -> np.ndarray:
    """Single query row."""
    return np.array([[1.0, 2.0]], dtype=np.float64)


@pytest.fixture
def numeric_gallery() -> np.ndarray:
    """Two-row gallery batch."""
    return np.array([[4.0, 6.0], [1.0, 2.0]], dtype=np.float64)


@pytest.fixture
def binary_batch() -> np.ndarray:
    """Boolean batch for pairwise binary tests."""
    return np.array([[True, False, True], [False, True, False]])


@pytest.fixture
def probability_query() -> np.ndarray:
    """Nonnegative mass vector for information-theoretic metrics."""
    return np.array([[0.5, 0.5]], dtype=np.float64)


@pytest.fixture
def probability_gallery() -> np.ndarray:
    """Gallery mass vectors for information-theoretic metrics."""
    return np.array([[0.25, 0.75], [0.5, 0.5]], dtype=np.float64)


@pytest.fixture
def identity_covariance() -> np.ndarray:
    """2x2 identity covariance matrix."""
    return np.eye(2, dtype=np.float64)


@pytest.fixture
def unit_variance() -> np.ndarray:
    """Per-coordinate unit variance for SEuclidean."""
    return np.array([1.0, 1.0], dtype=np.float64)
