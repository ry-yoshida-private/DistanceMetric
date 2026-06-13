# utils

## Overview

Shared low-level helpers for geodesic-sequence calculators (NumPy / Numba kernels).

## Components

| Path | Description |
|------|-------------|
| [`dtw_numba_kernels.py`](dtw_numba_kernels.py) | Numba JIT DTW recurrence, predecessor helper, and `dtw_cost_univariate` / `dtw_cost_multivariate` entry points. |
| [`dtw_pair_cost.py`](dtw_pair_cost.py) | Pair-level DTW dispatch (Numba vs NumPy fallback): local cost matrix, anti-diagonal DP, `compute_pair_dtw_cost`. |
