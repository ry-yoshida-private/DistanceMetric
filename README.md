# distance-metric

## Overview

**distance-metric** is a small NumPy-first library for **pairwise**, **cross**, and **elementwise** distances (and similarity-style scores exposed through the same API). Metrics are grouped into families; each family exposes an enum (`DistanceMetric` subclass) whose `.calculator` returns a ready-made `DistanceCalculator`. Numeric outputs are wrapped in `DistanceResult` with a semantic type (`DISTANCE` or `SIMILARITY`).

- **Python**: 3.10+
- **Dependency**: NumPy (see [`requirements.txt`](requirements.txt) and [`pyproject.toml`](pyproject.toml))

Further explanation and usage live under [`src/distance_metric/README.md`](src/distance_metric/README.md) and the family README files under [`src/distance_metric/calculators/`](src/distance_metric/calculators/).

## Installation

From the repository root:

```bash
pip install -e .
```

Or install dependencies only:

```bash
pip install -r requirements.txt
```

## Documentation map

| Topic | Location |
|-------|----------|
| Public exports (`__init__.py`) | [`src/distance_metric/__init__.py`](src/distance_metric/__init__.py) |
| Calculator contract (`pairwise`, `cross`, `elementwise`) | [`src/distance_metric/calculator.py`](src/distance_metric/calculator.py) |
| Minkowski family (longer tutorial-style examples) | [`src/distance_metric/calculators/minkowski/README.md`](src/distance_metric/calculators/minkowski/README.md) |
