# Binary Calculator Tests

## Overview

Pytest coverage for Hamming and Jaccard distance calculators, including boolean
and integer inputs, cross/pairwise/elementwise APIs, dtype validation, and
metric enum factory resolution.

## Components

| Component | Description |
| --- | --- |
| [`test_hamming.py`](test_hamming.py) | Hamming distance behavior and float-dtype rejection |
| [`test_jaccard.py`](test_jaccard.py) | Jaccard distance behavior and float-dtype rejection |

## Examples

```bash
pytest tests/calculators/binary -v
```
