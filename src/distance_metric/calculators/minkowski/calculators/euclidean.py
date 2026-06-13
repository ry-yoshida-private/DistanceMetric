"""
Euclidean (L2) distance calculator.

Delegates to MinkowskiDistanceCalculatorBase with fixed exponent two.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..calculator import MinkowskiDistanceCalculatorBase

if TYPE_CHECKING:
    from ..metric import MinkowskiDistanceMetric


class EuclideanDistanceCalculator(MinkowskiDistanceCalculatorBase):
    """
    Euclidean (L2) distance between equal-shaped samples.

    Notes:
    -----
    Uses MinkowskiDistanceCalculatorBase with norm_order 2.
    """

    @property
    def norm_order(self) -> float:
        """
        Fixed L2 exponent.

        Returns:
        --------
        float
            Always 2.0.
        """
        return 2.0

    @property
    def metric(self) -> MinkowskiDistanceMetric:
        """
        Enum tag for this calculator.

        Returns:
        --------
        MinkowskiDistanceMetric
            Always MinkowskiDistanceMetric.EUCLIDEAN.
        """
        from ..metric import MinkowskiDistanceMetric
        return MinkowskiDistanceMetric.EUCLIDEAN
