"""
Manhattan (L1) distance calculator.

Delegates to MinkowskiDistanceCalculatorBase with exponent one.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..calculator import MinkowskiDistanceCalculatorBase

if TYPE_CHECKING:
    from ..metric import MinkowskiDistanceMetric


class ManhattanDistanceCalculator(MinkowskiDistanceCalculatorBase):
    """
    Sum of absolute coordinate differences.

    Notes:
    -----
    Equivalent to Minkowski distance with p equal to 1.
    """

    @property
    def norm_order(self) -> float:
        """
        Fixed L1 exponent.

        Returns:
        --------
        float
            Always 1.0.
        """
        return 1.0

    @property
    def metric(self) -> MinkowskiDistanceMetric:
        """
        Enum tag for this calculator.

        Returns:
        --------
        MinkowskiDistanceMetric
            Always MinkowskiDistanceMetric.MANHATTAN.
        """
        from ..metric import MinkowskiDistanceMetric
        return MinkowskiDistanceMetric.MANHATTAN
