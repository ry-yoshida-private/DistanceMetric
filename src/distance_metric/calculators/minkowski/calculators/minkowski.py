"""
Configurable Minkowski (Lp) distance calculator.

Stores exponent p on the instance via norm_order.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..calculator import MinkowskiDistanceCalculatorBase

if TYPE_CHECKING:
    from ..metric import MinkowskiDistanceMetric


class MinkowskiDistanceCalculator(MinkowskiDistanceCalculatorBase):
    """
    Lp distance with user-supplied exponent.

    Notes:
    -----
    For Chebyshev (L-infinity) use ChebyshevDistanceCalculator instead of pushing p
    to infinity numerically.
    """

    def __init__(self, norm_order: float = 2.0) -> None:
        """
        Parameters:
        ----------
        norm_order: float, optional
            Minkowski exponent p used in both powering and the outer root.
        """
        self._norm_order = norm_order

    @property
    def norm_order(self) -> float:
        """
        Current Lp exponent.

        Returns:
        --------
        float
            Value passed at construction time.
        """
        return self._norm_order

    @property
    def metric(self) -> MinkowskiDistanceMetric:
        """
        Enum tag for this calculator.

        Returns:
        --------
        MinkowskiDistanceMetric
            Always MinkowskiDistanceMetric.MINKOWSKI.
        """
        from ..metric import MinkowskiDistanceMetric
        return MinkowskiDistanceMetric.MINKOWSKI
