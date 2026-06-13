"""
Binary-set distance metric identifiers (Hamming, Jaccard).

Each enum value corresponds to a DistanceCalculator implementation used when
resolving DistanceMetric from configuration or APIs.
"""

from enum import Enum

from ...calculator import DistanceCalculator
from ...metric import DistanceMetric


class BinaryDistanceMetric(DistanceMetric, Enum):
    """
    Enumeration of binary and set-style distances between aligned vectors.

    Members:
    --------
    HAMMING : str
        Coordinate mismatch rate between aligned vectors.
    JACCARD : str
        Set-style distance from boolean masks along features.

    Notes:
    -----
    String values match DistanceMetric naming; the calculator property resolves
    to concrete calculator instances with lazy imports.
    """

    HAMMING = "hamming"
    JACCARD = "jaccard"

    @property
    def calculator(self) -> DistanceCalculator:
        """
        Calculator instance associated with this metric name.

        Returns:
        --------
        DistanceCalculator
            HammingDistanceCalculator or JaccardDistanceCalculator.

        Raises:
        -------
        ValueError
            If the enum member is not mapped (internal consistency error).
        """
        match self:
            case BinaryDistanceMetric.HAMMING:
                from .calculators.hamming import HammingDistanceCalculator
                return HammingDistanceCalculator()
            case BinaryDistanceMetric.JACCARD:
                from .calculators.jaccard import JaccardDistanceCalculator
                return JaccardDistanceCalculator()
        raise ValueError(f"No calculator found for metric {self}")
