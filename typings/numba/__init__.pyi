"""
Minimal stubs for Numba so static analyzers resolve ``njit`` precisely enough.

The real Numba package provides runtime compilation; these declarations exist only
for type checking.
"""

from collections.abc import Callable
from typing import Any, TypeVar

_F = TypeVar("_F", bound=Callable[..., Any])


def njit(*args: Any, **kwargs: Any) -> Callable[[_F], _F]:
    """Compile a function with LLVM for scalar and NumPy array workloads."""
