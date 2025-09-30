"""
Library Integration for Automatic Differentiation.

This package provides integration between DaCe's autodiff system and various
libraries and frontends. It enables differentiation of code that uses
library operations and provides hooks for frontend-specific optimizations.
"""

import dace.library

from . import torch_integration
from . import library
from . import python_frontend

dace.library.register_library(__name__, "autodiff")

__all__ = [
    "torch_integration",
    "library",
    "python_frontend",
]
