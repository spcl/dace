"""
DaCe Automatic Differentiation (AD) System.

This module provides reverse-mode automatic differentiation for DaCe programs,
enabling automatic computation of gradients for optimized numerical kernels.

Main Components
---------------
- **add_backward_pass**: Main entry point for adding backward pass to an SDFG
- **BackwardPassGenerator**: Core algorithm for generating backward passes
- **BackwardImplementation**: ABC for implementing operation-specific backward rules
- **BackwardContext**: Context information for backward pass generation
- **BackwardResult**: Result of backward pass generation with forward/backward SDFGs
- **AutoDiffException**: Base exception for autodiff errors

Key Features
------------
- Support for control flow (loops, conditionals)
- Data forwarding strategies (store vs recompute tradeoffs)
- Extensible backward implementations for library nodes
- Integration with PyTorch autograd
- Automatic memory management for intermediate values


"""

from .base_abc import BackwardImplementation, BackwardContext, BackwardResult, AutoDiffException
from .backward_pass_generator import BackwardPassGenerator
from .autodiff import add_backward_pass

try:
    from .torch import make_backward_function
    TORCH_INTEGRATION_AVAILABLE = True
except ImportError:
    make_backward_function = None
    TORCH_INTEGRATION_AVAILABLE = False

import sys
from . import library

__all__ = [
    # Main API
    "add_backward_pass",
    # Core classes
    "BackwardPassGenerator",
    "BackwardContext",
    "BackwardResult",
    # Extension points
    "BackwardImplementation",
    # Exceptions
    "AutoDiffException",
    # Submodules
    "library",
]

if TORCH_INTEGRATION_AVAILABLE:
    __all__.append("make_backward_function")
