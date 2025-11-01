# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
"""
Library Integration for Automatic Differentiation.

This package provides integration between DaCe's autodiff system and various
libraries and frontends. It enables differentiation of code that uses
library operations and provides hooks for frontend-specific optimizations.
"""

import dace.library

from . import library

# PyTorch integrations are optional
try:
    from . import torch_integration
    from . import python_frontend
    TORCH_INTEGRATION_AVAILABLE = True
except ImportError:
    torch_integration = None
    python_frontend = None
    TORCH_INTEGRATION_AVAILABLE = False

dace.library.register_library(__name__, "autodiff")

__all__ = [
    "library",
]

if TORCH_INTEGRATION_AVAILABLE:
    __all__.extend(["torch_integration", "python_frontend"])
