"""
PyTorch Dispatchers for DaCe Modules.

This module provides different dispatcher implementations for executing DaCe SDFGs
from PyTorch. Dispatchers handle:
- Compiling SDFGs to native code
- Initializing runtime state and memory
- Converting between PyTorch tensors and DaCe arrays
- Calling forward and backward SDFG functions
- Managing the integration with PyTorch's autograd system

Available dispatchers:
- CTypes dispatcher: Uses ctypes for direct C function calls
- C++ PyTorch extension: Registers as a native PyTorch extension with custom autograd
"""

from .common import DaCeMLTorchFunction
from .cpp_torch_extension import register_and_compile_torch_extension
from .ctypes_module import get_ctypes_dispatcher

__all__ = [
    "DaCeMLTorchFunction",
    "register_and_compile_torch_extension",
    "get_ctypes_dispatcher"
]
