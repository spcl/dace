"""
DaCe PyTorch Integration Library.

This module provides integration between DaCe (Data-Centric Parallel Programming)
and PyTorch, enabling:
- Compilation of PyTorch operations to optimized DaCe SDFGs
- Interoperability between PyTorch tensors and DaCe arrays
- Support for both CPU (PyTorch) and GPU (PyTorchCUDA) execution
- DLPack-based zero-copy tensor sharing

The main exports are environment classes that define the PyTorch runtime
dependencies and configuration for code generation.
"""

try:
    from .environments import PyTorch, PyTorchCUDA
    __all__ = ["PyTorch", "PyTorchCUDA"]
except ImportError:
    # PyTorch not available
    PyTorch = None
    PyTorchCUDA = None
    __all__ = []
