# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
"""
DaCe PyTorch Integration Library.

This module provides integration between DaCe (Data-Centric Parallel Programming)
and PyTorch, enabling:
- Compilation of PyTorch operations to optimized DaCe SDFGs
- Interoperability between PyTorch tensors and DaCe arrays
- Support for both CPU (PyTorch) and GPU (PyTorchGPU) execution
- DLPack-based zero-copy tensor sharing

The main exports are environment classes that define the PyTorch runtime
dependencies and configuration for code generation.
"""

try:
    from .environments import PyTorch, PyTorchGPU
    __all__ = ["PyTorch", "PyTorchGPU"]
except ImportError:
    # PyTorch not available
    PyTorch = None
    PyTorchGPU = None
    __all__ = []
