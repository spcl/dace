# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
DaCe PyTorch Integration Library.

This module provides integration between DaCe (Data-Centric Parallel Programming)
and PyTorch, enabling:
- Compilation of PyTorch operations to optimized DaCe SDFGs
- Interoperability between PyTorch tensors and DaCe arrays
- Support for both CPU (PyTorch) and GPU (PyTorchGPU) execution
- DLPack-based zero-copy tensor sharing
- ATen operators as library nodes (:mod:`dace.libraries.torch.nodes`), which do not require PyTorch

The main exports are the ATen library nodes and the environment classes that define the PyTorch runtime
dependencies and configuration for code generation.
"""

from dace.library import register_library

from .nodes import (
    ATenOp,
    AvgPool,
    Convolution,
    ConvolutionBackward,
    MaxPoolWithIndices,
    MaxPoolWithIndicesBackward,
    NativeDropout,
    aten_node,
    node_class,
    registered_targets,
)

__all__ = [
    "ATenOp",
    "AvgPool",
    "Convolution",
    "ConvolutionBackward",
    "MaxPoolWithIndices",
    "MaxPoolWithIndicesBackward",
    "NativeDropout",
    "aten_node",
    "node_class",
    "registered_targets",
]

try:
    from .environments import PyTorch, PyTorchGPU

    __all__ += ["PyTorch", "PyTorchGPU"]
except ImportError:
    # PyTorch not available
    PyTorch = None
    PyTorchGPU = None

register_library(__name__, "torch")
