# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
"""
Backward Pass Implementations for SDFG Elements.

This package provides backward (gradient) implementations for various SDFG node types.
Each implementation defines how to compute gradients for specific operations.

Implementation Categories
-------------------------
1. **DaCe Nodes** (dace_nodes.py):
   - Core SDFG elements: Tasklet, MapEntry, AccessNode, etc.
   - Fundamental building blocks for all DaCe programs
   - Registered in DaceNodeBackwardImplementations

2. **DaCe Reduction Nodes** (dace_reduction_nodes.py):
   - Reduction operations: Sum, Max, Min
   - Registered using @autoregister decorator

3. **ONNX Operations** (onnx_ops.py):
   - ONNX-specific operations from dace.libraries.onnx
   - Neural network layers and operators
   - Supports ONNX model differentiation
"""

import dace.autodiff.implementations.dace_reduction_nodes
from dace.autodiff.implementations.dace_nodes import DaceNodeBackwardImplementations

# ONNX ops are optional
try:
    import dace.autodiff.implementations.onnx_ops
except ImportError:
    # ONNX backward implementations not available
    pass

__all__ = [
    "DaceNodeBackwardImplementations",
]
