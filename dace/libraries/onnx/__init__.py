# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
"""
DaCe ONNX Integration Library.

This module provides comprehensive support for importing and executing ONNX models
in DaCe. It enables:

- Importing ONNX models and converting them to DaCe SDFGs
- Implementing ONNX operations as DaCe library nodes
- Automatic shape inference for dynamic models
- Multiple implementation strategies (pure, optimized, etc.)

Main Components:
- ONNXModel: Main class for importing and manipulating ONNX models
- ONNXOp: Base class for ONNX operation nodes in SDFGs
- Schema system: Type checking and validation for ONNX operations

The library is registered with DaCe and uses 'pure' as the default implementation
strategy for ONNX operations.
"""

from dace.library import register_library, _DACE_REGISTERED_LIBRARIES

try:
    # Import schema and node utilities (nodes are lazy-loaded via __getattr__)
    from .schema import onnx_representation, ONNXAttributeType, ONNXAttribute, ONNXTypeConstraint, ONNXParameterType, ONNXSchema, ONNXParameter
    from .nodes import get_onnx_node, has_onnx_node

    register_library(__name__, "dace.libraries.onnx")
    _DACE_REGISTERED_LIBRARIES["dace.libraries.onnx"].default_implementation = "pure"

    ONNX_AVAILABLE = True

    def __getattr__(name):
        """Lazy attribute access for ONNX node classes, ONNXModel, and utilities."""
        if name == 'ONNXModel':
            from dace.frontend.ml.onnx import ONNXModel as _ONNXModel
            return _ONNXModel
        if name == 'parse_variadic_param':
            from .nodes.node_utils import parse_variadic_param as _parse_variadic_param
            return _parse_variadic_param
        if name.startswith('ONNX'):
            # Initialize registry and get the node class
            from .nodes.onnx_op_registry import _initialize_onnx_registry
            _initialize_onnx_registry()
            from .nodes import onnx_op_registry
            if hasattr(onnx_op_registry, name):
                return getattr(onnx_op_registry, name)
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

except ImportError:
    # ONNX library not available
    ONNXModel = None
    onnx_representation = None
    ONNXAttributeType = None
    ONNXAttribute = None
    ONNXTypeConstraint = None
    ONNXParameterType = None
    ONNXSchema = None
    ONNXParameter = None
    ONNX_AVAILABLE = False
