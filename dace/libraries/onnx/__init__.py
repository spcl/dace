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
    # Import all dynamically registered ONNX nodes (ONNXOp, ONNXEinsum, etc.)
    # This star import is necessary to expose all ONNX operation classes
    from .nodes import *
    from .schema import onnx_representation, ONNXAttributeType, ONNXAttribute, ONNXTypeConstraint, ONNXParameterType, ONNXSchema, ONNXParameter
    from .onnx_importer import ONNXModel

    register_library(__name__, "dace.libraries.onnx")
    _DACE_REGISTERED_LIBRARIES["dace.libraries.onnx"].default_implementation = "pure"

    ONNX_AVAILABLE = True
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
