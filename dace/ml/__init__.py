# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.

# Import PyTorch frontend
try:
    from dace.frontend.ml.torch import DaceModule, module
except ImportError:
    DaceModule = None
    module = None

# Import ONNX frontend
try:
    from dace.frontend.ml.onnx import ONNXModel
except ImportError:
    ONNXModel = None

__all__ = ['DaceModule', 'module', 'ONNXModel']
