# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.

try:
    from .torch import DaceModule
except ImportError:
    DaceModule = None

try:
    from .onnx import ONNXModel
except ImportError:
    ONNXModel = None

__all__ = ['DaceModule', 'ONNXModel']
