# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.

# Import PyTorch frontend
try:
    from dace.frontend.ml.torch import DaceModule
except ImportError:
    DaceModule = None

# Import ONNX frontend
try:
    from dace.frontend.ml.onnx import ONNXModel
except ImportError:
    ONNXModel = None

# Import TensorFlow frontend
try:
    from dace.frontend.ml.tensorflow import TFSession
except ImportError:
    TFSession = None

__all__ = ['DaceModule', 'ONNXModel', 'TFSession']
