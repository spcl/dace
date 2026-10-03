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

# Import TorchDynamo frontend (registers the 'dace' torch.compile backend)
try:
    from dace.frontend.ml.torch.dynamo import compile, DaceBackend, dace_backend
except ImportError:
    compile = None
    DaceBackend = None
    dace_backend = None

__all__ = ['DaceModule', 'module', 'ONNXModel', 'compile', 'DaceBackend', 'dace_backend']
