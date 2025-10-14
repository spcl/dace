# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
from dace.libraries.onnx.shape_inference.symbolic_shape_infer import SymbolicShapeInference, get_opset


def infer_shapes(onnx_model, auto_merge=False):
    result = SymbolicShapeInference.infer_shapes(onnx_model, auto_merge=auto_merge)
    if result is None:
        raise ValueError("Symbolic shape inference failed")
    return result
