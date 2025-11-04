# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
"""
Batch Norm is the only op that has a shared name between inputs and outputs. Test that prepending "in_" and "out_" works
"""

import pytest

pytest.importorskip("onnx", reason="ONNX not installed. Please install with: pip install dace[ml]")
pytest.importorskip("torch", reason="PyTorch not installed. Please install with: pip install dace[ml]")

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import dace
import dace.libraries.onnx as donnx
from dace.ml import DaceModule

from tests.utils import torch_tensors_close


@pytest.mark.onnx
@pytest.mark.parametrize("training_mode", [True, False])
def test_bn_standalone(training_mode: bool):

    if training_mode:

        @dace.program
        def test_bn_standalone(X: dace.float32[8, 3, 32,
                                               32], scale: dace.float32[3], B: dace.float32[3], mean: dace.float32[3],
                               var: dace.float32[3], running_mean: dace.float32[3], running_var: dace.float32[3]):
            Y = dace.define_local([8, 3, 32, 32], dace.float32)
            donnx.ONNXBatchNormalization(
                X=X,
                scale=scale,
                B=B,
                input_mean=mean,
                input_var=var,
                Y=Y,
                running_mean=running_mean,
                running_var=running_var,
                training_mode=True,
            )
            return Y
    else:

        @dace.program
        def test_bn_standalone(X: dace.float32[8, 3, 32, 32], scale: dace.float32[3], B: dace.float32[3],
                               mean: dace.float32[3], var: dace.float32[3]):

            Y = dace.define_local([8, 3, 32, 32], dace.float32)
            donnx.ONNXBatchNormalization(X=X,
                                         scale=scale,
                                         B=B,
                                         input_mean=mean,
                                         input_var=var,
                                         Y=Y,
                                         training_mode=training_mode)
            return Y

    X = torch.randn(8, 3, 32, 32)
    scale = torch.randn(3)
    B = torch.randn(3)
    mean = torch.randn(3)
    var = torch.abs(torch.randn(3))
    X_torch, scale_torch, B_torch, mean_torch, var_torch = X.clone(), scale.clone(), B.clone(), mean.clone(), var.clone(
    )
    if training_mode:
        running_mean = np.zeros(3, dtype=np.float32)
        running_var = np.ones(3, dtype=np.float32)
        dace_result = test_bn_standalone(X, scale, B, mean, var, running_mean, running_var)
    else:
        dace_result = test_bn_standalone(X, scale, B, mean, var)

    pt_result = F.batch_norm(X_torch, mean_torch, var_torch, scale_torch, B_torch, training=training_mode)
    torch_tensors_close("output", pt_result, torch.from_numpy(dace_result))


@pytest.mark.onnx
def test_bn_in_import():

    class Module(torch.nn.Module):

        def __init__(self):
            super(Module, self).__init__()
            self.bn = nn.BatchNorm2d(3, track_running_stats=True)

        def forward(self, x):
            return self.bn(x)

    pt_module = Module()
    pt_module.eval()
    dace_module = Module()
    dace_module.eval()

    dace_module.load_state_dict(pt_module.state_dict())

    dace_module = DaceModule(dace_module)

    X = torch.randn(8, 3, 32, 32)
    pt_result = pt_module(X)
    dace_result = dace_module(X)

    torch_tensors_close("output", pt_result, dace_result)


if __name__ == "__main__":
    for training_mode in [True, False]:
        test_bn_standalone(training_mode=training_mode)
    test_bn_in_import()
