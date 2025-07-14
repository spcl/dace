"""
Batch Norm is the only op that has a shared name between inputs and outputs. Test that prepending "in_" and "out_" works
"""

import pytest

import torch
from torch import nn
from torch.nn import functional as F

import dace
import dace.libraries.onnx as donnx
from dace.frontend.python.module import DaceModule

from dace.testing.utils import torch_tensors_close


@pytest.mark.ort
def test_bn_standalone():
    @dace.program
    def test_bn_standalone(X: dace.float32[8, 3, 32, 32],
                           scale: dace.float32[3], B: dace.float32[3],
                           mean: dace.float32[3], var: dace.float32[3]):

        Y = dace.define_local([8, 3, 32, 32], dace.float32)
        donnx.ONNXBatchNormalization(X=X,
                                     scale=scale,
                                     B=B,
                                     input_mean=mean,
                                     input_var=var,
                                     Y=Y)
        return Y

    X = torch.randn(8, 3, 32, 32)
    scale = torch.randn(3)
    B = torch.randn(3)
    mean = torch.randn(3)
    var = torch.randn(3)
    dace_result = test_bn_standalone(X, scale, B, mean, var)

    pt_result = F.batch_norm(X, mean, var, scale, B)
    torch_tensors_close("output", pt_result, torch.from_numpy(dace_result))


@pytest.mark.ort
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
