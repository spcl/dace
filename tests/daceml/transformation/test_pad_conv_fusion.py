import copy
import tempfile
import os

import pytest
import onnx
import onnxsim

import torch
from torch import nn
from torch.nn import functional as F

from dace.libraries.onnx import ONNXModel
from dace.testing import torch_tensors_close
from dace.transformation.onnx import PadConvFusion


class Module(nn.Module):
    def __init__(self):
        super(Module, self).__init__()
        self.conv = nn.Conv2d(3, 6, kernel_size=(3, 3))

    def forward(self, x):
        x = F.pad(x, [2, 2])
        return self.conv(x)


def test_onnx_import_only():
    # this uncovered an unrelated bug, adding a test here
    torch_module = Module()
    input = torch.rand(4, 3, 24, 24)
    with tempfile.TemporaryDirectory() as dir_name:
        export_name = os.path.join(dir_name, "export.onnx")
        torch.onnx.export(torch_module,
                          input,
                          export_name,
                          opset_version=12,
                          do_constant_folding=False,
                          keep_initializers_as_inputs=True)
        onnx_model = onnx.load(export_name)

    dace_model = ONNXModel("pad_conv", onnx_model, onnx_simplify=False)


@pytest.mark.ort
def test_pad_conv_fusion(sdfg_name):
    torch_module = Module()
    input = torch.rand(4, 3, 24, 24)

    with tempfile.TemporaryDirectory() as dir_name:
        export_name = os.path.join(dir_name, "export.onnx")
        torch.onnx.export(torch_module,
                          input,
                          export_name,
                          opset_version=12,
                          do_constant_folding=False,
                          keep_initializers_as_inputs=True)
        onnx_model = onnx.load(export_name)

    onnx_model, _ = onnxsim.simplify(onnx_model,
                                     skip_fuse_bn=True,
                                     skipped_optimizers=['fuse_pad_into_conv'])
    dace_model = ONNXModel("pad_conv", onnx_model, onnx_simplify=False)

    assert dace_model.sdfg.apply_transformations(PadConvFusion) == 1

    torch_output = torch_module(input)
    dace_output = dace_model(input)

    torch_tensors_close("output", torch_output, dace_output)
