import pytest
import torch

from dace.library import change_default
from efficientnet_pytorch import get_model_params
from efficientnet_pytorch.model import MBConvBlock
from torch import nn

import dace.libraries.onnx as donnx
from dace.libraries.onnx.op_implementations import CudnnConvolution
from dace.frontend.python.module import DaceModule
from dace.testing import torch_tensors_close


@pytest.mark.gpu
def test_bn_cudnn(sdfg_name):
    with change_default(donnx.ONNXBatchNormalization, "cuDNN"):
        torch_bn = nn.BatchNorm2d(3).cuda()
        dace_bn = DaceModule(nn.BatchNorm2d(3).cuda(),
                             backward=True,
                             training=True)

        with torch.no_grad():
            dace_inputs = torch.rand(8, 3, 224, 224).cuda()
            torch_inputs = torch.clone(dace_inputs)

        dace_inputs.requires_grad = True
        torch_inputs.requires_grad = True

        dace_output = dace_bn(dace_inputs)
        torch_output = torch_bn(torch_inputs)

        torch_tensors_close("output", torch_output, dace_output)


def faster_groupconv(mod):
    # Choose depthwise convolution implementation
    def _choose_impl(module):
        for state in module.sdfg.nodes():
            for node in state.nodes():
                if isinstance(node, donnx.ONNXConv):
                    if node.group > 1:
                        print("using pytorch dwise gconv:", node.label)
                        node.backward_implementation = "PyTorch-dwise"

    mod.append_post_onnx_hook("faster_groupconv", _choose_impl)



if __name__ == "__main__":
    test_bn_cudnn("test_bn_cudnn")
